from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import opt_einsum
import math
import numpy as np
from tqdm import tqdm
from ILRMA_D_FMM import ILRMA
from plyfile import PlyData
from make_simu_2 import (
    mic_array_locs, mic_array_geometry, SOUND_POSITIONS,
    SOUND_SPEED, n_mics_per_array, ans_R_N, gpu_ans_R_N, room_size
)

def read_ply_to_tensor(file_paths:str, translations) -> torch.Tensor:
    ply_locs = []
    for file_path, translation in zip(file_paths, translations):
        translation_tensor = torch.tensor(translation, dtype=torch.float32)
        plydata = PlyData.read(file_path)
        # vertex要素からx, y, z座標を取得
        vertex = plydata['vertex']
        x = np.array(vertex['x'], dtype=np.float32)
        y = np.array(vertex['y'], dtype=np.float32)
        z = np.array(vertex['z'], dtype=np.float32)
        # (点数, 3) の配列に変換し、torch.Tensorにする
        coords = np.vstack((x, y, z)).T  # 座標データを結合
        tensor = torch.tensor(coords) + translation_tensor   # Tensorに変換
        ply_locs.append(tensor)
    return ply_locs

file_paths = [
    "/home/yoshiilab1/soturon/3dgs/ply_data/406ca340-e/point_cloud/iteration_30000/point_cloud.ply",
    "/home/yoshiilab1/soturon/3dgs/ply_data/d77b305c-5/point_cloud/iteration_30000/point_cloud.ply"
]
translates = [(2.0, 1.0, 0.0), (2.0, 3.0, 0.0)]

torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
device = "cuda"
D = 3
I = 4

# ply_locs = read_ply_to_tensor(file_paths, translates)
# for i, tensor in enumerate(ply_locs):
#     print(f"File {i+1}: Tensor shape = {tensor.shape}")
#     print(tensor)
# print(len(ply_locs))
# print(ply_locs[0])

def set_U_NIVnD_ply(file_paths, translates):
    ply_locs = read_ply_to_tensor(file_paths, translates)
    U_NIVnD = []
    for R_N in ply_locs:
        array_locs = torch.tensor(mic_array_locs).to(device)  # (D+1, I)
        # R_N を I 次元に拡張
        R_N_expanded = R_N.expand(array_locs.shape[1], -1, -1)  # (I, N, D)
        # U_IND 計算
        U_IVnD = R_N_expanded - array_locs.T[:, None, :D]  # (I, N, D)
        # 回転行列を適用
        theta = -torch.deg2rad(array_locs[D])  # (I)
        R_mat = torch.zeros((I, D, D), dtype=torch.float64).to(device)  # (I, D, D)
        R_mat[:, 0, 0] = torch.cos(theta)
        R_mat[:, 0, 1] = -torch.sin(theta)
        R_mat[:, 1, 0] = torch.sin(theta)
        R_mat[:, 1, 1] = torch.cos(theta)
        if D == 3:
            R_mat[:, 2, 2] = 1.0
        # U_IND に回転を適用
            U_IVnD = (R_mat[:, None, :D, :D] @ U_IVnD[..., None])[..., 0]  # squeezeを使用せず直接取り出す
        U_NIVnD.append(U_IVnD)
    return U_NIVnD

# U_NIQnD = set_U_NIQnD_ply(file_paths, translates)
# print(len(U_NIQnD[1]))
n_freq = 513

def get_steering_vec_ply(U_NIVnD, ply_locs, array_distance=False, gpu=True) -> torch.Tensor:
    steering_vec = []
    i = 0
    for U_IVnD in U_NIVnD:
        gpu_array_geometry = torch.tensor(mic_array_geometry).to(device)
        gpu_mic_array_locs = torch.tensor(mic_array_locs).to(device)
        array_geometry = torch.zeros((D, n_mics_per_array), dtype=torch.float64, device=device)
        array_geometry[:2, :] = gpu_array_geometry # (2, Mi)
        omega = 2 * torch.pi * torch.arange(n_freq, device=device)
        delay = - opt_einsum.contract("dm,ind->inm", array_geometry, U_IVnD) / SOUND_SPEED # (I, N, Mi)
        Q_IN = torch.linalg.norm(ply_locs[i][None,] - gpu_mic_array_locs.T[:,None,:3], axis=2)
        i += 1
        if array_distance:
            delay += Q_IN[:,:,None] / SOUND_SPEED # (I, N, Mi)
        else:
            steering_vec_n = torch.exp(-1j * opt_einsum.contract("f,inm->infm", omega, delay)) # (I, N, F, Mi)
            steering_vec.append(steering_vec_n)
    return steering_vec # (N, I, Vn, F, Mi)

ply_locs = read_ply_to_tensor(file_paths, translates)
U_NIVnD = set_U_NIVnD_ply(file_paths, translates)
steering_vec = get_steering_vec_ply(U_NIVnD, ply_locs, array_distance=False)
# print(len(steering_vec[0][1]))

def get_hat_G(steering_vec_ply):
    eps = 1e-3
    B_NIVnFMi = steering_vec_ply
    for B_IVnFMi in B_NIVnFMi:
        hat_G_beta = torch.einsum("infa,infb->infab", B_IVnFMi, B_IVnFMi.conj()) + eps * torch.eye(n_mics_per_array)[None, None, None, ...]
    return hat_G_beta

def compute_scm(file_paths, translates, n_freq, n_mics, n_mics_per_array, weights=None, eps=1e-3):
    # 点群データの読み込み
    U_NIVnD = set_U_NIVnD_ply(file_paths, translates)  # (N_obj, I, Qn, D)
    ply_locs = read_ply_to_tensor(file_paths, translates)

    # 指向性ベクトルの計算
    steering_vec = get_steering_vec_ply(U_NIVnD, ply_locs)  # (N_obj, I, Qn, F, Mi)

    # SCM計算: einsumによる点ごとのSCMを計算
    scm_all = torch.zeros((len(steering_vec), n_freq, n_mics, n_mics), dtype=torch.complex128, device="cuda")
    for i, steering_vec_i in enumerate(steering_vec):
        # steering_vec_i: (I, Vn, F, Mi)
        hat_G_INFMiMi = torch.einsum(
            "iqfm,iqfn->iqfmn", steering_vec_i, steering_vec_i.conj()
        )  # (I, Qn, F, Mi, Mi)
        
        # 重み付け
        if weights is not None:
            weight = weights[i]
            weighted_hat_G = hat_G_INFMiMi * weight[:, None, None, None]  # (I, Vn, F, Mi, Mi)に重みを掛ける
        else:
            weighted_hat_G = hat_G_INFMiMi  # 重みなし
        
        # 各マイクアレイのブロック対角成分を計算
        for j in range(weighted_hat_G.shape[0]):  # I: マイクアレイ数
            start = n_mics_per_array * j
            end = n_mics_per_array * (j + 1)
            scm_all[i, :, start:end, start:end] = (
                weighted_hat_G[j].sum(dim=0) / steering_vec_i.shape[1]
            )  # Vnで平均化

        # 対角安定化
        scm_all[i] += eps * torch.eye(n_mics, device="cuda")[None, :, :]

    return scm_all  # (N_obj, F, M, M)

def compute_scm_einsum(file_paths, translates, n_freq, n_mics, n_mics_per_array, weights=None, eps=1e-3):
    # 点群データの読み込み
    U_NIVnD = set_U_NIVnD_ply(file_paths, translates)  # (N_obj, I, Qn, D)
    ply_locs = read_ply_to_tensor(file_paths, translates)

    # 指向性ベクトルの計算
    steering_vec = get_steering_vec_ply(U_NIVnD, ply_locs)  # (N_obj, I, Qn, F, Mi)

    # SCM計算
    scm_all = torch.zeros((len(steering_vec), n_freq, n_mics, n_mics), dtype=torch.complex128, device="cuda")
    
    for i, steering_vec_i in enumerate(steering_vec):
        # steering_vec_i: (I, Vn, F, Mi)
        # einsumを使ってSCMを計算し、重み付けを加えます。
        # 各マイクアレイごとのブロック対角行列を一度に計算する
        hat_G_INFMiMi = torch.einsum(
            "iqfm,iqfn->iqfmn", steering_vec_i, steering_vec_i.conj()
        )  # (I, Vn, F, Mi, Mi)
        
        # 重みを掛けて加算する。weightsがNoneの場合はそのまま、ある場合は重みを掛ける。
        if weights is not None:
            weight = weights[i]
            # einsumで重みを掛けて加算する
            weighted_hat_G = torch.einsum(
                "iqfmn,v->iqfmn", hat_G_INFMiMi, weight
            )  # (I, Vn, F, Mi, Mi)に重みを掛ける
        else:
            weighted_hat_G = hat_G_INFMiMi  # 重みなし
        
        # SCMにブロック対角成分を加算
        scm_all[i] = torch.einsum(
            "iqfmn->qfmm", weighted_hat_G.sum(dim=1) / steering_vec_i.shape[1]
        )  # Vnで平均化し、ブロック対角行列を作成
        
        # 対角安定化
        scm_all[i] += eps * torch.eye(n_mics, device="cuda")[None, :, :]
    
    return scm_all  # (N_obj, F, M, M)

def compute_scm_with_inplace_addition(file_paths, translates, n_freq, n_mics, n_mics_per_array, weights=None, eps=1e-3):
    U_NIVnD = set_U_NIVnD_ply(file_paths, translates)  # (N_obj, I, Vn, D)
    ply_locs = read_ply_to_tensor(file_paths, translates)
    steering_vec = get_steering_vec_ply(U_NIVnD, ply_locs)  # (N_obj, I, Vn, F, Mi)
    scm_all = torch.zeros((len(steering_vec), n_freq, n_mics, n_mics), dtype=torch.complex128, device="cuda")
    
    # if weights is not None:
    #     weights = torch.nn.utils.rnn.pad_sequence(weights, batch_first=True, padding_value=0).to("cuda")
    # for i, w in enumerate(weights):
    #     print(f"padded_weights[{i}].shape: {w.shape}")

    # steering_vec に対して逐次的に計算を行う
    for i, steering_vec_i in enumerate(steering_vec):
        # それぞれのマイクアレイに対して計算を行う
        for j in range(steering_vec_i.shape[0]):  # I: マイクアレイ数
            # steering_vec_i[j]: (Vn, F, Mi) (各マイクアレイに対応するデータ)
            part_result = torch.einsum("qfm,qfn->qfmn", steering_vec_i[j], steering_vec_i[j].conj())  # (Vn, F, Mi, Mi)
            # 重み付け
            if weights is not None:
                weight_tensor = weights[i].to("cuda")
                print(f"weight_tensor.shape: {weight_tensor.shape}")
                part_result *= weight_tensor[:, None, None, None]
            # 点群数で加算
            part_result = part_result.sum(dim=0)  # (F, Mi, Mi)

            # # 重み付け
            # if weights is not None:
            #     part_result *= weights[i, j][:, None, None]  # 重みを掛ける (F次元で拡張)

            # SCMの更新
            start = n_mics_per_array * j
            end = n_mics_per_array * (j + 1)
            scm_all[i, :, start:end, start:end] += part_result / steering_vec_i.shape[1]  # Vnで平均化

            # メモリ解放
            torch.cuda.empty_cache()

    # 対角安定化
    scm_all += eps * torch.eye(n_mics, device="cuda")[None, :, :]

    return scm_all  # (N_obj, F, M, M)

# 使用例
# n_mics = I * n_mics_per_array  # 総チャネル数 (例: 4 * 3 = 12)
# scm_all = compute_scm_with_inplace_addition(file_paths, translates, n_freq, n_mics, n_mics_per_array)
# print(f"SCM shape: {scm_all.shape}")

# # 使用例
# n_mics = I * n_mics_per_array  # 総チャネル数 (例: 4 * 3 = 12)
# scm_all = compute_scm_with_inplace_addition(file_paths, translates, n_freq, n_mics, n_mics_per_array)
# print(f"SCM shape: {scm_all.shape}")

# hat_G = get_hat_G(steering_vec)
# print(len(hat_G))

# 例としてランダムな重み行列を生成


# # 重みの生成: (N_obj, I, F) の形状
# weights = torch.rand(2, I, n_freq, device="cuda")  # [0, 1]の範囲でランダムな重み

# 関数の呼び出し
n_mics = n_mics_per_array * I
# scm_all = compute_scm_with_inplace_addition(file_paths, translates, n_freq, n_mics=n_mics, n_mics_per_array=n_mics_per_array, weights=weights)

# 結果を表示
# print(f"SCM shape: {scm_all.shape}")

ply_locs = read_ply_to_tensor(file_paths, translates)
# print(len(ply_locs[0]))

weights = []
for i in range(len(ply_locs)):
    weight = torch.rand(len(ply_locs[i]), device="cuda")
    weight /= weight.sum()  # 正規化
    weights.append(weight)

# print(len(weights[1]))
# print(weights)
for i, w in enumerate(weights):
    print(f"weights[{i}].shape: {w.shape}")


G_hat_NFMM = compute_scm_with_inplace_addition(file_paths, translates, n_freq, n_mics, n_mics_per_array, weights=weights)
print(f"SCM.shape : {G_hat_NFMM.shape}")