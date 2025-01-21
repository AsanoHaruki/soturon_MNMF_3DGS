import math
import numpy as np
import datetime
import soundfile as sf
import matplotlib.pyplot as plt
import pickle as pic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plot_fig
import matplotlib.animation as animation
import mir_eval
plt.style.use('ud.mplstyle')
import matplotlib.pyplot as plt
from pathlib import Path
# import IPython.display as ipd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plyfile import PlyData
from tqdm  import tqdm
from ILRMA_D_FMM import ILRMA
from ILRMA_Su import ILRMA_Su
from make_simu_2_Sumura_corner import (
    mic_array_locs, mic_array_geometry, SOUND_POSITIONS,
    SOUND_SPEED, n_mics_per_array, ans_R_N, gpu_ans_R_N, room_size
)

import opt_einsum
from matplotlib import rc
rc('animation', html='html5')  # HTML5ビデオ用の設定
rc('animation', writer='imagemagick', codec='gif')  # ImageMagickを指定

D = 3 # ３次元で実装eps = 1e-5
        # # print(f"n_source = {self.n_source}")
        # # print(f"n_mic = {self.n_mic}")
        # G_size = [self.n_source, self.n_freq, self.n_mic, self.n_mic]
        # G_NFMM = torch.zeros(G_size, dtype=torch.complex128)
        # G_NFMM[:, :] = torch.eye(self.n_mic)
        # # print(f"zero_G_NFMM : {G_NFMM.shape}")
        # print("ILRMA")
        # ilrma = ILRMA(n_basis=16)
        # ilrma.initialize()
        # for _ in range(70):
        #     ilrma.update()
        # ilrma.separate()
        # print("ILRMA finished")
        # # a_FMMは混合行列print(f"R_N.shape : {self.gpu_R_N.shape}")
        # # 分離行列の逆行列を計算することで混合行列を得る。
        # a_FNM = torch.linalg.inv(ilrma.D_FMM)
        # # print(f"a_FNM.shape : {a_FNM.shape}")
        # #! ILRMAは決定系。この辺で次元の違いが生まれてる？
        # #! ILRMAは16マイク16音源で動かしてる
        # separated_spec_power = torch.abs(ilrma.Z_FTN_ILRMA).mean(axis=(0, 1))
        # # print(f"separated_spec_power.shape : {separated_spec_power.shape}")
        # # separated_spec_power = torch.abs(ilrma.Z_FTN).mean(axis=(0, 1))
        # max_index = torch.argsort(separated_spec_power, descending=True)[:2]    #? [:2]を書き足した
        # G_NFMM = torch.einsum('fmn,fln->nfml',a_FNM,a_FNM.conj())
        # # if not G_NFMM.shape == G_size:
        # #     print("G_NFMM.shape is not (N,F,M,M)")  #! shapeが想定と違かったら表示
        # G_NFMM = G_NFMM[max_index]
        # # print(f"G_NFMM.shape : {G_NFMM.shape}") #! ここではもうshapeがおかしい
        # # self.G_NFMM = hat_G_NFMM.to(device)
        # G_NFMM = (G_NFMM + G_NFMM.conj().transpose(-2, -1)) / 2
        # G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=device)  # エルミート化
        # del ilrma
    # return device
def torch_setup():
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
        device = "mps"
    else:
        torch.set_default_device("cpu")
        device = "cpu"
    print(f"Using device: {device}")
    return device

class STFT:
    def __init__(self) -> None:
        self.n_fft = 1024
        self.hop_length = self.n_fft // 4
        self.window = torch.hann_window(self.n_fft)

    def stft(self, wav: torch.tensor):
        # wave: (B, L) => spec:(B, F, T)
        # B: batch size, F: frequency, T: time
        return torch.stft(wav,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          window=self.window,
                          return_complex=True)

    def istft(self, spec: torch.tensor):
        # spec:(B, F, T) => wave: (B, L)
        # B: batch size, F: frequency, T: time
        n_freq = spec.shape[0]
        spec = spec.permute(2, 0, 1)
        hop_length = (n_freq-1) // 2
        tmp = torch.istft(spec,
                          n_fft=self.n_fft,
                          hop_length=hop_length,
                          window=self.window)
        return tmp / tmp.abs().max()
    
today = datetime.datetime.now()
timestamp = today.strftime('%Y%m%d_%H%M')
data_root = Path("self_data")
save_root = Path(f"result/{timestamp}")
save_root.mkdir(exist_ok=True)
data_file = data_root / "mixture_time_domain_2_corner.wav"
data, samplerate = sf.read(data_file)
data = data - data.mean()
data = torch.tensor(data.T)
stft_tool = STFT()
x_stft = stft_tool.stft(data).permute(1, 2, 0)
x_stft[x_stft.abs() < 1e-15] = 1e-15
x_stft = x_stft / x_stft.abs().max()

def create_gif():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, room_size[0])
    ax.set_ylim(0, room_size[1])
    ax.set_xticks(np.arange(0, room_size[0] + 0.1, 1))
    ax.set_yticks(np.arange(0, room_size[1] + 0.1, 1))
    ax.grid()
    mic_array_x = mic_array_locs[0, :]
    mic_array_y = mic_array_locs[1, :]
    ax.scatter(mic_array_x, mic_array_y, color="black", label="Mic Arrays", s=15)
    ax.scatter(mnmf.R_N_init[:, 0].cpu(), mnmf.R_N_init[:, 1].cpu(), 
               marker="*", color="blue", label="Initial Sources", s=25)
    ax.scatter(ans_R_N[:, 0], ans_R_N[:, 1], 
               marker="x", color="red", label="Ground Truth", s=60)
    scat = ax.scatter([], [], marker="D", color="green", label="Estimated Sources", s=25)
    ax.legend()

    def gifupdate(frame):
        scat.set_offsets(frame[:, :2])  # Ensure tensor is converted
        return scat,
    anim = animation.FuncAnimation(
        fig, gifupdate, frames=mnmf.gif_positions, blit=True)
    gif_filename = save_root / f"R_N_optimization.gif"
    try:
        anim.save(gif_filename, writer="imagemagick", fps=2)
        print(f"GIF saved at {gif_filename}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")

def multi_log_prob_complex_normal(x: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor):
    d = x.shape[-1]
    return - 1/2 * torch.slogdet(Sigma)[1] - 1/2 * (x-mu)[..., None, :].conj() @ torch.linalg.inv(Sigma) @ (x-mu)[..., None] - d/2 * math.log(2*math.pi)


def multi_log_prob_invwishart(S: torch.Tensor, nu: int, Phi: torch.Tensor):
    """
    return log inverse Wishart distribution W^-1(S | nu, Phi)
    """
    d = S.shape[-1]
    # return nu/2 * torch.slogdet(Phi)[1] - (nu+d+1)/2 * torch.slogdet(S)[1] + trace(- 1/2 * Phi @ torch.linalg.inv(S))
    # return nu/2 * torch.slogdet(Phi)[1] - (nu*d)/2*math.log(2) - torch.special.multigammaln(torch.tensor(nu/2.), d) - (nu+d+1)/2. * torch.slogdet(S)[1] -1/2. * opt_einsum.contract("...mm->...", Phi @ torch.linalg.inv(S))
    return nu/2 * torch.slogdet(Phi.real)[1] - (nu*d)/2*math.log(2) - torch.special.multigammaln(torch.tensor(nu/2.), d) - (nu+d+1)/2. * torch.slogdet(S.real)[1] -1/2. * torch.einsum("...mm->...", Phi.real @ torch.linalg.inv(S.real))


class MNMF:
    # コンストラクタ　X信号 F周波数 Tフレーム Mマイク数
    def __init__(self, X_FTM, n_source=1, n_basis=8):
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        # print(f"n_mic : {self.n_mic}")
        self.X_FTM = X_FTM
        # x_stft:(F, T, M) => xx_stft:(F, T, M, M)
        self.XX_FTMM = torch.einsum('ftm,ftn->ftmn',
                                    self.X_FTM, self.X_FTM.conj())
        self.n_source = n_source
        self.n_basis = n_basis  # 分解時の基底数
        self.mic_array_locs = mic_array_locs  # クラスに mic_array_locs を保持
        self.I = int(self.n_mic / n_mics_per_array)
        self.method = 'MNMF'
        self.eps = 1e-12
        self.l1 = []
        self.l2 = []
        self.loss = []
        self.R_distance = {n: [] for n in range(self.n_source)}  # 音源ごとの距離を保存
        self.gif_positions = []
        self.l1_when_wh_updated = []
        self.l2_when_wh_updated = []
        self.loss_when_wh_updated = []
        self.sdr_list = []
        self.D = 3
        self.device = torch_setup()
        self.file_paths = [
            "/home/yoshiilab1/soturon/3dgs/ply_data/406ca340-e/point_cloud/iteration_30000/point_cloud.ply",
            "/home/yoshiilab1/soturon/3dgs/ply_data/2ebe532d-0/point_cloud/iteration_30000/point_cloud.ply"
        ]
        self.ply_translate =[(2.0, 6.0, 0.0), (6.0, 2.0, 0.0)]
        self.ply_rotation_angles = [315, 135]

    # パラメータの初期化
    def initialize(self, G_init_mode, eta_init_mode):
        self.initialize_R(init_loc=SOUND_POSITIONS)
        self.initialize_WH()
        self.initialize_G(init_mode=G_init_mode)
        # self.normalize_WHG()
        self.normalize_WH()
        self.read_ply_to_tensor()
        self.initialize_eta(init_mode=eta_init_mode)   #! ply用に書き足し
        self.plot_ply_eta(file_name="initial")

    def multiplicative_update_WH(self):
        # self.update_aux()
        self.update_WH()

    def update_G(self):
        self.L_optim.zero_grad()
        l1 = -self.log_prob_X()
        l2 = -self.log_prob_SCM()
        self.l1.append(l1.item())
        self.l2.append(l2.item())
        loss = (l1 + self.alpha*l2)
        self.loss.append(loss.item())
        loss.backward()
        self.L_optim.step()
    
    def update_R(self):
        self.R_optim.zero_grad()
        l2 = -self.log_prob_SCM(step=2)  # SCM損失計算
        self.l2.append(l2.item())
        l2.backward()
        self.R_optim.step()


    def initialize_R(self, init_loc=None,  ans_R_N=True, random_mode=False, noise_std=0.7, seed=6):
        if random_mode:
            # Generate random positions within the room
            rng = np.random.default_rng(seed)  # Ensure reproducibility
            random_positions = [[4., 3., 1.6]]
            self.R_N = torch.tensor(random_positions, dtype=torch.float64)
            self.R_N_init = torch.tensor(random_positions, dtype=torch.float64)
        elif init_loc is not None:
            # Generate positions near init_loc with noise in x, y directions only
            assert init_loc.shape == (self.n_source, D)  # (n_source, D)
            rng = np.random.default_rng()  # Ensure reproducibility
            noise_xy = rng.normal(0, noise_std, (self.n_source, 2))  # Shape: (N, 2)
            noisy_positions = init_loc.copy()
            noisy_positions[:, :2] += noise_xy  # Add noise to x, y
            # noisy_positions = [[4., 3., 1.6]]
            self.R_N = torch.tensor(noisy_positions, dtype=torch.float64)
            self.R_N_init = torch.tensor(noisy_positions, dtype=torch.float64)
        # self.R_N_init = self.R_N_init.detach()
        self.ans_R_N = ans_R_N
        self.gpu_ans_R_N = gpu_ans_R_N
        self.gpu_R_N = self.R_N.to(self.device)
        self.gpu_R_N.requires_grad_(True)

    def steering_vector(self, array_distance=False):
        U_IND = self.set_U_IND(gpu=True)
        return self.get_steering_vec(U_IND=U_IND, array_distance=array_distance, gpu=True)
    
    def set_U_IND(self, gpu=True):
        '''calculate U_IND from current R_N'''
        R_N = self.gpu_R_N
        array_locs = torch.tensor(mic_array_locs).to(self.device)  # (D+1, I)
        # R_N を I 次元に拡張
        R_N_expanded = R_N.expand(array_locs.shape[1], -1, -1)  # (I, N, D)
        # U_IND 計算
        U_IND = R_N_expanded - array_locs.T[:, None, :D]  # (I, N, D)
        # 回転行列を適用
        theta = -torch.deg2rad(array_locs[D])  # (I)
        R_mat = torch.zeros((self.I, D, D), dtype=torch.float64).to(self.device)  # (I, D, D)
        R_mat[:, 0, 0] = torch.cos(theta)
        R_mat[:, 0, 1] = -torch.sin(theta)
        R_mat[:, 1, 0] = torch.sin(theta)
        R_mat[:, 1, 1] = torch.cos(theta)
        if D == 3:
            R_mat[:, 2, 2] = 1.0
        # U_IND に回転を適用
            U_IND = (R_mat[:, None, :D, :D] @ U_IND[..., None])[..., 0]  # squeezeを使用せず直接取り出す
        return U_IND

    def get_steering_vec(self, U_IND, array_distance=False, gpu=True) -> torch.Tensor:
        self.gpu_array_geometry = torch.tensor(mic_array_geometry).to(self.device)
        self.mic_array_locs = torch.tensor(self.mic_array_locs).to(self.device)
        array_geometry = torch.zeros((D, n_mics_per_array), dtype=torch.float64, device=self.device)
        array_geometry[:2, :] = self.gpu_array_geometry # (3, Mi)
        omega = 2 * torch.pi * torch.arange(self.n_freq, device=self.device)
        delay = - torch.einsum("dm,ind->inm", array_geometry, U_IND) / SOUND_SPEED # (I, N, Mi)
        Q_IN = torch.linalg.norm(self.gpu_R_N[None,] - self.mic_array_locs.T[:,None,:3], axis=2)
        if array_distance:
            delay += Q_IN[:,:,None] / SOUND_SPEED # (I, N, Mi)
        else:
            steering_vec = torch.exp(-1j * torch.einsum("f,inm->infm", omega, delay)) # (I, N, F, Mi)
            return steering_vec

    # WHのランダム初期化
    def initialize_WH(self):
        W_size = [self.n_source, self.n_freq, self.n_basis]
        H_size = [self.n_source, self.n_basis, self.n_time]
        # self.W_NFK = torch.rand(W_size).detach()
        self.W_NFK = torch.rand(W_size)
        # Wの正規化条件に合わせて，周波数ごとに正規化
        self.W_NFK = self.W_NFK / self.W_NFK.sum(dim=1)[:, None]
        # self.H_NKT = torch.rand(H_size).detach()
        self.H_NKT = torch.rand(H_size)
        self.W_NFK = self.W_NFK.to(torch.complex128)
        self.H_NKT = self.H_NKT.to(torch.complex128)
        self.lambda_NFT = self.W_NFK @ self.H_NKT
        # print(f"lambda.shape : {self.lambda_NFT.shape}") #? (2,513,231)だった。想定通りのshape。確認するのはGかな？

    # 空間共分散行列Gを須村論文式(6)のように初期化。(mnmfの初期化方法)
    def initialize_G_beta(self):
        eps = 1e-5
        G_size = [self.n_source, self.n_freq, self.n_mic, self.n_mic]
        G_NFMM = torch.zeros(G_size, dtype=torch.complex128)
        G_NFMM[:, :] = torch.eye(self.n_mic)
        print("ILRMA")
        ilrma = ILRMA(n_basis=16)
        ilrma.initialize()
        for _ in range(70):
            ilrma.update()
        ilrma.separate(mic_index=0)
        print("ILRMA finished")
        # a_FMMは混合行列print(f"R_N.shape : {self.gpu_R_N.shape}")
        # 分離行列の逆行列を計算することで混合行列を得る。
        a_FNM = torch.linalg.pinv(ilrma.D_FMM)
        # print(f"a_FNM.shape : {a_FNM.shape}")
        separated_spec_power = torch.abs(ilrma.Z_FTN_ILRMA).mean(axis=(0, 1))
        
        for n in range(self.n_source):
            # a_FNMを使ってG_NFMMを計算
            G_NFMM[n, :, :, :] = torch.einsum('fm,fl->fml', 
                                            a_FNM[:, :, separated_spec_power.argmax()], 
                                            a_FNM[:, :, separated_spec_power.argmax()].conj())
            # 最もパワーが大きい成分を除外して次に進む
            separated_spec_power[separated_spec_power.argmax()] = 0
        
        #? 一旦これじゃないやり方でやってみる
        # max_index = torch.argsort(separated_spec_power, descending=True)[:2]
        # G_NFMM = torch.einsum('fmn,fln->nfml',a_FNM,a_FNM.conj())
        # G_NFMM = G_NFMM[max_index]
        
        # for n in range(self.n_source):
        #         G_NFMM[n, :, :, :] = torch.einsum('fm,fl->fml', 
        #                             a_FNM[:, :, separated_spec_power.argmax()], 
        #                             a_FNM[:, :, separated_spec_power.argmax()].conj())
        #         # ↓すでに処理した音源を次回のループで再度選ばれないようにするために0にしておく
        #         separated_spec_power[separated_spec_power.argmax()] = 0
        
        G_NFMM = (G_NFMM + G_NFMM.conj().transpose(-2, -1)) / 2
        G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)  # エルミート化
        del ilrma
        torch.clear_autocast_cache() #不要なキャッシュを削除
        self.gpu_L_NFMM = torch.linalg.cholesky(G_NFMM)
        self.gpu_L_NFMM = torch.log(self.gpu_L_NFMM).detach()
        self.gpu_L_NFMM.requires_grad_(True)
        
    def initialize_G(self, init_mode='ILRMA'):
        eps = 1e-5
        G_size = [self.n_source, self.n_freq, self.n_mic, self.n_mic]
        G_NFMM = torch.zeros(G_size, dtype=torch.complex128)
        G_NFMM[:, :] = torch.eye(self.n_mic)
        
        if init_mode == 'ILRMA':
            print("ILRMA")
            ilrma = ILRMA(n_basis=16)
            ilrma.initialize()
            for _ in range(70):
                ilrma.update()
            ilrma.separate(mic_index=0)
            print("ILRMA finished")
            a_FNM = torch.linalg.pinv(ilrma.D_FMM)
            separated_spec_power = torch.abs(ilrma.Z_FTN_ILRMA).mean(axis=(0, 1))
            for n in range(self.n_source):
                G_NFMM[n, :, :, :] = torch.einsum('fm,fl->fml', 
                                                a_FNM[:, :, separated_spec_power.argmax()], 
                                                a_FNM[:, :, separated_spec_power.argmax()].conj())
                separated_spec_power[separated_spec_power.argmax()] = 0
            G_NFMM = (G_NFMM + G_NFMM.conj().transpose(-2, -1)) / 2
            G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)  # エルミート化
            del ilrma
            torch.clear_autocast_cache() #不要なキャッシュを削除
            
        elif init_mode == 'random':
            print("random initialize")
            random_matrix = torch.randn(self.n_source, self.n_freq, self.n_mic, self.n_mic, dtype=torch.complex128)
            G_NFMM = torch.matmul(random_matrix, random_matrix.conj().transpose(-2, -1))
            # print(G_NFMM)
            G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)  # エルミート化
            torch.clear_autocast_cache()
            
        elif init_mode == 'ILRMA_Su':
            print("ILRMA")
            ilrma = ILRMA_Su()
            stft_tool = STFT()
            ilrma.import_obsdata(self.X_FTM.to(self.device), torch.tensor(self.n_freq).to(self.device))
            ilrma.initialize_WH(n_basis=16)
            ilrma.initialize_D()
            for i in range(50):
                ilrma.update_WH()
                ilrma.update_D()
                ilrma.normalize_WHD()
            print("ILRMA finished")
            spoint, epoint = 0.1, 0.7
            A_FMN = torch.linalg.inv(ilrma.gpu_D_FNM)
            separated_spec_power = torch.abs(ilrma.separate()[:, int(ilrma.T*spoint):int(ilrma.T*epoint), :]).mean(dim=(0, 1))
            n_args = torch.zeros((self.n_source), dtype=torch.int64, device=self.device)
            Z_FTN = ilrma.separate()
            
            print(f"Z_FTN.shape: {Z_FTN.shape}")
            print(f"X_FTM.shape: {self.X_FTM.shape}")
            X_FTM_flat = self.X_FTM.reshape(self.X_FTM.shape[0] * self.X_FTM.shape[1], self.X_FTM.shape[2]).to(self.device)
            Z_FTN_flat = Z_FTN.reshape(Z_FTN.shape[0] * Z_FTN.shape[1], Z_FTN.shape[2]).to(self.device)
            
            inner_product = torch.sum(Z_FTN_flat * X_FTM_flat.conj(), dim=1)
            norm_Z = torch.norm(Z_FTN_flat, dim=1)
            norm_X = torch.norm(X_FTM_flat, dim=1)
            
            true_data_1 = "data/arctic_a0002.wav"
            true_data_2 = "data/arctic_b0540.wav"
            data_1, samplerate_1 = sf.read(true_data_1)
            data_2, samplerate_2 = sf.read(true_data_2)
            data_1 = data_1 - data_1.mean()
            data_2 = data_2 - data_2.mean()
            max_length = max(len(data_1), len(data_2))
            data_1 = torch.tensor(data_1.T)
            data_2 = torch.tensor(data_2.T)
            data_1 = torch.nn.functional.pad(data_1, (0, max_length - len(data_1)))
            data_2 = torch.nn.functional.pad(data_2, (0, max_length - len(data_2)))
            x_stft_true_1 = stft_tool.stft(data_1)
            x_stft_true_2 = stft_tool.stft(data_2)
            x_stft_true = torch.cat([x_stft_true_1.unsqueeze(-1), x_stft_true_2.unsqueeze(-1)], dim=-1)
            print(f"x_stft_true.shape: {x_stft_true.shape}")

            arg_idx_list = []
            for i in range(self.n_source):
                # x_stft_trueのi番目の音源のスペクトルをフラットにする
                x_stft_true_i = x_stft_true[:, :, i].reshape(-1).to(self.device)
                x_stft_true_i = x_stft_true_i.unsqueeze(0)
                print(f"x_stft_true_i: {x_stft_true_i.shape}")
                # Z_FTNの各音源との内積とノルムを計算
                cosine_similarities = []
                for j in range(Z_FTN.shape[-1]):  # N個の音源
                    Z_FTN_j = Z_FTN[:, :, j].reshape(-1).to(self.device)
                    Z_FTN_j = Z_FTN_j.unsqueeze(0)
                    # Z_FTN_j と x_stft_true_i の形状を確認・一致させる
                    F_true, T_true = x_stft_true_i.shape
                    F_Z, T_Z = Z_FTN_j.shape

                    # 時間フレーム数を合わせる (パディングまたはリサンプリング)
                    if T_true < T_Z:
                        padding = T_Z - T_true
                        x_stft_true_i = torch.nn.functional.pad(x_stft_true_i, (0, padding))
                    elif T_true > T_Z:
                        padding = T_true - T_Z
                        Z_FTN_j = torch.nn.functional.pad(Z_FTN_j, (0, padding))

                    # 周波数ビン数を合わせる (パディングまたはトリミング)
                    if F_true < F_Z:
                        padding = F_Z - F_true
                        x_stft_true_i = torch.nn.functional.pad(x_stft_true_i, (0, padding))
                    elif F_true > F_Z:
                        Z_FTN_j = Z_FTN_j[:F_true]
                    # 内積とノルム
                    inner_product = torch.sum(Z_FTN_j * x_stft_true_i.conj())
                    norm_Z = torch.norm(Z_FTN_j)
                    norm_X = torch.norm(x_stft_true_i)
                    # コサイン類似度
                    cosine_similarity = inner_product / (norm_Z * norm_X)
                    cosine_similarities.append(cosine_similarity)
                    cosine_similarities_abs = torch.abs(torch.tensor(cosine_similarities))
                # 最も類似度が高いインデックスを取得
                best_idx = torch.argmax(cosine_similarities_abs).item()
                arg_idx_list.append(best_idx)

            print(f"arg_idx_list: {arg_idx_list}")
            
            print(f"arg_idx_list: {arg_idx_list}")
            for n in range(self.n_source):
                G_NFMM[n] = A_FMN[:, :, arg_idx_list[n]][:, :, None] @ A_FMN[:, :, arg_idx_list[n]][:, None].conj() \
                    + 1e-2 * torch.eye(self.n_mic, dtype=torch.complex128, device=self.device)[None]
                n_args[n] = arg_idx_list[n]
            print(f"n_arg: {n_args}")
            self.W_NFK = ilrma.gpu_W_NFK[n_args, :, :].to(torch.complex128)
            self.H_NKT = ilrma.gpu_H_NKT[n_args, :, :].to(torch.complex128)
            print(f"W_NFK.shape: {self.W_NFK.shape}")
            print(f"H_NKT.shape: {self.H_NKT.shape}")
            torch.clear_autocast_cache()
            
        elif init_mode == 'GS':
            print("initialize G from 3DGS")
            self.set_U_NIVnD_ply()
            self.get_steering_vec_ply()
            G_NFMM = self.compute_hat_G_ply()
            G_NFMM = (G_NFMM + G_NFMM.conj().transpose(-2, -1)) / 2
            G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)  # エルミート化
            torch.clear_autocast_cache()
        
        print("self.W_NFK.dtype:", self.W_NFK.dtype)
        print("self.H_NKT.dtype:", self.H_NKT.dtype)
        self.gpu_L_NFMM = torch.linalg.cholesky(G_NFMM)
        self.gpu_L_NFMM = torch.log(self.gpu_L_NFMM).detach()
        self.gpu_L_NFMM.requires_grad_(True)
        
    def gather_N_from_ILRMA(self, output_N=2):
        
        spoint, epoint = 0.1, 0.7
        separated_spec_power = torch.abs(self.ilrma.separate()[:, int(self.ilrma.T*spoint):int(self.ilrma.T*epoint), :]).mean(dim=(0,1))
        n_args = torch.zeros((output_N))
        # hat_G_pNFMM = torch.zeros((output_N, self.F, self.M, self.M), dtype=torch.complex128) 
        for n in range(output_N):
            n_args[n] = separated_spec_power.argmax()
            separated_spec_power[separated_spec_power.argmax()] = 0
        return n_args.type(torch.int64)
    
    def update_WH(self):
        #update aux
        with torch.no_grad():
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        print(f"lambda.shape: {self.lambda_NFT.shape}")
        print(f"G_NFMM.shape: {G_NFMM.shape}")
        print("self.lambda_NFT.dtype:", self.lambda_NFT.dtype)
        print("G_NFMM.dtype:", G_NFMM.dtype)
        iY_FTMM = torch.einsum('nft,nfml->ftml',
                                    self.lambda_NFT, G_NFMM).inverse()
        Yx_FTM1 = iY_FTMM @ self.X_FTM[..., None]
        iY_X_iY_FTMM = Yx_FTM1 @ Yx_FTM1.conj().permute(0, 1, 3, 2)
        G_iY_X_iY_NFT = torch.einsum('nfab,ftbc->nftac',
                                        G_NFMM,
                                        iY_X_iY_FTMM)
        tr_G_iY_X_iY_NFT = torch.einsum('...ii', G_iY_X_iY_NFT).real
        G_iY_NFT = torch.einsum('nfab,ftbc->nftac',G_NFMM,iY_FTMM)
        tr_G_iY_NFT = torch.einsum('...ii', G_iY_NFT).real
        
        # 乗法更新
        a_1 = (self.H_NKT.permute(0, 2, 1)[
               :, None] * tr_G_iY_X_iY_NFT[:, :, :, None]).sum(axis=2)
        b_1 = (self.H_NKT.permute(0, 2, 1)[
               :, None] * tr_G_iY_NFT[:, :, :, None]).sum(axis=2)
        a_2 = (self.W_NFK[..., None] *
               tr_G_iY_X_iY_NFT[:, :, None]).sum(axis=1)
        b_2 = (self.W_NFK[..., None] *
               tr_G_iY_NFT[:, :, None]).sum(axis=1)
        self.W_NFK = self.W_NFK * torch.sqrt(a_1 / b_1)
        self.H_NKT = self.H_NKT * torch.sqrt(a_2 / b_2)
        self.normalize_WH()
    
    def log_prob_X(self) -> torch.Tensor:
        G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj())) 
        mu_NF = torch.einsum('...ii', G_NFMM).real
        G_NFMM = (G_NFMM / mu_NF[:, :, None, None])
        Y_FTMM = torch.einsum('nft,nfml->ftml', self.lambda_NFT, G_NFMM)
        iY_FTMM = Y_FTMM.inverse()
        iY_XX_FTMM = - torch.einsum('ftml,ftln->ftmn', iY_FTMM, self.XX_FTMM)
        tr_iY_XX_FTMM = torch.einsum('...ii', iY_XX_FTMM).real
        lk = (tr_iY_XX_FTMM + torch.log(torch.linalg.det(iY_FTMM).real)).sum()
        return lk

    def log_prob_SCM(self, step=1) -> torch.Tensor:
        eps = 1e-3
        nu = self.n_mic + 1
        B_INFMi = self.steering_vector() # R_Nに依存
        hat_G_INFMiMi = torch.einsum("infa,infb->infab", B_INFMi, B_INFMi.conj()) + eps * torch.eye(n_mics_per_array)[None, None, None, ...]
        gpu_hat_G_NFMM = torch.zeros((self.n_source, self.n_freq, self.n_mic, self.n_mic), dtype=torch.complex128)
        # ブロック対角生成
        for i in range(self.I):
            gpu_hat_G_NFMM[:, :, n_mics_per_array*i:n_mics_per_array*(i+1), n_mics_per_array*i:n_mics_per_array*(i+1)] = hat_G_INFMiMi[i]
        eps_I = 1e-6
        gpu_hat_G_NFMM[:,:] = gpu_hat_G_NFMM[:,:] + eps_I
        # スケール調整が必要
        tr_gpu_hat_G_NF = torch.einsum("nfmm->nf", gpu_hat_G_NFMM).real
        gpu_hat_G_NFMM = gpu_hat_G_NFMM / (tr_gpu_hat_G_NF[...,None,None] / self.n_mic) # tr(hatG) = M = n_mic
        # パワーに応じて重み付けして総和
        if step==2:
            with torch.no_grad():
                G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj())) 
                
        else:
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        mu_NF = torch.einsum('...ii', G_NFMM).real
        G_NFMM = (G_NFMM / mu_NF[:, :, None, None])
        log_p = torch.sum(multi_log_prob_invwishart(G_NFMM, nu, (nu+self.n_mic) * gpu_hat_G_NFMM)) # N, Fで総和
        return log_p.real
    
    def normalize_WHG(self):
        with torch.no_grad():
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj())) 
            mu_NF = torch.einsum('...ii', G_NFMM).real
            self.G_NFMM = (G_NFMM / mu_NF[:, :, None, None])
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps    
    
    def normalize_WH(self):
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps
        
    def separate(self, mic_index=0):
        G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        Omega_NFTMM = torch.einsum('nft,nfml->nftml',
                    self.lambda_NFT, G_NFMM)
        Omega_sum_inv_FTMM = torch.inverse(Omega_NFTMM.sum(dim=0))
        self.Z_FTN = torch.einsum('nftpq,ftqr,ftr->ftnp',
                                  Omega_NFTMM,
                                  Omega_sum_inv_FTMM,
                                  self.X_FTM)[..., mic_index]

    def optimize_WHGR(self, lr_l=1e-3, lr_r=1e-3, n_wh_update=500, n_g_update=100 ,n_R_update=10, gif_frames=50, alpha=1e-7):
        print("Start WH-G-R optimization loop")
        # 各種初期化
        torch.autograd.detect_anomaly(True)
        self.alpha = alpha
        self.n_R_update = n_R_update
        self.n_wh_update = n_wh_update
        self.R_optim = optim.Adam([self.gpu_R_N], lr=lr_r)
        self.L_optim = optim.Adam([self.gpu_L_NFMM], lr=lr_l)
        frame_interval = max(1, n_wh_update // gif_frames)  # フレーム間隔


        with tqdm(range(n_wh_update), desc='WH Updates', leave=True) as pbar_wh:
            for j in pbar_wh:
                self.multiplicative_update_WH()
                for k in range(n_g_update):
                    self.update_G()  # Gの更新
                for _ in range(n_R_update):
                    self.update_R()
                self.l1_when_wh_updated.append(self.l1[-1])
                self.l2_when_wh_updated.append(self.l2[-1])
                self.loss_when_wh_updated.append(self.loss[-1])
                for n in range(self.n_source):
                    R_dist = torch.norm(self.gpu_R_N[n].detach() - self.gpu_ans_R_N[n].to(self.device))
                    self.R_distance[n].append(R_dist.item())  # 各音源ごとの距離を保存
                
                # 損失ログと更新
                if j % frame_interval == 0 or j == n_wh_update - 1:
                        self.gif_positions.append(self.gpu_R_N.detach().cpu().numpy().copy())
        self.tortal_it = len(self.loss)
        create_gif()
        print("WH-G-R Optimization Complete")
        
    def calculate_l2_grid(self, room_size, grid_interval=0.1, height=1.6):
        # グリッド点の生成 (x, y 固定間隔で生成)
        x = torch.arange(0, room_size[0] + grid_interval, grid_interval, device=self.device)
        y = torch.arange(0, room_size[1] + grid_interval, grid_interval, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')  # 明示的にインデックス順を指定
        
        # 平面内のグリッド座標を構築
        R_N_grid = torch.stack([X, Y, torch.full_like(X, height)], dim=-1)  # (H, W, 3)
        
        # l2 を計算するためのテンソルを用意
        grid_shape = R_N_grid.shape[:2]
        l2_grid = torch.zeros(grid_shape)
        
        R_N_after_step1 = self.gpu_R_N.detach()
        # グリッド点ごとに l2 を計算
        print("Calculating l2 values for each grid point...")
        total_points = grid_shape[0] * grid_shape[1]
        with tqdm(total=total_points, desc="Processing Grid Points") as pbar:
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    # 各グリッド点に gpu_R_N を設定
                    # mnmf.gpu_R_N.data = torch.tensor(grid_point, dtype=torch.float64, device=device).unsqueeze(0)  # Shape: (1, 3)
                    self.gpu_R_N = R_N_grid[i, j].unsqueeze(0)
                    # l2 を計算
                    l2 = -mnmf.log_prob_SCM()
                    l2_grid[i, j] = l2.item()
                    # プログレスバーを更新
                    pbar.update(1)
    
        # l2 をカラーマップとしてプロット
        plt.figure(figsize=(6, 6))
        X_cpu = X.cpu().numpy()
        Y_cpu = Y.cpu().numpy()
        l2_grid_cpu = l2_grid.cpu().numpy()
        plt.contourf(X_cpu, Y_cpu, l2_grid_cpu, levels=50, cmap="viridis")
        plt.colorbar(label="l2 value")
        ans_R_N_xy = ans_R_N[:, :2]  # x, y 座標のみを取得
        self.gpu_R_N = R_N_after_step1[:, :2] # gpu_R_N を元に戻す
        # mic_array_locs を重ねてプロット（黒い点）
        mic_array_locs_xy = mic_array_locs[:2, :].T  # x, y 座標のみを取得して転置
        
        plt.scatter(mic_array_locs_xy[:, 0], mic_array_locs_xy[:, 1], color="black", marker="o", label="Mic Arrays", s=50)
        plt.scatter(ans_R_N_xy[:, 0], ans_R_N_xy[:, 1], color="white", marker="x", label="Ground Truth", s=100)
        plt.scatter(self.gpu_R_N[:, 0].cpu().numpy(), self.gpu_R_N[:, 1].cpu().numpy(), color="red", marker="x", label="Estimated", s=100)
        plt.title("30th WH heatmap l2 Values for Grid Points at Height {:.1f}m ILRMA_70times".format(height))
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True)
        plt.savefig(save_root / "30th_WH_l2_heatmap normalize_nothing ILRMA_70times.png")
        plt.show()
        
    def read_ply_to_tensor(self) -> torch.Tensor:
        """バイナリ形式のplyファイルから座標を平行移動し, Tensorとして読み込む
        Return: torch.tensor(N, V_n, 3)
        """
        self.ply_locs = []
        for file_path, translation, angle in zip(self.file_paths, self.ply_translate, self.ply_rotation_angles):
            translation_torch = torch.tensor(translation, dtype=torch.float32)
            plydata = PlyData.read(file_path)
            # vertex要素からx, y, z座標を取得
            vertex = plydata['vertex']
            x = np.array(vertex['x'], dtype=np.float32)
            y = np.array(vertex['y'], dtype=np.float32)
            z = np.array(vertex['z'], dtype=np.float32)
            coords = np.vstack((x, y, z)).T  # (点群数, 3) に変換
            theta = np.radians(angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            coords = coords @ rotation_matrix.T
            tensor = torch.tensor(coords) + translation_torch   # Tensorに変換
            
            # 1/3ごと
            N = tensor.shape[0]
            part_size = N // 4
            part1 = tensor[:part_size]  # 最初の1/3
            part2 = tensor[part_size:2*part_size]  # 2番目の1/3
            part3 = tensor[2*part_size:3*part_size]  # 最後の1/3
            part4 = tensor[3*part_size:]
                
            self.ply_locs.append(part1)
            # self.ply_locs.append(tensor)
            
            #? 関数内にパディングを追加
        #     max_len = max(max_len, len(tensor))
        # padded_ply_locs = []
        # for loc in self.ply_locs:
        #     padded_loc = torch.cat([loc, torch.zeros(max_len - len(loc), 3, device=self.device)], dim=0)
        #     padded_ply_locs.append(padded_loc)
        # self.ply_locs = torch.stack(self.ply_locs)
        # print(f"ply_locs.shape: {self.ply_locs.shape}")
        return self.ply_locs
    
    def set_U_NIVnD_ply(self):
        self.read_ply_to_tensor()
        self.U_NIVnD = []
        for R_N in self.ply_locs:
            array_locs = torch.tensor(mic_array_locs).to(self.device)  # (D+1, I)
            # R_N を I 次元に拡張
            R_N_expanded = R_N.expand(array_locs.shape[1], -1, -1)  # (I, N, D)
            # U_IND 計算
            U_IQnD = R_N_expanded - array_locs.T[:, None, :self.D]  # (I, N, D)
            # 回転行列を適用
            theta = -torch.deg2rad(array_locs[self.D])  # (I)
            R_mat = torch.zeros((self.I, self.D, self.D), dtype=torch.float64).to(self.device)  # (I, D, D)
            R_mat[:, 0, 0] = torch.cos(theta)
            R_mat[:, 0, 1] = -torch.sin(theta)
            R_mat[:, 1, 0] = torch.sin(theta)
            R_mat[:, 1, 1] = torch.cos(theta)
            if self.D == 3:
                R_mat[:, 2, 2] = 1.0
            # U_IND に回転を適用
                U_IVnD = (R_mat[:, None, :self.D, :self.D] @ U_IQnD[..., None])[..., 0]  # squeezeを使用せず直接取り出す
            self.U_NIVnD.append(U_IVnD)
    
    def get_steering_vec_ply(self, array_distance=False, gpu=True) -> torch.Tensor:
        self.steering_vec_ply = []
        i = 0
        for U_IVnD in self.U_NIVnD:
            gpu_array_geometry = torch.tensor(mic_array_geometry).to(self.device)
            gpu_mic_array_locs = torch.tensor(mic_array_locs).to(self.device)
            array_geometry = torch.zeros((D, n_mics_per_array), dtype=torch.float64, device=self.device)
            array_geometry[:2, :] = gpu_array_geometry # (2, Mi)
            omega = 2 * torch.pi * torch.arange(self.n_freq, device=self.device)
            delay = - opt_einsum.contract("dm,ind->inm", array_geometry, U_IVnD) / SOUND_SPEED # (I, N, Mi)
            Q_IN = torch.linalg.norm(self.ply_locs[i][None,] - gpu_mic_array_locs.T[:,None,:3], axis=2)
            i += 1
            if array_distance:
                delay += Q_IN[:,:,None] / SOUND_SPEED # (I, N, Mi)
            else:
                steering_vec_n = torch.exp(-1j * opt_einsum.contract("f,inm->infm", omega, delay)) # (I, N, F, Mi)
                self.steering_vec_ply.append(steering_vec_n)
        return self.steering_vec_ply # (N, I, Vn, F, Mi)
    
    def compute_hat_G_ply(self, weights=None, eps=1e-5):
        #! U_NIVnD, ply_locs, steering_vec_plyが用意されている状態を想定
        hat_G_ply_NFMM = torch.zeros((len(self.steering_vec_ply), self.n_freq, self.n_mic, self.n_mic), dtype=torch.complex128, device=self.device)
        # 重みベクトルをゼロパディング
        # if weights is not None:
        #     #? パディングした部分をマスク処理するための準備 使わないかも
        #     mask = (weights != 0).float()
        #? 重み係数の初期化の時点で長さを揃えておく形にする可能性もある パディングは一旦この関数内でやろう
        #? 重み係数のベクトルは正規化されている必要ありそう
        # if weights is not None:
        #     weights = torch.nn.utils.rnn.pad_sequence(weights, batch_first=True, padding_value=0).to(device)
        for i, steering_vec_i in enumerate(self.steering_vec_ply):
            max_length = max([s.shape[0] for s in steering_vec_i])
            padded_steering_vec_i = []
            for j in range(len(steering_vec_i)):
                current_len = steering_vec_i[j].shape[0]
                # padding_size = max_length - current_len
                # padded_steering_vec_i.append(torch.cat([steering_vec_i[j], torch.zeros(padding_size, *steering_vec_i[j].shape[1:], dtype=steering_vec_i[j].dtype, device=self.device)], dim=0))
                padded_tensor = torch.zeros((max_length, *steering_vec_i[j].shape[1:]), dtype=steering_vec_i[j].dtype, device=self.device)
                padded_tensor[:current_len] = steering_vec_i[j]
                padded_steering_vec_i.append(padded_tensor)
            
            for j in range(steering_vec_i.shape[0]):
                # steering_vec_i[j]: (Vn, F, Mi)    (1オブジェクトの各マイクアレイに対応)
                part_result = torch.einsum("vfm,vfn->vfmn", padded_steering_vec_i[j], padded_steering_vec_i[j].conj())    # (Vn,F,Mi,Mi)
                # print(f"part_result.shape: {part_result.shape}")
                # print(f"eta.shape: {weights.shape}")
                if weights is not None: 
                    weight_tensor = weights[i, :]
                    
                    padded_part_result = torch.zeros((weight_tensor.shape[0], *part_result.shape[1:]), dtype=part_result.dtype, device=self.device) #? メモリ落ち
                    padded_part_result[:part_result.shape[0]] = part_result
                    # part_resultのパディング
                    if part_result.shape[0] < weight_tensor.shape[0]:
                        padding_size = weight_tensor.shape[0] - part_result.shape[0]
                        part_result = torch.cat([part_result, torch.zeros(padding_size, *part_result.shape[1:], dtype=part_result.dtype, device=self.device)], dim=0)
                    # print(f"weight_tensor.shape: {weight_tensor.shape}")
                    # part_result *= weight_tensor[:, None, None, None]
                    part_result = torch.einsum('v,vfmn->vfmn', weight_tensor, part_result)
                else:
                    # 重みがない場合は単純平均
                    part_result = part_result / part_result.shape[0]
                    
                part_result = part_result.sum(dim=0)    # (F, Mi, Mi)
                start = n_mics_per_array * j
                end = n_mics_per_array * (j+1)
                hat_G_ply_NFMM[i, :, start:end, start:end] += part_result / steering_vec_i.shape[1]
                del part_result
                torch.cuda.empty_cache()
        hat_G_ply_NFMM += eps * torch.eye(self.n_mic, device=self.device)[None, :, :]
        return hat_G_ply_NFMM
    
    def initialize_eta(self, init_mode='random'):
        length = [len(ply) for ply in self.ply_locs]
        max_length = max(length)
        if init_mode == 'random':
            # self.eta = torch.stack([
            #     torch.rand(len(self.ply_locs[i]), device=self.device, requires_grad=True) 
            #     for i in range(len(self.ply_locs))])  #? ２次元テンソルとして定義
            # eta_list = [
            #     torch.rand(len(self.ply_locs[i]), device=self.device, requires_grad=True)
            #     for i in range(len(self.ply_locs))
            # ]
            eta_full = torch.rand((len(self.ply_locs), max_length), device=self.device)
        elif init_mode == 'constant':
            eta_full = torch.ones((len(self.ply_locs), max_length), device=self.device)
        # ゼロパディング
        for i, length in enumerate(length):
            eta_full[i, length:] = 0
        # 正規化
        # with torch.no_grad():
        #     self.eta = self.eta / self.eta.sum(dim=1, keepdim=True)
        eta_full = eta_full / eta_full.sum(dim=1, keepdim=True)
        self.eta = eta_full.clone().detach().requires_grad_(True)
        print(f"eta.shape: {self.eta.shape}")
    
    def log_prob_SCM_ply(self, step=1) -> torch.Tensor:
        # eps = 1e-3
        self.set_U_NIVnD_ply()
        self.get_steering_vec_ply()
        nu = self.n_mic + 1
        hat_G_ply_NFMM = self.compute_hat_G_ply(weights=self.eta)
        tr_hat_G_ply_NF = torch.einsum("nfmm->nf", hat_G_ply_NFMM).real
        hat_G_ply_NFMM /= tr_hat_G_ply_NF[...,None,None] / self.n_mic
        if step == 2:
            with torch.no_grad():
                G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        else:
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        mu_NF = torch.einsum("...ii", G_NFMM).real
        G_NFMM = (G_NFMM / mu_NF[:, :, None, None])
        log_p = torch.sum(multi_log_prob_invwishart(G_NFMM, nu, (nu+self.n_mic) * hat_G_ply_NFMM))
        return log_p
    
    def update_eta(self):
        self.eta_optim.zero_grad()
        l2 = -self.log_prob_SCM_ply(step=2)
        self.l2.append(l2.item())
        l2.backward()
        self.eta_optim.step()
    
    def train(self, lr_l=2e-3, lr_eta=2e-3, n_wh_update=500, n_g_update=100, n_eta_update=50, gif_frames=50, alpha=1e-4):
        print("start optimization")
        torch.autograd.detect_anomaly(True)
        self.alpha = alpha
        self.n_eta_update = n_eta_update
        self.n_wh_update = n_wh_update
        self.eta_optim = optim.Adam([self.eta], lr=lr_eta)
        self.L_optim = optim.Adam([self.gpu_L_NFMM], lr=lr_l)
        frame_interval = max(1, n_wh_update // gif_frames)
        
        with tqdm(range(n_wh_update), desc='WH Updates', leave=True) as pbar_wh:
            for j in pbar_wh:
                self.multiplicative_update_WH()
                self.separated_sound_eval_SDR(mic_index=0)
                for k in range(n_g_update):
                    self.update_G()
                for _ in range(n_eta_update):
                    self.update_eta()
                self.l1_when_wh_updated.append(self.l1[-1])
                self.l2_when_wh_updated.append(self.l2[-1])
                self.loss_when_wh_updated.append(self.loss[-1])
                # 損失ログと更新
                if j % frame_interval == 0 or j == n_wh_update - 1:
                        self.gif_positions.append(self.gpu_R_N.detach().cpu().numpy().copy())
        self.tortal_it = len(self.loss)
        create_gif()
        print("optimization is completed")
        self.params = (
            f"lr_l={lr_l}, "
            f"lr_eta={lr_eta}, "
            f"n_wh_update={n_wh_update}, "
            f"n_g_update={n_g_update}, "
            f"alpha={alpha}"
        )
        
    def plot_ply_eta(self, file_name="initial"):
        """
        ply_locs の各オブジェクトを描画し、対応する eta を色としてプロット
        """
        viewpoints = [(30, 30), (30, 210)]
        for i, (elev, azim) in enumerate(viewpoints):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for obj_idx, ply_tensor in enumerate(self.ply_locs):
                points = ply_tensor.cpu().numpy()  # (N_obj, 3)
                eta_obj = self.eta[obj_idx].detach().cpu().numpy()  # (N_obj,)
                if points.shape[0] > eta_obj.shape[0]:
                    points = points[:eta_obj.shape[0], :]
                elif points.shape[0] < eta_obj.shape[0]:
                    eta_obj = eta_obj[:points.shape[0]]
                sc = ax.scatter(points[:, 0],  # X座標
                                points[:, 1],  # Y座標
                                points[:, 2],  # Z座標
                                c=eta_obj,      # 色情報
                                cmap='viridis', # カラーマップ
                                label=f'Object {obj_idx+1}')  # オブジェクトのラベル
            plt.colorbar(sc, ax=ax, label="Eta Values")
            # ラベルとタイトル
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("3D Point Cloud Visualization with Eta")
            ax.legend()
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            ax.set_zlim(0, 2.5)
            ax.view_init(elev=elev, azim=azim)
            file_path = save_root / f"{file_name}_eta_map_view{i+1}.png"
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.show()
    
    def separated_sound_eval_SDR(self, mic_index=0):
        with torch.no_grad():
            true_sources = [np.array(sf.read('data/arctic_a0002.wav')[0]),  # 正解データの読み込み
                            np.array(sf.read('data/arctic_b0540.wav')[0])]
            # separate関数
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
            Omega_NFTMM = torch.einsum('nft,nfml->nftml',
                        self.lambda_NFT, G_NFMM)
            Omega_sum_inv_FTMM = torch.inverse(Omega_NFTMM.sum(dim=0))
            Z_FTN = torch.einsum('nftpq,ftqr,ftr->ftnp', Omega_NFTMM, Omega_sum_inv_FTMM, self.X_FTM)[..., mic_index]
            recon_sound = stft_tool.istft(Z_FTN)
            separated_sources = [source.detach().cpu().numpy() for source in recon_sound]
            # 最大長を計算
            max_length = max(max([len(src) for src in true_sources]),
                             max([len(src) for src in separated_sources]))
            true_sources_padded = [np.pad(source, (0, max_length - len(source))) for source in true_sources] # パディング
            separated_sources_padded = [np.pad(source, (0, max_length - len(source))) for source in separated_sources]
            # 音源の対応関係を推定
            correlations = [np.corrcoef(true_source, est_source)[0, 1]
                            for true_source in true_sources_padded
                            for est_source in separated_sources_padded]
            correlations = np.array(correlations).reshape(len(true_sources_padded), len(separated_sources_padded))
            max_indices = np.argmax(correlations, axis=1)
            sorted_separated_sources = [separated_sources_padded[i] for i in max_indices]
            # SDRの計算
            sdr, _, _, _ = mir_eval.separation.bss_eval_sources(np.array(true_sources_padded),
                                                                np.array(sorted_separated_sources))
            # 音源ごとのSDRをリストに追加
            if len(self.sdr_list) == 0:
                self.sdr_list = [sdr.tolist()]
            else:
                for i, s in enumerate(sdr):
                    if len(self.sdr_list) <= i:
                        self.sdr_list.append([s])
                    else:
                        self.sdr_list[i].append(s)

def plot_sdr(sdr_list):
    num_sources = len(sdr_list)
    plt.figure(figsize=(10, 6))
    for i in range(num_sources):
        plt.plot(sdr_list[i], label=f"Source {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("SDR (dB)")
    plt.title("SDR per Source over Iterations")
    plt.legend()
    file_path = save_root / f"sdr.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()

#! main
mnmf = MNMF(x_stft, n_source=2, n_basis=16)
mnmf.initialize(G_init_mode='ILRMA_Su', eta_init_mode='constant')   
mnmf.train(lr_l=2e-3, lr_eta=2e-3, n_wh_update=10, n_g_update=10, n_eta_update=10, gif_frames=50, alpha=1)

mnmf.plot_ply_eta(file_name="optimized")
plot_sdr(mnmf.sdr_list)
mnmf.separate(mic_index=0)                                                          
n_source = mnmf.Z_FTN.shape[-1]
# n_source = 2
fig, axs = plt.subplots(n_source+1, 3, figsize=(10, 10))
cpu_loss = [loss for loss in mnmf.loss]
cpu_loss1 = [l1 for l1 in mnmf.l1]
cpu_loss2 = [l2 for l2 in mnmf.l2]
axs[0, 2].plot(cpu_loss)
axs[0, 0].plot(cpu_loss1)
axs[0, 1].plot(cpu_loss2)
for i in range(n_source):
    x_log = 10 * torch.log10(x_stft[..., i].abs()**2).detach().cpu().numpy()  
    z_log = 10 * torch.log10(mnmf.Z_FTN[..., i].abs()**2).detach().cpu().numpy() 
    axs[i+1, 0].imshow(x_log, aspect='auto', origin='lower')
    axs[i+1, 1].imshow(z_log, aspect='auto', origin='lower')
    axs[i+1, 2].imshow(x_log - z_log, aspect='auto', origin='lower')
axs[0,0].set_title("Loss1")
axs[0,1].set_title("Loss2")
axs[0,2].set_title("Total Loss")
fig.tight_layout()
fig.savefig(save_root/"separated_Sumura_model.png")
plt.show()
recon_sound = stft_tool.istft(mnmf.Z_FTN)
recon_sound = recon_sound.detach().to("cpu").numpy()
tortal_it = len(mnmf.loss)
for i in range(n_source):
    sf.write(save_root/f"separated_{i}_sound.wav",
                recon_sound[i, :], samplerate),

# 超パラメータ情報をテキストとして保存
params_file = save_root / "params.txt"
with open(params_file, "w") as f:
    f.write(mnmf.params)

# 保存した l1, l2, loss をグラフ化する
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(mnmf.l1_when_wh_updated, label="l1")
axs[0].set_title("l1 over n_wh_update")
axs[0].set_xlabel("Update Step")
axs[0].set_ylabel("Value")
axs[0].legend()
axs[1].plot(mnmf.l2_when_wh_updated, label="l2", color='orange')
axs[1].set_title("l2 over n_wh_update")
axs[1].set_xlabel("Update Step")
axs[1].set_ylabel("Value")
axs[1].legend()
axs[2].plot(mnmf.loss_when_wh_updated, label="Loss", color='red')
axs[2].set_title("Loss over n_wh_update")
axs[2].set_xlabel("Update Step")
axs[2].set_ylabel("Value")
axs[2].legend()
fig.tight_layout()
fig.savefig(save_root / "loss_l1_l2_over_n_wh_update.png")
plt.show() 


fig, axs = plt.subplots(2, 1, figsize=(8, 10))
R_distance_src0 = mnmf.R_distance[0]
R_distance_src1 = mnmf.R_distance[1]
axs[0].plot(R_distance_src0, label="Source 1")
axs[0].set_title("R_N Distance for Source 1")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Distance")
axs[1].plot(R_distance_src1, label="Source 2")
axs[1].set_title("R_N Distance for Source 2")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Distance")
fig.tight_layout()
fig.savefig(save_root/"R_distance.png")
plt.show()


plot_fig.plot_R_N_with_initialR(
    mnmf.R_N.detach().cpu().numpy(), room_size=room_size, ans_R_N=ans_R_N, 
    mic_locs = mic_array_locs, init_R_N=mnmf.R_N_init.detach().cpu().numpy(),
    filename = save_root / "R_N_with_init_R", it=mnmf.tortal_it, save=True)