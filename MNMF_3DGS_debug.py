import math
import numpy as np
import datetime
import soundfile as sf
import matplotlib.pyplot as plt
import pickle as pic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import plot_fig
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
import itertools
from plyfile import PlyData
from tqdm  import tqdm
from torch.optim import lr_scheduler
from ILRMA_D_FMM import ILRMA
from ILRMA_Su import ILRMA_Su
from make_simu_2_Sumura_2_coner import (
    mic_array_locs, mic_array_geometry, SOUND_POSITIONS,
    SOUND_SPEED, n_mics_per_array, ans_R_N, gpu_ans_R_N, room_size
)

import opt_einsum
from matplotlib import rc
rc('animation', html='html5')  # HTML5ビデオ用の設定
rc('animation', writer='imagemagick', codec='gif')  # ImageMagickを指定

D = 3 # ３次元で実装

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
        self.hop_length = self.n_fft // 8  # ホップ長を小さくするほど計算コスト↑、時間分解能↑
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
        # hop_length = (n_freq-1) // 2
        hop_length = self.hop_length
        tmp = torch.istft(spec,
                          n_fft=self.n_fft,
                          hop_length=hop_length,
                          window=self.window)
        return tmp / tmp.abs().max()
    
today = datetime.datetime.now()
timestamp = today.strftime('%Y%m%d_%H%M')
data_root = Path("self_data")
save_root = Path(f"result/{timestamp}_only_separate")
save_root.mkdir(exist_ok=True)
data_file = data_root / "mixture_time_domain_2_corner.wav"
data, samplerate = sf.read(data_file)
print(data[28400][0])
data = data - data.mean()
data = torch.tensor(data.T)
print(f"len(data): {len(data[0])}")
print(data[0][28400])
stft_tool = STFT()
x_stft = stft_tool.stft(data).permute(1, 2, 0)
x_stft[x_stft.abs() < 1e-15] = 1e-15
x_stft = x_stft / x_stft.abs().max()
print(f"x_stft.shape: {x_stft.shape}")

def multi_log_prob_complex_normal(x: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor):
    d = x.shape[-1]
    return - 1/2 * torch.slogdet(Sigma)[1] - 1/2 * (x-mu)[..., None, :].conj() @ torch.linalg.inv(Sigma) @ (x-mu)[..., None] - d/2 * math.log(2*math.pi)


def multi_log_prob_invwishart(S: torch.Tensor, nu: int, Phi: torch.Tensor):
    """
    return log inverse Wishart distribution W^-1(S | nu, Phi)
    """
    d = S.shape[-1]
    return nu/2 * torch.slogdet(Phi.real)[1] - (nu*d)/2*math.log(2) - torch.special.multigammaln(torch.tensor(nu/2.), d) - (nu+d+1)/2. * torch.slogdet(S.real)[1] -1/2. * torch.einsum("...mm->...", Phi.real @ torch.linalg.inv(S.real))

class MNMF:
    # コンストラクタ　X信号 F周波数 Tフレーム Mマイク数
    def __init__(self, X_FTM, n_source=1, n_basis=8):
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
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
        self.true_sources = [np.array(sf.read('data/a01.wav')[0]),  # 正解データの読み込み
                        np.array(sf.read('data/a03.wav')[0])]
        self.ply_translate =[(2.0, 6.0, 0.0), (6.0, 2.0, 0.0)]
        self.ply_rotation_angles = [315, 135]

    # パラメータの初期化
    def initialize(self, G_init_mode, eta_init_mode):
        self.initialize_WH()
        self.initialize_G(init_mode=G_init_mode)
        self.normalize_WHG()
        self.read_ply_to_tensor()
        self.initialize_eta(init_mode=eta_init_mode)
        # self.plot_ply_eta(file_name="initial")

    def multiplicative_update_WH(self):
        self.update_WH()

    def update_G(self):
        self.L_optim.zero_grad()
        loss = -self.log_prob_X()
        # l1 = -self.log_prob_X()
        # l2 = -self.log_prob_SCM_ply()
        # self.l1.append(l1.item())
        # self.l2.append(l2.item())
        # loss = (l1 + self.alpha*l2)
        self.loss.append(loss.item())
        loss.backward()
        self.L_optim.step()
    
    # WHのランダム初期化
    def initialize_WH(self):
        
        torch.manual_seed(0)    # デバッグ用にシード値を固定
        
        W_size = [self.n_source, self.n_freq, self.n_basis]
        H_size = [self.n_source, self.n_basis, self.n_time]
        self.W_NFK = torch.rand(W_size)
        # Wの正規化条件に合わせて，周波数ごとに正規化
        self.W_NFK = self.W_NFK / self.W_NFK.sum(dim=1)[:, None]
        self.H_NKT = torch.rand(H_size)
        self.W_NFK = self.W_NFK.to(torch.complex128)
        self.H_NKT = self.H_NKT.to(torch.complex128)
        self.lambda_NFT = self.W_NFK @ self.H_NKT
        
    def initialize_G(self, init_mode='ILRMA'):
        eps = 1e-3
        G_size = [self.n_source, self.n_freq, self.n_mic, self.n_mic]
        G_NFMM = torch.zeros(G_size, dtype=torch.complex128)
        # print(f"init_G_NFMM.shape: {G_NFMM.shape}")
        G_NFMM[:, :] = torch.eye(self.n_mic)    # すべて単位行列で初期化（雑音相当は単位行列のまま）
        
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
            G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)  # エルミート化
            torch.clear_autocast_cache()
            
        elif init_mode == 'GS':
            print("initialize G from 3DGS")
            self.set_U_NIVnD_ply()
            self.get_steering_vec_ply()
            G_NFMM = self.compute_hat_G_ply()
            # print(f"after_GS_G_NFMM.shape: {G_NFMM.shape}")
            # G_NFMM = (G_NFMM + G_NFMM.conj().transpose(-2, -1)) / 2   # エルミート化
            
            #? epsの加算でフルランク化．この加算は正規化後に行う．
            # G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)  # エルミート化
            
            # 雑音に割り当てる音源は単位行列で初期化
            n_noise = self.n_source - G_NFMM.shape[0]
            if n_noise > 0:
                I = torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)
                G_noise = torch.einsum('ij,kf->kfij', I, torch.ones((n_noise, self.n_freq), dtype=torch.complex128, device=self.device))
                G_NFMM = torch.cat((G_NFMM, G_noise), dim=0)
            self.normalize_initial(G_NFMM)
            G_NFMM = G_NFMM + eps * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128, device=self.device)
            torch.clear_autocast_cache()
            
        elif init_mode == 'ILRMA_Su':
            print("ILRMA")
            n_args = torch.zeros((len(self.true_sources)), dtype=torch.int64, device=self.device)
            ilrma = ILRMA_Su()
            ilrma.import_obsdata(self.X_FTM.to(self.device), torch.tensor(self.n_freq).to(self.device))
            ilrma.initialize_WH(n_basis=4)
            ilrma.initialize_D()
            for i in range(50):
                ilrma.update_WH()
                ilrma.update_D()
                ilrma.normalize_WHD()
            print("ILRMA finished")
            spoint, epoint = 0.1, 0.7
            A_FMN = torch.linalg.inv(ilrma.gpu_D_FNM)
            separated_spec_power = torch.abs(ilrma.separate()[:, int(ilrma.T*spoint):int(ilrma.T*epoint), :]).mean(dim=(0, 1))
            sorted_idx = torch.argsort(separated_spec_power, descending=True)
            print(f"selected_idx: {sorted_idx}")
            for n in range(len(self.true_sources)):
                G_NFMM[n] = A_FMN[:, :, sorted_idx[n]][:, :, None] @ A_FMN[:, :, sorted_idx[n]][:, None].conj() \
                    + 1e-2 * torch.eye(self.n_mic, dtype=torch.complex128, device=self.device)[None]
                n_args[n] = sorted_idx[n]
            for n in range(self.n_source):
                self.W_NFK[n] = ilrma.gpu_W_NFK[sorted_idx[n], :, :].to(torch.complex128)
                self.H_NKT[n] = ilrma.gpu_H_NKT[sorted_idx[n], :, :].to(torch.complex128)
            print(f"W_NFK.shape: {self.W_NFK.shape}")
            print(f"H_NKT.shape: {self.H_NKT.shape}")
            # self.W_NFK = ilrma.gpu_W_NFK[n_args, :, :].to(torch.complex128)
            # self.H_NKT = ilrma.gpu_H_NKT[n_args, :, :].to(torch.complex128)
            # ILRMAでの分離音を保存
            Z_FTN = ilrma.separate(mic_index=0)
            recon_sound = stft_tool.istft(Z_FTN.to(self.device))
            recon_sound = recon_sound.detach().to("cpu").numpy()
            ILRMA_save_root = Path(save_root/f"ILRMA_Sound")
            ILRMA_save_root.mkdir(exist_ok=True)
            for i in range(Z_FTN.shape[-1]):
                sf.write(ILRMA_save_root/f"ILRMA_separated_{i}_sound.wav",
                            recon_sound[i, :], samplerate),
            print(G_NFMM)
            torch.clear_autocast_cache()
        
        self.gpu_L_NFMM = torch.linalg.cholesky(G_NFMM) # エルミート行列である必要
        self.gpu_L_NFMM = torch.log(self.gpu_L_NFMM).detach()
        self.gpu_L_NFMM.requires_grad_(True)
        
    def gather_N_from_ILRMA(self, output_N=2):
        spoint, epoint = 0.1, 0.7
        separated_spec_power = torch.abs(self.ilrma.separate()[:, int(self.ilrma.T*spoint):int(self.ilrma.T*epoint), :]).mean(dim=(0,1))
        n_args = torch.zeros((output_N))
        for n in range(output_N):
            n_args[n] = separated_spec_power.argmax()
            separated_spec_power[separated_spec_power.argmax()] = 0
        return n_args.type(torch.int64)
    
    def update_WH(self):
        with torch.no_grad():
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        # print(f"G_NFMM.shape: {G_NFMM.shape}")
        # print(f"lambda_NFT.shape: {self.lambda_NFT.shape}")
        iY_FTMM = torch.einsum('nft,nfml->ftml', self.lambda_NFT, G_NFMM).inverse()
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
        # self.normalize_WH()
    
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

    def normalize_WHG(self):
        with torch.no_grad():
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj())) 
            mu_NF = torch.einsum('...ii', G_NFMM).real
            self.G_NFMM = G_NFMM / mu_NF[..., None, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT # + self.eps   
    
    def normalize_initial(self, G_NFMM):
        #? 計算グラフを切るかどうか
        mu_NF = torch.einsum('...ii', G_NFMM).real
        G_NFMM = G_NFMM / mu_NF[:, :, None, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT #+ self.eps
    
    def normalize_WH(self):
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps
        
    def normalize_G(self):
        with torch.no_grad():
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj())) 
            mu_NF = torch.einsum('...ii', G_NFMM).real
            self.G_NFMM = (G_NFMM / mu_NF[:, :, None, None])
        
    def separate(self, mic_index=0):
        G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
        Omega_NFTMM = torch.einsum('nft,nfml->nftml',
                    self.lambda_NFT, G_NFMM)
        Omega_sum_inv_FTMM = torch.inverse(Omega_NFTMM.sum(dim=0))
        self.Z_FTN = torch.einsum('nftpq,ftqr,ftr->ftnp',
                                  Omega_NFTMM,
                                  Omega_sum_inv_FTMM,
                                  self.X_FTM)[..., mic_index]
        
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
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            coords = coords @ rotation_matrix.T
            tensor = torch.tensor(coords) + translation_torch
            
            # # 1/3ごと
            # N = tensor.shape[0]
            # part_size = N // 5
            
            # part_sizeを数で指定
            part_size = 10000
            part1 = tensor[:part_size]  # 最初の1/3
            # part2 = tensor[part_size:2*part_size]  
            # part3 = tensor[2*part_size:3*part_size]  
            # part4 = tensor[3*part_size:]
                
            self.ply_locs.append(part1)
        return self.ply_locs
    
    def set_U_NIVnD_ply(self):
        self.read_ply_to_tensor()
        self.U_NIVnD = []
        for R_N in self.ply_locs:
            array_locs = torch.tensor(mic_array_locs).to(self.device)  # (D+1, I)
            R_N_expanded = R_N.expand(array_locs.shape[1], -1, -1)  # (I, N, D)
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
        hat_G_ply_NFMM = torch.zeros((len(self.steering_vec_ply), self.n_freq, self.n_mic, self.n_mic), dtype=torch.complex128, device=self.device)
        for i, steering_vec_i in enumerate(self.steering_vec_ply):
            max_length = max([s.shape[0] for s in steering_vec_i])
            padded_steering_vec_i = []
            for j in range(len(steering_vec_i)):
                current_len = steering_vec_i[j].shape[0]
                padded_tensor = torch.zeros((max_length, *steering_vec_i[j].shape[1:]), dtype=steering_vec_i[j].dtype, device=self.device)
                padded_tensor[:current_len] = steering_vec_i[j]
                padded_steering_vec_i.append(padded_tensor)
            
            for j in range(steering_vec_i.shape[0]):
                # steering_vec_i[j]: (Vn, F, Mi)    (1オブジェクトの各マイクアレイに対応)
                part_result = torch.einsum("vfm,vfn->vfmn", padded_steering_vec_i[j], padded_steering_vec_i[j].conj())    # (Vn,F,Mi,Mi)
                if weights is not None: 
                    weight_tensor = weights[i, :]
                    
                    padded_part_result = torch.zeros((weight_tensor.shape[0], *part_result.shape[1:]), dtype=part_result.dtype, device=self.device) #? メモリ落ち
                    padded_part_result[:part_result.shape[0]] = part_result
                    # part_resultのパディング
                    if part_result.shape[0] < weight_tensor.shape[0]:
                        padding_size = weight_tensor.shape[0] - part_result.shape[0]
                        part_result = torch.cat([part_result, torch.zeros(padding_size, *part_result.shape[1:], dtype=part_result.dtype, device=self.device)], dim=0)
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
            eta_full = torch.rand((len(self.ply_locs), max_length), device=self.device)
        elif init_mode == 'constant':
            eta_full = torch.ones((len(self.ply_locs), max_length), device=self.device)
        # ゼロパディング
        for i, length in enumerate(length):
            eta_full[i, length:] = 0
        eta_full = eta_full / eta_full.sum(dim=1, keepdim=True)
        self.eta = eta_full.clone().detach().requires_grad_(True)
        # print(f"eta.shape: {self.eta.shape}")
    
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
                # if j % frame_interval == 0 or j == n_wh_update - 1:
                #         self.gif_positions.append(self.gpu_R_N.detach().cpu().numpy().copy())
        self.tortal_it = len(self.loss)
        # create_gif()
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
                                s=10,           # 点の大きさ
                                label=f'Object {obj_idx+1}')  # オブジェクトのラベル
            plt.colorbar(sc, ax=ax, label="Eta Values")
            # ラベルとタイトル
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("3D Point Cloud Visualization with Eta")
            ax.legend()
            ax.set_xlim(0, room_size[0])
            ax.set_ylim(0, room_size[1])
            ax.set_zlim(0, room_size[3])
            ax.view_init(elev=elev, azim=azim)
            file_path = save_root / f"{file_name}_eta_map_view{i+1}.png"
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.show()
    
    def separated_sound_eval_SDR(self, mic_index=0):
        with torch.no_grad():
            # separate関数
            G_NFMM = torch.einsum("nfac,nfbc->nfab", torch.exp(self.gpu_L_NFMM), torch.exp(self.gpu_L_NFMM.conj()))
            Omega_NFTMM = torch.einsum('nft,nfml->nftml',
                        self.lambda_NFT, G_NFMM)
            Omega_sum_inv_FTMM = torch.inverse(Omega_NFTMM.sum(dim=0))
            Z_FTN = torch.einsum('nftpq,ftqr,ftr->ftnp', Omega_NFTMM, Omega_sum_inv_FTMM, self.X_FTM)[..., mic_index]
            
            # パワーが大きい順に選ぶ
            separated_spec_power = torch.abs(Z_FTN).mean(axis=(0, 1))
            selected_index = torch.argsort(separated_spec_power, descending=True)[:len(self.true_sources)]
            Z_FTN = Z_FTN[:, :, selected_index]
            
            recon_sound = stft_tool.istft(Z_FTN)
            separated_sources = [source.detach().cpu().numpy() for source in recon_sound]
            # 正規化
            self.true_sources = [source / np.linalg.norm(source) for source in self.true_sources]
            separated_sources = [source / np.linalg.norm(source) for source in separated_sources]
            
            # 最大長を計算
            max_length = max(max([len(src) for src in self.true_sources]),
                             max([len(src) for src in separated_sources]))
            true_sources_padded = [np.pad(source, (0, max_length - len(source))) for source in self.true_sources] # パディング
            separated_sources_padded = [np.pad(source, (0, max_length - len(source))) for source in separated_sources]
            # SDRが最も大きくなる組み合わせを選ぶ
            all_permutations = list(itertools.permutations(range(len(self.true_sources))))
            sdr_perm_list = []
            for perm in all_permutations:
                sorted_separated_sources = [separated_sources_padded[perm[i]] for i in range(len(self.true_sources))]
                sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
                    np.array(true_sources_padded),
                    np.array(sorted_separated_sources)
                )
                sdr_perm_list.append((sdr, perm))
            best_sdr, best_perm = max(sdr_perm_list, key=lambda x: np.mean(x[0]))
            
            # 音源ごとのSDRをリストに追加
            if len(self.sdr_list) == 0:
                self.sdr_list = [best_sdr.tolist()]
            else:
                for i, s in enumerate(best_sdr):
                    if len(self.sdr_list) <= i:
                        self.sdr_list.append([s])
                    else:
                        self.sdr_list[i].append(s)
    
    def train_only_separate(self, lr_l=2e-3, n_wh_update=500, n_g_update=50):
        print("start optimization")
        torch.autograd.detect_anomaly(True)
        self.L_optim = optim.Adam([self.gpu_L_NFMM], lr=lr_l)
        # 学習率スケジューラー
        scheduler = lr_scheduler.CosineAnnealingLR(self.L_optim, T_max=n_wh_update, eta_min=1e-6)
                
        with tqdm(range(n_wh_update), desc='WH Updates', leave=True) as pbar_wh:
            for j in pbar_wh:
                self.multiplicative_update_WH()
                self.normalize_WH()
                loss = -self.log_prob_X()
                self.loss.append(loss.item())
                self.separated_sound_eval_SDR(mic_index=0)
                
                # for k in range(n_g_update):
                #     self.update_G()
                
                if j > n_wh_update / 3:
                    for k in range(n_g_update):
                        self.update_G()
                        scheduler.step()
                #     self.normalize_WHG()
                self.loss_when_wh_updated.append(self.loss[-1])
        
        self.tortal_it = len(self.loss)
        print("optimization is completed")
        # self.normalize_WHG()
        self.params = f"lr_l={lr_l}, n_wh_update={n_wh_update}, n_source={self.n_source}, n_basis={self.n_basis}, n_g_update={n_g_update}, n_source={self.n_source}"

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
mnmf = MNMF(x_stft, n_source=3, n_basis=16)
mnmf.initialize(G_init_mode='GS', eta_init_mode='constant')   
mnmf.train_only_separate(lr_l=1e-2, n_wh_update=300, n_g_update=3)
# mnmf.plot_ply_eta(file_name="optimized")
plot_sdr(mnmf.sdr_list)
mnmf.separate(mic_index=0)                                                          
n_source = mnmf.n_source
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


# fig, axs = plt.subplots(2, 1, figsize=(8, 10))
# R_distance_src0 = mnmf.R_distance[0]
# R_distance_src1 = mnmf.R_distance[1]
# axs[0].plot(R_distance_src0, label="Source 1")
# axs[0].set_title("R_N Distance for Source 1")
# axs[0].set_xlabel("Iteration")
# axs[0].set_ylabel("Distance")
# axs[1].plot(R_distance_src1, label="Source 2")
# axs[1].set_title("R_N Distance for Source 2")
# axs[1].set_xlabel("Iteration")
# axs[1].set_ylabel("Distance")
# fig.tight_layout()
# fig.savefig(save_root/"R_distance.png")
# plt.show()


# plot_fig.plot_R_N_with_initialR(
#     mnmf.R_N.detach().cpu().numpy(), room_size=room_size, ans_R_N=ans_R_N, 
#     mic_locs = mic_array_locs, init_R_N=mnmf.R_N_init.detach().cpu().numpy(),
#     filename = save_root / "R_N_with_init_R", it=mnmf.tortal_it, save=True)