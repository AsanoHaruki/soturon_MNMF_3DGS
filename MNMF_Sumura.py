from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import opt_einsum
import math
import numpy as np
from tqdm import tqdm
from ILRMA_D_FMM import ILRMA
from make_simu_2 import (
    mic_array_locs, mic_array_geometry, SOUND_POSITIONS,
    SOUND_SPEED, n_mics_per_array, ans_R_N, gpu_ans_R_N, room_size
)

def torch_setup():
    torch.set_default_dtype(torch.float64)  #! float32のほうがいいかも
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

#! 先にDとdeviceを定義しなきゃない。これの場所は調整するべき
# device = torch_setup()

class STFT:
    def __init__(self) -> None:
        self.n_fft = 1024
        self.hop_length = self.n_fft // 4   # オーバーラップ 75%
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
        """
        # spec:(B, F, T) => wave: (B, L)
        # B: batch size, F: frequency, T: time
        """
        spec = spec.permute(2, 0, 1)
        hop_length = self.hop_length
        tmp = torch.istft(spec,
                          n_fft=self.n_fft,
                          hop_length=hop_length,
                          window=self.window)
        return tmp / tmp.abs().max()    # 正規化


class MNMF:
    # コンストラクタ　X信号 F周波数 Tフレーム Mマイク数
    def __init__(self, X_FTM, n_source=2, n_basis=8):   #? 音源数を指定
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = X_FTM
        # x_stft:(F, T, M) => xx_stft:(F, T, M, M)
        self.XX_FTMM = torch.einsum('ftm,ftn->ftmn',
                                    self.X_FTM, self.X_FTM.conj())  # 自己相関行列を計算
        self.n_source = n_source
        self.n_basis = n_basis
        self.D = 3  # 3次元
        self.device = torch_setup()
        self.method = 'MNMF'
        self.eps = 1e-15
        #? 損失・音源位置を格納するリストもここで初期化した方がいいかも？
        self.log_likelihood = []

    # パラメータの初期化
    #! ステアリングベクトルとかの初期化もここで
    def initialize(self, init_mode="identity"):
        self.initialize_WH()
        self.initialize_G(init_mode=init_mode)
        self.normalize_WHG()
    
    #! 個別のupdate関数を呼び出す形の方が良さげ
    # def update(self):
    #     self.update_WH()
    #     self.update_G()
    #     self.normalize_WHG()
    #     self.update_aux()
    #     self.update_log_likelihood()

    # WHのランダム初期化
    def initialize_WH(self):
        W_size = [self.n_source, self.n_freq, self.n_basis]
        H_size = [self.n_source, self.n_basis, self.n_time]
        self.W_NFK = torch.rand(W_size).detach()
        # Wの正規化条件に合わせて，周波数ごとに正規化
        self.W_NFK = self.W_NFK / self.W_NFK.sum(dim=1)[:, None]
        self.H_NKT = torch.rand(H_size).detach()
        self.W_NFK = self.W_NFK.to(torch.complex128)
        self.H_NKT = self.H_NKT.to(torch.complex128)
        self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps

    # 空間共分散行列Gを須村論文式(6)のように初期化。(mnmfの初期化方法)
    def initialize_G(self, init_mode="identity"):
        G_size = [self.n_source, self.n_freq, self.n_mic, self.n_mic]
        self.G_NFMM_init = torch.zeros(G_size, dtype=torch.complex128)
        self.G_NFMM_init[:, :] = torch.eye(self.n_mic)   # n,f要素を(m,m)の単位行列で初期化
        
        if init_mode == "ILRMA":
            print("ILRMA")
            ilrma = ILRMA(self.X_FTM, n_basis=4)
            ilrma.initialize()
            for _ in range(100):
                ilrma.update()
            ilrma.separate()
            print("ILRMA finished")
            # a_FMMは混合行列
            # 分離行列の逆行列を計算することで混合行列を得る。
            a_FMM = torch.linalg.inv(ilrma.D_FNM)
            separated_spec_power = torch.abs(ilrma.Z_FTN).mean(axis=(0, 1))
            # 各音源に対応するインデックスを取得
            for n in range(self.n_source):
                self.G_NFMM_init[n, :, :, :] = torch.einsum('fm,fl->fml', 
                                    a_FMM[:, :, separated_spec_power.argmax()], 
                                    a_FMM[:, :, separated_spec_power.argmax()].conj())
                # ↓すでに処理した音源を次回のループで再度選ばれないようにするために0にしておく
                separated_spec_power[separated_spec_power.argmax()] = 0
            
            # Gの対称性を保つためにエルミート化
            self.G_NFMM_init = (self.G_NFMM_init + self.G_NFMM_init.conj().transpose(-2,-1)) / 2
            # 正則化
            eps_regu = 1e-9
            self.G_NFMM_init += torch.eye(self.G_NFMM_init.shape[-1], dtype=self.G_NFMM_init.dtype, device=self.G_NFMM_init.device) * eps_regu
            eigvals = torch.linalg.eigvalsh(self.G_NFMM_init)
            if (eigvals <= 0).any():
                raise ValueError("G_NFMM is not positive-define")
            print("G_NFMM is positive-define")
            # コレスキー分解
            self.L_NFMM = torch.linalg.cholesky(self.G_NFMM_init).detach()
            self.L_NFMM = torch.log(self.L_NFMM).detach()
            self.L_NFMM.requires_grad_(True)
    
    #! 使うかわからん。音源位置を推定するわけではない？
    def initialize_R(self, init_loc=None,  ans_R_N=True, random_mode=False, noise_std=0.1   , seed=143):
        if random_mode:
            # Generate random positions within the room
            rng = np.random.default_rng(seed)  # Ensure reproducibility
            random_positions = rng.uniform(
                low=[0, 0, 0],
                high=[room_size[0], room_size[1], room_size[2]],
                size=(self.n_source, self.D)
            )
            self.R_N = torch.tensor(random_positions, dtype=torch.float64)
            # self.R_N_init = torch.tensor(random_positions, dtype=torch.float64)
            self.R_N_init = self.R_N.clone()
        elif init_loc is not None:
            # Generate positions near init_loc with noise in x, y directions only
            assert init_loc.shape == (self.n_source, self.D)  # (n_source, D)
            rng = np.random.default_rng()  # Ensure reproducibility
            noise_xy = rng.normal(0, noise_std, (self.n_source, 2))  # Shape: (N, 2)
            noisy_positions = init_loc.copy()
            noisy_positions[:, :2] += noise_xy  # Add noise to x, y
            self.R_N = torch.tensor(noisy_positions, dtype=torch.float64)
            # self.R_N_init = torch.tensor(noisy_positions, dtype=torch.float64)
            self.R_N_init = self.R_N.clone()
        self.init_R_N = self.R_N_init.detach()
        self.ans_R_N = ans_R_N
        self.gpu_ans_R_N = gpu_ans_R_N
        self.gpu_R_N = self.R_N.to(self.device)
        self.gpu_R_N.requires_grad_(True)
        
    def set_init_U_IND(self, gpu=True):
        '''calculate U_IND from current R_N'''
        R_N = self.gpu_R_N
        array_locs = torch.tensor(mic_array_locs).to(self.device)  # (D+1, I)
        # R_N を I 次元に拡張
        R_N_expanded = R_N.expand(array_locs.shape[1], -1, -1)  # (I, N, D)
        # U_IND 計算
        U_IND = R_N_expanded - array_locs.T[:, None, :self.D]  # (I, N, D)
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
        self.U_IND = (R_mat[:, None, :self.D, :self.D] @ U_IND[..., None])[..., 0]  # squeezeを使用せず直接取り出す

    def set_U_IND_for_optR(self, gpu=True):
        '''calculate U_IND from current R_N'''
        R_N = self.gpu_R_N
        array_locs = torch.tensor(mic_array_locs).to(self.device)  # (D+1, I)
        # R_N を I 次元に拡張
        R_N_expanded = R_N.expand(array_locs.shape[1], -1, -1)  # (I, N, D)
        # U_IND 計算
        U_IND = R_N_expanded - array_locs.T[:, None, :self.D]  # (I, N, D)
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
            self.U_IND = (R_mat[:, None, :self.D, :self.D] @ U_IND[..., None])[..., 0]  # squeezeを使用せず直接取り出す

    def get_init_steering_vec(self, array_distance=False, gpu=True) -> torch.Tensor:
        """
        calcurate theoretical steering vectors b_INFMi from current U_IND
        array_distance : the distance between array and source is considered for delay calculation
        """
        self.gpu_array_geometry = torch.tensor(mic_array_geometry).to(self.device)
        self.mic_array_locs = torch.tensor(self.mic_array_locs).to(self.device)
        array_geometry = torch.zeros((self.D, n_mics_per_array), dtype=torch.float64, device=self.device)
        array_geometry[:2, :] = self.gpu_array_geometry # (2, Mi)
        omega = 2 * torch.pi * torch.arange(self.n_freq, device=self.device)
        delay = - opt_einsum.contract("dm,ind->inm", array_geometry, self.U_IND) / SOUND_SPEED # (I, N, Mi)
        Q_IN = torch.linalg.norm(self.gpu_R_N[None,] - self.mic_array_locs.T[:,None,:3], axis=2)
        if array_distance:
            delay += Q_IN[:,:,None] / SOUND_SPEED # (I, N, Mi)
        else:
            self.steering_vec = torch.exp(-1j * opt_einsum.contract("f,inm->infm", omega, delay)) # (I, N, F, Mi)
            return self.steering_vec


    def get_steering_vec_for_optR(self, array_distance=False, gpu=True) -> torch.Tensor:
        self.gpu_array_geometry = torch.tensor(mic_array_geometry).to(self.device)
        self.mic_array_locs = torch.tensor(self.mic_array_locs).to(self.device)
        array_geometry = torch.zeros((self.D, n_mics_per_array), dtype=torch.float64, device=self.device)
        array_geometry[:2, :] = self.gpu_array_geometry # (2, Mi)
        omega = 2 * torch.pi * torch.arange(self.n_freq, device=self.device)
        delay = - opt_einsum.contract("dm,ind->inm", array_geometry, self.U_IND) / SOUND_SPEED # (I, N, Mi)
        Q_IN = torch.linalg.norm(self.gpu_R_N[None,] - self.mic_array_locs.T[:,None,:3], axis=2)
        if array_distance:
            delay += Q_IN[:,:,None] / SOUND_SPEED # (I, N, Mi)
        else:
            self.steering_vec = torch.exp(-1j * opt_einsum.contract("f,inm->infm", omega, delay)) # (I, N, F, Mi)
            return self.steering_vec
    
    def log_prob_SCM(self) -> torch.Tensor:
        eps = 1e-3
        nu = self.n_mic + 1
        G_NFMM = self.G_NFMM
        B_INFMi = self.get_init_steering_vec(gpu=True) # R_Nに依存
        hat_G_INFMiMi = opt_einsum.contract("infa,infb->infab", B_INFMi, B_INFMi.conj()) + eps * torch.eye(n_mics_per_array, device=self.device)[None, None, None, ...]
        self.gpu_hat_G_NFMM = torch.zeros((self.n_source, self.n_freq, self.n_mic, self.n_mic), dtype=torch.complex128, device=self.device)
        # ブロック対角生成
        for i in range(self.I):
            self.gpu_hat_G_NFMM[:, :, n_mics_per_array*i:n_mics_per_array*(i+1), n_mics_per_array*i:n_mics_per_array*(i+1)] = hat_G_INFMiMi[i]
        eps_I = 1e-6
        self.gpu_hat_G_NFMM[:,:] += eps_I
        # スケール調整が必要
        tr_gpu_hat_G_NF = opt_einsum.contract("nfmm->nf", self.gpu_hat_G_NFMM).real
        self.gpu_hat_G_NFMM /= (tr_gpu_hat_G_NF[...,None,None] / self.n_mic) # tr(hatG) = M = n_mic
        # パワーに応じて重み付けして総和
        log_p = torch.sum(multi_log_prob_invwishart(G_NFMM, nu, (nu+self.n_mic) * self.gpu_hat_G_NFMM)) # N, Fで総和
        return log_p.real
    
    def log_prob_SCM_init(self) -> torch.Tensor:
        '''True
        whole_mic : use all mics as one big array.False
        block_diag : build hatG with block elements corresponding to the SCM of each array
        block_diag=Trueでよい。各マイクアレイ内での相関のみを考慮し、アレイ間の相関を無視する設定が適する。独立したマイクアレイの分離モデルを構築。
        '''
        eps = 1e-3
        nu = self.n_mic + 1
        G_NFMM = self.G_NFMM
        B_INFMi = self.get_steering_vec_for_optR(gpu=True) # R_Nに依存
        hat_G_INFMiMi = opt_einsum.contract("infa,infb->infab", B_INFMi, B_INFMi.conj()) + eps * torch.eye(n_mics_per_array, device=self.device)[None, None, None, ...]
        self.gpu_hat_G_NFMM = torch.zeros((self.n_source, self.n_freq, self.n_mic, self.n_mic), dtype=torch.complex128, device=self.device)
        # ブロック対角生成
        for i in range(self.I):
            self.gpu_hat_G_NFMM[:, :, n_mics_per_array*i:n_mics_per_array*(i+1), n_mics_per_array*i:n_mics_per_array*(i+1)] = hat_G_INFMiMi[i]
        eps_I = 1e-6
        self.gpu_hat_G_NFMM[:,:] += eps_I
        # スケール調整が必要
        tr_gpu_hat_G_NF = opt_einsum.contract("nfmm->nf", self.gpu_hat_G_NFMM).real
        self.gpu_hat_G_NFMM /= (tr_gpu_hat_G_NF[...,None,None] / self.n_mic) # tr(hatG) = M = n_mic
        # パワーに応じて重み付けして総和
        log_p = torch.sum(multi_log_prob_invwishart(G_NFMM, nu, (nu+self.n_mic) * self.gpu_hat_G_NFMM)) # N, Fで総和
        return log_p.real
    
    #! get_steering_vecとSCMの計算方法の次元とかをちゃんと確認

    def update_aux(self):
        """乗法更新のための補助変数を事前に計算
        lambda_NFT・G_NFMMがすでに計算されていることが必要
        """
        self.iY_FTMM = torch.einsum('nft,nfml->ftml',
                                    self.lambda_NFT, self.G_NFMM).inverse()
        Yx_FTM1 = self.iY_FTMM @ self.X_FTM[..., None]
        self.iY_X_iY_FTMM = Yx_FTM1 @ Yx_FTM1.conj().permute(0, 1, 3, 2)
        G_iY_X_iY_NFT = torch.einsum('nfab,ftbc->nftac',
                                     self.G_NFMM,
                                     self.iY_X_iY_FTMM)
        self.tr_G_iY_X_iY_NFT = torch.einsum('...ii', G_iY_X_iY_NFT).real
        G_iY_NFT = torch.einsum('nfab,ftbc->nftac',
                                self.G_NFMM,
                                self.iY_FTMM)
        self.tr_G_iY_NFT = torch.einsum('...ii', G_iY_NFT).real

    def update_WH(self):
        a_1 = (self.H_NKT.permute(0, 2, 1)[
               :, None] * self.tr_G_iY_X_iY_NFT[:, :, :, None]).sum(axis=2)
        b_1 = (self.H_NKT.permute(0, 2, 1)[
               :, None] * self.tr_G_iY_NFT[:, :, :, None]).sum(axis=2)
        a_2 = (self.W_NFK[..., None] *
               self.tr_G_iY_X_iY_NFT[:, :, None]).sum(axis=1)
        b_2 = (self.W_NFK[..., None] *
               self.tr_G_iY_NFT[:, :, None]).sum(axis=1)
        self.W_NFK = self.W_NFK * torch.sqrt(a_1 / b_1)
        self.H_NKT = self.H_NKT * torch.sqrt(a_2 / b_2)

    def update_G(self):
        A_NFMM = self.G_NFMM @ torch.einsum('nft,ftlm->nflm',
                                            self.lambda_NFT, self.iY_X_iY_FTMM)
        A_NFMM = A_NFMM @ self.G_NFMM
        A_NFMM += (torch.eye(self.n_mic) * self.eps)[None, None]
        B_NFMM = torch.einsum('nft,ftlm->nflm',
                              self.lambda_NFT, self.iY_FTMM)
        self.G_NFMM = torch.linalg.inv(B_NFMM) @ matrix_sqrt(B_NFMM @ A_NFMM)
        self.G_NFMM = (
            self.G_NFMM + self.G_NFMM.permute(0, 1, 3, 2).conj()) / 2

    def normalize_WHG(self):
        mu_NF = torch.einsum('...ii', self.G_NFMM).real
        self.G_NFMM = self.G_NFMM / mu_NF[:, :, None, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps

    def normalize_WH(self):
        """
        正規化条件に従ってW・Hのみ正規化
        """
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]

    def update_log_likelihood(self):
        Y_FTMM = torch.einsum('nft,nfml->ftml', self.lambda_NFT, self.G_NFMM)
        iY_FTMM = torch.linalg.inv(Y_FTMM)
        iY_XX_FTMM = - torch.einsum('ftml,ftln->ftmn', iY_FTMM, self.XX_FTMM)
        tr_iY_XX_FTMM = torch.einsum('...ii', iY_XX_FTMM).real
        lk = (tr_iY_XX_FTMM + torch.log(torch.linalg.det(iY_FTMM).real)).sum()
        self.log_likelihood.append(lk.abs().detach())    # 正規分布の対数確率密度関数の絶対値・対数尤度になる
        return lk

    def separate(self, mic_index=0):
        Omega_NFTMM = torch.einsum('nft,nfml->nftml',
                                   self.lambda_NFT, self.G_NFMM)
        Omega_sum_inv_FTMM = torch.inverse(Omega_NFTMM.sum(dim=0))
        # 'nftpq,ftqr-> nftpr,ftr->nftp'
        self.Z_FTN = torch.einsum('nftpq,ftqr,ftr->ftnp',
                                  Omega_NFTMM,
                                  Omega_sum_inv_FTMM,
                                  self.X_FTM)[..., mic_index]   # Zは分離音スペクトル

#! train()
    def forward(self):
        self.G_NFMM = opt_einsum.contract("nfac,nfbc->nfab", torch.exp(self.L_NFMM), torch.exp(self.L_NFMM.conj()))
        self.Y_FTMM = torch.einsum('nft,nfml->ftml', self.lambda_NFT, self.G_NFMM)
    
    def train(self, iter_WH, iter_G, lr, eval=True):
        self.forward()
        optimizer = torch.optim.Adam([self.L_NFMM], lr=lr)
        if eval:
            stft_tool_eval = STFT()
            self.eval_signal = []
        
        with tqdm(total=iter_WH, desc='WH updates') as pbar_WH:
            for i in range(iter_WH):    # WHの更新
                with torch.no_grad():
                    self.update_aux()
                    self.update_WH()
                    self.normalize_WHG()
                loss = -self.get_log_likelihood().detach()
                pbar_WH.set_postfix({'Loss': loss.item()})
                pbar_WH.update()
                
                with tqdm(total=iter_G, desc='G Updates', leave=False) as pbar_G:
                    for j in range(iter_G):
                        optimizer.zero_grad()
                        self.forward()
                        loss = - self.get_log_likelihood()  # 尤度のマイナスが損失
                        loss.backward()
                        optimizer.step()
                        pbar_G.set_postfix({'Loss': loss.item()})
                        pbar_G.update()
    
    def get_log_likelihood(self):
        iY_FTMM = torch.linalg.inv(self.Y_FTMM)
        iY_XX_FTMM = - torch.einsum('ftml,ftln->ftmn', iY_FTMM, self.XX_FTMM)
        tr_iY_XX_FTMM = torch.einsum('...ii', iY_XX_FTMM).real
        lk = (tr_iY_XX_FTMM + torch.log(torch.linalg.det(iY_FTMM).real)).sum()
        # self.log_likelihood.append(lk.abs().detach())    # 正規分布の対数確率密度関数の絶対値・対数尤度になる
        self.log_likelihood.append(lk.detach())
        return lk

def matrix_sqrt(A):
    eig_val, eig_vec = torch.linalg.eig(A)
    tmp = torch.zeros_like(A)
    M = A.shape[-1]
    eig_val = torch.sqrt(eig_val)
    for m in range(M):
        tmp[:, :, m, m] = eig_val[:, :, m]
    A_sqrt = eig_vec @ tmp @ torch.linalg.inv(eig_vec)
    return A_sqrt

def multi_log_prob_invwishart(S: torch.Tensor, nu: int, Psi: torch.Tensor):
    """複素逆ウィシャート分布の対数確率密度関数
    Return: scalar
    """
    d = S.shape[-1]
    return nu/2 * torch.slogdet(Psi)[1] - (nu*d)/2*math.log(2) - torch.special.multigammaln(torch.tensor(nu/2), d) - (nu+d+1)/2 * torch.slogdet(S)[1] -1/2 * opt_einsum.contract("...mm->...", Psi @ torch.linalg.inv(S))


def main():
    device = torch_setup()

    D = 3   # 3次元
    
    data_root = Path("self_data")
    save_root = Path("result_mnmf_test")
    save_root.mkdir(exist_ok=True)
    data_file = data_root / "mixture_time_domain_2.wav"
    data, samplerate = sf.read(data_file)
    data = data - data.mean()
    data = torch.tensor(data.T)
    stft_tool = STFT()
    x_stft = stft_tool.stft(data).permute(1, 2, 0)
    x_stft[x_stft.abs() < 1e-15] = 1e-15
    x_stft = x_stft / x_stft.abs().max()    # 正規化

    #! 実行
    mnmf = MNMF(x_stft, n_source=2, n_basis=16) #? 音源数・基底数を指定
    mnmf.initialize(init_mode="ILRMA")
    mnmf.train(iter_WH=3, iter_G=10, lr=1e-4, eval=False)
    
    mnmf.separate(mic_index=0)
    n_source = mnmf.Z_FTN.shape[-1]
    fig, axs = plt.subplots(n_source+1, 3, figsize=(10, 10))
    cpu_lk = [torch.log10(lk).to("cpu") for lk in mnmf.log_likelihood]
    axs[0, 1].plot(cpu_lk)
    axs[0, 1].set_title("Log_likelihood")
    for i in range(n_source):
        x_log = 10 * torch.log10(x_stft[..., i].abs()**2).detach().to("cpu")
        z_log = 10 * torch.log10(mnmf.Z_FTN[..., i].abs()**2).detach().to("cpu")
        axs[i+1, 0].imshow(x_log,
                           aspect='auto', origin='lower')
        axs[i+1, 1].imshow(z_log,
                           aspect='auto', origin='lower')
        axs[i+1, 2].imshow(x_log - z_log,
                           aspect='auto', origin='lower')
    fig.tight_layout()
    fig.savefig(save_root/"separated_stft_Sumura_test_1211.png")
    plt.show()
    recon_sound = stft_tool.istft(mnmf.Z_FTN).detach()
    recon_sound = recon_sound.to("cpu").detach()
    for i in range(n_source):
        sf.write(save_root/f"separated_{i}_Sumura_test_1211.wav",
                 recon_sound[i, :], samplerate)

if __name__ == '__main__':
    main()