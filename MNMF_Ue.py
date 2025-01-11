from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ILRMA import ILRMA


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


class STFT:
    def __init__(self) -> None: # -> None: 型ヒント, 戻り値がない
        self.n_fft = 1024   # フーリエ変換のサイズ
        self.hop_length = self.n_fft // 4   # オーバーラップ 75%
        self.window = torch.hann_window(self.n_fft)

    def stft(self, wav: torch.tensor):
        """
        Return: tensor, (B,F,T)
        """
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
        # n_freq = spec.shape[0]  #! n_freq = spec.shape[1] では？
        n_freq = spec.shape[1]
        spec = spec.permute(2, 0, 1)
        # hop_length = (n_freq-1) // 2
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
        self.method = 'MNMF'
        self.eps = 1e-15

    # パラメータの初期化
    def initialize(self, init_mode="identity"):
        self.initialize_WH()
        self.initialize_G(init_mode=init_mode)
        self.normalize_WHG()
        self.update_aux()
        self.log_likelihood = []
        # self.update_log_likelihood()

    def update(self):
        self.update_WH()
        self.update_G()
        self.normalize_WHG()
        self.update_aux()

        self.update_log_likelihood()

    # WHのランダム初期化

    def initialize_WH(self):
        W_size = [self.n_source, self.n_freq, self.n_basis]
        H_size = [self.n_source, self.n_basis, self.n_time]
        self.W_NFK = torch.rand(W_size)
        # Wの正規化条件に合わせて，周波数ごとに正規化
        self.W_NFK = self.W_NFK / self.W_NFK.sum(dim=1)[:, None]
        self.H_NKT = torch.rand(H_size)
        self.W_NFK = self.W_NFK.to(torch.complex128)
        self.H_NKT = self.H_NKT.to(torch.complex128)
        self.l_NFT = self.W_NFK @ self.H_NKT

        # self.lambda_NFT = self.W_NFK @ self.H_NKT

    # 空間共分散行列Gを須村論文式(6)のように初期化。(mnmfの初期化方法)
    def initialize_G(self, init_mode="identity"):
        G_size = [self.n_source, self.n_freq, self.n_mic, self.n_mic]

        self.G_NFMM = torch.zeros(G_size, dtype=torch.complex128)
        self.G_NFMM[:, :] = torch.eye(self.n_mic)   # n,f要素を(m,m)の単位行列で初期化

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
            a_FNM = torch.linalg.inv(ilrma.D_FNM)
            separated_spec_power = torch.abs(ilrma.Z_FTN).mean(axis=(0, 1))

            max_index = torch.argsort(separated_spec_power, descending=True)

            self.G_NFMM_init = torch.einsum('fmn,fln->nfml',
                                            a_FNM,
                                            a_FNM.conj())
            self.G_NFMM_init = self.G_NFMM_init[max_index]

    def update_aux(self):
        """乗法更新のための補助変数を事前に計算
        iY_FTMM
        iY_X_iY_FTMM
        tr_G_iY_X_iY_NFT
        tr_G_iY_NFT
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

    def update_log_likelihood(self):
        Y_FTMM = torch.einsum('nft,nfml->ftml', self.lambda_NFT, self.G_NFMM)
        iY_FTMM = torch.linalg.inv(Y_FTMM)
        iY_XX_FTMM = - torch.einsum('ftml,ftln->ftmn', iY_FTMM, self.XX_FTMM)
        tr_iY_XX_FTMM = torch.einsum('...ii', iY_XX_FTMM).real
        lk = (tr_iY_XX_FTMM + torch.log(torch.linalg.det(iY_FTMM).real)).sum()

        self.log_likelihood.append(lk.abs())    # 正規分布の対数確率密度関数の絶対値・対数尤度になる

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


def matrix_sqrt(A):
    eig_val, eig_vec = torch.linalg.eig(A)
    tmp = torch.zeros_like(A)
    M = A.shape[-1]
    eig_val = torch.sqrt(eig_val)
    for m in range(M):
        tmp[:, :, m, m] = eig_val[:, :, m]
    A_sqrt = eig_vec @ tmp @ torch.linalg.inv(eig_vec)
    return A_sqrt


def main():
    torch_setup()

    data_root = Path("self_data")
    save_root = Path("result_mnmf")
    save_root.mkdir(exist_ok=True)
    data_file = data_root / "mixture_time_domain.wav"
    data, samplerate = sf.read(data_file)
    data = data - data.mean()
    data = torch.tensor(data.T)
    stft_tool = STFT()

    #  x:(M, L) -> x_stft:(M, F, T)
    #  x_stft:(M, F, T) -> x_stft:(F, T, M)
    x_stft = stft_tool.stft(data).permute(1, 2, 0)
    x_stft[x_stft.abs() < 1e-15] = 1e-15
    x_stft = x_stft / x_stft.abs().max()    # 正規化

    mnmf = MNMF(x_stft, n_source=2, n_basis=16) #? 音源数・基底数を指定
    mnmf.initialize(init_mode="ILRMA")
    with tqdm(range(100))as pbar:   #? エポック数を指定
        for e in pbar:
            mnmf.update()
            pbar.set_description("[Epoch %d]" % (e+1))
            pbar.set_postfix(
                {"lh": "{:.3e}".format(mnmf.log_likelihood[-1].abs())})
    mnmf.separate(mic_index=0)
    n_source = mnmf.Z_FTN.shape[-1]
    fig, axs = plt.subplots(n_source+1, 3, figsize=(10, 10))
    cpu_lk = [lk.to("cpu") for lk in mnmf.log_likelihood]
    axs[0, 1].plot(cpu_lk)
    for i in range(n_source):
        x_log = 10 * torch.log10(x_stft[..., i].abs()**2).to("cpu")
        z_log = 10 * torch.log10(mnmf.Z_FTN[..., i].abs()**2).to("cpu")
        axs[i+1, 0].imshow(x_log,
                           aspect='auto', origin='lower')
        axs[i+1, 1].imshow(z_log,
                           aspect='auto', origin='lower')
        axs[i+1, 2].imshow(x_log - z_log,
                           aspect='auto', origin='lower')
    fig.tight_layout()
    fig.savefig(save_root/"separated_stft_zyouhou.png")
    plt.show()

    recon_sound = stft_tool.istft(mnmf.Z_FTN)
    recon_sound = recon_sound.to("cpu")

    for i in range(n_source):
        sf.write(save_root/f"separated_{i}_zyouhou.wav",
                 recon_sound[i, :], samplerate)


if __name__ == '__main__':
    main()