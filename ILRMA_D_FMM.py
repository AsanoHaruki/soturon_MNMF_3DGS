import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
# plt.style.use('ud.mplstyle')
torch.set_default_device("cuda")
torch.random.manual_seed(0)
torch.set_default_dtype(torch.float64)


class STFT:
    def __init__(self) -> None:
        self.n_fft = 1024
        self.hop_length = self.n_fft // 4
        self.window = torch.hann_window(self.n_fft)

    def stft(self, wav: torch.tensor):
        # wave: (B, L) = > spec: (B, F, T)
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


class ILRMA:
    def __init__(self, n_basis=16):
        """ initialize ILRMA

        Parameters:
        -----------
            X_FTM: complex
                観測音の複素スペクトログラム
            n_basis: int
                NMFの基底数
        """
        data_root = Path("/home/yoshiilab1/soturon/mnmf/code/self_data")
        save_root = Path("result_a_source_Smuramodel")
        save_root.mkdir(exist_ok=True)
        data_file = data_root / "mixture_time_domain_2_corner.wav"
        # data, samplerate = sf.read(data_file)
        data, samplerate = sf.read(data_file)
        data = data - data.mean()
        data = torch.tensor(data.T)
        stft_tool = STFT()
        x_stft = stft_tool.stft(data).permute(1, 2, 0)
        x_stft[x_stft.abs() < 1e-15] = 1e-15
        x_stft = x_stft / x_stft.abs().max()
        self.n_freq, self.n_time, self.n_mic = x_stft.shape
        self.n_source = self.n_mic  #! 決定系 N=M
        # self.n_source = 1
        self.n_basis = n_basis
        self.X_FTM = x_stft
        self.XX_FTMM = torch.einsum('ftm,ftn->ftmn',
                                    self.X_FTM, self.X_FTM.conj())
        self.eps = 1e-15

    # パラメータの初期化
    def initialize(self):
        self.initialize_WHD()
        self.normalize_WHD()

        self.log_likelihood = []
        self.update_log_likelihood()

    # パラメータの更新
    def update(self):
        self.update_WHD()
        self.normalize_WHD()
        self.update_log_likelihood()

    def initialize_WHD(self):
        W_size = [self.n_source, self.n_freq, self.n_basis]
        H_size = [self.n_source, self.n_basis, self.n_time]
        self.W_NFK = torch.rand(W_size)
        # Wの正規化条件に合わせて，周波数ごとに正規化
        self.W_NFK = self.W_NFK / self.W_NFK.sum(dim=1)[:, None]
        self.H_NKT = torch.rand(H_size)
        self.W_NFK = self.W_NFK.to(torch.complex128)
        self.H_NKT = self.H_NKT.to(torch.complex128)
        self.l_NFT = self.W_NFK @ self.H_NKT
        self.D_FMM = torch.zeros(
            [self.n_freq, self.n_mic, self.n_mic], dtype=torch.complex128)
        self.D_FMM[:] = torch.eye(self.n_mic)

    def update_WHD(self):
        a = torch.einsum('nkt,nft->nfk',
                         self.H_NKT, (self.Z_pow_NFT / self.l_NFT ** 2))
        b = torch.einsum('nkt,nft->nfk',
                         self.H_NKT, (1 / self.l_NFT))
        self.W_NFK = self.W_NFK * torch.sqrt(a / b)
        self.W_NFK[self.W_NFK.abs() < self.eps] = self.eps
        self.l_NFT = torch.einsum('nfk,nkt->nft', self.W_NFK, self.H_NKT)

        a2 = torch.einsum('nfk,nft->nkt',
                          self.W_NFK, (self.Z_pow_NFT / self.l_NFT ** 2))
        b2 = torch.einsum('nfk,nft->nkt',
                          self.W_NFK, (1 / self.l_NFT))
        self.H_NKT = self.H_NKT * torch.sqrt(a2 / b2)
        self.H_NKT[self.H_NKT.abs() < self.eps] = self.eps
        self.l_NFT = torch.einsum('nfk,nkt->nft', self.W_NFK, self.H_NKT)

        V_SFMM = torch.einsum('ftmn,lft->lfmn',
                              self.XX_FTMM, 1/self.l_NFT)/self.n_time
        tmp_D_SFM = torch.einsum("fab,sfbc->sfac",
                                 self.D_FMM, V_SFMM)

        tmp_D_SFM = torch.einsum("sfms->sfm", torch.linalg.inv(tmp_D_SFM))

        norm = torch.einsum('sfa,sfab,sfb->sf',
                            tmp_D_SFM.conj(), V_SFMM, tmp_D_SFM)
        self.D_FMM = (tmp_D_SFM / torch.sqrt(norm)
                      [..., None]).conj().permute(1, 0, 2)

    def normalize_WHD(self):
        mu_FN = torch.einsum('fnm,fnm->fn',
                             self.D_FMM.conj(), self.D_FMM).real
        self.D_FMM = self.D_FMM / torch.sqrt(mu_FN[..., None])

        self.W_NFK = torch.einsum('nfk,fn->nfk', self.W_NFK, 1/mu_FN)

        nu_NK = torch.einsum('nfk->nk', self.W_NFK)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.l_NFT = torch.einsum('nfk,nkt->nft', self.W_NFK, self.H_NKT)

        self.Z_pow_NFT = torch.einsum('fnm,ftm->nft',
                                      self.D_FMM,
                                      self.X_FTM).abs()**2

    def separate(self, mic_index=0):
        self.Z_FTN_ILRMA = torch.einsum('fmn,ftn->ftm',  self.D_FMM, self.X_FTM)
        self.Z_FTN_ILRMA = torch.einsum('ftn,fmn->ftnm',
                                  self.Z_FTN_ILRMA,
                                  torch.linalg.inv(self.D_FMM))[..., mic_index]

    def update_log_likelihood(self):
        lk_ent = -(self.Z_pow_NFT / self.l_NFT + torch.log(self.l_NFT)).sum()
        DD = self.D_FMM @ self.D_FMM.adjoint()
        logdetDD = torch.log(torch.linalg.det(DD)).sum().real
        lk = self.n_time * logdetDD.to(torch.complex128) - self.n_mic * self.n_freq * self.n_time * torch.log(torch.tensor(torch.pi))
        lk += lk_ent
        self.log_likelihood.append(lk.abs())


def main():
    data_root = Path("/home/yoshiilab1/soturon/mnmf/code/MNMF_20241218/self_data_2")
    save_root = Path("result_clone")
    save_root.mkdir(exist_ok=True)
    data_file = data_root / "mixture_time_domain_2.wav"
    data, samplerate = sf.read(data_file)
    data = data - data.mean()
    data = torch.tensor(data.T)
    stft_tool = STFT()

    #  x:(M, L) -> x_stft:(M, F, T)
    #  x_stft:(M, F, T) -> x_stft:(F, T, M)

    x_stft = stft_tool.stft(data).permute(1, 2, 0)
    x_stft[x_stft.abs() < 1e-15] = 1e-15
    x_stft = x_stft / x_stft.abs().max()
    print(x_stft.shape)

    ilrma = ILRMA(n_basis=16)
    ilrma.initialize()
    with tqdm(range(100))as pbar:
        for e in pbar:
            ilrma.update()
            pbar.set_description("[Epoch %d]" % (e+1))
            pbar.set_postfix(
                {"lh": "{:.3e}".format(ilrma.log_likelihood[-1].abs())})
    ilrma.separate(mic_index=0)
    n_source = ilrma.Z_FTN_ILRMA.shape[-1]
    fig, axs = plt.subplots(n_source+1, 3, figsize=(10, 10))
    cpu_lk = [lk.to("cpu") for lk in ilrma.log_likelihood]
    axs[0, 1].plot(cpu_lk)
    for i in range(n_source):
        x_log = 10 * torch.log10(x_stft[..., i].abs()**2).to("cpu")
        z_log = 10 * torch.log10(ilrma.Z_FTN_ILRMA[..., i].abs()**2).to("cpu")
        axs[i+1, 0].imshow(x_log,
                           aspect='auto', origin='lower')
        axs[i+1, 1].imshow(z_log,
                           aspect='auto', origin='lower')
        axs[i+1, 2].imshow(x_log - z_log,
                           aspect='auto', origin='lower')
    fig.tight_layout()
    fig.savefig(save_root/"separated_stft.png")
    plt.show()

    recon_sound = stft_tool.istft(ilrma.Z_FTN_ILRMA)
    recon_sound = recon_sound.to("cpu")

    for i in range(n_source):
        sf.write(save_root/f"separated_{i}.wav",
                 recon_sound[i, :], samplerate)


if __name__ == '__main__':
    main()
