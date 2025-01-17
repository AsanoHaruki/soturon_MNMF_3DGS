import numpy as np
import torch

# ILRMAをPyTorchで動かすコード

class ILRMA_Su:

    def __init__(self, cpu_use=False, seed=1):
        # search gpu
        torch.manual_seed(seed)
        if cpu_use:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)
        # likelihood list
        self.lk_list = None

    def import_obsdata(self, X_FTM, f_bins=None):
        F, T, M = X_FTM.shape
        self.F = F
        self.T = T
        self.M = M
        self.gpu_X_FTM = torch.tensor(X_FTM, device=self.device)
        self.N = M # for ILRMA

    def initialize_WH(self, n_basis=4, w_mode="dirichlet"):
        EPS = 1e-6
        if w_mode == "rand":
            self.gpu_W_NFK = torch.rand(self.N, self.F, n_basis, dtype=torch.float64, device=self.device).clip(1e-6)
            self.gpu_W_NFK /= self.gpu_W_NFK.sum(dim=1, keepdim=True)
            self.gpu_W_NFK += EPS
        elif w_mode == "dirichlet":
            alpha0 = 2.
            self.gpu_W_NFK = torch.tensor(
                    np.random.dirichlet((alpha0,)*self.F, (self.N, n_basis)).transpose(0,2,1),
                    dtype=torch.float64, device=self.device,
            )        
        self.gpu_H_NKT = torch.rand(self.N, n_basis, self.T, dtype=torch.float64, device=self.device).clip(1e-6)
        self.K = n_basis

    def initialize_D(self):
        '''initialize D with identity matrix'''
        self.gpu_D_FNM = torch.zeros([self.F, self.N, self.M], dtype=torch.complex128, device=self.device)
        self.gpu_D_FNM[:] = torch.eye(self.M, dtype=torch.complex128, device=self.device)

    def update_WH(self):
        EPS = 1e-6
        Z_NFT = torch.einsum("fnm,ftm->nft", self.gpu_D_FNM, self.gpu_X_FTM)
        lambda_NFT = self.gpu_W_NFK @ self.gpu_H_NKT + EPS
        # update W
        a = torch.einsum("nkt,nft->nfk", self.gpu_H_NKT, torch.abs(Z_NFT)**2 / lambda_NFT**2)
        b = torch.einsum("nkt,nft->nfk", self.gpu_H_NKT, lambda_NFT**-1)
        self.gpu_W_NFK = self.gpu_W_NFK * torch.sqrt(a / b)
        self.gpu_W_NFK[self.gpu_W_NFK < EPS] = EPS
        lambda_NFT = self.gpu_W_NFK @ self.gpu_H_NKT + EPS
        # update H
        a = torch.einsum("nfk,nft->nkt", self.gpu_W_NFK, torch.abs(Z_NFT)**2 / lambda_NFT**2)
        b = torch.einsum("nfk,nft->nkt", self.gpu_W_NFK, lambda_NFT**-1)
        self.gpu_H_NKT = self.gpu_H_NKT * torch.sqrt(a / b)
        self.gpu_H_NKT[self.gpu_H_NKT < EPS] = EPS

    def update_D(self):
        lambda_NFT = (self.gpu_W_NFK @ self.gpu_H_NKT).type(torch.complex128)

        for m in range(self.M):
            V_FMM = torch.einsum("fta,ftb,ft->fab", self.gpu_X_FTM, self.gpu_X_FTM.conj(), lambda_NFT[m]**-1) / self.T 
            tmp_FM = torch.linalg.inv(self.gpu_D_FNM @ V_FMM)[:, :, m]
            self.gpu_D_FNM[:, m] = (tmp_FM / torch.sqrt(torch.einsum("fa,fab,fb->f", tmp_FM.conj(), V_FMM, tmp_FM)[:, None])).conj()

    def normalize_WHD(self):
        mu_NF = torch.einsum("fnm,fnm->nf", self.gpu_D_FNM.conj(), self.gpu_D_FNM).real
        self.gpu_D_FNM = self.gpu_D_FNM / torch.sqrt(mu_NF.transpose(0,1))[:,:,None]
        self.gpu_W_NFK = self.gpu_W_NFK / mu_NF[:, :, None]
        nu_NK = torch.sum(self.gpu_W_NFK, dim=1)
        self.gpu_W_NFK = self.gpu_W_NFK / nu_NK[:, None, :]
        self.gpu_H_NKT = self.gpu_H_NKT * nu_NK[:, :, None]

    def separate(self, mic_index=0) -> torch.Tensor:
        hat_S_FTN = torch.einsum("fnm,ftm->ftn", self.gpu_D_FNM, self.gpu_X_FTM) # d_fn^H x_ft
        Z_FTNM = torch.einsum("fmn,ftn->ftnm", torch.linalg.inv(self.gpu_D_FNM), hat_S_FTN)
        return Z_FTNM[:, :, :, mic_index].cpu()
    
    def get_log_likelihood(self) -> torch.Tensor:
        Z_NFT_power = torch.abs(torch.einsum("fnm,ftm->nft", self.gpu_D_FNM, self.gpu_X_FTM)) ** 2
        lambda_NFT = self.gpu_W_NFK @ self.gpu_H_NKT
        return -1 * (Z_NFT_power / lambda_NFT + torch.log(lambda_NFT)).sum() + self.T * torch.slogdet(torch.einsum("fam,fbm->fab", self.gpu_D_FNM, self.gpu_D_FNM.conj()))[1].sum()

    def add_log_likelihood_to_list(self):
        lk = self.get_log_likelihood()
        if self.lk_list is None:
            self.lk_list = [lk.cpu()]
        else:
            self.lk_list.append(lk.cpu())
