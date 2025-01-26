import torch

# G_nf = torch.randn([4, 4], dtype=torch.complex64)
# G_nf = G_nf @ G_nf.t().conj() + torch.eye(4)
# tr_G_nf = torch.trace(G_nf).real
# print(f"tr_G_nf, torch.trace : {tr_G_nf}")
# tr_G_nf_einsum = torch.einsum('ii->', G_nf).real
# print(f"tr_G_nf, torch.einsum : {tr_G_nf_einsum}")
# # print(f"G_NF : {G_NF}")
# L = torch.linalg.cholesky(G_nf)
# # print(f"L : {L}")
# tr_G_nf_L = torch.sum(torch.abs(L)**2)
# print(f"tr_G_nf_L : {tr_G_nf_L}")

# normal_G = G_nf / tr_G_nf
# # print(f"normal_G : {normal_G}")
# tr_normal_G = torch.trace(normal_G).real
# print(f"tr_normal_G : {tr_normal_G}")

# phi_L = torch.sqrt(tr_G_nf_L)
# L_normalize = L / phi_L
# G_normalize_L = torch.einsum('ij,jk->ik', L_normalize, L_normalize.t().conj())
# tr_normal_G_L = torch.trace(G_normalize_L).real
# print(f"tr_normal_G_L : {tr_normal_G_L}")

# 次元拡張版
G_NFMM = torch.randn([2, 3, 4, 4], dtype=torch.complex128)
G_NFMM = (G_NFMM + G_NFMM.conj().transpose(-2, -1)) / 2
G_NFMM = G_NFMM + 4 * torch.eye(G_NFMM.shape[-1], dtype=torch.complex128)
phi_NF = torch.einsum('...ii->...', G_NFMM).real
print(f"phi_NF : {phi_NF}")
L_NFMM = torch.linalg.cholesky(G_NFMM)
print(f"L_NFMM : {L_NFMM}")
# mu_L_NF = torch.sqrt(torch.einsum('nfab,nfab->nf', L_NFMM, L_NFMM))
mu_L_NF = torch.sum(L_NFMM**2, axis=(-1, -2))**0.5
# print(f"mu_L_NF : {mu_L_NF}")
print(f"mu_L_NF : {mu_L_NF.shape}")
L_normalize_NFMM = L_NFMM / mu_L_NF[:, :, None, None]
G_normalize_NFMM = torch.einsum('nfac,nfbc->nfab', L_normalize_NFMM, L_normalize_NFMM.conj())
phi_normalize_NF = torch.einsum('...ii->...', G_normalize_NFMM).real
print(f"phi_normalize_NF : {phi_normalize_NF}")