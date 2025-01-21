from cProfile import label
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import datetime
import torch
from pathlib import Path
from make_simu_2_Sumura_corner import (
    mic_array_locs, mic_array_geometry, SOUND_POSITIONS,
    SOUND_SPEED, n_mics_per_array, ans_R_N, gpu_ans_R_N, room_size
)
# import matplotlib.animation as animation
# from MNMF_Sumura_Model_add_estimation_R import mnmf

today = datetime.datetime.now()
timestamp = today.strftime('%Y%m%d_%H%M%S')
save_root = Path("result_Smura_model_add_estimation_R")
save_root.mkdir(exist_ok=True)

def plot_likelihood(log_likelihood, title="log likelihood", filename="likelihood", it=None, save=False, draw=False):
    plt.clf()
    if draw:
        plt.plot(log_likelihood[1:])
        plt.draw()
    else:
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        plt.title(title)
        plt.plot(log_likelihood[1:])
    if save:
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.png")
    # plt.show()
    plt.close()

def plot_sdr_score(sdr_score, plot_interval, filename="sdr_score", it=None, save=False):
    for n in range(sdr_score.shape[0]):
        plt.clf()
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        plt.plot(sdr_score[n], marker="D")
        plt.xticks(np.arange(0, sdr_score.shape[1]), np.arange(1, sdr_score.shape[1]+1)*plot_interval) 
        if save:
            plt.savefig(f"{filename}_N{str(n).zfill(2)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
    plt.close()


def plot_WH(w, h, psd, filename="WH_spectrogram",it=None, save=False):
    for n in range(w.shape[0]):
        plt.clf()

        fig = plt.figure(figsize=(11,9))
        axes = fig.subplot_mosaic(
            [
                [".", "A", "A"],
                ["B", "C", "C"],
                ["B", "C", "C"],
            ],
        )

        # データをプロット
        im1 = axes["B"].imshow(w[n], 
                        aspect = 'auto',
                        interpolation = "none",
                        origin = "lower"
                    )          
        # 軸設定
        axes["B"].set_xlabel('Basis')
        axes["B"].set_ylabel('Frequency')
        axes["B"].set_title('W_nKF')

        axins1 = inset_axes(axes["B"],
            width="5%",  # width = 5% of parent_bbox width
            height="50%",  # height : 50%
            loc='lower left',
            bbox_to_anchor=(-0.40, 0., 1, 1),
            bbox_transform=axes["B"].transAxes,
        )
        fig.colorbar(im1, cax=axins1)
                    
        # データをプロット
        im2 = axes["A"].imshow(h[n], 
                        interpolation = "none",
                        aspect = 'auto',
                    )          
        # 軸設定
        axes["A"].set_xlabel('Time')
        axes["A"].set_ylabel('Basis')
        axes["A"].set_title('H_nKT')

        axins2 = inset_axes(axes["A"],
            width="5%",  # width = 5% of parent_bbox width
            height="50%",  # height : 50%
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=axes["A"].transAxes,
        )
        fig.colorbar(im2, cax=axins2)


        # データをプロット
        im3 = axes["C"].imshow(psd[n], 
            aspect = 'auto',
            interpolation = "none",
            origin = "lower"
        )          
        # 軸設定
        axes["C"].set_ylabel('Frequency')
        axes["C"].set_xlabel('Time')
        axes["C"].set_title('log_lambda_nFT')

        # カラーバーを設定する。
        axins3 = inset_axes(axes["C"],
            width="5%",  # width = 5% of parent_bbox width
            height="50%",  # height : 50%
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=axes["C"].transAxes,
        )
        fig.colorbar(im3, cax=axins3)

        # グラフを表示する。
        if save:
            plt.savefig(f"{filename}_N{str(n).zfill(2)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
        # plt.show()
    plt.close()


def plot_X(X_NFT, filename="X_spectrogram", it=None, save=False):
    for n in range(X_NFT.shape[0]):
        plt.clf()
        fig = plt.figure(figsize=(11,8))
        ax1 = fig.add_subplot(1, 1, 1)

        # データをプロット
        im = ax1.imshow(X_NFT[n], 
                        aspect = 'auto',
                        interpolation = "none",
                        origin = "lower",
                        vmin = -30,
                        vmax = 10,
                    )          
        # 軸設定
        ax1.set_ylabel('Frequency', fontsize=16)
        ax1.set_xlabel('Time', fontsize=16)
        # ax1.set_title('X_nFT')
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # カラーバーを設定する。
        # axins = inset_axes(ax1,
        #     width="5%",  # width = 5% of parent_bbox width
        #     height="50%",  # height : 50%
        #     loc='lower left',
        #     bbox_to_anchor=(1.05, 0., 1, 1),
        #     bbox_transform=ax1.transAxes,
        # )
        # fig.colorbar(im, cax=axins)

        plt.tight_layout()
        # グラフを表示する。
        if save:
            plt.savefig(f"{filename}_N{str(n).zfill(2)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
            plt.savefig(f"{filename}_N{str(n).zfill(2)}{'' if it is None else '_I'+str(it).zfill(3)}.pdf")
        # plt.show()
    plt.close()


def plot_U_NT(U_NT, filename="U_NT", z=False, save=False):
    # plt.clf()
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set(xlim=(-1.1,1.1), ylim=(-1.1,1.1))
    ax1.set_xlabel("x")
    ax1.set_ylabel("z" if z else "y")        
    ax1.grid()
    ax1.plot(0, 0, marker='D', markersize=3, color='black', linestyle='solid')

    for n in range(U_NT.shape[0]):
        # print(U_NT[n, :, 0], U_NT[n, :, 1])
        ax1.plot(U_NT[n, :, 0], U_NT[n, :, 1], marker='D', markersize=3, linestyle='solid')
        ax1.plot(U_NT[n, 0, 0], U_NT[n, 0, 1], marker='D', markersize=3, color='red', linestyle='solid')

    if save:
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.eps")
    # plt.show()
    plt.close()


def plot_U_N(U_N, ans_U_N=None, filename="U_N", z=False, it=None, save=False):
    # plt.clf()
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set(xlim=(-1.1,1.1), ylim=(-1.1,1.1))
    ax1.set_xlabel("x")
    ax1.set_ylabel("z" if z else "y")        
    ax1.grid()
    ax1.plot(0, 0, marker='D', markersize=3, color='black', linestyle='solid')

    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax1.plot(x, y, color="gray")  # 円をプロット
    
    for n in range(U_N.shape[0]):
        # print(U_NT[n, :, 0], U_NT[n, :, 1])
        ax1.plot()
        ax1.scatter(U_N[n, 0], U_N[n, 2 if z else 1], marker='D', s=25, label=f"Source {n}", zorder=10)
    if ans_U_N is not None:
        ax1.scatter(ans_U_N[:, 0], ans_U_N[:, 2 if z else 1], marker='x', s=60, color='red', label="GT", zorder=10)

    plt.legend(loc='upper right', fontsize=14)

    if save:
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.png")
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.eps")
    # plt.show()
    plt.close()
    

def plot_R_N_with_initialR(R_N, init_R_N, room_size=None, ans_R_N=None, mic_locs=None, filename="R_N_with_init_R_N", z=False, it=None, save=False):
    axis2 = 2 if z else 1        
    fig, ax1 = plt.subplots(figsize=(room_size[0]+1.5,room_size[axis2]))
    # fig, ax1 = plt.subplots()
    if room_size is not None:
        ax1.set(xlim=(0.,room_size[0]), ylim=(0.,room_size[axis2]))
    ax1.set_xlabel("x")
    ax1.set_ylabel("z" if z else "y")
    ax1.set_xticks(np.arange(0, room_size[0]+0.1, 1))
    ax1.set_yticks(np.arange(0, room_size[axis2]+0.1, 1))
    ax1.grid()

    if ans_R_N is not None:
        # for n in range(ans_R_N.shape[0]):
            # ax1.plot(ans_R_N[n, 0], ans_R_N[n, 2 if z else 1], marker='x', markersize=10, color='red', linestyle='solid')
        ax1.scatter(ans_R_N[:, 0], ans_R_N[:, axis2], marker='x', s=60, color='red', label="Ground truth")

    for n in range(R_N.shape[0]):
        # ax1.plot(R_N[n, 0], R_N[n, 2 if z else 1], marker='D', markersize=7, linestyle='solid')
        ax1.scatter(R_N[n, 0], R_N[n, axis2], marker='D', s=25, label=f"Source {n}")
        
    for n in range(init_R_N.shape[0]):
        # ax1.plot(R_N[n, 0], R_N[n, 2 if z else 1], marker='D', markersize=7, linestyle='solid')
        ax1.scatter(init_R_N[n, 0], init_R_N[n, axis2], marker='*', s=25, label=f"init_Source {n}")

    if mic_locs is not None:
        # for m in range(mic_locs.shape[1]):
        ax1.scatter(mic_locs[0, :], mic_locs[axis2, :], marker='o', color='black', s=15, label="Mic_Arrays")

    ax1.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()

    if save:
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.png")
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.eps")
    plt.show()
    plt.close()

# def create_gif(self):
#     """
#     GIFを生成する
#     """
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.set_xlim(0, room_size[0])
#     ax.set_ylim(0, room_size[1])
#     ax.set_xticks(np.arange(0, room_size[0] + 0.1, 1))
#     ax.set_yticks(np.arange(0, room_size[1] + 0.1, 1))
#     ax.grid()

#     mic_array_x = mic_array_locs[0, :]
#     mic_array_y = mic_array_locs[1, :]
#     ax.scatter(mic_array_x, mic_array_y, color="black", label="Mic Arrays", s=15)
#     ax.scatter(self.init_R_N[:, 0], self.init_R_N[:, 1], marker="*", color="blue", label="Initial Sources", s=25)
#     ax.scatter(ans_R_N[:, 0], ans_R_N[:, 1], marker="x", color="red", label="Ground Truth", s=60)

#     scat = ax.scatter([], [], marker="D", color="green", label="Estimated Sources", s=25)
#     ax.legend()

#     def update(frame):
#         scat.set_offsets(frame)
#         return scat,

#     anim = animation.FuncAnimation(fig, update, frames=mnmf.gif_positions, blit=True)
#     gif_filename = save_root / f"{timestamp}_R_N_optimization.gif"
#     anim.save(gif_filename, writer="imagemagick", fps=2)
#     print(f"GIF saved at {gif_filename}")


def plot_R_N(R_N, room_size=None, ans_R_N=None, mic_locs=None, filename="R_N", z=False, it=None, save=False):
    axis2 = 2 if z else 1        
    fig, ax1 = plt.subplots(figsize=(room_size[0]+1.5,room_size[axis2]))
    # fig, ax1 = plt.subplots()
    if room_size is not None:
        ax1.set(xlim=(0.,room_size[0]), ylim=(0.,room_size[axis2]))
    ax1.set_xlabel("x")
    ax1.set_ylabel("z" if z else "y")
    ax1.set_xticks(np.arange(0, room_size[0]+0.1, 1))
    ax1.set_yticks(np.arange(0, room_size[axis2]+0.1, 1))
    ax1.grid()

    if ans_R_N is not None:
        # for n in range(ans_R_N.shape[0]):
            # ax1.plot(ans_R_N[n, 0], ans_R_N[n, 2 if z else 1], marker='x', markersize=10, color='red', linestyle='solid')
        ax1.scatter(ans_R_N[:, 0], ans_R_N[:, axis2], marker='x', s=60, color='red', label="Ground truth")

    for n in range(R_N.shape[0]):
        # ax1.plot(R_N[n, 0], R_N[n, 2 if z else 1], marker='D', markersize=7, linestyle='solid')
        ax1.scatter(R_N[n, 0], R_N[n, axis2], marker='D', s=25, label=f"Source {n}")

    if mic_locs is not None:
        # for m in range(mic_locs.shape[1]):
        ax1.scatter(mic_locs[0, :], mic_locs[axis2, :], marker='o', color='black', s=15, label="Mic_Arrays")

    ax1.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()

    if save:
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.png")
        plt.savefig(f"{filename}{'' if it is None else '_I'+str(it).zfill(3)}.eps")
    # plt.show()
    plt.close()

def trace(input: torch.Tensor, axis1=-2, axis2=-1):
    assert input.shape[axis1] == input.shape[axis2], input.shape
    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]
    shape[axis2] = 1
    strides[axis2] = 0
    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(axis1, axis2))

def np_multi_log_prob_invwishart(S: np.ndarray, nu: int, Phi: np.ndarray):
    """
    return log inverse Wishart distribution W^-1(S | nu, Phi)
    """
    d = S.shape[-1]
    return nu/2 * np.log(np.linalg.det(Phi)) - (nu+d+1)/2 * np.log(np.linalg.det(S)) + np.trace(- 1/2 * Phi @ np.linalg.inv(S), axis1=-2, axis2=-1)

def multi_log_prob_invwishart(S: torch.Tensor, nu: int, Phi: torch.Tensor):
    """
    return log inverse Wishart distribution W^-1(S | nu, Phi)
    """
    d = S.shape[-1]
    return nu/2 * torch.slogdet(Phi)[1] - (nu+d+1)/2 * torch.slogdet(S)[1] + trace(- 1/2 * Phi @ torch.linalg.inv(S))

def plot_inv_wishart(f_bins, mic_locs, G_NFMM, N, U=None, D=2, filename="IW", it=None, save=False):
    c = 340
    degrees = np.linspace(-180, 180, num=361)
    rads = np.deg2rad(degrees)
    U_rads = np.zeros((361, D), dtype=np.float64)
    U_rads[:, 0] = np.cos(rads)
    U_rads[:, 1] = np.sin(rads)

    omega = 2 * np.pi * f_bins # (F)
    delay = - np.einsum("dm,nd->nm", mic_locs, U_rads) / c # (N, M)
    B_rads = np.exp(-1j * np.einsum("f,nm->nfm", omega, delay)) # (N, F, M)

    hat_G_rads = np.einsum("nfa,nfb->nfab", B_rads, B_rads.conj()) + 1e-3 * np.eye(mic_locs.shape[1])[None, None, ...]
    u_0 = np.array([1., 0.], dtype=np.float64)
    for n in range(N):
        dist = np_multi_log_prob_invwishart(G_NFMM[n], mic_locs.shape[1]+1, hat_G_rads)
        dist = np.sum(dist, axis=1).real
        if U is not None:
            estimated_theta = np.arccos((U[n,0:2] @ u_0)/(np.linalg.norm(U[n,0:2])))
            estimated_deg = np.rad2deg(estimated_theta)*np.sign(U[n,1])

        wid = dist.max() - dist.min()
        plt.clf()
        plt.title("Inverse Wishart")
        plt.plot(degrees, dist, label="dist")
        # plt.vlines(-135, ymin=dist.min()-wid*0.1, ymax=dist.max()+wid*0.1, colors="red", label="ans")
        # plt.vlines(-45, ymin=dist.min()-wid*0.1, ymax=dist.max()+wid*0.1, colors="red")
        if U is not None:
            plt.vlines(estimated_deg, ymin=dist.min()-wid*0.05, ymax=dist.max()+wid*0.05, colors="orange", label="estimated")
        plt.legend()
        if save:
            plt.savefig(f"{filename}_N{str(n)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
        # plt.show()
    plt.close()

c = 340

def plot_2array_inv_wishart(f_bins, array_geometry, G_NFMM, N, D=2, filename="IW_2array", it=None, save=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    c = 340
    M = G_NFMM.shape[-1]
    Mi = array_geometry.shape[1]
    gpu_G_NFMM = torch.tensor(G_NFMM[:N], device=device)
    #
    degrees = np.linspace(-180, 180, num=361)
    rads = np.deg2rad(degrees)
    U_rads = np.zeros((361, D), dtype=np.float64)
    U_rads[:, 0] = np.cos(rads)
    U_rads[:, 1] = np.sin(rads)
    #
    omega = 2 * np.pi * f_bins # (F)
    delay = - np.einsum("dm,nd->nm", array_geometry, U_rads) / c # (N, M)
    B_rads = np.exp(-1j * np.einsum("f,nm->nfm", omega, delay)) # (N, F, M)
    gpu_B_rads = torch.tensor(B_rads, device=device)
    # ブロック対角hat_Gの作成 # (deg1, deg2, F, M, M)
    gpu_hat_Gi_rads = torch.einsum("nfa,nfb->nfab", gpu_B_rads, gpu_B_rads.conj()) + 1e-3 * torch.eye(array_geometry.shape[1], device=device)[None, None, ...]
    inv_wishart_NXY = torch.zeros((N, 361, 361), dtype=torch.complex128, device=device)
    for deg1 in range(360):
        gpu_hat_G_rads = torch.zeros((361, f_bins.shape[0], M, M), dtype=torch.complex128, device=device)
        gpu_hat_G_rads[:, :, :Mi, :Mi] = gpu_hat_Gi_rads[deg1, ...]
        gpu_hat_G_rads[:, :, Mi:, Mi:] = gpu_hat_Gi_rads[None, ...]
        # IW
        inv_wishart_NY = torch.sum(
            multi_log_prob_invwishart(gpu_G_NFMM[:,None,...], M+1, (2*M+1)*gpu_hat_G_rads[None, ...]),
            dim=2)
        inv_wishart_NXY[:, deg1, :] = inv_wishart_NY
    inv_wishart_NXY = inv_wishart_NXY.cpu()
    # plot
    for n in range(N):
        plt.clf()
        plt.figure(figsize=(5,5))
        plt.title("Inverse Wishart")
        plt.imshow(inv_wishart_NXY.numpy().real[n].T,
            aspect = 'auto',
            interpolation = "none",
            origin = "lower"
        )
        plt.xlabel("i=1 [deg]")
        plt.ylabel("i=2 [deg]")
        plt.xticks(np.linspace(0, 360, num=13, dtype=int), np.linspace(-180, 180, num=13, dtype=int))
        plt.yticks(np.linspace(0, 360, num=13, dtype=int), np.linspace(-180, 180, num=13, dtype=int))
        if save:
            plt.savefig(f"{filename}_N{str(n)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
    plt.close()


def plot_mesh_inv_wishart(
        f_bins, array_locs, array_geometry, G_NFMM, N, room_size=None,
        filename="IW", it=None, save=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_f_bins = torch.tensor(f_bins, device=device)
    gpu_array_locs = torch.tensor(array_locs, device=device)
    gpu_array_geometry = torch.tensor(array_geometry, device=device)
    gpu_G_NFMM = torch.tensor(G_NFMM, device=device)
    if room_size is None:
        room_size = np.array([10, 10, 3])
    mesh = torch.zeros((int(room_size[0]*10+1), int(room_size[1]*10+1), 2), device=device)
    mesh[:, :, 0] = torch.linspace(0, room_size[0], int(room_size[0]*10+1), device=device)[:, None]
    mesh[:, :, 1] = torch.linspace(0, room_size[1], int(room_size[1]*10+1), device=device)[None, :]
    U_IXYD = mesh[None, ...] - gpu_array_locs.T[:, None, None, :2] # (I, X, Y, D)
    theta = - torch.deg2rad(gpu_array_locs[2]) # (I)
    I = gpu_array_locs.shape[1]
    # 平面回展行列の生成
    R_mat = torch.zeros((I, 2, 2), dtype=torch.float64, device=device) # (I, D, D)
    R_mat[:, 0, 0] =  torch.cos(theta)
    R_mat[:, 0, 1] = -torch.sin(theta)
    R_mat[:, 1, 0] =  torch.sin(theta)
    R_mat[:, 1, 1] =  torch.cos(theta)
    # 回転
    U_IXYD = torch.squeeze(R_mat[:, None, None,...] @ U_IXYD[..., None])
    U_IXYD /= torch.linalg.norm(U_IXYD, axis=3, keepdims=True)
    # steering vec
    omega = 2 * torch.pi * gpu_f_bins # (F)
    delay = - torch.einsum("dm,ixyd->ixym", gpu_array_geometry, U_IXYD) / c # (I, X, Y, Mi)
    B_IXYFMi = torch.exp(-1j * torch.einsum("f,ixym->ixyfm", omega, delay)) # (I, F, X, Y, Mi)
    _, X, Y, F, Mi = B_IXYFMi.shape
    del U_IXYD
    # hat_G
    inv_wishart_NXY = torch.zeros((N, int(room_size[0]*10+1), int(room_size[1]*10+1)), dtype=torch.complex128, device=device)
    for f in range(F):
        hat_G_IXYMiMi = torch.zeros((I, X, Y, Mi, Mi), dtype=torch.complex128)
        hat_G_IXYMiMi = torch.einsum("ixya,ixyb->ixyab", B_IXYFMi[:,:,:,f], B_IXYFMi.conj()[:,:,:,f]) + 1e-3 * torch.eye(Mi, device=device)[None, None, None, ...]
        hat_G_XYMM = torch.zeros((X, Y, Mi*I, Mi*I), dtype=torch.complex128, device=device)
        for i in range(I):
            hat_G_XYMM[:, :, Mi*i:Mi*(i+1), Mi*i:Mi*(i+1)] = hat_G_IXYMiMi[i,:,:]
        # 各座標の尤度を計算
        for n in range(N):
            inv_wishart_NXY[n, :, :] += multi_log_prob_invwishart(gpu_G_NFMM[n, None, None, f], Mi*I+1, (2*Mi*I+1)*hat_G_XYMM[:,:])
    inv_wishart_NXY = inv_wishart_NXY.cpu()
    del B_IXYFMi
    del hat_G_IXYMiMi
    del hat_G_XYMM
    #
    for n in range(N):
        plt.clf()
        plt.figure(figsize=(5,5))
        plt.title("Inverse Wishart")
        plt.imshow(inv_wishart_NXY.numpy().real[n].T,
            aspect = 'auto',
            interpolation = "none",
            origin = "lower"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(np.linspace(0, int(room_size[0]*10), num=int(room_size[0]+1)), np.linspace(0, room_size[0], num=int(room_size[0]+1), dtype=int))
        plt.yticks(np.linspace(0, int(room_size[1]*10), num=int(room_size[1]+1)), np.linspace(0, room_size[1], num=int(room_size[1]+1), dtype=int))
        if save:
            plt.savefig(f"{filename}_N{str(n)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
        # plt.show()
    plt.close()

# def plot_mesh_inv_wishart_3D(

# )


def plot_grid_search_result(
    xlim, ylim, zlim, inv_wishart_NXYZ: np.ndarray, filename="gird_search", title=None, it=None, save=False):
    N, X, Y, Z = inv_wishart_NXYZ.shape
    for n in range(N):
        for z, z_val in enumerate(np.linspace(zlim[0], zlim[1], zlim[2])):
            plt.clf()
            plt.figure(figsize=(xlim[1]-xlim[0],ylim[1]-ylim[0]))
            if title is None:
                plt.title(f"Grid Search n={n} z={round(z_val, 2)}")
            else:
                plt.title(f"{title} n={n} z={round(z_val, 2)}")
            plt.imshow(inv_wishart_NXYZ.real[n,:, :, z].T,
                aspect="auto",
                interpolation="none",
                origin="lower",
                vmin=np.min(inv_wishart_NXYZ.real[n]),
                vmax=np.max(inv_wishart_NXYZ.real[n]),
            )
            plt.xlabel("x")
            plt.ylabel("y")
            if X > 40:
                plt.xticks(np.linspace(0, X-1, int(X/10)+1), np.linspace(xlim[0], xlim[1], int(xlim[2]/10)+1).round(2))
                plt.yticks(np.linspace(0, Y-1, int(Y/10)+1), np.linspace(ylim[0], ylim[1], int(ylim[2]/10)+1).round(2))
            else:
                plt.xticks(np.linspace(0, X-1, X), np.linspace(xlim[0], xlim[1], xlim[2]).round(2))
                plt.yticks(np.linspace(0, Y-1, Y), np.linspace(ylim[0], ylim[1], ylim[2]).round(2))
            plt.xticks(rotation=90)
            if save:
                plt.savefig(f"{filename}_N{str(n)}_Z{str(z)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
    plt.close()


def plot_mesh_eval_result(
    xlim, ylim, zlim, 
    eval_NXYZ: np.ndarray, R_N: np.ndarray, ans_R_N=None, mic_locs=None, 
    filename="gird_search", title=None, it=None, save=False):
    N, X, Y, Z = eval_NXYZ.shape
    for n in range(N):
        for z, z_val in enumerate(np.linspace(zlim[0], zlim[1], zlim[2])):
            plt.clf()
            plt.figure(figsize=(xlim[1]-xlim[0],ylim[1]-ylim[0]))
            # if title is None:
                # plt.title(f"Grid Search n={n} z={round(z_val, 2)}")
            # else:
            #     plt.title(f"{title} n={n} z={round(z_val, 2)}")
            plt.imshow(eval_NXYZ[n,:, :, z].T,
                aspect="auto",
                interpolation="none",
                origin="lower",
                vmin=np.percentile(eval_NXYZ[n], 30),
                # vmin=np.min(eval_NXYZ[n]),
                vmax=np.max(eval_NXYZ[n]),
            )
            # 座標変換
            x_trans = lambda rx: (rx-xlim[0])/(xlim[1]-xlim[0])*(X-1)
            y_trans = lambda ry: (ry-ylim[0])/(ylim[1]-ylim[0])*(Y-1)        
            #
            for rn in range(R_N.shape[0]):
                plt.scatter(x_trans(R_N[rn, 0]), y_trans(R_N[rn, 1]), marker='D', s=30, ec="w", label="estimated")

            if ans_R_N is not None:
                # for n in range(ans_R_N.shape[0]):
                plt.scatter(x_trans(ans_R_N[:, 0]), y_trans(ans_R_N[:, 1]), marker='x', s=30, color='red', label="ground truth")

            if mic_locs is not None:
                # for m in range(mic_locs.shape[1]):
                plt.scatter(x_trans(mic_locs[0, :]), y_trans(mic_locs[1, :]), marker='o', color='black', s=15, label="mics")

            plt.xlabel("x")
            plt.ylabel("y")
            if X > 40:
                plt.xticks(np.linspace(0, X-1, int(X/10)+1), np.linspace(xlim[0], xlim[1], int(xlim[2]/10)+1).round(2))
                plt.yticks(np.linspace(0, Y-1, int(Y/10)+1), np.linspace(ylim[0], ylim[1], int(ylim[2]/10)+1).round(2))
            else:
                plt.xticks(np.linspace(0, X-1, X), np.linspace(xlim[0], xlim[1], xlim[2]).round(2))
                plt.yticks(np.linspace(0, Y-1, Y), np.linspace(ylim[0], ylim[1], ylim[2]).round(2))
            plt.xticks(rotation=90)
            plt.legend()
            if save:
                plt.savefig(f"{filename}_N{str(n)}_Z{str(z)}{'' if it is None else '_I'+str(it).zfill(3)}.png")
    plt.close()


