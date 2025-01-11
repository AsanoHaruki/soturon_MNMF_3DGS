import torch
import opt_einsum
from plyfile import PlyData
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

def get_R_GS(file_path) -> torch.Tensor:    # Return: (3, Q_n)
    ply_data = PlyData.read(file_path)
    if 'vertex' not in ply_data.elements[0].name:
        raise ValueError("The PLY file does not contain 'vertex' data.")
    vertex_data = ply_data['vertex']
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    coordinates = torch.tensor([x, y, z], dtype=torch.float32).T  # Shape (N, 3)
    return coordinates

D = 3

class Test:
    def __init__(self, n_source=2, n_basis=8):
        self.D = 3
        self.device = torch_setup()
    
    def get_G_hat(self):
        """
        Args: 
            R_GS: Tensor, (Q_n, 3), 中心座標群
        Return:
            G_hat_NFMM: Tensor, (N,F,M,M), Gaussianの中心座標を用いたSCMの理論値
        """

    def get_U_GS(self):
        """Args:
                R_GS: (3) 1点の中心座標
            Return: U_GS: (I,3) 1点とマイクアレイ内のI台のマイクへの方向ベクトル
        """
        ply_locs = get_R_GS()
        array_locs = torch.tensor(mic_array_locs).to(self.device)
        R_N_expanded = 