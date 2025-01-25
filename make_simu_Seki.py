import numpy as np
import torch
import soundfile as sf
import librosa
import pickle as pic
import scipy.signal
import matplotlib.pyplot as plt
import IPython.display as ipd
import pyroomacoustics as pra
import os
from scipy.io import wavfile

SOUND_SPEED = 340.
SAMPLING_RATE = 16000  # 保存するときのサンプリングレート
N_FFT = 1024  # 窓幅
HOP_LENGTH = N_FFT / 8  # シフト幅（オーバラップをどれだけにするか）
F = int(N_FFT/2+1)  # 周波数ビン数

ABSORPTION = 0.7
MAX_REFLECTION_ORDER = 3

ROOM_SIZE = [8, 8, 2.5]
center_x = 4
center_y = 4

# マイク準備
def rotate_coordinates(xy, deg: int) -> np.ndarray:
    theta = np.radians(deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return np.dot(R, xy)

def get_whole_3D_mic_locs(mic_array_locs, mic_array_geometry) -> np.ndarray:
    mic_locs = np.empty((3, 0))
    for mic_array_loc in mic_array_locs.T:
        mic_array = np.zeros((3, mic_array_geometry.shape[1])) # (3, Mi)
        mic_array[0:2, :] = rotate_coordinates(mic_array_geometry, mic_array_loc[3]) + mic_array_loc[0:2, None]
        mic_array[2, :] = mic_array_loc[2] # z
        mic_locs = np.append(mic_locs, mic_array, axis=1)
    return mic_locs # (3, M)

n_mics_per_array = 6  #! 各マイクアレイのチャンネル数
radius = 0.1          #! マイクアレイの半径（メートル）
center_z = 1.5        # マイクの高さ（人の耳の高さ）
# マイクアレイの幾何学 (2D座標系での半径方向の配置)
angles = np.linspace(0, 2 * np.pi, n_mics_per_array, endpoint=False)  # 各マイクの角度
mic_array_geometry = np.array([  # 形状: (2, n_mics_per_array)
    radius * np.cos(angles),  # x座標
    radius * np.sin(angles)   # y座標
])
# マイクアレイの中心位置と回転角度
mic_array_locs = np.array([  # 形状: (I, 4)
    [2.0, 2.0, center_z, 0], 
]).T

class array_centers(): #! マイクアレイの位置
    def set_array_centers(self):
        self.array_locs = np.array([
        [2.0, 2.0],         
        ])

        self.gpu_array_locs = torch.tensor([
        [2.0, 2.0],     
        ], dtype=torch.float64)

MIC_POSITIONS = get_whole_3D_mic_locs(mic_array_locs, mic_array_geometry)
print(MIC_POSITIONS)


# 音源設定
SOUND_POSITIONS = np.array([
    [2.18, 6.0, 1.61],
    [6.06, 1.87, 1.46]    
])
gpu_SOUND_POSITIONS = torch.tensor([
    [2.18, 6.0, 1.61],
    [6.06, 1.87, 1.46]
])
ans_R_N = SOUND_POSITIONS
gpu_ans_R_N = gpu_SOUND_POSITIONS


# データ準備
filename_list = ["data/a01.wav", "data/a02.wav"]    #! 正解音源
N = max(len(SOUND_POSITIONS), len(filename_list))
M = len(MIC_POSITIONS)


# 生成音声の可視化
def visualize_spec(Z_FTN):
    for n in range(Z_FTN.shape[2]):
        signal = librosa.core.istft(Z_FTN[:, :, n], hop_length=int((F-1)/2))
        plt.rcParams['figure.figsize'] = (15.0, 3.0)
        plt.imshow(np.log(np.abs(Z_FTN[:, :, n]) +
                          1.0e-8), origin='lower', aspect="auto")
        plt.show()
        ipd.display(ipd.Audio(signal, rate=SAMPLING_RATE))

def visualize_wav(wav_MT):
    M, T = wav_MT.shape
    for m in range(M):
        spec = librosa.core.stft(
            wav_MT[m], n_fft=N_FFT, hop_length=int(N_FFT/4))
        plt.rcParams['figure.figsize'] = (15.0, 3.0)
        plt.imshow(np.log(np.abs(spec) + 1.0e-8),
                   origin='lower', aspect="auto")
        plt.show()
        ipd.display(ipd.Audio(wav_MT[m], rate=SAMPLING_RATE))


wav_list = []
for filename in filename_list:
    wav, fs = sf.read(filename)
    wav_list.append(wav)
min_length = min(map(len, wav_list))
wav_NT = np.array([wav[:min_length] for wav in wav_list])
# visualize_wav(wav_NT)

# 関口式プロット. なんかダサい
# plt.figure(figsize=(4, 4))
# plt.plot(MIC_POSITIONS.T[:, 0], MIC_POSITIONS.T[:, 1], "o", label="Mic")
# plt.plot(SOUND_POSITIONS[:, 0], SOUND_POSITIONS[:, 1], "o", label="Sound")
# plt.xlim(0, 8)
# plt.ylim(0, 8)
# plt.legend(fontsize=15)
# plt.show()

#! 2D plot
plt.figure(figsize=(8, 8))  # 図のサイズを指定
plt.title("Microphones and Sources")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.xlim(0, 8)
plt.ylim(0, 8)
# マイクの位置をプロット
plt.scatter(MIC_POSITIONS[0],MIC_POSITIONS[1], label='Microphones', color='blue', marker='o')
# 音源の位置をプロット
plt.scatter(SOUND_POSITIONS[:, 0], SOUND_POSITIONS[:, 1], label='Sound Sources', color='red', marker='x')
# ラベルの追加
for i, pos in enumerate(SOUND_POSITIONS):
    plt.text(pos[0], pos[1], f'Source {i + 1}, z={SOUND_POSITIONS[i,2]}', fontsize=10, verticalalignment='bottom', horizontalalignment='center')
for i, pos in enumerate(MIC_POSITIONS.T):
    if i == 1:
        plt.text(pos[0], pos[1], f'Mic {i + 1}  Microphone array, z=1.5', fontsize=10, verticalalignment='bottom')
    else:
        plt.text(pos[0], pos[1], f'Mic {i + 1}', fontsize=10, verticalalignment='bottom')
plt.axhline(y=center_y, color='gray', linestyle='--', linewidth=0.8)  # 中心線を描画
plt.grid()  # グリッドを表示
plt.legend()  # 凡例を表示
plt.show()  # 2Dプロット


# シミュレーション
def convolve_RIR(signal_NT):
    convolved_signals = []
    # 部屋のサイズ、壁による音の吸収率、音の反射回数などを指定して登録
    room = pra.ShoeBox(
        ROOM_SIZE, fs=SAMPLING_RATE,
        absorption=ABSORPTION, max_order=MAX_REFLECTION_ORDER)
    # マイク配置を登録
    mic_array = pra.MicrophoneArray(MIC_POSITIONS, room.fs)
    room.add_microphone_array(mic_array)
    # 音源配置と信号を登録
    for n in range(N):
        room.add_source(
            position=SOUND_POSITIONS[n], signal=signal_NT[n], delay=0)
    #room.compute_rir()
    # 指定した条件での音源からマイクまでの音の伝達をシミュレート
    room.simulate()
    mixture_signal_MT = room.mic_array.signals
    mixture_signal_MT /= np.abs(mixture_signal_MT).max() * 1.2
    return mixture_signal_MT

# 個別にシミュレートしてから足す. うまくいかん
def simu_add(signal_NT, sound_positions, mic_positions):
    N, T = signal_NT.shape
    M = mic_positions.shape[0]
    mixture_signal_MT = np.zeros((M, T))
    each_images_NMT = np.zeros((N,M,T))
    for n in range(N):
        temp_room = pra.ShoeBox(
            ROOM_SIZE, 
            fs = SAMPLING_RATE, 
            absorption = ABSORPTION, 
            max_order = MAX_REFLECTION_ORDER
        )
        temp_room.add_microphone_array(pra.MicrophoneArray(mic_positions.T, temp_room.fs))
        temp_room.add_source(position=sound_positions[n], signal=signal_NT[n], delay=0)
        temp_room.simulate()
        observed = temp_room.mic_array.signals[:, :T]
        each_images_NMT[n] = observed
        mixture_signal_MT += observed
    mixture_signal_MT /= np.abs(mixture_signal_MT).max() * 1.2
    return mixture_signal_MT, each_images_NMT

mixture_signal_MT = convolve_RIR(wav_NT)
output_dir = "/home/yoshiilab1/soturon/mnmf/code/self_data/mixture_wav_files/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# output_dir = os.path.dirname(output_path)
for i in range(mixture_signal_MT.shape[0]):
    output_path = os.path.join(output_dir, f"mic_{i+1}_mixture.wav")
    sf.write(output_path, mixture_signal_MT[i,:].T, SAMPLING_RATE)