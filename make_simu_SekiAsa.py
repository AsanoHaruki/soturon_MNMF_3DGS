import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import IPython.display as ipd
import scipy.signal
import pyroomacoustics as pra
import torch

# MIC_POSITIONS = np.array([
#     [4.0, 4.0, 0.0], 
#     [4.2, 4.2, 0.0], 
#     [4.0, 4.2, 0.0], 
#     [4.2, 4.0, 0.0]
# ])

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

# 各音源の位置
SOUND_POSITIONS = np.array([
    [5.0, 4.0, 0.0],
    [4.0 + np.cos(np.pi/3), 4.0 + np.sin(np.pi/3), 0.0],
])

SOUND_SPEED = 340.
SAMPLING_RATE = 16000  # 保存するときのサンプリングレート
N_FFT = 1024  # STFTをするときの窓幅
HOP_LENGTH = 256  # STFTをするときのシフト幅（オーバラップをどれだけにするか）
F = int(N_FFT/2+1)  # STFTしたあとの周波数ビンの数

# filename_list = ["data/a01.wav", "data/a02.wav", "data/a03.wav"]
filename_list = ["data/a01.wav", "data/a02.wav"]
N = max(len(SOUND_POSITIONS), len(filename_list))  # 音源数
M = len(MIC_POSITIONS.T)  # マイク数

# 生成した音声の可視化・確認
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
visualize_wav(wav_NT)

plt.figure(figsize=(4, 4))
plt.plot(MIC_POSITIONS.T[:, 0], MIC_POSITIONS.T[:, 1], "o", label="Mic")
plt.plot(SOUND_POSITIONS[:, 0], SOUND_POSITIONS[:, 1], "o", label="Sound")
plt.xlim(2.5, 5.5)
plt.ylim(2.5, 5.5)
plt.legend(fontsize=15)

ROOM_SIZE = [8, 8, 2]  # meter
ABSORPTION = 0.6
MAX_REFLECTION_ORDER = 10


def convolve_RIR(signal_NT):
    convolved_signals = []

    # 部屋のサイズ、壁による音の吸収率、音の反射回数などを指定して登録
    room = pra.ShoeBox(ROOM_SIZE, fs=SAMPLING_RATE,
                       absorption=ABSORPTION, max_order=MAX_REFLECTION_ORDER)
    # マイク配置を登録
    mic_array = pra.MicrophoneArray(MIC_POSITIONS, room.fs)
    room.add_microphone_array(mic_array)

    # 音源配置と信号を登録
    for n in range(N):
        print(n)
        room.add_source(
            position=SOUND_POSITIONS[n], signal=signal_NT[n], delay=0)
#     room.compute_rir()

    # 指定した条件での音源からマイクまでの音の伝達をシミュレート
    room.simulate()
    mixture_signal_MT = room.mic_array.signals
    mixture_signal_MT /= np.abs(mixture_signal_MT).max() * 1.2

    return mixture_signal_MT

# 個別にシミュレートしてから足す
def simu_add(signal_NT, sound_positions, mic_positions):
    N, T = signal_NT.shape
    M = mic_positions.shape[0]
    # min_length = min(signal.shape[0] for signal in signal_NT)
    # signal_NT = np.array([signal[:min_length] for signal in signal_NT])
    # T = min_length
    # room = pra.ShoeBox(
    #     ROOM_SIZE, 
    #     fs = SAMPLING_RATE, 
    #     absorption = ABSORPTION, 
    #     max_order = MAX_REFLECTION_ORDER
    # )
    # mic_array = pra.MicrophoneArray(mic_positions.T, room.fs)
    # room.add_microphone_array(mic_array)
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
sf.write("data/mixture_time_domain.wav", mixture_signal_MT.T, 16000)

visualize_wav(mixture_signal_MT)