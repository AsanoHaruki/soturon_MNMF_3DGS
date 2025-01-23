import numpy as np
import torch
import pyroomacoustics as pra
import soundfile as sf
import glob
from scipy.io import wavfile
from pathlib import Path
from sklearn.decomposition import PCA
import scipy.io.wavfile as wav
import scipy.signal as sig
from IPython.display import display, Audio
import os

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

data_dir = "self_data/mixture_wav_files"
N = 2
data_list = []
for f in glob.glob(data_dir+'/*.wav'):
    RATE, data = wav.read(f)
    data_list.append(data)
t_data = np.stack(data_list, axis=1)
print(t_data.shape)

pca = PCA(n_components=N, whiten=True)
H = pca.fit_transform(t_data)
print(H.shape)
seg = 1024
stft_list = []
for i in range(N):
    _,_,Z = sig.stft(H[:,i],nperseg=seg)
    stft_list.append(Z.T)
f_data = np.stack(stft_list, axis=2)
print(f_data.shape)
Array = pra.bss.ilrma(f_data, n_src=None, n_iter=100, proj_back=True, W0=None, n_components=N, return_filters=False, callback=None)

sep = []
for i in range(N):
    x=sig.istft(Array[:,:, -(i+1)].T, nperseg=seg)
    sep.append(x[1])
    # display(Audio(x[1],rate=RATE))

# 保存先のディレクトリを指定
output_dir = "result_pra_ILRMA"
os.makedirs(output_dir, exist_ok=True)

# 音源の保存
for i in range(N):
    output_file = os.path.join(output_dir, f"separated_source_{i+1}.wav")
    # scipy.io.wavfile.writeは整数型のデータを期待するので、適切なスケーリングを行います
    audio_data = np.int16(sep[i] / np.max(np.abs(sep[i]))*32767)
    # 音量を最大化し、データの範囲に収めます
    wav.write(output_file, RATE, sep[i].astype(np.int16))  # sep[i] は音源データ
    print(f"Source {i+1} saved to {output_file}")

for i in range(N):
    display(Audio(sep[i], rate=RATE))