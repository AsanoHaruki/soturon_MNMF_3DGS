import numpy as np
import soundfile as sf
import mir_eval

# 予測音源と参照音源の読み込み
predicted_file_1 = "result_pra_ILRMA/separated_source_1.wav"
predicted_file_2 = "result_pra_ILRMA/separated_source_2.wav"
reference_file_1 = "data/a02.wav"
reference_file_2 = "data/a03.wav"

# 音源の読み込み
predicted_audio_1, RATE_pred_1 = sf.read(predicted_file_1)
predicted_audio_2, RATE_pred_2 = sf.read(predicted_file_2)
reference_audio_1, RATE_ref_1 = sf.read(reference_file_1)
reference_audio_2, RATE_ref_2 = sf.read(reference_file_2)

print(f"predict: {predicted_audio_1}")
print(f"reference: {reference_audio_1}")
print("Predicted Audio 1 type:", predicted_audio_1.dtype)
print("Reference Audio 1 type:", reference_audio_1.dtype)


# サンプリングレートが一致しているか確認
if RATE_pred_1 != RATE_ref_1 or RATE_pred_2 != RATE_ref_2:
    print("Warning: Sample rates do not match.")

# 各音源の最大値を用いた正規化を行う場合
predicted_audio_1 /= np.max(np.abs(predicted_audio_1))
predicted_audio_2 /= np.max(np.abs(predicted_audio_2))
reference_audio_1 /= np.max(np.abs(reference_audio_1))
reference_audio_2 /= np.max(np.abs(reference_audio_2))
print(f"predict: {predicted_audio_1}")
print(f"reference: {reference_audio_1}")

# 正規化後の最大値と最小値を表示
print("Predicted Audio 1 - Max:", np.max(predicted_audio_1), "Min:", np.min(predicted_audio_1))
print("Predicted Audio 2 - Max:", np.max(predicted_audio_2), "Min:", np.min(predicted_audio_2))
print("Reference Audio 1 - Max:", np.max(reference_audio_1), "Min:", np.min(reference_audio_1))
print("Reference Audio 2 - Max:", np.max(reference_audio_2), "Min:", np.min(reference_audio_2))

# 音源の長さを揃える
max_length = max(len(predicted_audio_1), len(predicted_audio_2), len(reference_audio_1), len(reference_audio_2))
predicted_audio_1 = np.pad(predicted_audio_1, (0, max_length - len(predicted_audio_1)), mode='constant')
predicted_audio_2 = np.pad(predicted_audio_2, (0, max_length - len(predicted_audio_2)), mode='constant')
reference_audio_1 = np.pad(reference_audio_1, (0, max_length - len(reference_audio_1)), mode='constant')
reference_audio_2 = np.pad(reference_audio_2, (0, max_length - len(reference_audio_2)), mode='constant')

# SDRの評価
predicted_audio = np.array([predicted_audio_1, predicted_audio_2])
reference_audio = np.array([reference_audio_1, reference_audio_2])
sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(predicted_audio, reference_audio)

# 結果の表示（音源ごとのSDR）
for i in range(len(sdr)):
    print(f"SDR for Source {i+1}: {sdr[i]:.2f} dB")
    
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# プロット
plt.subplot(2, 1, 1)
plt.plot(predicted_audio_2)
plt.title("Predicted Audio 2")

plt.subplot(2, 1, 2)
plt.plot(reference_audio_2)
plt.title("Reference Audio 2")

plt.show()
