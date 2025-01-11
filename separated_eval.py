import numpy as np
import soundfile as sf
import mir_eval

# 元の音源
true_sources = [
    np.array(sf.read('data/arctic_a0002.wav')[0]),
    np.array(sf.read('data/arctic_b0540.wav')[0])
]
# 分離音
separated_sources = [
    np.array(sf.read('/home/yoshiilab1/soturon/mnmf/code/result_2_source_Sumuramodel/20250110_161224_separated_0_Sumura_model- lr_l=1e-3, lr_r=1e-3, n_wh_update=500, n_g_update=100 ,n_R_update=10, gif_frames=50, alpha=1e-7.wav')[0]),
    np.array(sf.read('/home/yoshiilab1/soturon/mnmf/code/result_2_source_Sumuramodel/20250110_161224_separated_1_Sumura_model- lr_l=1e-3, lr_r=1e-3, n_wh_update=500, n_g_update=100 ,n_R_update=10, gif_frames=50, alpha=1e-7.wav')[0])
]

# 最大長を計算
max_length = max(true_sources[0].shape[0], true_sources[1].shape[0], separated_sources[0].shape[0], separated_sources[1].shape[0])

# 長さを一致させるためのパディング
true_sources_padded = [np.pad(source, (0, max_length - source.shape[0])) for source in true_sources]
separated_sources_padded = [np.pad(source, (0, max_length - source.shape[0])) for source in separated_sources]

# 最大相関を用いて音源の対応関係を推定
correlations = [np.corrcoef(true_source, est_source)[0, 1]
                for true_source in true_sources_padded for est_source in separated_sources_padded]
# 最大相関を使って、分離された音源の順番を並べ替え
correlations = np.array(correlations).reshape(len(true_sources_padded), len(separated_sources_padded))
max_indices = np.argmax(correlations, axis=1)
# 元の音源と分離された音源を対応させる
sorted_separated_sources = [separated_sources_padded[i] for i in max_indices]

# リストを 2 次元の NumPy 配列に変換
true_sources_padded = np.array(true_sources_padded)
sorted_separated_sources = np.array(sorted_separated_sources)

# 音源分離の評価
sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(true_sources_padded, sorted_separated_sources)

# 評価結果を表示
print(f"SDR: {sdr}")
print(f"SIR: {sir}")
print(f"SAR: {sar}")

