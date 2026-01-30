import scipy.io as sio
import os
import numpy as np
from scipy.signal import welch, get_window
from scipy.stats import skew, kurtosis

# ================== 1. 配置 ==================
RAW_DATA_PATH = r"E:\DEAP\data_preprocessed_matlab"
SAVE_DIR = r'D:\EEGLAB\Processed_Data'
fs = 128
segment_cnt = 15  # 60s / 4s

os.makedirs(SAVE_DIR, exist_ok=True)

# ================== 2. 统计特征 ==================
def extract_complex_stats(eeg_seg):
    m = np.mean(eeg_seg, axis=1)
    v = np.var(eeg_seg, axis=1)
    s = skew(eeg_seg, axis=1)
    k = kurtosis(eeg_seg, axis=1)
    zcr = np.mean(np.diff(np.sign(eeg_seg), axis=1) != 0, axis=1)

    abs_eeg = np.abs(eeg_seg)
    norm_eeg = abs_eeg / (np.sum(abs_eeg, axis=1, keepdims=True) + 1e-10)
    shannon_en = -np.sum(norm_eeg * np.log2(norm_eeg + 1e-10), axis=1)
    log_energy_en = np.sum(np.log2(eeg_seg**2 + 1e-10), axis=1)

    return np.stack(
        [m, v, s, k, zcr, shannon_en, log_energy_en],
        axis=1
    ).flatten()  # (224,)

# ================== 3. 外周特征 ==================
def extract_peri_features(peri_seg):
    p_m = np.mean(peri_seg, axis=1)
    p_s = np.std(peri_seg, axis=1)
    p_v = np.var(peri_seg, axis=1)
    p_sk = skew(peri_seg, axis=1)
    p_ku = kurtosis(peri_seg, axis=1)
    p_diff = np.mean(np.diff(peri_seg, axis=1), axis=1)

    combined = np.concatenate([p_m, p_s, p_v, p_sk, p_ku, p_diff])
    feat = np.zeros(55)
    feat[:len(combined)] = combined
    return feat

# ================== 4. PSD (MATLAB 风格，50% overlap 保留) ==================
def calculate_psd(seg_eeg, fs):
    nperseg = fs
    noverlap = fs // 2   # ✅ 50% 覆盖率
    window = get_window('hamming', nperseg)

    bands = [(4, 8), (8, 13), (8, 10), (13, 30), (30, 45)]

    f, pxx = welch(
        seg_eeg,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    psd_feat = []
    for fmin, fmax in bands:
        idx = (f >= fmin) & (f <= fmax)
        band_power = np.mean(pxx[:, idx], axis=1)
        psd_feat.append(band_power)

    # 下限保护，避免后续 log 问题
    return np.maximum(np.concatenate(psd_feat), 1e-12)

# ================== 5. 主循环 ==================
all_psds, all_stats, all_peris = [], [], []

for subj in range(1, 33):
    path = os.path.join(RAW_DATA_PATH, f's{subj:02d}.mat')
    if not os.path.exists(path):
        continue

    print(f"Processing Subject {subj:02d}")
    data = sio.loadmat(path)['data']  # (40, 40, 8064)

    for trial in range(40):
        eeg = data[trial, :32, 3*fs:]
        peri = data[trial, 32:, 3*fs:]

        for seg in range(segment_cnt):
            st, ed = seg*4*fs, (seg+1)*4*fs
            seg_eeg = eeg[:, st:ed]
            seg_peri = peri[:, st:ed]

            all_psds.append(calculate_psd(seg_eeg, fs))
            all_stats.append(extract_complex_stats(seg_eeg))
            all_peris.append(extract_peri_features(seg_peri))

# ================== 6. 保存 ==================
np.save(os.path.join(SAVE_DIR, 'final_psds.npy'),
        np.asarray(all_psds, dtype=np.float32))
np.save(os.path.join(SAVE_DIR, 'final_stats.npy'),
        np.asarray(all_stats, dtype=np.float32))
np.save(os.path.join(SAVE_DIR, 'final_peris.npy'),
        np.asarray(all_peris, dtype=np.float32))

print("Code A finished.")
