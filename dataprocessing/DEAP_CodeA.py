import numpy as np
import scipy.io as sio
import os
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# --- 1. 配置参数 ---
RAW_DATA_PATH = r"E:\DEAP\data_preprocessed_matlab"
SAVE_DIR = r'D:\EEGLAB\Processed_Data'
fs = 128
segment_cnt = 15  # 60s/4s = 15段

if not os.path.exists(SAVE_DIR): 
    os.makedirs(SAVE_DIR)

# --- 2. 统计特征提取函数 (适配原 MATLAB 逻辑) ---
def extract_complex_stats(eeg_seg):
    # eeg_seg: (32, 512)
    m = np.mean(eeg_seg, axis=1)
    v = np.var(eeg_seg, axis=1)
    s = skew(eeg_seg, axis=1)
    k = kurtosis(eeg_seg, axis=1)
    zcr = np.mean(np.diff(np.sign(eeg_seg), axis=1) != 0, axis=1)
    # Shannon Entropy
    abs_eeg = np.abs(eeg_seg)
    norm_eeg = abs_eeg / (np.sum(abs_eeg, axis=1, keepdims=True) + 1e-10)
    shannon_en = -np.sum(norm_eeg * np.log2(norm_eeg + 1e-10), axis=1)
    # Log Energy Entropy
    log_energy_en = np.sum(np.log2(eeg_seg**2 + 1e-10), axis=1)
    
    # 拼接顺序：每个通道的 7 个特征连在一起，最后 flatten 得到 32*7=224 维
    return np.stack([m, v, s, k, zcr, shannon_en, log_energy_en], axis=1).flatten()

# --- 3. 外周特征提取 ---
def extract_peri_features(peri_seg):
    # peri_seg: (8, 512)
    p_m = np.mean(peri_seg, axis=1)
    p_s = np.std(peri_seg, axis=1)
    p_v = np.var(peri_seg, axis=1)
    p_sk = skew(peri_seg, axis=1)
    p_ku = kurtosis(peri_seg, axis=1)
    # 模拟原代码中的差分特征
    p_diff = np.mean(np.diff(peri_seg, axis=1), axis=1)
    
    combined = np.concatenate([p_m, p_s, p_v, p_sk, p_ku, p_diff]) # 48维
    feat = np.zeros(55) # 补齐 55 维
    feat[:len(combined)] = combined
    return feat

# --- 4. 主处理循环 ---
all_psds, all_en_stats, all_peris = [], [], []

for p in range(1, 33):
    file_path = os.path.join(RAW_DATA_PATH, f's{p:02d}.mat')
    if not os.path.exists(file_path): continue
    
    print(f"Extracting Raw Features from Subject S{p:02d}...")
    data = sio.loadmat(file_path)['data'] # (40, 40, 8064)
    
    for t in range(40):
        # 剔除 3 秒基线
        trial_eeg = data[t, :32, 3*fs:]
        trial_peri = data[t, 32:, 3*fs:]
        
        for s in range(segment_cnt):
            start, end = s*fs*4, (s+1)*fs*4
            seg_eeg = trial_eeg[:, start:end]
            seg_peri = trial_peri[:, start:end]
            
            # A. PSD 计算 (5频段: Theta, Alpha, Slow-Alpha, Beta, Gamma)
            # 注意：频段范围需根据你的 MATLAB 函数对齐，这里使用典型值
            bands = [(4, 8), (8, 13), (8, 10), (13, 30), (30, 45)] 
            psd_160 = []
            f, p_val = welch(seg_eeg, fs, nperseg=fs)
            
            for (fmin, fmax) in bands:
                idx = np.logical_and(f >= fmin, f <= fmax)
                band_de = np.mean(p_val[:, idx], axis=1) # 原始 PSD，先不取 log
                psd_160.append(band_de)
            
            all_psds.append(np.concatenate(psd_160)) # 160维
            all_en_stats.append(extract_complex_stats(seg_eeg)) # 224维
            all_peris.append(extract_peri_features(seg_peri)) # 55维

# --- 5. 最终汇总与保存 ---
print("\nSaving raw feature matrices...")
all_psds = np.nan_to_num(np.array(all_psds))
all_en_stats = np.nan_to_num(np.array(all_en_stats))
all_peris = np.nan_to_num(np.array(all_peris))

# 保存为代码 B 所需的原材料
np.save(os.path.join(SAVE_DIR, 'final_psds.npy'), all_psds.astype(np.float32))
np.save(os.path.join(SAVE_DIR, 'final_stats.npy'), all_en_stats.astype(np.float32))
np.save(os.path.join(SAVE_DIR, 'final_peris.npy'), all_peris.astype(np.float32))

print(f"Code A finished. Please run Code B to generate Topomaps.")