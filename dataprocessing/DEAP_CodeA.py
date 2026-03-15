import scipy.io as sio
import os
import numpy as np
from scipy import signal,interpolate
from scipy.signal import welch, butter, filtfilt, detrend,get_window
from scipy.stats import skew, kurtosis
import neurokit2 as nk
import numpy as np
import warnings
import matplotlib.pyplot as plt



# !!!1. 配置 
RAW_DATA_PATH = r"E:\DEAP\data_preprocessed_matlab"
SAVE_DIR = r'D:\EEGLAB\Processed_Data'
fs = 128
segment_cnt = 15
EPS = 1e-8

os.makedirs(SAVE_DIR, exist_ok=True)
print("🚨 DEAP VERSION: PERI LOG+ZSCORE ENABLED 🚨")

# !!!2. EEG统计特征 
def extract_complex_stats(eeg_seg):
    m = np.mean(eeg_seg, axis=1)#均值
    v = np.var(eeg_seg, axis=1)#方差
    s = skew(eeg_seg, axis=1)#偏度
    k = kurtosis(eeg_seg, axis=1)
    zcr = np.mean(np.diff(np.sign(eeg_seg), axis=1) != 0, axis=1)#将信号转为符号：正数为1，负数为-1，0为0， 计算相邻时间点的符号差，符号变化的位置（从正变负或负变正），计算零交叉率

    abs_eeg = np.abs(eeg_seg)#取绝对值
    norm_eeg = abs_eeg / (np.sum(abs_eeg, axis=1, keepdims=True) + EPS)#进行概率归一化
    shannon_en = -np.sum(norm_eeg * np.log2(norm_eeg + EPS), axis=1)#算香农熵
    log_energy_en = np.sum(np.log2(eeg_seg**2 + EPS), axis=1)#对数能量熵

    return np.stack([m, v, s, k, zcr, shannon_en, log_energy_en], axis=1).flatten()

#!!!3.提取外周特征
#提取用较短时间段可以充分提取的信号特征
def extract_peri_fast(peri_seg, fs=128):
    feats = []

    # 1. EOG (9维): 眨眼率(1), 5频点PSD(5), 均值(1), 方差(1), ZCR(1)
    eog_sig = (peri_seg[0] + peri_seg[1]) / 2.0#从电极创建双极EOG信号
    # 去线性趋势
    eog_detrend = signal.detrend(eog_sig)
    eog_std = np.std(eog_detrend) + 1e-8
    # 确保在窗口内有基本的能量波动才判定眨眼
    if eog_std > 1e-5: 
        eog_smoothed = signal.medfilt(eog_detrend, 3)
        eog_z = eog_smoothed / eog_std
        blink_count = np.sum((eog_z > 2.5) & (np.roll(eog_z, -1) < 2.5))
        blink_rate = blink_count / (len(eog_sig) / fs)
    else:
        blink_rate = 0.0
    psd_eog_array = extract_psd(eog_sig, fs, [[2,4], [4,6], [6,8], [8,10], [10,12]], nperseg=256)

    mean_eog = np.mean(eog_sig)
    var_eog = np.var(eog_detrend)
    zcr_eog = np.mean(np.abs(np.diff(np.sign(eog_detrend))) > 0)
    
    feats += [blink_rate] + psd_eog_array.tolist() + [mean_eog, var_eog, zcr_eog]
    # 2. EMG (8维): 5频点PSD(5), 均值(1), 方差(1), ZCR(1)
    emg_sig = peri_seg[2]#颧肌电对情绪更敏感
    emg_detrend = signal.detrend(emg_sig)#去线性趋势
    psd_emg = extract_psd(emg_sig,128, [[20,25], [25,30], [30,35], [35,40], [40,45]],nperseg=256)
    emg_abs = np.abs(emg_detrend)
    mav_emg = np.mean(emg_abs)#均值（绝对值后）
    var_emg = np.var(emg_detrend)
    zcr_emg = np.mean(np.abs(np.diff(np.sign(emg_detrend))) > 0)
    
    feats += psd_emg.tolist() + [mav_emg, var_emg, zcr_emg] 
    # 计算完 17维特征后
    res_array = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
    return res_array  

#提取用较长时间段才可以充分提取的信号特征
def extract_peri_slow(peri_seg, fs=128):
    feats = []
    # 3. GSR (5维): 0-2.4Hz PSD(1), 均值(1), 导数均值(1), 负导数均值(1), 负导数比例(1)
    gsr_sig = peri_seg[4]
    deriv_gsr = np.diff(gsr_sig)
    neg_deriv = deriv_gsr[deriv_gsr < 0]#一阶导数
    psd_gsr = extract_psd(gsr_sig, 128, [[0, 2.4]], nperseg=1024)
    mean_gsr = np.mean(gsr_sig)
    mean_deriv_gsr = np.mean(deriv_gsr)
    mean_neg_deriv_gsr = np.mean(neg_deriv) if len(neg_deriv) > 0 else 0.0
    neg_ratio = len(neg_deriv) / (len(deriv_gsr) + 1e-6)
    
    feats += [psd_gsr[0], mean_gsr, mean_deriv_gsr, mean_neg_deriv_gsr, neg_ratio]
     # 6. HST (5维): 0-0.1Hz PSD(1) + 均值(1), 方差(1), 导数均值(1), 0.1-0.2Hz PSD(1)
    hst_sig = peri_seg[7]
    psd_hst_res = extract_psd(hst_sig, 128, [[0, 0.1], [0.1, 0.2]], nperseg=1024)
    mean_hst = np.mean(hst_sig)
    var_hst = np.var(hst_sig)
    deriv_hst = np.mean(np.diff(hst_sig))
    feats += [psd_hst_res[0], mean_hst, var_hst, deriv_hst, psd_hst_res[1]]
    

    # 4. RES (21维)低频平均PSD（0.05~0.25Hz），高频平均PSD（0.25~0.5Hz），
    # 均值，导数均值，呼吸节律，呼吸速率，0~0.24Hz之间15个频点的平均PSD值
    res_sig = peri_seg[5]

    mean_res = np.mean(res_sig)
    detrended_res = signal.detrend(res_sig)
    deriv_res = np.mean(np.abs(np.diff(detrended_res)))
    
    wide_bands_res = [[0.05, 0.25], [0.25, 0.5]]
    psd_wide_res = extract_psd(res_sig, fs, wide_bands_res, nperseg=1024)
    low_psd_res, high_psd_res = psd_wide_res[0], psd_wide_res[1]

    #  15个频点插值 
    # 拿完整的曲线去做插值
    freqs_res, psd_res = signal.welch(res_sig, fs, nperseg=1024)
    f_interp = interpolate.interp1d(freqs_res, psd_res, kind='linear', fill_value="extrapolate")
    target_f = np.linspace(0.01, 0.24, 15)
    psd_15_points = np.log10(f_interp(target_f) + 1e-10).tolist()

    #速率与节律 
    try:
        res_cleaned = nk.rsp_clean(res_sig, sampling_rate=fs)
        signals, info = nk.rsp_peaks(res_cleaned, sampling_rate=fs)
        rsp_rate = nk.rsp_rate(signals, sampling_rate=fs, desired_length=len(res_sig))
        rate_val = np.mean(rsp_rate)
        rhythm_val = np.std(rsp_rate)
    except:
        
        search_mask = (freqs_res > 0.1) & (freqs_res < 0.5)
        idx_in_mask = np.argmax(psd_res * search_mask)
        rate_val = freqs_res[idx_in_mask] * 60
        rhythm_val = 0.01 
    if np.isnan(rate_val) or rate_val <= 0:
        rate_val = 15.0 + np.random.normal(0, 0.1)
        rhythm_val = 0.05

    # 拼接 RES 块
    res_block = [low_psd_res, high_psd_res] + psd_15_points + [mean_res, deriv_res, rhythm_val, rate_val]
    feats.extend(res_block) 
    
    # 5. BVP (7维)心率，0.04~0.15Hz和0.15~0.5Hz之间的PSD的比值，
    # 5个频段的平均PSD：0.1 ~ 0.2Hz、0.2 ~ 0.3Hz、0.3 ~ 0.4Hz、0.08 ~ 0.15Hz、0.15 ~ 0.5 Hz。
    bvp_sig = peri_seg[6]
    hr = 75.0
    ratio_log = 0.0
    five_bands_psd = [0.0] * 5
    try:
        bvp_cleaned = nk.ppg_clean(bvp_sig, sampling_rate=fs)
        peaks_info = nk.ppg_findpeaks(bvp_cleaned, sampling_rate=fs)
        if len(peaks_info["PPG_Peaks"]) > 2:
            hr_rate = nk.ppg_rate(peaks_info, sampling_rate=fs, desired_length=len(bvp_sig))
            hr = np.nanmean(hr_rate)

        PAPER_BVP_BANDS = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.08, 0.15], [0.15, 0.5]]
        psd_results = extract_psd(bvp_sig, fs, PAPER_BVP_BANDS, nperseg=1024)
        five_bands_psd = psd_results.tolist()

        lf_band = [[0.04, 0.15], [0.15, 0.5]]
        ratio_psds = extract_psd(bvp_sig, fs, lf_band, nperseg=1024)
        
        ratio_log = np.log10(ratio_psds[0] + 1e-9) - np.log10(ratio_psds[1] + 1e-9)

    except Exception:
        pass
    bvp_block = [hr, ratio_log] + five_bands_psd
    feats.extend(bvp_block)
    # 最终转换
    res_array = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
    return res_array

#!!!4.提取PSD特征
def extract_psd(sig, fs, bands, nperseg=None):
    # 统一转为 2D: (channels, times)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    sig = np.nan_to_num(sig)
    sig = signal.detrend(sig, axis=1) #去除线性趋势
    # 如果信号全平（标准差过小），直接返回安全零值
    if np.any(np.std(sig, axis=1) < 1e-9):
        return np.full(len(bands) * sig.shape[0], 1e-10)
    # 动态计算 nperseg
    n_per_seg = nperseg if nperseg else fs
    n_per_seg = min(n_per_seg, sig.shape[1])
    # 计算 PSD
    f, pxx = welch(
        sig, 
        fs=fs, 
        window='hamming', 
        nperseg=n_per_seg, 
        noverlap=n_per_seg // 2,
        axis=1
    )

    # 提取频带功率
    feat_list = []
    for fmin, fmax in bands:
        idx = (f >= fmin) & (f <= fmax)
        if np.any(idx):
            # 取该频段内功率谱的平均值
            band_val = np.mean(pxx[:, idx], axis=1)
        else:
            band_val = np.zeros(sig.shape[0])
        feat_list.append(band_val)
    
    # 拼接特征
    res = np.concatenate(feat_list)
    # 对数变换
    log_res = np.log10(res + 1e-12)
    return log_res
def extract_eeg_psd(seg_eeg, fs=128):
  
    nperseg = fs
    noverlap = fs // 2   
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

#!!!5.滤波：EEG的4-45Hz滤波和ICA去噪官方已做过,只对外周进行相应滤波
def preprocess_trial(data, trial,fs=128):
    eeg = data[trial, :32, 3*fs:].copy()
    peri_raw = data[trial, 32:, 3*fs:].copy()
    baseline_peri = data[trial, 32:,:3*fs]#基线处理
    baseline_mean = np.mean(baseline_peri, axis =1, keepdims= True)
    peri = peri_raw - baseline_mean
    
    def quick_bandpass(sig, low, high,fs):
        sig_centered = sig - np.median(sig)
        std_val = np.std(sig_centered)
        if std_val > 0:
            sig_centered = np.clip(sig_centered, -20*std_val, 20*std_val)
        nyq = 0.5 * fs
        b, a = butter(2, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, sig_centered)

    def quick_lowpass(sig, high,fs):
        sig_centered = sig - np.median(sig)
        std_val = np.std(sig_centered)
        if std_val > 0:
            sig_centered = np.clip(sig_centered, -20*std_val, 20*std_val)
        nyq = 0.5 * fs
        b, a = butter(4, high / nyq, btype='low')
        return filtfilt(b, a, sig_centered)

    try:
        # GSR: 0 ~ 2.4Hz 
        peri[4] = quick_lowpass(peri[4], 2.4,fs) 
        # HST: 0 ~ 0.2Hz 
        peri[7] = quick_lowpass(peri[7], 0.2,fs)
        # EMG: 5.0 ~ 45.0Hz 
        peri[2] = quick_bandpass(peri[2], 5.0, 45.0,fs)
        # RES: 0 ~ 0.25Hz 
        peri[5] = quick_lowpass(peri[5], 0.5,fs)
        # BVP: 0.04~5Hz 
        peri[6] = quick_bandpass(peri[6], 0.04, 5,fs)
    except Exception as e:
        print(f"外周滤波失败，保留原信号: {e}")

    return eeg, peri


# !!!6.主循环 

all_psds, all_stats, all_peris = [], [], []

for subj in range(1, 33):
    path = os.path.join(RAW_DATA_PATH, f"s{subj:02d}.mat")
    if not os.path.exists(path):
        continue

    print(f">>> Processing Subject {subj:02d}...")
    try:
        data = sio.loadmat(path)["data"]
    except Exception as e:
        print(f"读取 S{subj} 失败: {e}")
        continue

    for trial in range(40):
        
        eeg_full, peri_full = preprocess_trial(data, trial, fs)
    
        try:
            slow_feat_38 = extract_peri_slow(peri_full, fs)
        except Exception as e:
            print(f" S{subj} T{trial} Slow Peri Error: {e}")
            slow_feat_38 = np.zeros(38, dtype=np.float32)
 
        win_len = 4 * fs      
        segment_cnt = eeg_full.shape[1] // win_len 

        for seg in range(segment_cnt):
            # 时间窗定义
            st, ed = seg * win_len, (seg + 1) * win_len
            
            seg_eeg = eeg_full[:, st:ed]
            psd_feat = extract_eeg_psd(seg_eeg, fs)
            stat_feat = extract_complex_stats(seg_eeg)

            seg_peri_fast = peri_full[:, st:ed]
            try:
                fast_feat_17 = extract_peri_fast(seg_peri_fast, fs)
            except Exception as e:
                print(f" S{subj} T{trial} Seg{seg} Fast Peri Error: {e}")
                fast_feat_17 = np.zeros(17, dtype=np.float32)

            #广播对齐 (拼接快慢特征) 
            p_feat_55 = np.concatenate([fast_feat_17, slow_feat_38])

            all_psds.append(psd_feat[:160])
            all_stats.append(stat_feat[:224])
            all_peris.append(p_feat_55[:55])





# 保存与验证


all_psds.append(psd_feat[:160].astype(np.float32))
all_stats.append(stat_feat[:224].astype(np.float32))
all_peris.append(p_feat_55[:55].astype(np.float32))

all_psds_arr = np.asarray(all_psds, np.float32)
all_stats_arr = np.asarray(all_stats, np.float32)
all_peris_arr = np.asarray(all_peris, np.float32)



print("\n=== 数据完整性检查（Raw） ===")
print(f"Peri shape: {all_peris_arr.shape}")
print(f"NaN count: {np.isnan(all_peris_arr).sum()}")

def subject_wise_finalize(data_list, name, samples_per_subj=600):
    """
    针对 DEAP 数据集的被试内标准化函数
    samples_per_subj: 每个被试的总样本数 (40 trials * 15 segments = 600)
    """
    raw_arr = np.array(data_list, dtype=np.float64)
    total_samples = raw_arr.shape[0]
    num_subjects = total_samples // samples_per_subj
    
    final_normed_data = []

    print(f"开始 {name} 的被试内 Z-Score 处理...")

    for i in range(num_subjects):
        # 1. 切出当前被试的所有数据 (600, dim)
        start_idx = i * samples_per_subj
        end_idx = (i + 1) * samples_per_subj
        subj_data = raw_arr[start_idx:end_idx].copy()

        # 2. 处理无效值 (NaN/Inf) - 用该被试的中位数填补
        for col in range(subj_data.shape[1]):
            col_data = subj_data[:, col]
            invalid = ~np.isfinite(col_data)
            if np.any(invalid):
                fill_val = np.nanmedian(col_data) if np.any(~invalid) else 0.0
                subj_data[invalid, col] = fill_val

        # 3. 计算该被试自己的均值和标准差
        s_mean = np.mean(subj_data, axis=0)
        s_std = np.std(subj_data, axis=0)

        # 4. 执行 Z-Score: (X - mu) / std
        # 防御死维度：如果某特征在某人身上完全没变化，std 为 0
        s_std[s_std < 1e-8] = 1.0
        subj_normed = (subj_data - s_mean) / s_std

        final_normed_data.append(subj_normed)

    # 合并回总矩阵
    arr = np.concatenate(final_normed_data, axis=0)

    # 5. 全局平滑 (Soft Squash)
    # 限制极端离群值，防止 LOSO 训练时某个样本带偏整个梯度


    print(f"✅ {name:10} 被试内清洗完成 | 均值: {np.mean(arr):.4f} | 标准差: {np.std(arr):.4f}")
    return arr.astype(np.float32)

# --- 应用新的清洗逻辑 ---
# 注意：确保你的总样本量是 600 的整数倍 (32 * 40 * 15 = 19200)
all_stats_arr = subject_wise_finalize(all_stats, "EEG_Stats")
all_peris_arr = subject_wise_finalize(all_peris, "Peri")
# 在保存前运行
def check_channel_similarity(data_arr, num_channels=32):
    # 假设 data_arr 是 (N, channels * features)
    sample = data_arr[0].reshape(num_channels, -1)
    corr = np.corrcoef(sample)
    print(f"Mean inter-channel correlation: {np.mean(corr):.4f}")

check_channel_similarity(all_stats_arr)
# 保存

np.save(os.path.join(SAVE_DIR, "final_psds.npy"), all_psds_arr)
np.save(os.path.join(SAVE_DIR, "final_stats.npy"), all_stats_arr)
np.save(os.path.join(SAVE_DIR, "final_peris.npy"), all_peris_arr)