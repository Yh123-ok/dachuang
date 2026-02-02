import scipy.io as sio
import os
import numpy as np
from scipy import signal
from scipy.signal import welch, butter, filtfilt, detrend,get_window
from scipy.stats import skew, kurtosis
import neurokit2 as nk

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
    # 确保在 12s 窗口内有基本的能量波动才判定眨眼
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
    
    # 计算完 27 维特征后
    res_array = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
    return res_array  

#提取用较长时间段才可以充分提取的信号特征
def extract_peri_slow(peri_60s,fs=128):
    feats = []
    # 4. RES (21维): 低频 0.05-0.25Hz PSD(1) + 高频 0.25-0.5Hz(1) + 均值(1) + 导数均值(1)
    # + 呼吸节律(1) + 呼吸速率 (1) + 0-0.24Hz 之间15个频点的平均PSD值(15)  
    res_sig = peri_60s[5]

    # 低频(0.05-0.25) & 高频(0.25-0.5)
    # 传入 nperseg=1024 确保呼吸信号的分辨率
    wide_bands_res = [[0.05, 0.25], [0.25, 0.5]]
    psd_wide_res = extract_psd(res_sig, 128, wide_bands_res, nperseg=1024)
    low_psd_res, high_psd_res = psd_wide_res[0], psd_wide_res[1]

    # 0-0.24Hz 之间的 15 个频点
    # 传入“极窄频带”来模拟频点采样
    narrow_freqs_res = np.linspace(0, 0.24, 15)
    # 构造极窄区间，例如 [0.01, 0.011]
    narrow_freqs_res = np.linspace(0.01, 0.24, 15)  # 从0.01开始避开DC偏置
    narrow_bands_res = [[f, f + 0.01] for f in narrow_freqs_res]
    psd_15_points = extract_psd(res_sig, 128, narrow_bands_res, nperseg=1024).tolist()
    mean_res = np.mean(res_sig)
    deriv_res = np.mean(np.abs(np.diff(signal.detrend(res_sig))))

    # 呼吸速率与节律 
    try:
        res_cleaned = nk.rsp_clean(res_sig, sampling_rate=fs)
        signals, info = nk.rsp_peaks(res_cleaned, sampling_rate=fs)
        rsp_rate = nk.rsp_rate(signals, sampling_rate=fs, desired_length=len(res_sig))
        rate_val = np.mean(rsp_rate)
        rhythm_val = np.std(rsp_rate)
        if np.isnan(rate_val) or rate_val <= 0:
            rate_val, rhythm_val = 15.0, 0.0
    except Exception:
        rate_val, rhythm_val = 15.0, 0.0

    feats +=  [low_psd_res, high_psd_res] + psd_15_points + [mean_res, deriv_res, rhythm_val, rate_val]
    
    # 5. BVP (7维): 心率(1), 频带比(1), 5波带PSD(5)
    bvp_sig = peri_60s[6]

    # 1. 估算心率 (使用 NeuroKit2)
    try:
        # 清洗 BVP 信号（去噪、基线校正）
        bvp_cleaned = nk.ppg_clean(bvp_sig, sampling_rate=fs)
        # 寻找心跳峰值 (PPG 信号通常对应心室收缩)
        peaks_info = nk.ppg_findpeaks(bvp_cleaned, sampling_rate=fs)
        # 计算瞬时心率
        hr_rate = nk.ppg_rate(peaks_info, sampling_rate=fs, desired_length=len(bvp_sig))
        hr = np.mean(hr_rate)
     
    except Exception:
        # 兜底：DEAP 正常心率约 60-90
        hr= 75.0

    # 2. 频域特征提取 (LF, HF 及 5个独立频带)
    # 直接传入原始信号，extract_psd 会处理 detrend
    bvp_bands = [
    [0.04, 0.15], [0.15, 0.4], # LF, HF (标准 HRV 频段)
    [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.08, 0.15], [0.15, 0.5] # 5个特定频段
    ]
    # 注意：psd_bvp_results 里的值已经是 log10(Power)
    psd_bvp_results = extract_psd(bvp_sig, fs, bvp_bands, nperseg=1024)

    
    # log(LF)，log(HF)
    lf_log = psd_bvp_results[0]
    hf_log = psd_bvp_results[1]
    ratio_log = lf_log - hf_log 
    five_bands_psd = psd_bvp_results[2:].tolist()

    feats += [hr, ratio_log] + five_bands_psd

    # 计算完 28 维特征后
    res_array = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
    return res_array 
   

#!!!4.提取PSD特征
def extract_psd(sig, fs, bands, nperseg=None):
    # 统一转为 2D: (channels, times)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    
    # 计算 PSD 前必须去均值，否则 0Hz 的直流分量会淹没其他频段
    sig = np.nan_to_num(sig)
    sig = signal.detrend(sig, axis=1) # 关键：去除线性趋势和均值
    
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
    
    # 对数变换补丁 
    log_res = np.log10(res + 1e-12)
    
    return log_res

#!!!5.滤波：EEG的4-45Hz滤波和ICA去噪官方已做过,只对外周进行相应滤波
def preprocess_trial(data, trial,fs=128):
    eeg = data[trial, :32, 3*fs:].copy()
    peri = data[trial, 32:, 3*fs:].copy()
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
        # HST: 0 ~ 0.        res_cleaned = nk.rsp_clean(res_sig, sampling_rate=fs)2Hz 
        peri[7] = quick_lowpass(peri[7], 0.2,fs)
        # EMG: 5.0 ~ 45.0Hz 
        peri[2] = quick_bandpass(peri[2], 5.0, 45.0,fs)
        # RES: 0 ~ 0.25Hz 
        peri[5] = quick_lowpass(peri[5], 0.25,fs)
        # BVP: 0.04 ~ 0.5Hz 
        peri[6] = quick_bandpass(peri[6], 0.04, 0.5,fs)
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
        # 滤波：得到完整的 eeg (32, 7680) 和 peri (8, 7680)
        eeg_full, peri_full = preprocess_trial(data, trial, fs)
        # 1.提取慢特征 (针对 60s 全量信号，Trial 级别只算一次)
        # 提取 RES (21维) + BVP (7维) = 28维
        try:
            slow_feat_28 = extract_peri_slow(peri_full, fs)
        except Exception as e:
            print(f"⚠️ S{subj} T{trial} Slow Peri Error: {e}")
            slow_feat_28 = np.zeros(28, dtype=np.float32)

        # 2.进入 4s 切片循环 
        win_len = 4 * fs      # 512 个点
        segment_cnt = eeg_full.shape[1] // win_len 

        for seg in range(segment_cnt):
            # 时间窗定义
            st, ed = seg * win_len, (seg + 1) * win_len
            
            # 1. 脑电特征提取 (4s)
            seg_eeg = eeg_full[:, st:ed]
            eeg_bands = [(4, 8), (8, 13), (8, 10), (13, 30), (30, 45)]
            psd_feat = extract_psd(seg_eeg, fs, eeg_bands)
            stat_feat = extract_complex_stats(seg_eeg)

            # 2. 外周快特征提取 (4s)
            # 提取 EOG (9维) + EMG (8维) + GSR (5维) + HST (5维) = 27维
            seg_peri_fast = peri_full[:, st:ed]
            try:
                fast_feat_27 = extract_peri_fast(seg_peri_fast, fs)
            except Exception as e:
                print(f"⚠️ S{subj} T{trial} Seg{seg} Fast Peri Error: {e}")
                fast_feat_27 = np.zeros(27, dtype=np.float32)

            #3.广播对齐 (拼接快慢特征) 
            # 拼接 27 维快特征 + 28 维慢特征 = 55 维
            # 此时同一个 Trial 的 15 个 Segment 共享相同的 slow_feat_28
            p_feat_55 = np.concatenate([fast_feat_27, slow_feat_28])

            # 4.存入容器，确保已对齐
            all_psds.append(psd_feat[:160])
            all_stats.append(stat_feat[:224])
            all_peris.append(p_feat_55[:55])

print(f"处理完成！总样本量: {len(all_peris)}")
# 保存与验证
all_psds_arr = np.asarray(all_psds, np.float32)
all_stats_arr = np.asarray(all_stats, np.float32)
all_peris_arr = np.asarray(all_peris, np.float32)

print("\n处理完成！数据形状：")
print(f"EEG PSD Shape:   {all_psds_arr.shape}")   # 应为 (19200, 160)
print(f"EEG Stats Shape: {all_stats_arr.shape}") # 应为 (19200, 224)
print(f"Peri Shape:      {all_peris_arr.shape}")  # 应为 (19200, 55)


print("\n=== 数据完整性检查（Raw） ===")
print(f"Peri shape: {all_peris_arr.shape}")
print(f"NaN count: {np.isnan(all_peris_arr).sum()}")


# 强力清洗与保存

def robust_finalize(data_list, name):
    # 1. 基础转换
    arr = np.array(data_list, dtype=np.float64)
    
    # 2. 拦截并处理异常值 (NaN, Inf, 极端离群值)
    for col in range(arr.shape[1]):
        col_data = arr[:, col]
        # 识别无效值
        invalid = ~np.isfinite(col_data)
        # 识别极端离离群值 (例如数值超过 5 个标准差且量级巨大)
        if np.any(~invalid):
            median = np.median(col_data[~invalid])
            std = np.std(col_data[~invalid])
            outlier_mask = np.abs(col_data - median) > (10 * std + 1e6)
            invalid |= outlier_mask
            
        if np.any(invalid):
            # 用中位数填充，若全无效则填 0
            fill_val = np.median(col_data[~invalid]) if np.any(~invalid) else 0.0
            arr[invalid, col] = fill_val

    # 3. 动态压缩逻辑 
    # 检查数据是否已经过对数处理（你的新 PSD 提取后通常在 -12 到 5 之间）
    # 如果最大值 > 100，说明这可能是原始统计量（如 GSR 能量或 BVP 原始值），需要 log1p 压缩
    for col in range(arr.shape[1]):
        col_max = np.max(np.abs(arr[:, col]))
        if col_max > 50: # 经验阈值：如果波动很大，说明没经过 log 压缩
            arr[:, col] = np.sign(arr[:, col]) * np.log1p(np.abs(arr[:, col]))

    # 4. 执行稳健归一化 
    # 计算均值和标准差用于最终缩放
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    
    # 防御“死维度”：如果标准差极小，说明该列几乎无信息
    dead_mask = std < 1e-8
    std[dead_mask] = 1.0 
    
    arr = (arr - mean) / std
    
    # 对于死维度，注入微小噪声激活（防止后续训练梯度消失）
    if np.any(dead_mask):
        arr[:, dead_mask] += np.random.normal(0, 1e-4, size=(arr.shape[0], np.sum(dead_mask)))

    # 5.将特征限制在 [-5, 5] 之间
    arr = np.clip(arr, -5, 5)
    
    print(f" {name:10} 清洗完成 | 维度: {arr.shape[1]} | 均值: {np.mean(arr):.3f} | 范围: [{arr.min():.2f}, {arr.max():.2f}]")
    return arr.astype(np.float32)

# 应用清洗
all_psds_arr = robust_finalize(all_psds, "EEG_PSD")
all_stats_arr = robust_finalize(all_stats, "EEG_Stats")
all_peris_arr = robust_finalize(all_peris, "Peri")

# 保存

np.save(os.path.join(SAVE_DIR, "final_psds.npy"), all_psds_arr)
np.save(os.path.join(SAVE_DIR, "final_stats.npy"), all_stats_arr)
np.save(os.path.join(SAVE_DIR, "final_peris.npy"), all_peris_arr)


# 外周特征检查
print("\n" + "="*30)
print(" PERI DISTRIBUTION QUALITY CHECK")
print("="*30)

# 1. 基础统计
p_min, p_max = all_peris_arr.min(), all_peris_arr.max()
p_mean, p_std = all_peris_arr.mean(), all_peris_arr.std()

# 2. 计算在 [-3, 3] 范围内的比例 (理论上应 > 99%)
within_3std = np.sum((all_peris_arr >= -3.0) & (all_peris_arr <= 3.0)) / all_peris_arr.size * 100

# 3. 计算在 [-1, 1] 范围内的比例 (理论上应在 68% 左右)
within_1std = np.sum((all_peris_arr >= -1.0) & (all_peris_arr <= 1.0)) / all_peris_arr.size * 100

print(f"Overall Range: [{p_min:.4f}, {p_max:.4f}]")
print(f"Mean: {p_mean:.6f} | Std: {p_std:.6f}")
print(f"Data within [-1.0, 1.0]: {within_1std:.2f}% (Normal: ~68%)")
print(f"Data within [-3.0, 3.0]: {within_3std:.2f}% (Target: >99%)")

# 4. 检查是否有全为 0 的异常维度
zero_dims = np.where(np.all(all_peris_arr == 0, axis=0))[0]
if len(zero_dims) > 0:
    print(f"警告: 发现 {len(zero_dims)} 个特征维度全为 0: {zero_dims}")
else:
    print("没有发现全为 0 的死特征维度")

# 5. 最终判定建议
if within_3std < 97.0:
    print("建议: 数据分布偏离严重，请检查 log1p 逻辑或数据源。")
elif abs(p_mean) > 0.01:
    print("建议: 均值不为 0，被试内归一化可能未完全覆盖。")
else:
    print(" 状态: 数据分布非常健康！")
print("="*30 + "\n")

