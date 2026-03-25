import os
import numpy as np
import pyedflib
from scipy.signal import welch, get_window
from scipy.stats import skew, kurtosis

# ================== 1. 核心配置==================
RAW_DATA_PATH = r"D:\QQ\hci-tagging-database_download_2021-12-14_12_32_35"
SAVE_DIR = r'D:\EEGLAB\HCI_Processed_Data'
fs = 512  # HCI采样率512Hz
segment_len = 4  # 4秒分段
segment_samples = segment_len * fs


# 被试 ID 与会话文件夹的映射关系表
subj_session_mapping = {
    "1": ["2"],
    "2": ["4"],
    "3": ["6"],
    "4": ["8"],
    "5": ["10"],
    "6": ["12"],
    "7": ["14"],
    "8": ["16"],
    "9": ["18"],
    "10": ["20"],
    "11": ["22"],
    "12": ["24"],
    "13": ["26"],
    "14": ["28"],
    "15": ["30"],
    "16": ["32"],
    "17": ["34"],
    "18": ["36"],
    "19": ["38"],
    "20": ["40"],
    "21": ["132"],
    "22": ["134"],
    "23": ["136"],
    "24": ["138"],
    "25": ["140"],
    "26": ["142"],
    "27": ["144"],
    "28": ["146"],
    "29": ["148"],
    "30": ["150"],
}
os.makedirs(SAVE_DIR, exist_ok=True)


# ================== 2. 特征提取函数==================
def extract_complex_stats(eeg_seg):
    """
    提取 EEG 信号的复杂统计特征

    参数:
        eeg_seg (numpy.ndarray): EEG 信号片段，形状为 (通道数，时间点数)

    返回:
        numpy.ndarray: 包含 7 类统计特征的扁平化数组，具体包括:
            - 均值 (mean): 描述信号的中心趋势
            - 方差 (variance): 描述信号的离散程度
            - 偏度 (skewness): 描述信号分布的对称性
            - 峰度 (kurtosis): 描述信号分布的陡峭程度
            - 过零率 (zero-crossing rate): 描述信号频率特性
            - 香农熵 (Shannon entropy): 描述信号的复杂度
            - 对数能量熵 (Log energy entropy): 描述信号的能量分布
    """
    # 计算基本统计特征
    m = np.mean(eeg_seg, axis=1)
    v = np.var(eeg_seg, axis=1)
    s = skew(eeg_seg, axis=1)
    k = kurtosis(eeg_seg, axis=1)
    # 计算过零率：表征信号穿过零点的频率
    zcr = np.mean(np.diff(np.sign(eeg_seg), axis=1) != 0, axis=1)

    # 计算信息论特征：香农熵和对数能量熵
    abs_eeg = np.abs(eeg_seg)
    norm_eeg = abs_eeg / (np.sum(abs_eeg, axis=1, keepdims=True) + 1e-10)
    shannon_en = -np.sum(norm_eeg * np.log2(norm_eeg + 1e-10), axis=1)
    log_energy_en = np.sum(np.log2(eeg_seg ** 2 + 1e-10), axis=1)

    # 堆叠所有特征并扁平化返回
    return np.stack([m, v, s, k, zcr, shannon_en, log_energy_en], axis=1).flatten()


def extract_peri_features(peri_seg, fs=512):
    """
       提取外周生理信号的特征

       参数:
           peri_seg (numpy.ndarray): 外周生理信号片段，形状为 (通道数，时间点数)
               - 通道 0: ECG (心电图)
               - 通道 1: GSR (皮肤电反应)
               - 通道 2: RES (呼吸)
               - 通道 3: HST (体温)
           fs (int): 采样频率，默认值为 512Hz

       返回:
           numpy.ndarray: 37 维特征向量，包含:
               - ECG 特征 (6 维): 低频/中频/高频功率谱密度、总功率谱密度、均值、方差
               - GSR 特征 (5 维): 功率谱密度、均值、一阶差分均值、负差分均值、负差分比率
               - RES 特征 (21 维): 低频/高频功率谱密度、均值、一阶差分均值、15 个子频带功率谱密度
               - HST 特征 (5 维): 低频/高频功率谱密度、均值、方差、一阶差分均值
       """
    peri_feats = []
    eps = 1e-10

    # ECG心电图 6维
    if peri_seg.shape[0] >= 1:
        ecg = peri_seg[0, :]
        f_ecg, pxx_ecg = welch(ecg, fs=fs, nperseg=fs, noverlap=fs // 2, scaling='density')
        idx_low = (f_ecg >= 0.01) & (f_ecg <= 0.08)
        idx_mid = (f_ecg >= 0.08) & (f_ecg <= 0.15)
        idx_high = (f_ecg >= 0.15) & (f_ecg <= 0.5)
        psd_low = np.mean(pxx_ecg[idx_low]) if np.any(idx_low) else 0
        psd_mid = np.mean(pxx_ecg[idx_mid]) if np.any(idx_mid) else 0
        psd_high = np.mean(pxx_ecg[idx_high]) if np.any(idx_high) else 0
        peri_feats.extend([psd_low, psd_mid, psd_high, np.mean(pxx_ecg), np.mean(ecg), np.var(ecg)])

    # GSR 皮肤电反应 5维
    if peri_seg.shape[0] >= 2:
        gsr = peri_seg[1, :]
        f_gsr, pxx_gsr = welch(gsr, fs=fs, nperseg=fs, noverlap=fs // 2, scaling='density')
        idx_gsr = (f_gsr >= 0) & (f_gsr <= 2.4)
        gsr_diff = np.diff(gsr)
        gsr_neg_diff = gsr_diff[gsr_diff < 0]
        peri_feats.extend([np.mean(pxx_gsr[idx_gsr]), np.mean(gsr), np.mean(gsr_diff),
                           np.mean(gsr_neg_diff) if len(gsr_neg_diff) > 0 else 0,
                           len(gsr_neg_diff) / (len(gsr_diff) + eps) if len(gsr_diff) > 0 else 0])

    # RES呼吸  21维
    if peri_seg.shape[0] >= 3:
        res = peri_seg[2, :]
        f_res, pxx_res = welch(res, fs=fs, nperseg=fs, noverlap=fs // 2, scaling='density')
        idx_low = (f_res >= 0.05) & (f_res <= 0.25)
        idx_high = (f_res >= 0.25) & (f_res <= 0.5)
        res_diff = np.diff(res)
        idx_bands = (f_res >= 0) & (f_res <= 0.24)
        res_band_psds = [0] * 15
        if np.any(idx_bands):
            res_band_vals = pxx_res[idx_bands]
            res_band_split = np.array_split(res_band_vals, 15)
            res_band_psds = [np.mean(b) for b in res_band_split[:15]]
        peri_feats.extend([np.mean(pxx_res[idx_low]), np.mean(pxx_res[idx_high]),
                           np.mean(res), np.mean(res_diff)] + res_band_psds)

    # HST体温  5维
    if peri_seg.shape[0] >= 4:
        hst = peri_seg[3, :]
        f_hst, pxx_hst = welch(hst, fs=fs, nperseg=fs, noverlap=fs // 2, scaling='density')
        idx_low = (f_hst >= 0) & (f_hst <= 0.1)
        idx_high = (f_hst >= 0.1) & (f_hst <= 0.2)
        hst_diff = np.diff(hst)
        peri_feats.extend([np.mean(pxx_hst[idx_low]), np.mean(pxx_hst[idx_high]),
                           np.mean(hst), np.var(hst), np.mean(hst_diff) if len(hst_diff) > 0 else 0])

    feat = np.zeros(37, dtype=np.float32)
    feat[:len(peri_feats)] = peri_feats
    return feat


def calculate_psd(seg_eeg, fs):
    """
       计算 EEG 信号的功率谱密度特征

       参数:
           seg_eeg (numpy.ndarray): EEG 信号片段，形状为 (通道数，时间点数)
           fs (int): 采样频率

       返回:
           numpy.ndarray: 5 个频段的 PSD 特征，形状为 (样本数，5*通道数)
               频段划分:
               - Theta: 4-8 Hz
               - Alpha: 8-13 Hz
               - Alpha1: 8-10 Hz
               - Beta: 13-30 Hz
               - Gamma: 30-45 Hz
       """
    nperseg = fs
    noverlap = fs // 2
    window = get_window('hamming', nperseg)
    bands = [(4, 8), (8, 13), (8, 10), (13, 30), (30, 45)]
    f, pxx = welch(seg_eeg, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density')

    # 计算每个频段的平均功率谱密度
    psd_feat = []
    for fmin, fmax in bands:
        idx = (f >= fmin) & (f <= fmax)
        band_power = np.mean(pxx[:, idx], axis=1)
        psd_feat.append(band_power)

    return np.maximum(np.concatenate(psd_feat), 1e-12).astype(np.float32)


# ================== 3. BDF读取函数==================
def read_bdf_from_folder(session_folder, sessions_root):

    """
    从指定文件夹中读取BDF文件并提取EEG和外周信号数据
    参数:
        session_folder (str): 会话文件夹名称
        sessions_root (str): 会话文件根目录路径
    返回:
        tuple: (eeg, peri)
            eeg: 32通道EEG信号数据，形状为(32, min_samples)
            peri: 4通道外周信号数据(ECG/GSR/RES/HST)，形状为(4, min_samples)
    """
    # 构建会话文件的完整路径
    session_path = os.path.join(sessions_root, session_folder)
    # 获取文件夹中所有.bdf扩展名的文件
    bdf_files = [f for f in os.listdir(session_path) if f.lower().endswith('.bdf')]
    # 如果没有找到BDF文件，打印提示并返回空数组
    if not bdf_files:
        print(f"  会话{session_folder}：无BDF文件")
        return np.array([]), np.array([])

    # 构建第一个BDF文件的完整路径
    bdf_path = os.path.join(session_path, bdf_files[0])
    try:
        # 使用pyedflib打开BDF文件
        f = pyedflib.EdfReader(bdf_path)
        # 获取文件中的通道数
        n_channels = f.signals_in_file
        # 如果没有有效通道，关闭文件并返回空数组
        if n_channels == 0:
            print(f"  会话{session_folder}：BDF无有效通道")
            f.close()
            return np.array([]), np.array([])

        # 获取所有通道的标签并去除空格转换为大写
        signal_labels = [label.strip().upper() for label in f.getSignalLabels()]
        # 关键1：获取每个通道的采样点数（解决通道长度不一致问题）
        channel_samples = f.getNSamples()  # 每个通道的采样点数列表
        min_samples = min(channel_samples)  # 取最小采样点数作为统一长度
        # 如果最小采样点数不足4秒，关闭文件并返回空数组
        if min_samples < segment_samples:
            print(f"  会话{session_folder}：最小通道长度不足4秒")
            f.close()
            return np.array([]), np.array([])

        # 关键2：读取每个通道并裁剪到最小采样点数，保证所有通道长度一致
        all_signals = []
        for ch in range(n_channels):
            sig = f.readSignal(ch)[:min_samples]  # 裁剪到最小长度
            all_signals.append(sig.astype(np.float32))
        all_signals = np.array(all_signals)
        f.close()

        # 提取32通道EEG（不足补0，多余截断）
        eeg_channel_idx = [i for i, label in enumerate(signal_labels) if 'EEG' in label][:32]
        eeg = all_signals[eeg_channel_idx] if len(eeg_channel_idx) > 0 else np.array([])
        # 如果没有找到EEG通道，创建32xmin_samples的零数组
        if eeg.size == 0:
            eeg = np.zeros((32, min_samples), dtype=np.float32)
        elif eeg.shape[0] < 32:
            # 如果通道数不足32，用0填充
            eeg = np.pad(eeg, ((0, 32 - eeg.shape[0]), (0, 0)), mode='constant')
        eeg = eeg[:32, :]  # 强制截断为32通道

        # 提取4通道外周（ECG/GSR/RES/HST，不足补0）
        peri_mapping = {'ECG': None, 'GSR': None, 'RES': None, 'HST': None}
        for i, label in enumerate(signal_labels):
            # 查找并映射ECG通道
            if 'ECG' in label and not peri_mapping['ECG']:
                peri_mapping['ECG'] = i
            # 查找并映射GSR/EDA通道
            elif ('GSR' in label or 'EDA' in label) and not peri_mapping['GSR']:
                peri_mapping['GSR'] = i
            # 查找并映射RES/RESP通道
            elif ('RES' in label or 'RESP' in label) and not peri_mapping['RES']:
                peri_mapping['RES'] = i
            # 查找并映射HST/TEMP通道
            elif ('HST' in label or 'TEMP' in label) and not peri_mapping['HST']:
                peri_mapping['HST'] = i
        # 获取有效的外周通道索引
        peri_idx = [v for v in peri_mapping.values() if v is not None and v < n_channels]
        peri = all_signals[peri_idx] if len(peri_idx) > 0 else np.array([])
        # 如果没有找到外周通道，创建4xmin_samples的零数组
        if peri.size == 0:
            peri = np.zeros((4, min_samples), dtype=np.float32)
        elif peri.shape[0] < 4:
            # 如果通道数不足4，用0填充
            peri = np.pad(peri, ((0, 4 - peri.shape[0]), (0, 0)), mode='constant')
        peri = peri[:4, :]  # 强制截断为4通道

        # 打印成功信息并返回结果
        print(f"   会话{session_folder}：BDF读取成功（统一长度{min_samples}点，EEG32通道，外周4通道）")
        return eeg, peri
    except Exception as e:
        err_info = str(e)[:60]  # 缩短错误信息，避免打印过长
        print(f"   会话{session_folder}：BDF读取失败 - {err_info}")
        return np.array([]), np.array([])


# ================== 4. 主程序==================
if __name__ == "__main__":
    all_psds, all_stats, all_peris = [], [], []
    sessions_root = os.path.join(RAW_DATA_PATH, 'Sessions')
    if not os.path.exists(sessions_root):
        raise FileNotFoundError(f" 未找到Sessions文件夹：{sessions_root}")

    # 遍历所有被试和会话，处理 HCI 数据集
    print(f"开始处理HCI数据集，手动映射被试数：{len(subj_session_mapping)}")
    print("=" * 60)

    for subj_id, session_folders in subj_session_mapping.items():
        print(
            f"[{list(subj_session_mapping.keys()).index(subj_id) + 1}/{len(subj_session_mapping)}] 被试 subject{subj_id}")
        for session_folder in session_folders:
            eeg, peri = read_bdf_from_folder(session_folder, sessions_root)
            if eeg.size == 0 or peri.size == 0:
                continue

            # 4秒分段提取特征
            total_samples = eeg.shape[1]
            valid_segment_cnt = total_samples // segment_samples
            if valid_segment_cnt == 0:
                print(f"  会话{session_folder}：有效数据不足4秒，跳过")
                continue

            # 对每个有效分段提取三种特征：PSD、统计特征、外周特征
            for seg in range(valid_segment_cnt):
                st = seg * segment_samples
                ed = st + segment_samples
                seg_eeg = eeg[:, st:ed]
                seg_peri = peri[:, st:ed]
                all_psds.append(calculate_psd(seg_eeg, fs))
                all_stats.append(extract_complex_stats(seg_eeg))
                all_peris.append(extract_peri_features(seg_peri, fs))

            print(f"  会话{session_folder}：处理完成，有效分段 {valid_segment_cnt} 个")

    # 打印总体处理结果
    print("=" * 60)
    total_samples = len(all_psds)
    print(f"特征提取完成，总样本数：{total_samples}")
    if total_samples == 0:
        raise ValueError("未提取到任何有效特征")

    # 保存特征
    all_psds_np = np.array(all_psds, dtype=np.float32)
    all_stats_np = np.array(all_stats, dtype=np.float32)
    all_peris_np = np.array(all_peris, dtype=np.float32)
    np.save(os.path.join(SAVE_DIR, 'hci_psds.npy'), all_psds_np)
    np.save(os.path.join(SAVE_DIR, 'hci_stats.npy'), all_stats_np)
    np.save(os.path.join(SAVE_DIR, 'hci_peris.npy'), all_peris_np)

    # 打印结果
    print(f"特征维度：")
    print(f"  PSD特征：{all_psds_np.shape} [样本数, 160维]")
    print(f"  EEG统计特征：{all_stats_np.shape} [样本数, 224维]")
    print(f"  外周特征：{all_peris_np.shape} [样本数, 37维]")
    print(f"特征保存路径：{SAVE_DIR}")
    print(" HCI第一步特征提取（最终修复版）成功完成！")