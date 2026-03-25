import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ================== 1. 核心配置==================
# 第一步特征保存目录
STEP1_SAVE_DIR = r'D:\EEGLAB\HCI_Processed_Data'
STEP2_SAVE_DIR = r'D:\EEGLAB\HCI_Processed_Data_Step2'

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

fs = 512
segment_len = 4
n_channels = 32  # EEG通道数
n_bands = 5  # PSD5个频段
n_peri_feat = 37  # 外周37维
n_stat_feat = 224  # EEG统计224维
# 32通道EEG拓扑图坐标
eeg_coords = np.array([
    [0, 0.8], [-0.2, 0.6], [0.2, 0.6], [-0.4, 0.4], [0, 0.4], [0.4, 0.4],
    [-0.6, 0.2], [-0.2, 0.2], [0.2, 0.2], [0.6, 0.2], [-0.8, 0], [-0.4, 0],
    [0, 0], [0.4, 0], [0.8, 0], [-0.6, -0.2], [-0.2, -0.2], [0.2, -0.2],
    [0.6, -0.2], [-0.4, -0.4], [0, -0.4], [0.4, -0.4], [-0.2, -0.6], [0.2, -0.6],
    [-0.5, -0.8], [0, -0.8], [0.5, -0.8], [-0.1, -1.0], [0.1, -1.0],
    [-0.3, -1.2], [0, -1.2], [0.3, -1.2]
])
# 创建输出目录
os.makedirs(STEP2_SAVE_DIR, exist_ok=True)


# ================== 2. 工具函数 ==================
def load_step1_features(step1_dir):
    """加载第一步生成的3类特征"""
    psd_path = os.path.join(step1_dir, 'hci_psds.npy')
    stats_path = os.path.join(step1_dir, 'hci_stats.npy')
    peri_path = os.path.join(step1_dir, 'hci_peris.npy')

    # 检查文件是否存在
    for path in [psd_path, stats_path, peri_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"第一步特征文件缺失：{path}")

    psds = np.load(psd_path).astype(np.float32)  # (total_samples, 160) = 32*5
    stats = np.load(stats_path).astype(np.float32)  # (total_samples, 224)
    peris = np.load(peri_path).astype(np.float32)  # (total_samples, 37)

    print(f" 加载第一步特征成功")
    print(f"  PSD特征：{psds.shape} | EEG统计特征：{stats.shape} | 外周特征：{peris.shape}")
    return psds, stats, peris


def split_features_by_subj(psds, stats, peris, mapping):

    subj_feat_dict = {}
    current_idx = 0
    total_subj = len(mapping)

    for subj_id, sessions in mapping.items():
        # 计算每个被试的总分段数（每个会话 48 个 4 秒分段）
        subj_total_segs = len(sessions) * 48

        # 拆分特征
        end_idx = current_idx + subj_total_segs
        if end_idx > psds.shape[0]:
            end_idx = psds.shape[0]  # 防止越界
        subj_psds = psds[current_idx:end_idx]
        subj_stats = stats[current_idx:end_idx]
        subj_peris = peris[current_idx:end_idx]
        current_idx = end_idx

        # 过滤空特征
        if subj_psds.shape[0] == 0:
            print(f" 被试subject{subj_id}：无有效特征，跳过")
            continue

        subj_feat_dict[subj_id] = {
            'psd': subj_psds,
            'stats': subj_stats,
            'peri': subj_peris
        }
        print(f" 被试subject{subj_id}：拆分特征 {subj_psds.shape[0]} 个样本")

    # 检查是否有剩余特征
    if current_idx < psds.shape[0]:
        print(f"剩余未拆分特征：{psds.shape[0] - current_idx} 个样本")
    return subj_feat_dict


def subject_wise_standardize(feat):
    """
        受试者内标准化（每个被试单独标准化，避免被试间差异）

        参数:
            feat (numpy.ndarray): 待标准化的特征数组
                - 二维数组：形状为 (n_samples, n_feats)
                - 三维数组：形状为 (n_samples, n_channels, n_bands)

        返回:
            numpy.ndarray: 标准化后的特征数组，已处理无穷值和空值，并裁剪到[-5, 5]范围

    受试者内标准化（每个被试单独标准化，避免被试间差异）
    """
    scaler = StandardScaler()
    # 处理二维特征 (n_samples, n_feats)
    if feat.ndim == 2:
        standardized = scaler.fit_transform(feat)
    # 处理三维特征 (n_samples, n_channels, n_bands)
    elif feat.ndim == 3:
        n_samples, n_c, n_b = feat.shape
        feat_2d = feat.reshape(n_samples, -1)
        standardized_2d = scaler.fit_transform(feat_2d)
        standardized = standardized_2d.reshape(n_samples, n_c, n_b)
    else:
        standardized = feat
    # 替换无穷值和空值
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)
    # 裁剪极值（防止异常值）
    standardized = np.clip(standardized, -5, 5)
    return standardized


def psd_to_topomap(psd_feat, coords, n_channels=32, n_bands=5):
    """将PSD特征转换为拓扑图特征 (n_samples, n_bands, H, W)
    psd_feat: (n_samples, 160) → 转换为 (n_samples, 32, 5)
    coords: 32通道坐标，映射为64×64的拓扑图
    """
    n_samples = psd_feat.shape[0]
    # 重塑为 (n_samples, n_channels, n_bands)
    psd_3d = psd_feat.reshape(n_samples, n_channels, n_bands)
    # 拓扑图尺寸（固定64×64，兼顾分辨率和计算量）
    map_h, map_w = 64, 64
    # 坐标归一化到0~map_h-1, 0~map_w-1
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-10)
    coords[:, 0] = coords[:, 0] * (map_w - 1)
    coords[:, 1] = coords[:, 1] * (map_h - 1)
    coords = np.round(coords).astype(int)

    # 初始化拓扑图
    topomap = np.zeros((n_samples, n_bands, map_h, map_w), dtype=np.float32)
    for i in range(n_samples):
        for band in range(n_bands):
            for ch in range(n_channels):
                x, y = coords[ch]
                # 防止坐标越界
                x = np.clip(x, 0, map_w - 1)
                y = np.clip(y, 0, map_h - 1)
                topomap[i, band, y, x] = psd_3d[i, ch, band]

    return topomap


# ================== 3. 主程序：第二步特征处理 ==================
if __name__ == "__main__":
    print(" HCI数据集特征处理第二步）")


    # 1. 加载第一步特征
    psds, stats, peris = load_step1_features(STEP1_SAVE_DIR)

    # 2. 按被试拆分特征
    print("\n" + "-" * 50)
    print(" 按被试拆分特征")
    print("-" * 50)
    subj_feat_dict = split_features_by_subj(psds, stats, peris, subj_session_mapping)
    if not subj_feat_dict:
        raise ValueError(" 无有效被试特征")

    # 3. 逐被试处理：标准化 + PSD转拓扑图 + 保存npz
    print(" 逐被试特征处理（标准化+拓扑图转换）")

    total_processed = 0
    for subj_id, feat in subj_feat_dict.items():
        subj_id_int = int(subj_id)
        psd = feat['psd']
        stat = feat['stats']
        peri = feat['peri']
        n_samples = psd.shape[0]

        # 受试者内标准化（核心：每个被试单独标准化）
        psd_standardized = subject_wise_standardize(psd)
        stat_standardized = subject_wise_standardize(stat)
        peri_standardized = subject_wise_standardize(peri)

        # PSD特征转拓扑图 (n_samples, 5, 64, 64)
        psd_topomap = psd_to_topomap(psd_standardized, eeg_coords)

        # 保存为npz文件（命名：subjectX.npz，对接DataLoader）
        save_path = os.path.join(STEP2_SAVE_DIR, f"subject{subj_id}.npz")
        np.savez_compressed(
            save_path,
            psd_topomap=psd_topomap,  # 5频段拓扑图 (N,5,64,64)
            eeg_stats=stat_standardized,  # EEG统计特征 (N,224)
            peri_feats=peri_standardized,  # 外周特征 (N,37)
            subj_id=subj_id_int,  # 被试ID
            n_samples=n_samples  # 样本数
        )

        total_processed += n_samples
        print(
            f" 被试subject{subj_id}：处理完成 | 样本数{n_samples} | 拓扑图{psd_topomap.shape} | 保存至{os.path.basename(save_path)}")

    # 4. 处理完成汇总
    print("\n" + "=" * 70)
    print(f" HCI特征处理第二步全部完成！")
    print(f" 处理结果：共{len(subj_feat_dict)}个有效被试 | 总样本数{total_processed}")
    print(f" 特征保存目录：{STEP2_SAVE_DIR}")
    print(f"文件格式：subjectX.npz（包含psd_topomap/eeg_stats/peri_feats）")