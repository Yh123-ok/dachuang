import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# ================== 核心配置==================
FEAT_ROOT = r'D:\EEGLAB\HCI_Processed_Data_Step2'  # 第二步特征+真实标签目录
LABEL_TYPE = 'valence'  # 可选：'valence'（效价）/ 'arousal'（唤醒度）
BATCH_SIZE = 32  # 适配11个被试的小样本量
NUM_WORKERS = 0  # Windows系统必设0，避免多进程报错
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 标签二分类阈值（HCI数据集默认1-9标度，取中间值5，>5为高/1，≤5为低/0，可根据数据集调整）
LABEL_THRESHOLD = 5.0

# 校验目录和标签文件
if not os.path.exists(FEAT_ROOT):
    raise FileNotFoundError(f"特征/标签目录不存在：{FEAT_ROOT}")
# 检查真实标签文件是否存在
label_file_exist = os.path.exists(os.path.join(FEAT_ROOT, 'hci_labels.csv')) or \
                   any([f.endswith('_label.npz') for f in os.listdir(FEAT_ROOT)])
if not label_file_exist:
    raise FileNotFoundError(f"特征目录下无真实标签文件，将hci_labels.csv或subjectX_label.npz放到{FEAT_ROOT}")


# ================== HCI数据集类（核心修复：加载真实标签）==================
class HCIDataset(Dataset):
    """
       HCI 数据集类：读取第二步 npz 特征 + 真实标签

       功能:
           - 支持两种标签格式：hci_labels.csv（总标签）/ subjectX_label.npz（按被试标签）
           - 自动进行标签二分类转换（基于阈值）
           - 严格校验特征与标签的样本数匹配
           - 输出张量自动推送到指定设备（CPU/GPU）

       输出数据维度:
           - psd_topomap: (5, 64, 64) - 5 频段 EEG 拓扑图
           - eeg_stats: (224,) - EEG 统计特征
           - peri_feats: (37,) - 外周生理信号特征
           - label: (1,) - 二分类标签（0 或 1）

       属性:
           feat_files (list): 特征文件路径列表
           label_type (str): 标签类型，'valence'或'arousal'
           threshold (float): 二分类阈值
           data (tuple): 加载后的所有数据 (psd, stats, peri, labels)
       """

    def __init__(self, feat_files, label_type=LABEL_TYPE, threshold=LABEL_THRESHOLD):
        self.feat_files = feat_files
        self.label_type = label_type
        self.threshold = threshold
        self.data = self._load_feat_and_label()  # 加载特征+真实标签

    def _load_single_subj_label(self, subj_id):
        """加载单个被试的npz标签：subjectX_label.npz"""
        label_path = os.path.join(FEAT_ROOT, f"subject{subj_id}_label.npz")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"被试{subj_id}的标签文件缺失：{label_path}")
        label_data = np.load(label_path, allow_pickle=True)
        # 读取指定类型标签并二分类
        raw_label = label_data[self.label_type].astype(np.float32)
        binary_label = np.where(raw_label > self.threshold, 1, 0).reshape(-1, 1)
        return binary_label

    def _load_feat_and_label(self):
        """
        加载所有被试的特征和真实标签，严格校验样本数匹配

        处理流程:
            1. 优先加载 csv 总标签文件（若存在）
            2. 遍历每个被试的 npz 特征文件
            3. 根据被试 ID 匹配对应标签（csv 优先，npz 备用）
            4. 严格校验标签数与特征样本数一致
            5. 处理空值和无穷值
            6. 转换为 PyTorch 张量并拼接

        返回:
            tuple: (psd_concat, stats_concat, peri_concat, labels_concat)
                psd_concat: PSD 拓扑图张量，形状为 (total_samples, 5, 64, 64)
                stats_concat: EEG 统计特征张量，形状为 (total_samples, 224)
                peri_concat: 外周特征张量，形状为 (total_samples, 37)
                labels_concat: 标签张量，形状为 (total_samples, 1)

        异常:
            AssertionError: 当标签数与特征样本数不匹配时抛出
        """
        all_psd, all_stats, all_peri, all_labels = [], [], [], []
        # 若有总csv标签，先加载备用（按被试匹配）
        csv_label_dict = {}
        csv_label_path = os.path.join(FEAT_ROOT, 'hci_labels.csv')
        if os.path.exists(csv_label_path):
            df = pd.read_csv(csv_label_path, header=None, names=['subj_id', 'valence', 'arousal'])
            for subj in df['subj_id'].unique():
                subj_df = df[df['subj_id'] == subj]
                csv_label_dict[subj] = subj_df[self.label_type].values.reshape(-1, 1)

        for feat_file in self.feat_files:
            # 解析被试ID
            fname = os.path.basename(feat_file)
            subj_id = int(fname.replace('subject', '').replace('.npz', ''))
            # 加载特征
            npz_data = np.load(feat_file, allow_pickle=True)
            psd = npz_data['psd_topomap']  # (N,5,64,64)
            stats = npz_data['eeg_stats']  # (N,224)
            peri = npz_data['peri_feats']  # (N,37)
            n_samples = psd.shape[0]

            # 加载真实标签（csv优先，无则加载npz）
            if subj_id in csv_label_dict:
                labels = csv_label_dict[subj_id][:n_samples]  # 截断到特征样本数
            else:
                labels = self._load_single_subj_label(subj_id)[:n_samples]

            # 严格校验：标签数必须等于特征样本数
            assert len(labels) == n_samples, \
                f"被试{subj_id}标签数({len(labels)})与特征样本数({n_samples})不匹配！"

            # 空值/无穷值处理（防止训练报错）
            psd = np.nan_to_num(psd, nan=0.0, posinf=10.0, neginf=-10.0)
            stats = np.nan_to_num(stats, nan=0.0, posinf=10.0, neginf=-10.0)
            peri = np.nan_to_num(peri, nan=0.0, posinf=10.0, neginf=-10.0)

            # 转换为PyTorch张量（float32，适配训练精度）
            all_psd.append(torch.from_numpy(psd).float())
            all_stats.append(torch.from_numpy(stats).float())
            all_peri.append(torch.from_numpy(peri).float())
            all_labels.append(torch.from_numpy(labels).float())

        # 拼接所有被试数据
        psd_concat = torch.cat(all_psd, dim=0)
        stats_concat = torch.cat(all_stats, dim=0)
        peri_concat = torch.cat(all_peri, dim=0)
        labels_concat = torch.cat(all_labels, dim=0)
        return (psd_concat, stats_concat, peri_concat, labels_concat)

    def __len__(self):
        """返回总样本数"""
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        """
               按索引获取单个样本并推送到指定设备

               参数:
                   idx (int): 样本索引

               返回:
                   tuple: (psd_topomap, eeg_stats, peri_feats, label)
                       所有张量均已移动到 DEVICE 指定的设备（CPU/GPU）
        """
        psd_topomap = self.data[0][idx].to(DEVICE)
        eeg_stats = self.data[1][idx].to(DEVICE)
        peri_feats = self.data[2][idx].to(DEVICE)
        label = self.data[3][idx].to(DEVICE)
        return psd_topomap, eeg_stats, peri_feats, label


# ================== LOSO留一被试拆分==================
def get_loso_dataloaders(test_subj_id, feat_root=FEAT_ROOT, batch_size=BATCH_SIZE):
    """
        实现留一被试交叉验证（Leave-One-Subject-Out）的数据加载器拆分

        功能:
            - 指定一个测试被试，其余被试作为训练集
            - 确保训练集和测试集无样本重叠
            - 自动过滤无效的标签文件

        参数:
            test_subj_id (int): 测试被试 ID（范围：1 到总被试数）
            feat_root (str): 特征文件根目录路径，默认使用 FEAT_ROOT
            batch_size (int): 批次大小，默认使用 BATCH_SIZE

        返回:
            tuple: (train_loader, test_loader)
                train_loader: 训练集 DataLoader
                test_loader: 测试集 DataLoader

        异常:
            FileNotFoundError: 当特征目录下无有效特征文件时抛出
            ValueError: 当测试被试无对应文件或训练集为空时抛出
    """
    # 获取所有特征文件（subjectX.npz）
    all_feat_files = [os.path.join(feat_root, f) for f in os.listdir(feat_root)
                      if f.startswith('subject') and f.endswith('.npz') and not f.endswith('_label.npz')]
    if not all_feat_files:
        raise FileNotFoundError(f"特征目录下无有效特征文件：{feat_root}")

    # 硬分离：训练集=除测试被试外所有，测试集=仅测试被试
    test_files = [f for f in all_feat_files if f"subject{test_subj_id}.npz" in f]
    train_files = [f for f in all_feat_files if f"subject{test_subj_id}.npz" not in f]

    if not test_files:
        raise ValueError(f"测试被试{test_subj_id}无对应特征文件，检查被试ID")
    if not train_files:
        raise ValueError(f"训练集为空")

    # 创建数据集
    train_dataset = HCIDataset(train_files)
    test_dataset = HCIDataset(test_files)

    # 创建DataLoader（小样本适配：drop_last=False，不丢弃最后一批）
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False
    )
    print(
        f" LOSO拆分完成 | 训练集：{len(train_dataset)}样本 | 测试集：{len(test_dataset)}样本 | 测试被试：{test_subj_id}")
    return train_loader, test_loader


# ================== 获取所有有效被试ID==================
def get_all_valid_subj_ids(feat_root=FEAT_ROOT):
    """
        获取所有有效被试 ID（同时具备特征文件和标签文件的被试）

        功能:
            - 扫描特征目录下的所有 subjectX.npz 文件
            - 验证每个被试是否有对应的标签文件（csv 或 npz 格式）
            - 返回升序排列的有效被试 ID 列表

        参数:
            feat_root (str): 特征文件根目录路径，默认使用 FEAT_ROOT

        返回:
            list: 有效被试 ID 的升序列表，如 [1, 2, 3, ..., 11]

        异常:
            ValueError: 当无有效被试时抛出
    """
    all_files = os.listdir(feat_root)
    subj_ids = []
    for f in all_files:
        if f.startswith('subject') and f.endswith('.npz') and not f.endswith('_label.npz'):
            try:
                subj_id = int(f.replace('subject', '').replace('.npz', ''))
                # 校验该被试是否有标签
                has_label = os.path.exists(os.path.join(feat_root, f"subject{subj_id}_label.npz")) or \
                            (os.path.exists(os.path.join(feat_root, 'hci_labels.csv')) and
                             subj_id in pd.read_csv(os.path.join(feat_root, 'hci_labels.csv'), header=None)[0].unique())
                if has_label:
                    subj_ids.append(subj_id)
            except:
                continue
    subj_ids = sorted(subj_ids)
    if not subj_ids:
        raise ValueError("无有效被试（特征+标签均存在的被试）")
    print(f" 检测到有效被试（特征+标签均存在）：{subj_ids} | 共{len(subj_ids)}个")
    return subj_ids


# 测试代码（运行该文件验证：特征+标签加载正常、维度匹配）
if __name__ == "__main__":
    subj_ids = get_all_valid_subj_ids()
    train_loader, test_loader = get_loso_dataloaders(test_subj_id=1)
    # 遍历一批数据验证维度
    for batch in train_loader:
        psd, stats, peri, label = batch
        print(f" 批样本数：{psd.shape[0]}")
        print(f" 拓扑图维度：{psd.shape} (N,5,64,64) - 匹配第二步输出")
        print(f" 统计特征维度：{stats.shape} (N,224) - 匹配第二步输出")
        print(f" 外周特征维度：{peri.shape} (N,37) - 匹配第二步输出")
        print(f" 标签维度：{label.shape} (N,1) - 二分类标签（0/1）")
        print(f" 标签取值：{torch.unique(label)} - 仅0/1，符合二分类要求")
        break
