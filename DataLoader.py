import os
import torch
import numpy as np
import pickle
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ==========================================
# 1. Dataset 定义 (加入训练扰动与类型转换)
# ==========================================
class CustomDataset(Dataset):
    def __init__(self, eeg_map, eeg_stat, peri, labels, is_train=False):
        # 确保数据为 Float32 Tensor，标签为 Long Tensor
        self.eeg_map = torch.from_numpy(eeg_map.astype(np.float32))
        self.eeg_stat = torch.from_numpy(eeg_stat.astype(np.float32))
        self.peri = torch.from_numpy(peri.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.is_train = is_train

    def __len__(self):
        return len(self.eeg_map)

    def __getitem__(self, index):
        m, s, p, l = self.eeg_map[index], self.eeg_stat[index], self.peri[index], self.labels[index]
        
        # 训练集增加微小扰动 (Data Augmentation)
        if self.is_train:
            m = m + torch.randn_like(m) * 1e-4
            
        return m, s, p, l

# ==========================================
# 2. 数据加载函数 (利用代码 B 生成的 .npz 文件)
# ==========================================
def load_data_from_npz(dataset_name, label_type='valence', 
                       feat_root=r'D:\EEGLAB\Processed_Data_Python', 
                       label_root=r'E:\DEAP\data_preprocessed_python'):
    """
    根据受试者编号，依次读取代码 B 生成的 .npz 文件，并匹配原始标签
    """
    # 配置各数据集参数
    config = {
        "DEAP": {"num": 32, "stat_ch": 32, "peri_dim": 55},
        "HCI": {"num": 24, "stat_ch": 32, "peri_dim": 49},
        "SEED-IV": {"num": 15, "stat_ch": 62, "peri_dim": 31},
        "SEED-V": {"num": 16, "stat_ch": 62, "peri_dim": 33}
    }
    
    if dataset_name not in config:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    cfg = config[dataset_name]
    all_maps, all_stats, all_peris, all_labels = [], [], [], []

    for i in range(1, cfg["num"] + 1):
        # A. 加载特征 (.npz 文件由代码 B 生成)
        feat_path = os.path.join(feat_root, f's{i:02d}_features.npz')
        if not os.path.exists(feat_path):
            print(f"Warning: File not found {feat_path}")
            continue
        
        payload = np.load(feat_path)
        # 自动 reshape 以符合模型输入 (N, C, H, W) 和 (N, Ch, Feat)
        m = payload['eeg_allband_feature_map']
        s = payload['eeg_en_stat'].reshape(-1, cfg["stat_ch"], 7)
        p = payload['peri_feature'].reshape(-1, cfg["peri_dim"], 1)

        # B. 加载标签 (逻辑保持与原代码一致)
        # 注意：此处需根据实际标签存储路径微调
        if dataset_name == "DEAP":
            dat_file = os.path.join(label_root, f's{i:02d}.dat')
            with open(dat_file, 'rb') as f:
                subject_content = pickle.load(f, encoding='latin1')
            tmp_l = subject_content['labels']
            col = 0 if label_type == 'valence' else 1
            # 扩展标签：DEAP 40视频 -> 600样本 (15 segments/video)
            l = (np.repeat(tmp_l[:, col], 15) > 5.0).astype(np.int64)
        else:
            # HCI/SEED 系列通常直接有 labels.mat，此处演示通用读取
            # 建议将标签也预处理进 .npz 以简化此处逻辑
            l = payload['labels'] if 'labels' in payload.files else np.zeros(len(m))

        all_maps.append(m)
        all_stats.append(s)
        all_peris.append(p)
        all_labels.append(l)

    print(f"Loaded {dataset_name} successfully from .npz files!")
    return all_maps, all_stats, all_peris, all_labels

# ==========================================
# 3. 数据集切分与 Loader 生成 (LOSO 逻辑)
# ==========================================
def dataset_loaders(dataset_name, batch_size=128, label_type='valence'):
    # 获取按受试者拆分的列表
    maps_list, stats_list, peris_list, labels_list = load_data_from_npz(dataset_name, label_type)
    num_subjects = len(maps_list)

    # 先为每个受试者创建对应的 Dataset 对象
    train_datasets = []
    test_datasets = []
    for i in range(num_subjects):
        train_datasets.append(CustomDataset(maps_list[i], stats_list[i], peris_list[i], labels_list[i], is_train=True))
        test_datasets.append(CustomDataset(maps_list[i], stats_list[i], peris_list[i], labels_list[i], is_train=False))

    train_loaders = []
    test_loaders = []

    # 构造留一法 (LOSO) 组合
    for test_idx in range(num_subjects):
        # 训练集：拼接除 test_idx 以外的所有训练 Dataset
        train_combined = ConcatDataset([train_datasets[i] for i in range(num_subjects) if i != test_idx])
        test_target = test_datasets[test_idx]

        train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_target, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders

# 测试调用
if __name__ == "__main__":
    t_loaders, v_loaders = dataset_loaders("DEAP", batch_size=128)
    print(f"Generated {len(t_loaders)} pairs of LOSO loaders.")