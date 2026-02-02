import os
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


# 1. 结构化 Dataset 

class CustomDataset(Dataset):
    def __init__(self, eeg_map, eeg_stat, peri, labels, trial_ids, is_train=False):
        self.eeg_map = torch.from_numpy(eeg_map).float()
        self.eeg_stat = torch.from_numpy(eeg_stat).float()
        self.peri = torch.from_numpy(peri).float()
        
        # 标签处理：如果是 2D (N, 1) 转 1D (N,)，确保 CrossEntropyLoss 能用
        if labels.ndim == 2:
            self.labels = torch.from_numpy(labels).long().squeeze()
        else:
            self.labels = torch.from_numpy(labels).long()
            
        self.trial_ids = torch.from_numpy(trial_ids).long()
        self.is_train = is_train

    def __len__(self):
        return len(self.eeg_map)

    def __getitem__(self, index):
        # 对应模型输入: (Map, Stat, Peri)
        return (self.eeg_map[index], self.eeg_stat[index], 
                self.peri[index], self.labels[index], self.trial_ids[index])


# 2. 增强型标准化器 

class RobustScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data_list):
        # data_list: [array(N1, F), array(N2, F), ...]
        combined = np.concatenate(data_list, axis=0)
        # 针对每个特征维度独立计算均值方差
        flat_data = combined.reshape(combined.shape[0], -1)
        self.mean = np.mean(flat_data, axis=0, keepdims=True)
        self.std = np.std(flat_data, axis=0, keepdims=True) + 1e-8

    def transform(self, data):
        original_shape = data.shape
        flat_data = data.reshape(original_shape[0], -1)
        scaled = (flat_data - self.mean) / self.std
        return scaled.reshape(original_shape)


# 3. 数据加载

def load_data_raw(dataset_name, feat_root, label_root):
    # 配置不同数据集的通道数，方便扩展
    config = {
        "DEAP": {"num": 32, "stat_ch": 32, "peri_dim": 55},
        "SEED-IV": {"num": 15, "stat_ch": 62, "peri_dim": 31}, # 示例预留
    }
    
    if dataset_name not in config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    cfg = config[dataset_name]
    print(f"开始读取 {dataset_name} 数据，共 {cfg['num']} 位被试...")
    
    maps_list, stats_list, peris_list, labels_list = [], [], [], []
    valid_subjects = []

    for i in range(1, cfg["num"] + 1):
        
        feat_path = os.path.join(feat_root, f's{i:02d}_features.npz')
        mat_file = os.path.join(label_root, f's{i:02d}.mat')
        
        if not (os.path.exists(feat_path) and os.path.exists(mat_file)):
            print(f"警告: 缺失文件 s{i:02d}, 跳过。")
            continue
        
        try:
            # 1. 加载 Python 预处理的特征
            payload = np.load(feat_path)
            
            # Map: Log1p 预处理 (保留空间能量差异)
            m = np.log1p(np.maximum(payload['eeg_allband_feature_map'], 0))
            
            # Stat: 形状重塑 + Log1p
            s_raw = payload['eeg_en_stat'].reshape(-1, cfg["stat_ch"], 7)
            s = np.log1p(np.abs(s_raw)) * np.sign(s_raw)
            
            # Peri: 形状重塑 (N, 55, 1)
            p = payload['peri_feature'].reshape(-1, cfg["peri_dim"], 1)

            # 2. 加载 Matlab 原始标签
            subject_content = loadmat(mat_file)
            tmp_l = subject_content['labels'] # DEAP: (40, 4)
            
            # 标签扩展：将 trial 级标签广播到 segment 级
            # 假设每个 trial 被切分成了 M 个片段，这里自动计算倍数
            num_segments_total = m.shape[0]
            num_trials = tmp_l.shape[0]
            
            # 计算每个 Trial 切了多少个 segments 
            segments_per_trial = int(round(num_segments_total / num_trials)) 
            
            # 广播标签
            l_expanded = np.repeat(tmp_l, segments_per_trial, axis=0)
            
            # 安全截断：防止四舍五入导致的 1-2 帧误差
            l = l_expanded[:m.shape[0], :]
            
            # 二分类阈值处理
            l = (l > 5.0).astype(np.int64)

            maps_list.append(m)
            stats_list.append(s)
            peris_list.append(p)
            labels_list.append(l)
            valid_subjects.append(i)
            
        except Exception as e:
            print(f"读取 s{i:02d} 出错: {e}")

    print(f"成功加载 {len(valid_subjects)} 位被试数据。")
    return maps_list, stats_list, peris_list, labels_list


# 4. LOSO 生成器

def dataset_loaders(dataset_name, batch_size=128, label_type='valence',
                    feat_root=r'D:\EEGLAB\Processed_Data_Python', 
                    label_root=r'E:\DEAP\data_preprocessed_matlab'):
    
    # 加载所有数据
    maps_list, stats_list, peris_list, labels_list = load_data_raw(dataset_name, feat_root, label_root)
    
    num_subjects = len(maps_list)
    if num_subjects == 0:
        raise RuntimeError("未加载到任何数据，请检查路径！")

    # 选择标签列：0: Valence, 1: Arousal
    l_idx = 0 if label_type == 'valence' else 1
    
    train_loaders, test_loaders = [], []

    print(f"构建 LOSO 数据流 (Target: {label_type})...")

    # Leave-One-Subject-Out 循环
    for test_idx in range(num_subjects):
        # A. 划分训练/测试 ID
        train_indices = [i for i in range(num_subjects) if i != test_idx]
        
        # B. 训练集拟合 Scaler (绝对不看测试集！)
        stat_scaler = RobustScaler()
        peri_scaler = RobustScaler()
        stat_scaler.fit([stats_list[i] for i in train_indices])
        peri_scaler.fit([peris_list[i] for i in train_indices])

        # C. 数据处理闭包 (Closure)
        def process_sub(idx, sub_offset):
            m = maps_list[idx]
            # Map Z-Score: 被试内独立归一化 (Domain Adaptation 的基础)
            m = (m - m.mean(axis=(1,2,3), keepdims=True)) / (m.std(axis=(1,2,3), keepdims=True) + 1e-8)
            
            # Stat & Peri: 使用训练集的分布进行变换
            s = stat_scaler.transform(stats_list[idx])
            p = peri_scaler.transform(peris_list[idx])
            
            # Label
            l = labels_list[idx][:, l_idx]
            
            # Trial ID: 加上偏移量保证全局唯一
            num_samples = m.shape[0]
            segments_per_trial = 15 
            # 生成 ID: [100, 100, ..., 101, 101, ...]
            raw_ids = np.repeat(np.arange(np.ceil(num_samples/segments_per_trial)), segments_per_trial)
            t_ids = sub_offset + raw_ids[:num_samples]
            
            return m, s, p, l, t_ids

        # D. 组装训练集 (Merge all train subjects)
        t_m, t_s, t_p, t_l, t_ids = [], [], [], [], []
        for i, sub_idx in enumerate(train_indices):
            m, s, p, l, ids = process_sub(sub_idx, sub_offset=i*1000) # 偏移大一点防止重叠
            t_m.append(m); t_s.append(s); t_p.append(p); t_l.append(l); t_ids.append(ids)

        # E. 组装测试集
        v_m, v_s, v_p, v_l, v_ids = process_sub(test_idx, sub_offset=99999)

        # F. 构建 Torch Dataset
        train_ds = CustomDataset(np.concatenate(t_m), np.concatenate(t_s), 
                                 np.concatenate(t_p), np.concatenate(t_l), 
                                 np.concatenate(t_ids), is_train=True)
        
        test_ds = CustomDataset(v_m, v_s, v_p, v_l, v_ids, is_train=False)

        # G. DataLoader
        # drop_last=True 在训练时很重要，特别是当使用了 BatchNorm 时，防止最后一个小 batch 导致统计不稳定
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True))
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False))

    return train_loaders, test_loaders

if __name__ == "__main__":
    # 测试代码
    try:
        t_loaders, v_loaders = dataset_loaders("DEAP", batch_size=32)
        print(f"数据流构建完成，共 {len(t_loaders)} 折。")
        
        # 取出一个 Batch 检查维度
        example_loader = t_loaders[0]
        batch = next(iter(example_loader))
        eeg_map, eeg_stat, peri, label, tid = batch
        
        print("\n Batch 维度检查:")
        print(f"1. EEG Map (SDLN输入):  {eeg_map.shape}  -> 应为 (32, 5, 32, 32)")
        print(f"2. EEG Stat (Trans输入): {eeg_stat.shape} -> 应为 (32, 32, 7)")
        print(f"3. Peri (Trans输入):     {peri.shape}     -> 应为 (32, 55, 1)")
        print(f"4. Label:                {label.shape}    -> 应为 (32,)")
        print(f"5. Trial ID:             {tid.shape}")
        
        print("\n 状态: 数据准备完毕！")
        
    except Exception as e:
        print("请确保 'feat_root' 和 'label_root' 指向了正确的文件夹。")