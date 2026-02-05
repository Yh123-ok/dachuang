import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import scipy.io as sio

# ==========================================
# 1. 核心模型：Deep-MT-DGF-GNN (带权重输出)
# ==========================================
class ResDGFLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x, return_adj=False):
        dist_sq = torch.cdist(x, x, p=2)**2
        sigmas = torch.exp(self.log_sigmas)
        adjs = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6))
        adj_final = adjs.mean(dim=1)
        
        out = torch.matmul(adj_final, x)
        out = self.proj(out)
        res = self.norm(self.activation(out) + x)
        
        if return_adj: return res, adj_final
        return res

class Deep_MT_DGF_GNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 脑电地形图分支
        self.cnn_net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.cnn_proj = nn.Linear(128, hidden_dim)

        # 深层脑电统计特征分支 (3层 GNN)
        self.gnn_mapping = nn.Linear(7, hidden_dim)
        self.gnn_stack = nn.ModuleList([ResDGFLayer(hidden_dim) for _ in range(3)])

        # 外周信号分支
        self.peri_net = nn.Sequential(nn.Linear(55, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())

        # 跨模态融合层
        self.fusion_layer = ResDGFLayer(hidden_dim, num_heads=8)
        
        # 多任务头与权重
        self.v_head = nn.Sequential(nn.Linear(hidden_dim*3, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(hidden_dim*3, 64), nn.ELU(), nn.Linear(64, 2))
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, maps, stats, peri, return_weights=False):
        h_cnn = self.cnn_proj(self.cnn_net(maps)).unsqueeze(1)
        
        h_gnn = self.gnn_mapping(stats)
        for layer in self.gnn_stack: h_gnn = layer(h_gnn)
        h_gnn = h_gnn.mean(dim=1, keepdim=True)
        
        h_peri = self.peri_net(peri).unsqueeze(1)

        # 融合三个模态节点: [CNN, GNN, Peri]
        combined = torch.cat([h_cnn, h_gnn, h_peri], dim=1)
        fused, weights = self.fusion_layer(combined, return_adj=True)
        
        flat = fused.view(fused.size(0), -1)
        v_out, a_out = self.v_head(flat), self.a_head(flat)
        
        if return_weights: return v_out, a_out, weights
        return v_out, a_out

# ==========================================
# 2. LOSO 数据集管理器
# ==========================================
class DeapLOSODataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir):
        self.npz_dir = npz_dir
        self.file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.raw_mat_dir = raw_mat_dir
        # 预加载标签
        self.labels_v = []
        self.labels_a = []
        for f in self.file_list:
            subj_id = f[:3]
            mat = sio.loadmat(os.path.join(raw_mat_dir, f"{subj_id}.mat"))['labels']
            self.labels_v.append(np.repeat((mat[:, 0] > 5).astype(int), 15))
            self.labels_a.append(np.repeat((mat[:, 1] > 5).astype(int), 15))

    def get_subject_data(self, subj_idx):
        """获取单个被试的所有数据用于测试"""
        with np.load(os.path.join(self.npz_dir, self.file_list[subj_idx])) as d:
            m = torch.from_numpy(d['eeg_allband_feature_map']).float()
            s = torch.from_numpy(d['eeg_en_stat']).view(-1, 32, 7).float()
            p = torch.from_numpy(d['peri_feature']).float()
        lv = torch.from_numpy(self.labels_v[subj_idx]).long()
        la = torch.from_numpy(self.labels_a[subj_idx]).long()
        return m, s, p, lv, la

    def get_train_indices(self, test_subj_idx):
        """获取除测试被试外所有样本的全局索引"""
        indices = []
        for i in range(len(self.file_list)):
            if i != test_subj_idx:
                indices.extend(range(i * 600, (i + 1) * 600))
        return indices

    def __len__(self): return len(self.file_list) * 600

    def __getitem__(self, idx):
        subj_idx, inner_idx = idx // 600, idx % 600
        with np.load(os.path.join(self.npz_dir, self.file_list[subj_idx])) as d:
            m = torch.from_numpy(d['eeg_allband_feature_map'][inner_idx]).float()
            s = torch.from_numpy(d['eeg_en_stat'][inner_idx]).view(32, 7).float()
            p = torch.from_numpy(d['peri_feature'][inner_idx]).float()
        return m, s, p, self.labels_v[subj_idx][inner_idx], self.labels_a[subj_idx][inner_idx]

# ==========================================
# 3. LOSO 训练与权重分析主引擎
# ==========================================
def run_loso_experiment():
    NPZ_PATH = r'D:\Users\cyz\dc\222'
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = DeapLOSODataset(NPZ_PATH, RAW_PATH)
    num_subjects = len(full_dataset.file_list)
    
    all_results = []
    
    # 遍历每一个被试作为测试集
    for test_subj in range(num_subjects):
        print(f"\n{'='*20} LOSO Fold: Subject {test_subj+1}/{num_subjects} {'='*20}")
        
        # 划分训练集和测试集
        train_idx = full_dataset.get_train_indices(test_subj)
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=64, shuffle=True)
        # 测试集直接一次性加载
        tm, ts, tp, tlv, tla = full_dataset.get_subject_data(test_subj)
        
        model = Deep_MT_DGF_GNN().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)

        # 快速训练 (LOSO 耗时较长，建议 epoch 不要设太大)
        for epoch in range(1, 16):
            model.train()
            for m, s, p, lv, la in train_loader:
                m, s, p, lv, la = m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), lv.to(DEVICE), la.to(DEVICE)
                ov, oa = model(m, s, p)
                loss = (F.cross_entropy(ov, lv) * torch.exp(-model.log_vars[0]) + model.log_vars[0]) + \
                       (F.cross_entropy(oa, la) * torch.exp(-model.log_vars[1]) + model.log_vars[1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 测试并提取模态权重
        model.eval()
        with torch.no_grad():
            tm, ts, tp = tm.to(DEVICE), ts.to(DEVICE), tp.to(DEVICE)
            ov, oa, weights = model(tm, ts, tp, return_weights=True)
            
            # 计算模态重要性 (对 Batch 和 Head 取平均)
            # 权重矩阵维度 [Batch, 3, 3] -> 每一列代表对输入模态的关注度
            # 0: EEG-CNN, 1: EEG-GNN, 2: Peripheral
            modality_importance = weights.mean(dim=0).sum(dim=0).cpu().numpy()
            modality_importance /= modality_importance.sum() # 归一化

            f1_v = f1_score(tlv.numpy(), ov.argmax(1).cpu().numpy(), average='macro')
            f1_a = f1_score(tla.numpy(), oa.argmax(1).cpu().numpy(), average='macro')
            
            print(f"Subject {test_subj+1} Results -> F1_V: {f1_v:.4f}, F1_A: {f1_a:.4f}")
            print(f"Modality Weights -> EEG-CNN: {modality_importance[0]:.2%}, "
                  f"EEG-GNN: {modality_importance[1]:.2%}, Peri: {modality_importance[2]:.2%}")
            
            all_results.append([f1_v, f1_a])

    # 最终汇总
    avg_res = np.mean(all_results, axis=0)
    print(f"\n{'+'*20} LOSO 最终检验结果 {'+'*20}")
    print(f"平均 Valence F1: {avg_res[0]:.4f}")
    print(f"平均 Arousal F1: {avg_res[1]:.4f}")

if __name__ == "__main__":
    run_loso_experiment()