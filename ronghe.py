import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import os
import scipy.io as sio

# ==========================================
# 1. 核心层：Res-DGF 动态图卷积层
# ==========================================
class ResDGFLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        # log_sigmas 确保高斯核参数 sigma 始终为正
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x):
        b, n, d = x.size()
        # 计算欧氏距离平方
        dist_sq = torch.cdist(x, x, p=2)**2
        # 计算多头动态邻接矩阵 (RBF Kernel)
        sigmas = torch.exp(self.log_sigmas)
        # [B, H, N, N]
        adjs = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6))
        adj_final = adjs.mean(dim=1)
        
        # 图信息聚合
        out = torch.matmul(adj_final, x)
        out = self.proj(out)
        return self.norm(self.activation(out) + x)

# ==========================================
# 2. 模型：多模态中端融合网络 (Res-DGF-Fusion)
# ==========================================
class ResDGF_FusionNet(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2):
        super().__init__()
        
        # 分支 A: 脑电地形图分支 (CNN) -> [5, 32, 32]
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten() # 输出: 512
        )
        
        # 分支 B: 脑电统计特征分支 (Res-DGF-GNN) -> [32, 7]
        self.gnn_mapping = nn.Linear(7, hidden_dim)
        self.gnn_layer = ResDGFLayer(hidden_dim, num_heads=4)
        
        # 分支 C: 外周生理特征分支 (MLP) -> [55]
        self.peri_branch = nn.Sequential(
            nn.Linear(55, 32),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        
        # 融合分类器 (512 + 64 + 32 = 608)
        combined_dim = 512 + hidden_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, maps, stats, peri):
        f_cnn = self.cnn_branch(maps)
        f_gnn = self.gnn_layer(self.gnn_mapping(stats)).mean(dim=1)
        f_peri = self.peri_branch(peri)
        
        combined = torch.cat([f_cnn, f_gnn, f_peri], dim=1)
        return self.classifier(combined)

# ==========================================
# 3. 数据：DEAP 特征与真实标签关联加载器
# ==========================================
class DeapMultiModalDataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir, target_type='valence'):
        self.npz_dir = npz_dir
        self.file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.target_type = target_type
        self.samples_per_trial = 15 # 每个Trial切成了15个4s片段
        self.total_samples_per_subj = 600

        self.all_labels = []
        print(f"正在加载 {target_type} 标签并关联原始数据...")
        
        for f_name in self.file_list:
            subj_id = f_name.split('_')[0] # 提取 s01
            mat_path = os.path.join(raw_mat_dir, f"{subj_id}.mat")
            
            # 加载原始标签: (40 trials, 4 dimensions)
            # labels顺序: Valence, Arousal, Dominance, Liking
            raw_labels = sio.loadmat(mat_path)['labels']
            label_col = 0 if target_type == 'valence' else 1
            
            # 阈值 5 分割为高/低两类
            binary_labels = (raw_labels[:, label_col] > 5).astype(np.int64)
            # 扩展标签以对齐切片样本 (40 -> 600)
            expanded = np.repeat(binary_labels, self.samples_per_trial)
            self.all_labels.append(expanded)
            
        self.all_labels = np.concatenate(self.all_labels)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        file_idx = idx // self.total_samples_per_subj
        inner_idx = idx % self.total_samples_per_subj
        
        file_path = os.path.join(self.npz_dir, self.file_list[file_idx])
        with np.load(file_path) as data:
            maps = torch.from_numpy(data['eeg_allband_feature_map'][inner_idx]).float()
            # 重塑统计特征为 [电极数32, 特征数7]
            stats = torch.from_numpy(data['eeg_en_stat'][inner_idx]).view(32, 7).float()
            peri = torch.from_numpy(data['peri_feature'][inner_idx]).float()
        
        label = torch.tensor(self.all_labels[idx])
        return maps, stats, peri, label

# ==========================================
# 4. 训练与检验主逻辑
# ==========================================
def run_experiment(target='valence'):
    # 配置路径 (请根据你的实际路径修改)
    NPZ_PATH = r'D:\Users\cyz\dc\222'
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 任务: {target.upper()} 分类 ---")

    # 数据加载
    dataset = DeapMultiModalDataset(NPZ_PATH, RAW_PATH, target_type=target)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)

    # 模型初始化
    model = ResDGF_FusionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(1, 21):
        model.train()
        t_loss = 0
        for m, s, p, l in train_loader:
            m, s, p, l = m.to(device), s.to(device), p.to(device), l.to(device)
            optimizer.zero_grad()
            loss = criterion(model(m, s, p), l)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for m, s, p, l in val_loader:
                m, s, p, l = m.to(device), s.to(device), p.to(device), l.to(device)
                out = model(m, s, p)
                preds.extend(torch.max(out, 1)[1].cpu().numpy())
                targets.extend(l.cpu().numpy())
        
        f1 = f1_score(targets, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"best_model_{target}.pth")
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Loss: {t_loss/len(train_loader):.4f} | Val F1: {f1:.4f}")

    # 最终报告
    print(f"\n{target.upper()} 最终检验报告:")
    model.load_state_dict(torch.load(f"best_model_{target}.pth"))
    # 再次运行验证集获取详细报告
    print(classification_report(targets, preds, target_names=['Low', 'High']))

if __name__ == "__main__":
    # 可以依次运行两个任务
    run_experiment(target='valence')
    run_experiment(target='arousal')