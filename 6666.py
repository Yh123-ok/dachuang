import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 强化归一化的特征提取
# ==========================================
def extract_robust_feat(data, fs=128):
    bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
    processed = []
    for trial in range(40):
        trial_feats = []
        for ch in range(34):
            f, psd = welch(data[trial, ch], fs, nperseg=fs)
            # 微分熵特征
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[(f>=b[0])&(f<=b[1])]) + 1e-8) for b in bands]
            trial_feats.append(de)
        arr = np.array(trial_feats)
        # 【关键改进】对节点特征进行 L2 归一化，消除绝对幅值差异
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = arr / norm
        processed.append(arr)
    return np.array(processed)

class DEAPUltimateDataset(Dataset):
    def __init__(self, data_dir, sub_range):
        self.x, self.y = [], []
        for i in sub_range:
            path = os.path.join(data_dir, f's{i:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            self.x.append(extract_robust_feat(raw['data']))
            self.y.append((raw['labels'][:, 1] > 5).astype(int))
        self.x = torch.FloatTensor(np.concatenate(self.x))
        self.y = torch.LongTensor(np.concatenate(self.y))

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 2. 空间感知的 DGF-GNN
# ==========================================
class FinalGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 64)
        # 动态图生成：增加多尺度感知
        self.sigma_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.gnn_w = nn.Parameter(torch.randn(64, 64))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.6), # 强化 Dropout 应对泛化
            nn.Linear(64, 2)
        )

    def forward(self, x):
        h = F.elu(self.proj(x))
        # DGF 动态构图
        b, n, d = h.size()
        xi = h.unsqueeze(2).repeat(1, 1, n, 1)
        xj = h.unsqueeze(1).repeat(1, n, 1, 1)
        sigmas = self.sigma_net(torch.cat([xi, xj], dim=-1)).squeeze(-1)
        adj = torch.exp(-torch.sum((xi - xj)**2, dim=-1) / (2 * sigmas**2 + 1e-6))
        
        # 聚合 + 残差
        h_agg = torch.matmul(adj, h) @ self.gnn_w
        h_res = F.elu(h_agg + h)
        
        # 特征池化
        out = torch.cat([h_res.mean(1), h_res.max(1)[0]], dim=1)
        return self.classifier(out)

# ==========================================
# 3. 自动调优训练器
# ==========================================
def train_final():
    DATA_DIR = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    # 扩大训练集，缩减测试集，模拟真实泛化
    train_ds = DEAPUltimateDataset(DATA_DIR, range(1, 29))
    test_ds = DEAPUltimateDataset(DATA_DIR, range(29, 33))
    
    loader_args = {'batch_size': 32, 'num_workers': 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = FinalGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)
    # 引入学习率衰减，防止在 55% 附近震荡
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n--- 终极泛化模式启动 ---")
    best_acc = 0
    for epoch in range(60):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                p = model(x.to(DEVICE)).argmax(1)
                correct += (p == y.to(DEVICE)).sum().item()
        
        acc = 100 * correct / len(test_ds)
        if acc > best_acc: best_acc = acc
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

if __name__ == "__main__":
    train_final()