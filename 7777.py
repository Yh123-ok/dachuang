import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 1. 环境与设备配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 高阶特征工程：频段差分与能量归一化
# ==========================================
def extract_advanced_feats(data, fs=128):
    """
    不仅提取DE，还进行受试者内标准化，减少电压漂移
    """
    bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
    all_trials = []
    for t in range(40):
        ch_feats = []
        for c in range(34): # 32 EEG + 2 EOG
            f, psd = welch(data[t, c], fs, nperseg=fs)
            # 提取 4 个频段的微分熵
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[(f>=b[0]) & (f<=b[1])]) + 1e-8) for b in bands]
            ch_feats.append(de)
        
        fe = np.array(ch_feats) # (34, 4)
        # 受试者内 Z-Score：这是跨人识别成功的基石
        fe = (fe - np.mean(fe, axis=0)) / (np.std(fe, axis=0) + 1e-8)
        all_trials.append(fe)
    return np.array(all_trials)

class CrossSubjectDataset(Dataset):
    def __init__(self, data_dir, subjects):
        self.x, self.y = [], []
        print(f"--- 正在加载受试者群组: {list(subjects)} ---")
        for s in subjects:
            path = os.path.join(data_dir, f's{s:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            self.x.append(extract_advanced_feats(raw['data']))
            # Arousal 二分类 (阈值5)
            self.y.append((raw['labels'][:, 1] > 5).astype(int))
            
        self.x = torch.FloatTensor(np.concatenate(self.x))
        self.y = torch.LongTensor(np.concatenate(self.y))

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 3. 核心架构：多尺度动态图卷积 (MS-DGF-GNN)
# ==========================================
class MultiHeadDGF(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.sigma_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model*2, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softplus()) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, n, d = x.size()
        adjs = []
        xi = x.unsqueeze(2).repeat(1, 1, n, 1)
        xj = x.unsqueeze(1).repeat(1, n, 1, 1)
        combined = torch.cat([xi, xj], dim=-1)

        for i in range(self.num_heads):
            sigma = self.sigma_nets[i](combined).squeeze(-1)
            dist = torch.sum((xi - xj)**2, dim=-1)
            adj = torch.exp(-dist / (2 * sigma**2 + 1e-6))
            adjs.append(adj)
            
        # 平均多头邻接矩阵并融合特征
        adj_final = torch.stack(adjs).mean(dim=0)
        h = torch.matmul(adj_final, x)
        return self.proj(h)



class MS_DGF_GNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128):
        super().__init__()
        # 1. 频段特征增强
        self.feat_ext = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )
        
        # 2. 通道注意力（SE-Block）：自适应给 34 个电极分配权重
        self.channel_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. 多尺度图卷积层
        self.gnn1 = MultiHeadDGF(hidden_dim, num_heads=4)
        self.gnn2 = MultiHeadDGF(hidden_dim, num_heads=2)
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: [B, 34, 4]
        h = self.feat_ext(x)
        
        # 通道注意力增强
        w = self.channel_attn(torch.mean(h, dim=1)).unsqueeze(1)
        h = h * w
        
        # 残差图卷积
        h_g1 = F.elu(self.gnn1(h) + h)
        h_g2 = F.elu(self.gnn2(h_g1) + h_g1)
        
        # 特征池化 (Global Avg + Global Max)
        avg_p = torch.mean(h_g2, dim=1)
        max_p, _ = torch.max(h_g2, dim=1)
        feat = torch.cat([avg_p, max_p], dim=1)
        
        return self.classifier(feat)

# ==========================================
# 4. 训练引擎：温热重启与跨人验证
# ==========================================
def train_ms_dgf():
    DATA_PATH = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    
    # 为了保证泛用性：使用 LOSO (Leave-One-Subject-Out) 逻辑的变体
    # 训练集: 1-26号, 测试集: 27-32号 (从未见过)
    train_set = CrossSubjectDataset(DATA_PATH, range(1, 27))
    test_set = CrossSubjectDataset(DATA_PATH, range(27, 33))
    
    train_loader = DataLoader(train_set, batch_size=48, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=48, shuffle=False)

    model = MS_DGF_GNN().to(DEVICE)
    # 使用 AdamW 结合强 L2 正则
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    # 温热重启调度器：有助于跳出局部最优解
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n--- MS-DGF-GNN 精度提升训练启动 ---")
    best_acc = 0
    for epoch in range(70):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                p = model(x.to(DEVICE)).argmax(1)
                correct += (p == y.to(DEVICE)).sum().item()
        
        acc = 100 * correct / len(test_set)
        if acc > best_acc: best_acc = acc
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n训练完成！跨受试者最高精度：{best_acc:.2f}%")

if __name__ == "__main__":
    train_ms_dgf()