import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset, random_split

# 环境补丁
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. 鲁棒性特征提取
# ==========================================
def extract_robust_features(data, fs=128, segment_secs=10):
    total_samples = data.shape[-1]
    step = segment_secs * fs
    segments = []
    for start in range(0, total_samples - step + 1, step):
        chunk = data[:, start:start+step]
        bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
        feat_list = []
        for i in range(32):
            f, psd = welch(chunk[i], fs, nperseg=fs)
            # 提取 4 频段 DE 特征
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[np.logical_and(f>=b[0], f<=b[1])]) + 1e-8) for b in bands]
            feat_list.append(de)
        # EOG 特征
        for i in [32, 33]:
            sig = chunk[i]
            feat_list.append([np.mean(sig), np.std(sig), np.ptp(sig), np.sum(sig**2)/len(sig)])
        
        feat_arr = np.array(feat_list)
        # 局部归一化：消除单次实验内的量纲差异
        feat_arr = (feat_arr - np.mean(feat_arr, axis=1, keepdims=True)) / (np.std(feat_arr, axis=1, keepdims=True) + 1e-8)
        segments.append({'x': torch.FloatTensor(feat_arr)})
    return segments

class DEAPGeneralDataset(Dataset):
    def __init__(self, data_dir, subject_range):
        self.samples = []
        print(f"--- 正在构建跨受试者泛化数据集 (s{subject_range[0]:02d}-s{subject_range[-1]:02d}) ---")
        for i in subject_range:
            path = os.path.join(data_dir, f's{i:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            for j in range(40):
                label = 1 if raw['labels'][j, 1] > 5 else 0 # Arousal
                segs = extract_robust_features(raw['data'][j])
                for s in segs:
                    s['y'] = torch.LongTensor([label])[0]
                    self.samples.append(s)
        print(f"数据集构建完毕，当前总样本: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 2. 泛化型 DGF-GNN 架构 (DGF-GNN-Gen)
# ==========================================
class DGFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sigma_net = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.LayerNorm(64), # 引入 LayerNorm 增加泛化力
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    def forward(self, x):
        b, n, d = x.size()
        xi = x.unsqueeze(2).repeat(1, 1, n, 1)
        xj = x.unsqueeze(1).repeat(1, n, 1, 1)
        # 动态 sigma
        sigmas = self.sigma_net(torch.cat([xi, xj], dim=-1)).squeeze(-1)
        dist_sq = torch.sum((xi - xj)**2, dim=-1)
        # 增加温度系数 0.5，让注意力更集中
        return torch.exp(-dist_sq / (2 * sigmas**2 * 0.5 + 1e-6))



class GeneralDGFGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        self.node_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.dgf = DGFLayer(hidden_dim)
        
        # 多尺度图卷积
        self.gnn_w = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.gnn_w)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.6),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # 1. 跨人分布对齐
        x = self.node_norm(x)
        h_init = F.relu(self.proj(x))
        
        # 2. 动态拓扑学习
        adj = self.dgf(h_init)
        
        # 3. 特征聚合与残差
        h_fused = torch.matmul(adj, h_init)
        h_fused = torch.matmul(h_fused, self.gnn_w)
        h_res = F.elu(h_fused + h_init)
        
        # 4. 混合池化
        graph_emb = torch.cat([torch.mean(h_res, dim=1), torch.max(h_res, dim=1)[0]], dim=1)
        return self.classifier(graph_emb)

# ==========================================
# 3. 泛化测试：留一受试者法 (LOSO) 模拟
# ==========================================
def train_generalization():
    DATA_DIR = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 为了演示泛用性，我们用 1-28 号做训练，29-32 号完全作为“从未见过”的受试者进行测试
    train_range = range(1, 29)
    test_range = range(29, 33)
    
    train_ds = DEAPGeneralDataset(DATA_DIR, train_range)
    test_ds = DEAPGeneralDataset(DATA_DIR, test_range)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = GeneralDGFGNN().to(DEVICE)
    # 降低学习率，增加 L2 正则化 (weight_decay) 来提升泛用性
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 标签平滑提升泛化

    print("\n[开始泛化性训练] 训练集: S01-S28, 测试集: 未见过的 S29-S32")
    best_test_acc = 0
    
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
        
        test_acc = 100 * correct / len(test_ds)
        if test_acc > best_test_acc: best_test_acc = test_acc
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | 跨人测试精度: {test_acc:.2f}% | 历史最高: {best_test_acc:.2f}%")

if __name__ == "__main__":
    train_generalization()