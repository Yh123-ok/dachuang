import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset, random_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. 数据处理 (保持切段，增加频段能量归一化)
# ==========================================
def extract_final_features(data, fs=128, segment_secs=10):
    total_samples = data.shape[-1]
    step = segment_secs * fs
    segments = []
    for start in range(0, total_samples - step + 1, step):
        chunk = data[:, start:start+step]
        bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
        feat_list = []
        for i in range(32):
            f, psd = welch(chunk[i], fs, nperseg=fs)
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[np.logical_and(f>=b[0], f<=b[1])]) + 1e-8) for b in bands]
            feat_list.append(de)
        for i in [32, 33]: # EOG
            sig = chunk[i]
            feat_list.append([np.mean(sig), np.std(sig), np.ptp(sig), np.sum(sig**2)/len(sig)])
        
        feat_arr = np.array(feat_list)
        # 强力归一化：将所有受试者投影到标准正态分布
        feat_arr = (feat_arr - feat_arr.mean()) / (feat_arr.std() + 1e-8)
        segments.append({'x': torch.FloatTensor(feat_arr)})
    return segments

class DEAPFinalDataset(Dataset):
    def __init__(self, data_dir, subject_range):
        self.samples = []
        for i in subject_range:
            path = os.path.join(data_dir, f's{i:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            for j in range(40):
                label = 1 if raw['labels'][j, 1] > 5 else 0 # Arousal
                segs = extract_final_features(raw['data'][j])
                for s in segs:
                    s['y'] = torch.LongTensor([label])[0]
                    self.samples.append(s)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 2. 空间注意力 + DGF-GNN
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, in_channels=34):
        super().__init__()
        self.kv_proj = nn.Linear(4, 16)
        self.attn_weight = nn.Parameter(torch.ones(in_channels, 1))

    def forward(self, x):
        # x: [B, 34, 4]
        weights = torch.sigmoid(self.attn_weight).unsqueeze(0) # [1, 34, 1]
        return x * weights

class FinalDGFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sigma_net = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    def forward(self, x):
        b, n, d = x.size()
        xi = x.unsqueeze(2).repeat(1, 1, n, 1)
        xj = x.unsqueeze(1).repeat(1, n, 1, 1)
        sigmas = self.sigma_net(torch.cat([xi, xj], dim=-1)).squeeze(-1)
        dist_sq = torch.sum((xi - xj)**2, dim=-1)
        return torch.exp(-dist_sq / (2 * sigmas**2 * 0.5 + 1e-6))

class AdvancedDGFGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        self.attention = SpatialAttention(34)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.dgf = FinalDGFLayer(hidden_dim)
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
        x = self.attention(x)
        h_init = F.relu(self.proj(x))
        adj = self.dgf(h_init)
        h_fused = torch.matmul(adj, h_init)
        h_fused = torch.matmul(h_fused, self.gnn_w)
        h_res = F.elu(h_fused + h_init)
        graph_emb = torch.cat([torch.mean(h_res, dim=1), torch.max(h_res, dim=1)[0]], dim=1)
        return self.classifier(graph_emb)

# ==========================================
# 3. 实验启动
# ==========================================
def run_final():
    DATA_DIR = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 留一法交叉验证的思想：训练 S01-S28，测试 S29-S32
    train_ds = DEAPFinalDataset(DATA_DIR, range(1, 29))
    test_ds = DEAPFinalDataset(DATA_DIR, range(29, 33))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = AdvancedDGFGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n[Final Mode] 正在进行空间增强型泛化训练...")
    best_acc = 0
    for epoch in range(60):
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
        if test_acc > best_acc: best_acc = test_acc
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

if __name__ == "__main__":
    run_final()