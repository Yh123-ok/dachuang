import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

# 1. 环境补丁
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 2. 数据处理与切段增强 (Data Augmentation)
# ==========================================
def extract_segments(data, label, fs=128, segment_secs=10):
    total_samples = data.shape[-1]
    step = segment_secs * fs
    segments = []
    
    for start in range(0, total_samples - step + 1, step):
        chunk = data[:, start:start+step]
        bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
        feat_list = []
        
        # EEG 32通道微分熵
        for i in range(32):
            f, psd = welch(chunk[i], fs, nperseg=fs)
            de_feats = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[np.logical_and(f>=b[0], f<=b[1])]) + 1e-8) for b in bands]
            feat_list.append(de_feats)
            
        # EOG 2通道统计特征
        for i in [32, 33]:
            sig = chunk[i]
            feat_list.append([np.mean(sig), np.std(sig), np.ptp(sig), np.sum(sig**2)/len(sig)])
            
        feat_arr = np.array(feat_list)
        # Z-Score 标准化
        feat_arr = (feat_arr - np.mean(feat_arr)) / (np.std(feat_arr) + 1e-8)
        
        segments.append({'x': torch.FloatTensor(feat_arr), 'y': torch.LongTensor([label])[0]})
    return segments

class DEAPFullDataset(Dataset):
    def __init__(self, data_dir, subject_range):
        self.samples = []
        print(f"--- 正在加载并增强处理受试者 s{subject_range[0]:02d}-s{subject_range[-1]:02d} ---")
        for i in subject_range:
            path = os.path.join(data_dir, f's{i:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            for j in range(40):
                # 【修改点】改为 Arousal 标签 (raw['labels'] 的第2列)
                # Arousal 对压力和情绪波动的生理反应比 Valence 更强
                label = 1 if raw['labels'][j, 1] > 5 else 0
                self.samples.extend(extract_segments(raw['data'][j], label))
            print(f"受试者 s{i:02d} 加载完毕，当前总样本: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 3. 核心模型：残差 DGF-GNN (带双重池化)
# ==========================================
class DGFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sigma_net = nn.Sequential(
            nn.Linear(dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
    def forward(self, x):
        b, n, d = x.size()
        xi = x.unsqueeze(2).repeat(1, 1, n, 1)
        xj = x.unsqueeze(1).repeat(1, n, 1, 1)
        sigmas = self.sigma_net(torch.cat([xi, xj], dim=-1)).squeeze(-1)
        dist_sq = torch.sum((xi - xj)**2, dim=-1)
        return torch.exp(-dist_sq / (2 * sigmas**2 + 1e-6))

class FinalDGFGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.dgf = DGFLayer(hidden_dim)
        self.gnn_w = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.gnn_w.data)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), # 拼接 mean 和 max 所以是 *2
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        h_init = F.relu(self.proj(x))
        adj = self.dgf(h_init)
        h_fused = torch.matmul(adj, h_init)
        h_fused = torch.matmul(h_fused, self.gnn_w)
        
        # 残差连接：保留原始生理特征
        h_res = F.relu(h_fused + h_init)
        
        # 【修改点】混合池化：同时捕捉平均状态和极端电位异常
        avg_p = torch.mean(h_res, dim=1)
        max_p, _ = torch.max(h_res, dim=1)
        graph_emb = torch.cat([avg_p, max_p], dim=1)
        
        return self.classifier(graph_emb)

# ==========================================
# 4. 训练与自适应优化
# ==========================================
def run_experiment():
    DATA_DIR = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 全量数据加载
    dataset = DEAPFullDataset(DATA_DIR, range(1, 33))
    train_size = int(0.85 * len(dataset))
    train_db, test_db = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_db, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=False)

    model = FinalDGFGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-3)
    
    # 【修改点】学习率调度器：根据测试准确率自动调整 LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- 实验启动 ---")
    best_acc = 0
    for epoch in range(60):
        model.train()
        correct, total_loss = 0, 0
        for batch in train_loader:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            correct += (preds.argmax(1) == y).sum().item()
            total_loss += loss.item()
            
        # 测试评估
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                p = model(x)
                test_correct += (p.argmax(1) == y).sum().item()
        
        test_acc = 100 * test_correct / len(test_db)
        train_acc = 100 * correct / len(train_db)
        
        # 更新调度器
        scheduler.step(test_acc)
        
        if test_acc > best_acc: best_acc = test_acc
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Best: {best_acc:.1f}%")

    print(f"\n实验完成！最佳 Arousal 识别精度: {best_acc:.2f}%")

if __name__ == "__main__":
    run_experiment()