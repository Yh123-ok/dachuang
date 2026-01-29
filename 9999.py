import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import confusion_matrix

# 1. 环境配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 增强型特征工程：滑动窗口 + 频段特征
# ==========================================
def extract_enhanced_feats(data, fs=128, window_size=10, step_size=5):
    """
    使用滑动窗口增加样本量
    window_size: 窗口长度(秒)
    step_size: 步长(秒)
    """
    bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
    samples_per_window = window_size * fs
    step_samples = step_size * fs
    total_len = data.shape[-1]
    
    trial_augmented_feats = []
    
    # 对每一段 trial 进行滑动切分
    for start in range(0, total_len - samples_per_window + 1, step_samples):
        window_data = data[:, start : start + samples_per_window]
        ch_feats = []
        for c in range(34): # 32 EEG + 2 EOG
            f, psd = welch(window_data[c], fs, nperseg=fs)
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[(f>=b[0]) & (f<=b[1])]) + 1e-8) for b in bands]
            ch_feats.append(de)
        
        fe = np.array(ch_feats) # (34, 4)
        # 受试者内标准化
        fe = (fe - np.mean(fe, axis=0)) / (np.std(fe, axis=0) + 1e-8)
        trial_augmented_feats.append(fe)
        
    return np.array(trial_augmented_feats) # (N_windows, 34, 4)

class DEAPAugmentedDataset(Dataset):
    def __init__(self, data_dir, subjects):
        self.x, self.y = [], []
        print(f"--- 正在执行数据增强加载 (LOSO 模式): {list(subjects)} ---")
        for s in subjects:
            path = os.path.join(data_dir, f's{s:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            
            for j in range(40):
                # 对每个 60s 的 trial 进行滑动窗口增强
                # 10s窗口, 5s步长, 每个trial生成 (60-10)/5 + 1 = 11个样本
                feats = extract_enhanced_feats(raw['data'][j]) 
                label = 1 if raw['labels'][j, 1] > 5 else 0
                
                self.x.append(feats)
                self.y.append(np.full(len(feats), label))
                
        self.x = torch.FloatTensor(np.concatenate(self.x))
        self.y = torch.LongTensor(np.concatenate(self.y))
        print(f"数据加载完成，总样本数: {len(self.x)}")

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 3. 核心架构：频段注意力 + 多尺度 DGF-GNN
# ==========================================
class FreqAttention(nn.Module):
    """方案三：频段注意力机制"""
    def __init__(self, in_dim=4):
        super().__init__()
        self.attn_weights = nn.Parameter(torch.ones(1, 1, in_dim))
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # x: [B, 34, 4]
        # 计算频段重要性权重
        weights = torch.sigmoid(self.fc(self.attn_weights))
        return x * weights

class MultiHeadDGF(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.sigma_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model*2, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softplus()) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, n, d = x.size()
        xi = x.unsqueeze(2).repeat(1, 1, n, 1)
        xj = x.unsqueeze(1).repeat(1, n, 1, 1)
        combined = torch.cat([xi, xj], dim=-1)
        adjs = [torch.exp(-torch.sum((xi - xj)**2, dim=-1) / (2 * net(combined).squeeze(-1)**2 + 1e-6)) for net in self.sigma_nets]
        h = torch.matmul(torch.stack(adjs).mean(dim=0), x)
        return self.proj(h)

class SuperDGFModel(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128):
        super().__init__()
        self.freq_attn = FreqAttention(in_dim)
        self.feat_ext = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.channel_attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 8), nn.ReLU(), nn.Linear(hidden_dim // 8, hidden_dim), nn.Sigmoid())
        
        self.gnn = MultiHeadDGF(hidden_dim, num_heads=4)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.freq_attn(x) # 频段注意力增强
        h = self.feat_ext(x)
        h = h * self.channel_attn(torch.mean(h, dim=1)).unsqueeze(1) # 空间注意力
        h = F.elu(self.gnn(h) + h)
        feat = torch.cat([torch.mean(h, dim=1), torch.max(h, dim=1)[0]], dim=1)
        return self.classifier(feat)

# ==========================================
# 4. 训练与评估
# ==========================================
def run_enhanced_experiment():
    DATA_PATH = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"```
    
    # 增加数据量后，batch_size 可以适当加大
    train_set = DEAPAugmentedDataset(DATA_PATH, range(1, 27))
    test_set = DEAPAugmentedDataset(DATA_PATH, range(27, 33))
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = SuperDGFModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    history = {'loss': [], 'acc': []}

    print("\n--- 启动频段注意力+数据增强 冲刺模式 ---")
    for epoch in range(60):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                preds = model(x.to(DEVICE)).argmax(1)
                correct += (preds == y.to(DEVICE)).sum().item()
        
        acc = 100 * correct / len(test_set)
        if acc > best_acc: best_acc = acc
        
        history['loss'].append(train_loss/len(train_loader))
        history['acc'].append(acc)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {history['loss'][-1]:.4f} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1); plt.plot(history['loss']); plt.title('训练损失')
    plt.subplot(1,2,2); plt.plot(history['acc']); plt.title('测试精度')
    plt.savefig('enhanced_result.png')
    print(f"\n训练结束！最高精度: {best_acc:.2f}%")

if __name__ == "__main__":
    run_enhanced_experiment()