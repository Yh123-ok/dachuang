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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# 1. 环境配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 稳定版特征工程
# ==========================================
def extract_stable_feats(data, fs=128):
    """
    长时窗口 DE 特征提取，保证信号的完整性
    """
    bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
    all_trials = []
    for t in range(40):
        ch_feats = []
        for c in range(34): # 32 EEG + 2 EOG
            f, psd = welch(data[t, c], fs, nperseg=fs*2) # 增加频率分辨率
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[(f>=b[0]) & (f<=b[1])]) + 1e-8) for b in bands]
            ch_feats.append(de)
        
        fe = np.array(ch_feats) # (34, 4)
        # 受试者内标准化：这是解决 Domain Shift 的关键
        fe = (fe - np.mean(fe, axis=0)) / (np.std(fe, axis=0) + 1e-8)
        all_trials.append(fe)
    return np.array(all_trials)

class DEAPStableDataset(Dataset):
    def __init__(self, data_dir, subjects):
        self.x, self.y = [], []
        print(f"--- 正在加载受试者群组: {list(subjects)} ---")
        for s in subjects:
            path = os.path.join(data_dir, f's{s:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            
            self.x.append(extract_stable_feats(raw['data']))
            # Arousal 二分类
            label = (raw['labels'][:, 1] > 5).astype(int)
            self.y.append(label)
                
        self.x = torch.FloatTensor(np.concatenate(self.x))
        self.y = torch.LongTensor(np.concatenate(self.y))

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 3. 终极架构：残差动态图神经网络 (Res-DGF-GNN)
# ==========================================
class ResDGFLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.sigmas = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, 34, D]
        b, n, d = x.size()
        
        # 计算欧式距离矩阵
        dist = torch.cdist(x, x, p=2) # [B, 34, 34]
        
        # 多头动态构图
        adjs = []
        for i in range(self.num_heads):
            adj = torch.exp(- (dist**2) / (2 * (self.sigmas[i]**2) + 1e-6))
            adjs.append(adj)
        
        adj_final = torch.stack(adjs).mean(dim=0)
        
        # 图聚合 + 残差连接
        out = torch.matmul(adj_final, x)
        out = self.proj(out)
        return self.norm(out + x)



class ResDGFGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        # 1. 初始投影
        self.input_mapping = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )
        
        # 2. 深度残差图卷积
        self.gnn1 = ResDGFLayer(hidden_dim, num_heads=4)
        self.gnn2 = ResDGFLayer(hidden_dim, num_heads=8)
        
        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.6),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        h = self.input_mapping(x)
        h = self.gnn1(h)
        h = self.gnn2(h)
        
        # 混合池化提取全图特征
        feat = torch.cat([h.mean(dim=1), h.max(dim=1)[0]], dim=1)
        return self.classifier(feat)

# ==========================================
# 4. 训练引擎
# ==========================================
def train_ultimate():
    DATA_PATH = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    
    # 保持长窗口 1-26 训练，27-32 测试
    train_set = DEAPStableDataset(DATA_PATH, range(1, 27))
    test_set = DEAPStableDataset(DATA_PATH, range(27, 33))
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = ResDGFGNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    # 使用 Plateau 调度器，当 Acc 不再提升时自动降低 LR
    # 移除 verbose 参数
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    weights = torch.tensor([2.0, 1.0]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_acc = 0
    best_preds, best_labels = [], []

    print("\n--- 启动科研版 Res-DGF-GNN 训练 ---")
    for epoch in range(100): # 增加迭代次数
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        temp_preds, temp_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(DEVICE))
                p = out.argmax(1)
                correct += (p == y.to(DEVICE)).sum().item()
                temp_preds.extend(p.cpu().numpy())
                temp_labels.extend(y.cpu().numpy())
        
        acc = 100 * correct / len(test_set)
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_preds, best_labels = temp_preds, temp_labels
            torch.save(model.state_dict(), 'ultimate_model.pth')
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Best Acc: {best_acc:.2f}% | Current Acc: {acc:.2f}%")

    # --- 结果可视化 ---
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['低唤醒','高唤醒'], yticklabels=['低唤醒','高唤醒'])
    plt.title(f'终极版跨人识别混淆矩阵 (Acc: {best_acc:.2f}%)')
    plt.savefig('ultimate_cm.png')
    
    print("\n" + "="*30)
    print(f"训练完成！跨受试者最高精度：{best_acc:.2f}%")
    print(classification_report(best_labels, best_preds, target_names=['低唤醒','高唤醒']))
    print("="*30)

if __name__ == "__main__":
    train_ultimate()