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
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文标签

# ==========================================
# 2. 高阶特征工程与数据加载
# ==========================================
def extract_advanced_feats(data, fs=128):
    bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
    all_trials = []
    for t in range(40):
        ch_feats = []
        for c in range(34):
            f, psd = welch(data[t, c], fs, nperseg=fs)
            de = [0.5 * np.log(2 * np.pi * np.e * np.mean(psd[(f>=b[0]) & (f<=b[1])]) + 1e-8) for b in bands]
            ch_feats.append(de)
        fe = np.array(ch_feats)
        # 受试者内标准化：消除个体电压基准差异
        fe = (fe - np.mean(fe, axis=0)) / (np.std(fe, axis=0) + 1e-8)
        all_trials.append(fe)
    return np.array(all_trials)

class CrossSubjectDataset(Dataset):
    def __init__(self, data_dir, subjects):
        self.x, self.y = [], []
        print(f"--- 正在预加载受试者群组: {list(subjects)} ---")
        for s in subjects:
            path = os.path.join(data_dir, f's{s:02d}.dat')
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')
            self.x.append(extract_advanced_feats(raw['data']))
            self.y.append((raw['labels'][:, 1] > 5).astype(int))
        self.x = torch.FloatTensor(np.concatenate(self.x))
        self.y = torch.LongTensor(np.concatenate(self.y))

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 3. MS-DGF-GNN 模型架构
# ==========================================
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

class FinalProjectModel(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128):
        super().__init__()
        self.feat_ext = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.channel_attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 8), nn.ReLU(), nn.Linear(hidden_dim // 8, hidden_dim), nn.Sigmoid())
        self.gnn1 = MultiHeadDGF(hidden_dim, num_heads=4)
        self.gnn2 = MultiHeadDGF(hidden_dim, num_heads=2)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.5), nn.Linear(64, 2))

    def forward(self, x):
        h = self.feat_ext(x)
        h = h * self.channel_attn(torch.mean(h, dim=1)).unsqueeze(1)
        h = F.elu(self.gnn1(h) + h)
        h = F.elu(self.gnn2(h) + h)
        feat = torch.cat([torch.mean(h, dim=1), torch.max(h, dim=1)[0]], dim=1)
        return self.classifier(feat)

# ==========================================
# 4. 训练、验证与可视化系统
# ==========================================
def train_and_visualize():
    DATA_PATH = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    # 划分训练/测试集 (模拟跨受试者泛化)
    train_set = CrossSubjectDataset(DATA_PATH, range(1, 27))
    test_set = CrossSubjectDataset(DATA_PATH, range(27, 33))
    train_loader = DataLoader(train_set, batch_size=48, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=48, shuffle=False)

    model = FinalProjectModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {'train_loss': [], 'test_acc': []}
    best_acc = 0
    all_preds, all_labels = [], []

    print("\n--- 启动大创结题级实验方案 ---")
    for epoch in range(60):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        temp_preds, temp_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(DEVICE))
                preds = out.argmax(1)
                correct += (preds == y.to(DEVICE)).sum().item()
                temp_preds.extend(preds.cpu().numpy())
                temp_labels.extend(y.cpu().numpy())

        acc = 100 * correct / len(test_set)
        history['train_loss'].append(epoch_loss/len(train_loader))
        history['test_acc'].append(acc)

        if acc > best_acc:
            best_acc = acc
            all_preds, all_labels = temp_preds, temp_labels
            torch.save(model.state_dict(), 'best_dgf_model.pth')

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {history['train_loss'][-1]:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    # --- 可视化生成 ---
    # 1. 损失与精度曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('训练损失收敛曲线')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], color='orange', label='Test Acc')
    plt.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.1f}%')
    plt.title('跨受试者测试准确率趋势')
    plt.legend()
    plt.savefig('learning_curve.png')
    
    # 2. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['低唤醒', '高唤醒'], yticklabels=['低唤醒', '高唤醒'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'跨受试者识别混淆矩阵 (Best Acc: {best_acc:.2f}%)')
    plt.savefig('confusion_matrix.png')

    print(f"\n实验完成！最优准确率: {best_acc:.2f}%")
    print("图表 'learning_curve.png' 和 'confusion_matrix.png' 已保存在当前目录。")

if __name__ == "__main__":
    train_and_visualize()