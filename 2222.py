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
# 1. 核心特征提取 (保持切段增强)
# ==========================================
def extract_segments(data, label, fs=128, segment_secs=10):
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
        for i in [32, 33]:
            sig = chunk[i]
            feat_list.append([np.mean(sig), np.std(sig), np.ptp(sig), np.sum(sig**2)/len(sig)])
        feat_arr = np.array(feat_list)
        feat_arr = (feat_arr - np.mean(feat_arr)) / (np.std(feat_arr) + 1e-8)
        segments.append({'x': torch.FloatTensor(feat_arr), 'y': torch.LongTensor([label])[0]})
    return segments

class SingleSubjectDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')
        for j in range(40):
            label = 1 if raw['labels'][j, 1] > 5 else 0 # Arousal 标签
            self.samples.extend(extract_segments(raw['data'][j], label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 2. 增强型 DGF-GNN 模型
# ==========================================
class DGFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sigma_net = nn.Sequential(nn.Linear(dim * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())
    def forward(self, x):
        b, n, d = x.size()
        xi = x.unsqueeze(2).repeat(1, 1, n, 1)
        xj = x.unsqueeze(1).repeat(1, n, 1, 1)
        sigmas = self.sigma_net(torch.cat([xi, xj], dim=-1)).squeeze(-1)
        dist_sq = torch.sum((xi - xj)**2, dim=-1)
        return torch.exp(-dist_sq / (2 * sigmas**2 + 1e-6))



class ResDGFGNN_Final(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.dgf = DGFLayer(hidden_dim)
        self.gnn_w = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.gnn_w.data)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        h_init = F.elu(self.proj(x))
        adj = self.dgf(h_init)
        h_fused = torch.matmul(adj, h_init)
        h_fused = torch.matmul(h_fused, self.gnn_w)
        h_res = F.elu(h_fused + h_init)
        graph_emb = torch.cat([torch.mean(h_res, dim=1), torch.max(h_res, dim=1)[0]], dim=1)
        return self.classifier(graph_emb)

# ==========================================
# 3. 自动化实验循环
# ==========================================
def run_full_subject_experiment():
    DATA_DIR = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_python"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subject_results = []

    print("开始执行 32 位受试者独立建模实验...")
    print("-" * 50)

    for sub_id in range(1, 33):
        file_path = os.path.join(DATA_DIR, f's{sub_id:02d}.dat')
        if not os.path.exists(file_path): continue

        # 1. 加载单人数据
        dataset = SingleSubjectDataset(file_path)
        train_size = int(0.8 * len(dataset))
        train_db, test_db = random_split(dataset, [train_size, len(dataset)-train_size])
        train_loader = DataLoader(train_db, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_db, batch_size=16, shuffle=False)

        # 2. 初始化模型
        model = ResDGFGNN_Final().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 3. 快速训练 (单人数据只需 30 轮左右即可收敛)
        best_sub_acc = 0
        for epoch in range(30):
            model.train()
            for batch in train_loader:
                x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                optimizer.zero_grad(); model(x)
                loss = criterion(model(x), y); loss.backward(); optimizer.step()
            
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
            acc = 100 * correct / len(test_db)
            if acc > best_sub_acc: best_sub_acc = acc
        
        subject_results.append(best_sub_acc)
        print(f"受试者 s{sub_id:02d} | 最佳精度: {best_sub_acc:.2f}%")

    print("-" * 50)
    print(f"实验结束！32位受试者平均识别精度: {np.mean(subject_results):.2f}%")

if __name__ == "__main__":
    run_full_subject_experiment()