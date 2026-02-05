import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score
import numpy as np
import os
import scipy.io as sio

# ==========================================
# 🛠️ 模块 1: 基础组件 - 残差动态图层
# ==========================================
class ResDGFLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x, return_adj=False):
        # 修正缩进：确保 forward 在类内部
        b, n, d = x.size()
        dist_sq = torch.cdist(x, x, p=2)**2
        sigmas = torch.exp(self.log_sigmas)
        adjs = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6))
        adj_final = adjs.mean(dim=1) 
        
        out = torch.matmul(adj_final, x)
        out = self.proj(out)
        res = self.norm(self.activation(out) + x)
        
        if return_adj:
            return res, adj_final
        return res

# ==========================================
# 🛠️ 模块 2: 深层 GNN 堆叠块
# ==========================================
class DeepGNNBlock(nn.Module):
    def __init__(self, d_model, layers=3, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            ResDGFLayer(d_model, num_heads) for _ in range(layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            # 内部堆叠层不需要返回邻接矩阵
            x = layer(x)
        return x

# ==========================================
# 🧠 模块 3: Deep_MT_DGF_GNN 模型主体
# ==========================================
class Deep_MT_DGF_GNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 分支 A: CNN
        self.cnn_net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), 
            nn.Flatten()
        )
        self.cnn_proj = nn.Linear(128, hidden_dim)

        # 分支 B: Deep GNN
        self.gnn_mapping = nn.Linear(7, hidden_dim)
        self.deep_gnn = DeepGNNBlock(d_model=hidden_dim, layers=3, num_heads=4)

        # 分支 C: MLP
        self.peri_net = nn.Sequential(
            nn.Linear(55, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ELU()
        )

        # 融合模块
        self.fusion_layer = ResDGFLayer(hidden_dim, num_heads=8)
        
        # 多任务输出头
        combined_dim = hidden_dim * 3
        self.v_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.log_vars = nn.Parameter(torch.zeros(2)) 

    def forward(self, maps, stats, peri, return_weights=False):
        # 1. 特征提取
        h_cnn = self.cnn_proj(self.cnn_net(maps)).unsqueeze(1)
        
        gnn_in = self.gnn_mapping(stats) 
        h_gnn_nodes = self.deep_gnn(gnn_in) 
        h_gnn = h_gnn_nodes.mean(dim=1, keepdim=True)
        
        h_peri = self.peri_net(peri).unsqueeze(1)

        # 2. 动态融合
        combined = torch.cat([h_cnn, h_gnn, h_peri], dim=1) 
        
        # 修正：支持根据 return_weights 返回邻接矩阵
        if return_weights:
            fused, weights = self.fusion_layer(combined, return_adj=True)
        else:
            fused = self.fusion_layer(combined, return_adj=False)
        
        # 3. 展平并分类
        flat_feat = fused.view(fused.size(0), -1)
        v_out, a_out = self.v_head(flat_feat), self.a_head(flat_feat)
        
        if return_weights:
            return v_out, a_out, weights
        return v_out, a_out

# ==========================================
# 💾 模块 4: 数据加载与标签对齐 (LOSO)
# ==========================================
class DeapLOSODataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir):
        self.npz_dir = npz_dir
        self.raw_mat_dir = raw_mat_dir
        self.file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.samples_per_subject = 600
        
        self.v_labels_list = []
        self.a_labels_list = []
        
        print(f"📦 正在初始化数据管理器... 共检测到 {len(self.file_list)} 个被试")
        
        for f_name in self.file_list:
            subj_id = f_name[:3] 
            mat_path = os.path.join(raw_mat_dir, f"{subj_id}.mat")
            if not os.path.exists(mat_path):
                raise FileNotFoundError(f"未找到标签文件: {mat_path}")
            raw_labels = sio.loadmat(mat_path)['labels']
            v_bin = (raw_labels[:, 0] > 5).astype(np.int64)
            a_bin = (raw_labels[:, 1] > 5).astype(np.int64)
            self.v_labels_list.append(np.repeat(v_bin, 15))
            self.a_labels_list.append(np.repeat(a_bin, 15))
            
        self.all_v = np.concatenate(self.v_labels_list)
        self.all_a = np.concatenate(self.a_labels_list)

    def __len__(self):
        return len(self.all_v)

    def get_train_indices(self, test_subj_idx):
        all_indices = np.arange(len(self))
        test_start = test_subj_idx * self.samples_per_subject
        test_end = (test_subj_idx + 1) * self.samples_per_subject
        train_mask = np.ones(len(self), dtype=bool)
        train_mask[test_start:test_end] = False
        return all_indices[train_mask].tolist()

    def get_subject_data(self, subj_idx):
        file_path = os.path.join(self.npz_dir, self.file_list[subj_idx])
        with np.load(file_path) as data:
            m = torch.from_numpy(data['eeg_allband_feature_map']).float()
            s = torch.from_numpy(data['eeg_en_stat']).view(-1, 32, 7).float()
            p = torch.from_numpy(data['peri_feature']).float()
        v = torch.from_numpy(self.v_labels_list[subj_idx]).long()
        a = torch.from_numpy(self.a_labels_list[subj_idx]).long()
        return m, s, p, v, a

    def __getitem__(self, idx):
        subj_idx = idx // self.samples_per_subject
        inner_idx = idx % self.samples_per_subject
        file_path = os.path.join(self.npz_dir, self.file_list[subj_idx])
        with np.load(file_path) as data:
            maps = torch.from_numpy(data['eeg_allband_feature_map'][inner_idx]).float()
            stats = torch.from_numpy(data['eeg_en_stat'][inner_idx]).view(32, 7).float()
            peri = torch.from_numpy(data['peri_feature'][inner_idx]).float()
        return maps, stats, peri, self.all_v[idx], self.all_a[idx]

# ==========================================
# 🚀 模块 5: 训练引擎与评估 (LOSO)
# ==========================================
def train_deep_mt_dgf_loso():
    NPZ_PATH = r'D:\Users\cyz\dc\222'
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    BATCH_SIZE = 64
    EPOCHS = 10 
    LR = 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = DeapLOSODataset(NPZ_PATH, RAW_PATH)
    num_subjects = len(dataset.file_list)
    all_subject_f1 = []

    print(f"\n⚡ 启动 LOSO 验证流程 | 被试总数: {num_subjects} | 设备: {DEVICE}")

    for test_subj_idx in range(num_subjects):
        subj_name = dataset.file_list[test_subj_idx][:3]
        print(f"\n>>> [Fold {test_subj_idx+1}/{num_subjects}] 测试被试: {subj_name}")
        
        train_idx = dataset.get_train_indices(test_subj_idx)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        
        tm, ts, tp, tlv, tla = dataset.get_subject_data(test_subj_idx)
        tm, ts, tp = tm.to(DEVICE), ts.to(DEVICE), tp.to(DEVICE)

        model = Deep_MT_DGF_GNN(hidden_dim=64).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for maps, stats, peri, lv, la in train_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)
                
                ov, oa = model(maps, stats, peri) # 默认 return_weights=False
                
                loss_v = F.cross_entropy(ov, lv)
                loss_a = F.cross_entropy(oa, la)
                loss = (loss_v * torch.exp(-model.log_vars[0]) + model.log_vars[0]) + \
                       (loss_a * torch.exp(-model.log_vars[1]) + model.log_vars[1])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            # 评估时开启权重提取
            ov, oa, weights = model(tm, ts, tp, return_weights=True)
            
            f1_v = f1_score(tlv.numpy(), ov.argmax(dim=1).cpu().numpy(), average='macro')
            f1_a = f1_score(tla.numpy(), oa.argmax(dim=1).cpu().numpy(), average='macro')
            
            # 模态贡献度分析
            imp = weights.mean(dim=0).sum(dim=0).cpu().numpy()
            imp /= imp.sum()
            
            print(f"   Done. Valence F1: {f1_v:.4f} | Arousal F1: {f1_a:.4f}")
            print(f"   🧠 模态贡献度: EEG-CNN: {imp[0]:.1%} | EEG-GNN: {imp[1]:.1%} | Peri: {imp[2]:.1%}")
            
            all_subject_f1.append((f1_v + f1_a) / 2)
            
        del model, optimizer
        torch.cuda.empty_cache()

    print("\n" + "=".center(40, "="))
    print(f"🏆 LOSO 最终平均 F1: {np.mean(all_subject_f1):.4f}")
    print("=".center(40, "="))

if __name__ == "__main__":
    train_deep_mt_dgf_loso()