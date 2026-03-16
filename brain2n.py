import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score
import numpy as np
import os
import scipy.io as sio

torch.backends.cudnn.benchmark = True

# ===============================
# 脑区划分
# ===============================
BRAIN_REGIONS = {
    'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],
    'Parietal': [6, 7, 8, 20, 21, 22],
    'Temporal': [9, 10, 24, 25],
    'Occipital': [28, 29, 30],
    'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]
}

# ===============================
# 动态自适应 EEG GNN (改进点1)
# ===============================
class DynamicGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, max(1, out_dim // 4))
        self.k = nn.Linear(in_dim, max(1, out_dim // 4))
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        # x shape: [B, 32, in_dim]
        q = self.q(x)  # [B, 32, C']
        k = self.k(x)  # [B, 32, C']
        # 动态计算图邻接矩阵
        attn_adj = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        adj = F.softmax(attn_adj, dim=-1)
        
        out = torch.matmul(adj, x)
        out = self.proj(out)
        return self.norm(self.act(out) + x)

# ===============================
# 模态特征门控 (改进点2)
# ===============================
class FeatureGate(nn.Module):
    def __init__(self, dim, num_modalities=4):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Sigmoid()
        )

    def forward(self, modal_list):
        # 将各模态特征池化后拼接，计算动态门控权重
        pooled = torch.cat([m.mean(dim=1) for m in modal_list], dim=-1)
        gates = self.gate_net(pooled)  # [B, 4]
        
        fused = []
        for i, m in enumerate(modal_list):
            fused.append(m * gates[:, i].unsqueeze(1).unsqueeze(2))
        return torch.cat(fused, dim=1)

# ===============================
# Graph Transformer
# ===============================
class GraphTransformer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        x = self.norm2(x + ffn)
        return x

# ===============================
# 主模型 MH-DGFNet
# ===============================
class MH_DGFNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # EEG (增加 Dropout 防止过拟合)
        self.eeg_proj = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(7, hidden_dim)
        )
        self.eeg_gnn = nn.Sequential(
            DynamicGNNLayer(hidden_dim, hidden_dim),
            DynamicGNNLayer(hidden_dim, hidden_dim)
        )
        self.region_nets = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
            for r in BRAIN_REGIONS.keys()
        })

        # Microstate CNN
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.ms_node_gen = nn.Linear(64 * 4 * 4, 8 * hidden_dim)

        # Peripheral
        self.peri_node_gen = nn.Sequential(
            nn.Linear(55, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 8 * hidden_dim)
        )

        # Visual (增加 LayerNorm 缓解量级差异)
        self.vis_node_gen = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 4 * hidden_dim)
        )

        # 动态模态门控
        self.feature_gate = FeatureGate(hidden_dim, num_modalities=4)

        # 节点 embedding
        self.modality_emb = nn.Parameter(torch.randn(1, 25, hidden_dim) * 0.02)

        # Graph Transformer
        self.graph_transformer = GraphTransformer(hidden_dim)
        self.dropout = nn.Dropout(0.4)

        self.v_head = nn.Sequential(
            nn.Linear(25 * hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )
        self.a_head = nn.Sequential(
            nn.Linear(25 * hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        # 多任务不确定性损失权重参数
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, maps, stats, peri, visual):
        # EEG
        h = self.eeg_proj(stats)
        h = self.eeg_gnn(h)
        region_nodes = []
        for r, idx in BRAIN_REGIONS.items():
            node = self.region_nets[r](h[:, idx, :].mean(dim=1))
            region_nodes.append(node)
        h_neuro = torch.stack(region_nodes, dim=1)

        # Microstate
        h_ms = self.ms_node_gen(self.ms_encoder(maps)).view(-1, 8, self.hidden_dim)

        # Peripheral
        h_peri = self.peri_node_gen(peri).view(-1, 8, self.hidden_dim)

        # Visual
        h_vis = self.vis_node_gen(visual).view(-1, 4, self.hidden_dim)

        # 使用动态特征门控代替全局静态门控
        nodes = self.feature_gate([h_neuro, h_ms, h_peri, h_vis])
        nodes = nodes + self.modality_emb

        # Graph Transformer
        fused = self.graph_transformer(nodes)
        feat = self.dropout(fused.view(fused.size(0), -1))

        return self.v_head(feat), self.a_head(feat)

# ===============================
# Dataset (支持全量加载并记录受试者ID)
# ===============================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files, vis_dim=512):
        self.vis_dim = vis_dim
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        
        # 记录每个样本对应的受试者索引，方便 LOSO 切分
        self.subject_ids = []

        for sub_idx, f in enumerate(files):
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            
            sub_v = np.repeat((lbl[:, 0] > 5).astype(int), 15)
            sub_a = np.repeat((lbl[:, 1] > 5).astype(int), 15)
            v_list.append(sub_v)
            a_list.append(sub_a)
            
            # 记录当前受试者的所有样本 ID
            self.subject_ids.extend([sub_idx] * len(sub_v))

            with np.load(os.path.join(npz_path, f)) as d:
                m_list.append(d['eeg_allband_feature_map'])
                s_list.append(d['eeg_en_stat'])
                p_list.append(d['peri_feature'])

            subj_visuals = []
            for t in range(1, 41):
                name = f"{sid}_trial{t:02d}_features.npy"
                path = os.path.join(visual_npy_path, name)
                if os.path.exists(path):
                    feat = np.load(path)
                    if feat.ndim == 1:
                        feat = np.tile(feat, (15, 1))
                    elif feat.shape[0] != 15:
                        splits = np.array_split(feat, 15)
                        feat = np.vstack([np.mean(x, axis=0) for x in splits])
                    subj_visuals.append(feat)
                else:
                    subj_visuals.append(np.zeros((15, vis_dim)))

            vis_list.append(np.concatenate(subj_visuals))

        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float()
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()
        self.subject_ids = np.array(self.subject_ids)

    def __len__(self):
        return len(self.vl)

    def __getitem__(self, i):
        return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i]

# ===============================
# 优化后的 LOSO 训练 (改进点3)
# ===============================
def run_loso():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    RAW_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VISUAL_DIR = r'D:\Users\cyz\dc\see'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 64
    EPOCHS = 80
    LR = 5e-4

    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    num_subjects = len(files)

    print("开始加载全量数据至内存，这可能需要一两分钟...")
    # 一次性加载所有受试者数据，告别反复读盘
    full_dataset = DeapLoaderRAM(NPZ_DIR, RAW_DIR, VISUAL_DIR, files)
    print(f"数据加载完成，总样本数: {len(full_dataset)}")

    v_accs = []
    a_accs = []

    print("开始 LOSO 验证")
    for test_sub_idx in range(num_subjects):
        
        # 获取当前受试者的索引作为测试集，其他作为训练集
        test_idx = np.where(full_dataset.subject_ids == test_sub_idx)[0]
        train_idx = np.where(full_dataset.subject_ids != test_sub_idx)[0]

        train_ds = Subset(full_dataset, train_idx)
        test_ds = Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_ds, BATCH, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, BATCH, shuffle=False)

        model = MH_DGFNet().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        
        # 添加余弦退火学习率调度器，让后期收敛更平滑
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        for ep in range(EPOCHS):
            model.train()
            for m, s, p, v, lv, la in train_loader:
                m, s, p, v = m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)

                opt.zero_grad()
                ov, oa = model(m, s, p, v)

                loss_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                loss_a = F.cross_entropy(oa, la, label_smoothing=0.1)

                # 修正后的多任务不确定性损失函数公式: 1/(2*exp(s)) * Loss + s/2
                # log_vars 是 log(sigma^2)，exp(-log_vars) 就是 1/sigma^2
                loss = (0.5 * torch.exp(-model.log_vars[0]) * loss_v + 0.5 * model.log_vars[0]) + \
                       (0.5 * torch.exp(-model.log_vars[1]) * loss_a + 0.5 * model.log_vars[1])

                loss.backward()
                opt.step()
            
            scheduler.step()

        model.eval()
        vc, ac, total = 0, 0, 0
        with torch.no_grad():
            for m, s, p, v, lv, la in test_loader:
                ov, oa = model(m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE))
                
                vc += (ov.argmax(1) == lv.to(DEVICE)).sum().item()
                ac += (oa.argmax(1) == la.to(DEVICE)).sum().item()
                total += lv.size(0)

        v_acc = vc / total
        a_acc = ac / total
        v_accs.append(v_acc)
        a_accs.append(a_acc)

        print(f"[{test_sub_idx+1}/{num_subjects}] Test_Sub: {files[test_sub_idx][:3]} | Val: {v_acc:.4f} Aro: {a_acc:.4f}")

    print("================================")
    print(f"Final Valence Mean Acc: {np.mean(v_accs):.4f}")
    print(f"Final Arousal Mean Acc: {np.mean(a_accs):.4f}")

if __name__ == "__main__":
    run_loso()