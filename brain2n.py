import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Function
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
# 梯度反转层 (GRL) - 用于 MT 域对抗
# ===============================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，梯度乘一个负数 alpha，实现对抗学习
        output = grad_output.neg() * ctx.alpha
        return output, None

# ===============================
# 网络基础模块
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
        q = self.q(x)
        k = self.k(x)
        attn_adj = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        adj = F.softmax(attn_adj, dim=-1)
        out = torch.matmul(adj, x)
        return self.norm(self.act(self.proj(out)) + x)

class FeatureGate(nn.Module):
    def __init__(self, dim, num_modalities=4):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Sigmoid()
        )

    def forward(self, modal_list):
        pooled = torch.cat([m.mean(dim=1) for m in modal_list], dim=-1)
        gates = self.gate_net(pooled)
        fused = [m * gates[:, i].unsqueeze(1).unsqueeze(2) for i, m in enumerate(modal_list)]
        return torch.cat(fused, dim=1)

class GraphTransformer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ffn(x))
        return x

# ===============================
# 主模型 MH-DGFNet (引入 MT 域对抗)
# ===============================
class MH_DGFNet(nn.Module):
    def __init__(self, hidden_dim=64, num_subjects=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.eeg_proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(7, hidden_dim))
        self.eeg_gnn = nn.Sequential(DynamicGNNLayer(hidden_dim, hidden_dim), DynamicGNNLayer(hidden_dim, hidden_dim))
        self.region_nets = nn.ModuleDict({r: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU()) for r in BRAIN_REGIONS.keys()})

        self.ms_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten()
        )
        self.ms_node_gen = nn.Linear(64 * 4 * 4, 8 * hidden_dim)

        self.peri_node_gen = nn.Sequential(nn.Linear(55, 128), nn.LayerNorm(128), nn.ELU(), nn.Linear(128, 8 * hidden_dim))
        self.vis_node_gen = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.LayerNorm(128), nn.ELU(), nn.Linear(128, 4 * hidden_dim))

        self.feature_gate = FeatureGate(hidden_dim, num_modalities=4)
        self.modality_emb = nn.Parameter(torch.randn(1, 25, hidden_dim) * 0.02)
        self.graph_transformer = GraphTransformer(hidden_dim)
        
        self.dropout = nn.Dropout(0.5) # 提高 Dropout 抑制过拟合

        # 情感分类头 (Valence & Arousal)
        self.v_head = nn.Sequential(nn.Linear(25 * hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(25 * hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        
        # 域分类头 (Subject ID Classification)
        self.domain_head = nn.Sequential(nn.Linear(25 * hidden_dim, 64), nn.ELU(), nn.Linear(64, num_subjects))

        # 3个任务的不确定性权重 (V, A, Domain)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, maps, stats, peri, visual, alpha=0.0):
        h = self.eeg_gnn(self.eeg_proj(stats))
        h_neuro = torch.stack([self.region_nets[r](h[:, idx, :].mean(dim=1)) for r, idx in BRAIN_REGIONS.items()], dim=1)
        h_ms = self.ms_node_gen(self.ms_encoder(maps)).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_node_gen(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_node_gen(visual).view(-1, 4, self.hidden_dim)

        nodes = self.feature_gate([h_neuro, h_ms, h_peri, h_vis]) + self.modality_emb
        feat = self.dropout(self.graph_transformer(nodes).view(nodes.size(0), -1))

        # 情感预测
        v_out = self.v_head(feat)
        a_out = self.a_head(feat)
        
        # 域/被试预测 (经过 GRL)
        feat_rev = GradientReversalLayer.apply(feat, alpha)
        d_out = self.domain_head(feat_rev)

        return v_out, a_out, d_out

# ===============================
# 数据集加载 (增加归一化)
# ===============================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files, vis_dim=512):
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        self.subject_ids = []

        for sub_idx, f in enumerate(files):
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            sub_v = np.repeat((lbl[:, 0] > 5).astype(int), 15)
            sub_a = np.repeat((lbl[:, 1] > 5).astype(int), 15)
            v_list.append(sub_v)
            a_list.append(sub_a)
            self.subject_ids.extend([sub_idx] * len(sub_v))

            with np.load(os.path.join(npz_path, f)) as d:
                # 独立 Z-score 归一化提取共性
                m_feat = d['eeg_allband_feature_map']
                m_feat = (m_feat - np.mean(m_feat, axis=0, keepdims=True)) / (np.std(m_feat, axis=0, keepdims=True) + 1e-5)
                m_list.append(m_feat)
                
                s_feat = d['eeg_en_stat']
                s_feat = (s_feat - np.mean(s_feat, axis=0, keepdims=True)) / (np.std(s_feat, axis=0, keepdims=True) + 1e-5)
                s_list.append(s_feat)

                p_feat = d['peri_feature']
                p_feat = (p_feat - np.mean(p_feat, axis=0, keepdims=True)) / (np.std(p_feat, axis=0, keepdims=True) + 1e-5)
                p_list.append(p_feat)

            subj_visuals = []
            for t in range(1, 41):
                path = os.path.join(visual_npy_path, f"{sid}_trial{t:02d}_features.npy")
                if os.path.exists(path):
                    feat = np.load(path)
                    if feat.ndim == 1: feat = np.tile(feat, (15, 1))
                    elif feat.shape[0] != 15: feat = np.vstack([np.mean(x, axis=0) for x in np.array_split(feat, 15)])
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
        self.sub_l = torch.tensor(self.subject_ids, dtype=torch.long) # Domain 标签

    def __len__(self): return len(self.vl)
    def __getitem__(self, i): return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i], self.sub_l[i]

# ===============================
# 5-Fold 交叉验证训练逻辑 (仅修改此处)
# ===============================
def run_5fold():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    RAW_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VISUAL_DIR = r'D:\Users\cyz\dc\see'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH = 64
    EPOCHS = 40 
    LR = 5e-4

    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    num_subjects = len(files)

    print("开始加载全量归一化数据至内存...")
    full_dataset = DeapLoaderRAM(NPZ_DIR, RAW_DIR, VISUAL_DIR, files)
    print(f"加载完成，总样本数: {len(full_dataset)}")

    # 设置 5 折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    subjects = np.arange(num_subjects)
    
    v_accs, a_accs = [], []
    scaler = torch.cuda.amp.GradScaler()

    # 按受试者 ID 进行划分，确保测试集是模型从未见过的“陌生人”
    for fold, (train_subs_idx, test_subs_idx) in enumerate(kf.split(subjects)):
        train_subjects = subjects[train_subs_idx]
        test_subjects = subjects[test_subs_idx]
        
        print(f"\n>>>> 开始第 {fold+1}/5 折验证 | 测试集受试者编号: {train_subjects}")

        # 根据划分好的受试者 ID 提取对应的样本索引
        train_idx = np.where(np.isin(full_dataset.sub_l.numpy(), train_subjects))[0]
        test_idx = np.where(np.isin(full_dataset.sub_l.numpy(), test_subjects))[0]

        train_ds = Subset(full_dataset, train_idx)
        test_ds = Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_ds, BATCH, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=4, pin_memory=True)

        # 初始化模型
        model = MH_DGFNet(num_subjects=num_subjects).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        # 训练循环
        for ep in range(EPOCHS):
            model.train()
            p = ep / EPOCHS
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            for m, s, p_feat, v, lv, la, ld in train_loader:
                m, s, p_feat, v = m.to(DEVICE), s.to(DEVICE), p_feat.to(DEVICE), v.to(DEVICE)
                lv, la, ld = lv.to(DEVICE), la.to(DEVICE), ld.to(DEVICE)

                opt.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    ov, oa, od = model(m, s, p_feat, v, alpha=alpha)

                    loss_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                    loss_a = F.cross_entropy(oa, la, label_smoothing=0.1)
                    loss_d = F.cross_entropy(od, ld)

                    loss = (0.5 * torch.exp(-model.log_vars[0]) * loss_v + 0.5 * model.log_vars[0]) + \
                           (0.5 * torch.exp(-model.log_vars[1]) * loss_a + 0.5 * model.log_vars[1]) + \
                           (0.5 * torch.exp(-model.log_vars[2]) * loss_d + 0.5 * model.log_vars[2])

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            
            scheduler.step()

        # 验证循环
        model.eval()
        vc, ac, total = 0, 0, 0
        with torch.no_grad():
            for m, s, p_feat, v, lv, la, _ in test_loader:
                ov, oa, _ = model(m.to(DEVICE), s.to(DEVICE), p_feat.to(DEVICE), v.to(DEVICE), alpha=0.0)
                vc += (ov.argmax(1) == lv.to(DEVICE)).sum().item()
                ac += (oa.argmax(1) == la.to(DEVICE)).sum().item()
                total += lv.size(0)

        v_acc, a_acc = vc / total, ac / total
        v_accs.append(v_acc)
        a_accs.append(a_acc)
        print(f"Fold {fold+1} 完成 | Valence Acc: {v_acc:.4f} | Arousal Acc: {a_acc:.4f}")

    print("\n" + "="*40)
    print(f"🏁 5-Fold 最终平均精度 (Subject-Independent):")
    print(f"Mean Valence Acc: {np.mean(v_accs):.4f} (±{np.std(v_accs):.4f})")
    print(f"Mean Arousal Acc: {np.mean(a_accs):.4f} (±{np.std(a_accs):.4f})")
    print("="*40)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run_5fold()