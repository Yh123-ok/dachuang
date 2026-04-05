import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Function
from sklearn.model_selection import KFold
import numpy as np
import os
import scipy.io as sio
import random

# ===============================
# 1. 核心层定义 (GRL & GNN)
# ===============================

class GradientReversalLayer(Function):
    """梯度反转层：正向传播保持不变，反向传播梯度取反并乘以 alpha"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class AdaptiveGNN(nn.Module):
    """自适应图卷积：动态学习脑区间的关联权重"""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        adj = F.softmax(attn, dim=-1)
        
        out = torch.matmul(adj, v).transpose(1, 2).reshape(B, N, -1)
        return self.norm(self.proj(out) + x)

# ===============================
# 2. 主模型：MH-DGFNet (全模块版)
# ===============================

class MH_DGFNet_Full(nn.Module):
    def __init__(self, hidden_dim=64, num_subjects=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.regions = {
            'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],
            'Parietal': [6, 7, 8, 20, 21, 22],
            'Temporal': [9, 10, 24, 25],
            'Occipital': [28, 29, 30],
            'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]
        }

        # EEG 支路
        self.eeg_proj = nn.Sequential(nn.Linear(7, hidden_dim), nn.ELU())
        self.eeg_gnn = nn.Sequential(AdaptiveGNN(hidden_dim, hidden_dim), AdaptiveGNN(hidden_dim, hidden_dim))
        self.region_ops = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim) for r in self.regions.keys()})

        # 多模态编码
        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(32*16, 8*hidden_dim)
        )
        self.peri_enc = nn.Sequential(nn.Linear(55, 128), nn.ELU(), nn.Linear(128, 8*hidden_dim))
        self.vis_enc = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 4*hidden_dim))

        # 融合中枢 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 多任务头
        self.v_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.domain_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_subjects))

        # 动态 Loss 权重 (Uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, maps, stats, peri, visual, alpha=1.0):
        # EEG 处理
        h_eeg = self.eeg_gnn(self.eeg_proj(stats))
        h_neuro = torch.stack([F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1))) for r, idx in self.regions.items()], dim=1)

        # 模态对齐
        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)

        # 全局融合
        combined = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1) # [B, 25, hidden]
        feat = self.transformer(combined).mean(dim=1) # 全局平均池化

        # 输出
        v_logits = self.v_head(feat)
        a_logits = self.a_head(feat)
        
        # 域对抗分支
        feat_rev = GradientReversalLayer.apply(feat, alpha)
        d_logits = self.domain_head(feat_rev)

        return v_logits, a_logits, d_logits

# ===============================
# 3. 数据集加载 (被试级归一化)
# ===============================

class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files):
        m_list, s_list, p_list, v_list, a_list, vis_list, sub_ids = [], [], [], [], [], [], []
        
        print("正在预加载数据至内存并进行受试者级 Z-score 归一化...")
        for sub_idx, f in enumerate(files):
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            v_list.append(np.repeat((lbl[:, 0] > 5).astype(int), 15))
            a_list.append(np.repeat((lbl[:, 1] > 5).astype(int), 15))
            
            with np.load(os.path.join(npz_path, f)) as d:
                # 执行被试级独立归一化 (消除个体阻抗差异的关键)
                def norm(x): return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
                m_list.append(norm(d['eeg_allband_feature_map']))
                s_list.append(norm(d['eeg_en_stat']))
                p_list.append(norm(d['peri_feature']))
            
            # 视觉特征
            v_feats = []
            for t in range(1, 41):
                p = os.path.join(visual_npy_path, f"{sid}_trial{t:02d}_features.npy")
                feat = np.load(p) if os.path.exists(p) else np.zeros((15, 512))
                if feat.ndim == 1: feat = np.tile(feat, (15, 1))
                elif feat.shape[0] != 15: feat = np.vstack([np.mean(x, 0) for x in np.array_split(feat, 15)])
                v_feats.append(feat)
            vis_list.append(np.concatenate(v_feats))
            sub_ids.extend([sub_idx] * 600)

        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float()
        self.vl, self.al = torch.from_numpy(np.concatenate(v_list)).long(), torch.from_numpy(np.concatenate(a_list)).long()
        self.sub_l = torch.tensor(sub_ids).long()

    def __len__(self): return len(self.vl)
    def __getitem__(self, i): return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i], self.sub_l[i]

# ===============================
# 4. 严谨的 5-Fold 训练主函数
# ===============================

def train_rigorous():
    # 路径配置
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VIS_DIR = r'D:\Users\cyz\dc\see'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    full_ds = DeapLoaderRAM(NPZ_DIR, MAT_DIR, VIS_DIR, files)
    
    # 5折划分逻辑
    subjects = np.arange(len(files))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    v_results, a_results = [], []

    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\n>>>> Fold {fold+1}/5 | 测试集受试者: {subjects[test_subs]}")
        
        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]
        
        train_loader = DataLoader(Subset(full_ds, train_idx), 128, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), 128, shuffle=False)

        model = MH_DGFNet_Full(num_subjects=len(subjects)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
        scaler = torch.cuda.amp.GradScaler() # AMP 加速器

        # 训练循环
        for ep in range(30):
            model.train()
            alpha = 2. / (1. + np.exp(-10 * (ep / 30))) - 1 # 动态 Alpha
            
            for m, s, p, v, lv, la, ld in train_loader:
                m, s, p, v, lv, la, ld = [x.to(DEVICE) for x in [m, s, p, v, lv, la, ld]]
                
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda'):
                    ov, oa, od = model(m, s, p, v, alpha=alpha)
                    
                    # 标签平滑与多任务损失
                    l_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                    l_a = F.cross_entropy(oa, la, label_smoothing=0.1)
                    l_d = F.cross_entropy(od, ld)
                    
                    # 动态权重平衡损失
                    loss = (torch.exp(-model.log_vars[0]) * l_v + model.log_vars[0]) + \
                           (torch.exp(-model.log_vars[1]) * l_a + model.log_vars[1]) + \
                           (torch.exp(-model.log_vars[2]) * l_d + model.log_vars[2])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # 评估本折结果
        model.eval()
        vc, ac, total = 0, 0, 0
        with torch.no_grad():
            for m, s, p, v, lv, la, _ in test_loader:
                ov, oa, _ = model(m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE))
                vc += (ov.argmax(1) == lv.to(DEVICE)).sum().item()
                ac += (oa.argmax(1) == la.to(DEVICE)).sum().item()
                total += lv.size(0)
        
        v_results.append(vc/total); a_results.append(ac/total)
        print(f"Fold {fold+1} 结果: Valence {vc/total:.4f}, Arousal {ac/total:.4f}")

    print("\n" + "="*40)
    print(f"🏁 5-Fold 最终平均精度:")
    print(f"Valence: {np.mean(v_results):.4f} (±{np.std(v_results):.4f})")
    print(f"Arousal: {np.mean(a_results):.4f} (±{np.std(a_results):.4f})")
    print("="*40)

if __name__ == "__main__":
    # 多进程启动保护
    torch.multiprocessing.freeze_support()
    train_rigorous()