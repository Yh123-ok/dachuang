# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
import os
import scipy.io as sio

# ===============================
# 0. 全局配置与加速
# ===============================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ===============================
# 1. 模型组件 (强化正则化版)
# ===============================
class AdaptiveGNN(nn.Module):
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
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.norm(self.proj(out) + x)

class MH_DGFNet_Final(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.regions = {
            'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],
            'Parietal': [6, 7, 8, 20, 21, 22],
            'Temporal': [9, 10, 24, 25],
            'Occipital': [28, 29, 30],
            'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]
        }

        self.eeg_proj = nn.Sequential(nn.Linear(7, hidden_dim), nn.ELU())
        self.eeg_gnn = nn.Sequential(AdaptiveGNN(hidden_dim, hidden_dim), AdaptiveGNN(hidden_dim, hidden_dim))
        self.region_ops = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim) for r in self.regions.keys()})

        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1), # 加入 stride=2 降维防止参数爆炸
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 再降一次
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten(),
            nn.Linear(64 * 16, 8 * hidden_dim) # 对应这里的 64 通道
        )
        self.peri_enc = nn.Sequential(nn.Linear(55, 128), nn.ELU(), nn.Linear(128, 8*hidden_dim))
        self.vis_enc = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.5), nn.Linear(512, 4*hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.6)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 强化版分类头：增加 Dropout 强制泛化
        self.v_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.6), nn.Linear(32, 2))
        self.a_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.6), nn.Linear(32, 2))

    def forward(self, maps, stats, peri, visual):
        h_eeg = self.eeg_gnn(self.eeg_proj(stats))
        h_neuro = torch.stack([F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1))) for r, idx in self.regions.items()], dim=1)
        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)
        combined = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1)
        feat = self.transformer(combined).mean(dim=1)
        return self.v_head(feat), self.a_head(feat)

# ===============================
# 2. 数据加载 (被试内归一化 + Trial ID 追踪)
# ===============================
class DeapLoaderFinal(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files):
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        sub_ids, trial_ids = [], []
        keep_last = 8

        def sub_norm(x): # 被试内标准化函数
            return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        print("正在进行被试内归一化加载...")
        for sub_idx, f in enumerate(files):
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            
            with np.load(os.path.join(npz_path, f)) as d:
                # 获取原始数据并进行标准化
                m_raw = sub_norm(d['eeg_allband_feature_map'])
                s_raw = sub_norm(d['eeg_en_stat'])
                p_raw = sub_norm(d['peri_feature'])
                m_sub = m_raw.reshape(40, 15, 5, 32, 32)[:, -keep_last:, ...]
                s_sub = s_raw.reshape(40, 15, 32, 7)[:, -keep_last:, ...]
                p_sub = p_raw.reshape(40, 15, 55)[:, -keep_last:, ...]
                
                m_list.append(m_sub.reshape(-1, 5, 32, 32)) # 注意这里改为 32, 32
                s_list.append(s_sub.reshape(-1, 32, 7))
                p_list.append(p_sub.reshape(-1, 55))

            v_list.append(np.repeat((lbl[:, 0] > 5).astype(int), keep_last))
            a_list.append(np.repeat((lbl[:, 1] > 5).astype(int), keep_last))
            
            # 记录 Trial ID 用于后续投票
            cur_trials = np.repeat(np.arange(40), keep_last) + (sub_idx * 40)
            trial_ids.extend(cur_trials)
            sub_ids.extend([sub_idx] * (40 * keep_last))

            # 视觉特征加载与截断
            v_feats_sub = []
            for t in range(1, 41):
                path = os.path.join(visual_npy_path, f"{sid}_trial{t:02d}_features.npy")
                feat = np.load(path) if os.path.exists(path) else np.zeros((15, 512))
                if feat.ndim == 1: feat = np.tile(feat, (15, 1))
                elif feat.shape[0] != 15: feat = np.vstack([np.mean(x, 0) for x in np.array_split(feat, 15)])
                v_feats_sub.append(feat[-keep_last:, :])
            vis_list.append(np.concatenate(v_feats_sub))

        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float()
        self.vl, self.al = torch.from_numpy(np.concatenate(v_list)).long(), torch.from_numpy(np.concatenate(a_list)).long()
        self.sub_l, self.trial_l = torch.tensor(sub_ids).long(), torch.tensor(trial_ids).long()

    def __len__(self): return len(self.vl)
    def __getitem__(self, i): return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i], self.trial_l[i]

# ===============================
# 3. 训练与投票验证逻辑
# ===============================
def train_and_vote():
    NPZ_DIR, MAT_DIR, VIS_DIR = r'D:\Users\cyz\dc\222', r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab', r'D:\Users\cyz\dc\see'
    DEVICE = torch.device("cuda")
    
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    full_ds = DeapLoaderFinal(NPZ_DIR, MAT_DIR, VIS_DIR, files)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    subjects = np.arange(len(files))

    v_accs, a_accs = [], []

    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\n--- Fold {fold+1} ---")
        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]

        train_loader = DataLoader(Subset(full_ds, train_idx), 128, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), 128, shuffle=False, num_workers=0)

        model = MH_DGFNet_Final().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        scaler = torch.amp.GradScaler('cuda')

        for ep in range(100):
            model.train()
            for m, s, p, v, lv, la, _ in train_loader:
                m, s, p, v, lv, la = m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE), lv.to(DEVICE), la.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)
                    loss = 0.5 * F.cross_entropy(ov, lv, label_smoothing=0.1) + 0.5 * F.cross_entropy(oa, la, label_smoothing=0.1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

        # ===== 核心：Trial 级投票验证 =====
        model.eval()
        trial_votes_v, trial_votes_a = {}, {}
        trial_ground_truth_v, trial_ground_truth_a = {}, {}

        

        with torch.no_grad():
            for m, s, p, v, lv, la, tid in test_loader:
                m, s, p, v = m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE)
                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)
                
                pv, pa = ov.argmax(1).cpu().numpy(), oa.argmax(1).cpu().numpy()
                tids, lsv, lsa = tid.numpy(), lv.numpy(), la.numpy()

                for i in range(len(tids)):
                    t_id = tids[i]
                    if t_id not in trial_votes_v:
                        trial_votes_v[t_id], trial_votes_a[t_id] = [], []
                    trial_votes_v[t_id].append(pv[i])
                    trial_votes_a[t_id].append(pa[i])
                    trial_ground_truth_v[t_id] = lsv[i]
                    trial_ground_truth_a[t_id] = lsa[i]

        # 计算投票后的准确率
        correct_v = sum([1 for t_id, preds in trial_votes_v.items() if Counter(preds).most_common(1)[0][0] == trial_ground_truth_v[t_id]])
        correct_a = sum([1 for t_id, preds in trial_votes_a.items() if Counter(preds).most_common(1)[0][0] == trial_ground_truth_a[t_id]])
        
        v_acc, a_acc = correct_v / len(trial_votes_v), correct_a / len(trial_votes_a)
        v_accs.append(v_acc); a_accs.append(a_acc)
        print(f"Fold {fold+1} 投票结果 -> Valence: {v_acc:.4f}, Arousal: {a_acc:.4f}")

    print(f"\n最终投票准确率: Valence {np.mean(v_accs):.4f}, Arousal {np.mean(a_accs):.4f}")

if __name__ == "__main__":
    train_and_vote()