import os
import gc
import copy
import random
import warnings
from collections import defaultdict

import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


# ===============================
# 0. 随机种子
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ===============================
# 1. 全局计算加速配置
# ===============================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


# ===============================
# 2. AdaptiveGNN
# 脑电模态内特征提取，不改变融合方法
# ===============================
class AdaptiveGNN(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, ffn_ratio=2):
        super().__init__()

        assert out_dim % heads == 0
        assert in_dim == out_dim

        self.heads = heads
        self.head_dim = out_dim // heads

        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, ffn_ratio * out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * out_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        residual = x
        x_norm = self.norm1(x)

        q = self.q(x_norm).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_norm).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        x = residual + out
        x = x + self.ffn(self.norm2(x))

        return x


# ===============================
# 3. Dynamic Graph Fusion
# 融合方法保持不变
# ===============================
class DynamicGraphFusion(nn.Module):
    def __init__(self, dim, k_neighbors=10, dropout=0.3):
        super().__init__()

        self.k_neighbors = k_neighbors

        self.theta = nn.Linear(dim, dim)
        self.phi = nn.Linear(dim, dim)
        self.gcn_weight = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        k = min(self.k_neighbors, N)

        theta_x = self.theta(x)
        phi_x = self.phi(x)

        sim_matrix = torch.matmul(theta_x, phi_x.transpose(1, 2)) / (C ** 0.5)

        if k < N:
            mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            topk_idx = torch.topk(sim_matrix, k=k, dim=-1).indices
            mask.scatter_(2, topk_idx, True)
            sim_matrix = sim_matrix.masked_fill(~mask, -1e4)

        adj = F.softmax(sim_matrix, dim=-1)
        adj = self.dropout(adj)

        gcn_x = self.gcn_weight(x)
        out = torch.matmul(adj, gcn_x)

        return self.norm(x + self.act(out))


# ===============================
# 4. 主模型 MH-DGFNet
# ===============================
class MH_DGFNet_Full(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.regions = {
            'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],
            'Parietal': [6, 7, 8, 20, 21, 22],
            'Temporal': [9, 10, 24, 25],
            'Occipital': [28, 29, 30],
            'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]
        }

        self.eeg_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 32, hidden_dim))

        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim, heads=4, dropout=dropout),
            AdaptiveGNN(hidden_dim, hidden_dim, heads=4, dropout=dropout)
        )

        self.region_ops = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            for r in self.regions.keys()
        })

        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(32 * 16, 8 * hidden_dim),
            nn.LayerNorm(8 * hidden_dim),
            nn.Dropout(dropout)
        )

        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 8 * hidden_dim),
            nn.LayerNorm(8 * hidden_dim),
            nn.Dropout(dropout)
        )

        self.vis_enc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),

            nn.Linear(512, 4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.modality_embed = nn.Parameter(torch.zeros(4, hidden_dim))
        self.super_node = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.super_node, std=0.02)

        self.dgf_layers = nn.Sequential(
            DynamicGraphFusion(hidden_dim, k_neighbors=12, dropout=dropout),
            DynamicGraphFusion(hidden_dim, k_neighbors=12, dropout=dropout)
        )

        self.v_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        self.a_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, maps, stats, peri, visual):
        # stats: B, 32, 7
        h_eeg = self.eeg_proj(stats) + self.pos_embed
        h_eeg = self.eeg_gnn(h_eeg)

        h_neuro = torch.stack([
            self.region_ops[r](h_eeg[:, idx, :].mean(dim=1))
            for r, idx in self.regions.items()
        ], dim=1)

        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)

        h_neuro = h_neuro + self.modality_embed[0]
        h_ms = h_ms + self.modality_embed[1]
        h_peri = h_peri + self.modality_embed[2]
        h_vis = h_vis + self.modality_embed[3]

        combined_nodes = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1)

        B = combined_nodes.size(0)
        super_n = self.super_node.expand(B, -1, -1)

        graph_nodes = torch.cat([super_n, combined_nodes], dim=1)

        fused_nodes = self.dgf_layers(graph_nodes)

        graph_repr = fused_nodes[:, 0] + fused_nodes[:, 1:].mean(dim=1)

        v_logits = self.v_head(graph_repr)
        a_logits = self.a_head(graph_repr)

        return v_logits, a_logits


# ===============================
# 5. 数据集
# ===============================
class DeapLoaderFixed(Dataset):
    def __init__(
        self,
        npz_path,
        mat_path,
        visual_npy_path,
        files,
        use_all_segments=True
    ):
        super().__init__()

        self.segments_per_trial = 15
        self.keep_last = 15 if use_all_segments else 8

        m_list = []
        s_list = []
        p_list = []
        vis_list = []

        v_label_list = []
        a_label_list = []

        sub_ids = []
        trial_ids = []

        print(f"加载数据：每个 Trial 使用 {self.keep_last} 个片段")
        print("标签规则：Valence/Arousal <3 为 0，>7 为 1，3~7 为 -1 并在训练/测试中忽略。")

        def encode_extreme_labels(scores, low_th=3.0, high_th=7.0):
            labels = np.full(scores.shape, -1, dtype=np.int64)
            labels[scores < low_th] = 0
            labels[scores > high_th] = 1
            return labels

        def filter_segments(arr):
            # 输入: 40 * 15, ...
            arr = arr.reshape(40, self.segments_per_trial, *arr.shape[1:])
            arr = arr[:, -self.keep_last:, ...]
            arr = arr.reshape(-1, *arr.shape[2:])
            return arr

        valid_sub_idx = 0

        for f in tqdm(files, desc="Loading subjects"):
            sid = f[:3]

            npz_file = os.path.join(npz_path, f)
            mat_file = os.path.join(mat_path, f"{sid}.mat")

            if not os.path.exists(npz_file):
                warnings.warn(f"找不到 NPZ 文件：{npz_file}，跳过")
                continue

            if not os.path.exists(mat_file):
                warnings.warn(f"找不到 MAT 标签文件：{mat_file}，跳过")
                continue

            lbl = sio.loadmat(mat_file)['labels']

            # DEAP labels 通常为 40 x 4
            # 第 0 列 Valence，第 1 列 Arousal
            valence_scores = lbl[:, 0]
            arousal_scores = lbl[:, 1]

            v_labels = encode_extreme_labels(valence_scores)
            a_labels = encode_extreme_labels(arousal_scores)

            with np.load(npz_file) as d:
                maps = filter_segments(d['eeg_allband_feature_map'])
                stats = filter_segments(d['eeg_en_stat'])
                peri = filter_segments(d['peri_feature'])

            visual_feats = []

            for t in range(1, 41):
                vis_path = os.path.join(
                    visual_npy_path,
                    f"{sid}_trial{t:02d}_features.npy"
                )

                if os.path.exists(vis_path):
                    feat = np.load(vis_path)
                else:
                    warnings.warn(f"缺失视觉特征：{vis_path}，使用 0 填充")
                    feat = np.zeros((15, 512), dtype=np.float32)

                if feat.ndim == 1:
                    feat = np.tile(feat, (15, 1))

                if feat.shape[0] != 15:
                    feat = np.vstack([
                        np.mean(x, axis=0)
                        for x in np.array_split(feat, 15)
                    ])

                visual_feats.append(feat[-self.keep_last:, :])

            visual_feats = np.concatenate(visual_feats, axis=0)

            repeated_v = np.repeat(v_labels, self.keep_last)
            repeated_a = np.repeat(a_labels, self.keep_last)

            assert len(repeated_v) == maps.shape[0]
            assert len(repeated_a) == maps.shape[0]
            assert visual_feats.shape[0] == maps.shape[0]

            m_list.append(maps.astype(np.float32))
            s_list.append(stats.astype(np.float32))
            p_list.append(peri.astype(np.float32))
            vis_list.append(visual_feats.astype(np.float32))

            v_label_list.append(repeated_v.astype(np.int64))
            a_label_list.append(repeated_a.astype(np.int64))

            num_samples = maps.shape[0]

            sub_ids.extend([valid_sub_idx] * num_samples)

            for t in range(40):
                global_trial_id = valid_sub_idx * 100 + t
                trial_ids.extend([global_trial_id] * self.keep_last)

            valid_sub_idx += 1

        print("合并数据...")

        self.m = torch.from_numpy(np.concatenate(m_list, axis=0)).float()
        self.s = torch.from_numpy(np.concatenate(s_list, axis=0)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list, axis=0)).float()
        self.vis = torch.from_numpy(np.concatenate(vis_list, axis=0)).float()

        self.vl = torch.from_numpy(np.concatenate(v_label_list, axis=0)).long()
        self.al = torch.from_numpy(np.concatenate(a_label_list, axis=0)).long()

        self.sub_l = torch.tensor(sub_ids).long()
        self.trial_ids = torch.tensor(trial_ids).long()

        print(f"数据加载完成，总样本数：{len(self.vl)}")
        print(f"有效被试数：{len(torch.unique(self.sub_l))}")
        print(f"maps shape : {self.m.shape}")
        print(f"stats shape: {self.s.shape}")
        print(f"peri shape : {self.p.shape}")
        print(f"vis shape  : {self.vis.shape}")

    def __len__(self):
        return len(self.vl)

    def __getitem__(self, idx):
        return (
            self.m[idx],
            self.s[idx],
            self.p[idx],
            self.vis[idx],
            self.vl[idx],
            self.al[idx],
            self.trial_ids[idx]
        )


# ===============================
# 6. EMA
# ===============================
class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for k in ema_state.keys():
            if ema_state[k].is_floating_point():
                ema_state[k].mul_(self.decay).add_(
                    model_state[k].detach(),
                    alpha=1.0 - self.decay
                )
            else:
                ema_state[k].copy_(model_state[k])


# ===============================
# 7. 工具函数
# ===============================
def add_gaussian_noise(tensor, std=0.03, prob=0.5):
    if torch.rand(1, device=tensor.device).item() < prob:
        return tensor + torch.randn_like(tensor) * std
    return tensor


def make_class_weight(labels, device):
    labels = labels[labels != -1]

    if len(labels) == 0:
        return torch.ones(2, dtype=torch.float32, device=device)

    cnt = np.bincount(labels, minlength=2).astype(np.float32)

    # 防止某一类为 0
    if cnt[0] == 0 or cnt[1] == 0:
        return torch.ones(2, dtype=torch.float32, device=device)

    weight = cnt.sum() / (2.0 * cnt + 1e-6)
    weight = weight / weight.mean()

    return torch.tensor(weight, dtype=torch.float32, device=device)


def compute_scaler_stats(full_ds, train_idx, device):
    train_m = full_ds.m[train_idx].numpy()
    train_s = full_ds.s[train_idx].numpy()
    train_p = full_ds.p[train_idx].numpy()
    train_vis = full_ds.vis[train_idx].numpy()

    # maps: N, 5, H, W
    m_mean = train_m.mean(axis=(0, 2, 3), keepdims=True)
    m_std = train_m.std(axis=(0, 2, 3), keepdims=True) + 1e-6

    # stats: N, 32, 7
    s_mean = train_s.reshape(-1, train_s.shape[-1]).mean(axis=0)
    s_std = train_s.reshape(-1, train_s.shape[-1]).std(axis=0) + 1e-6

    # peri: N, 55
    p_mean = train_p.mean(axis=0)
    p_std = train_p.std(axis=0) + 1e-6

    # visual: N, 512
    vis_mean = train_vis.mean(axis=0)
    vis_std = train_vis.std(axis=0) + 1e-6

    scaler_stats = {
        'm': (
            torch.from_numpy(m_mean).float().to(device),
            torch.from_numpy(m_std).float().to(device)
        ),
        's': (
            torch.from_numpy(s_mean).float().to(device),
            torch.from_numpy(s_std).float().to(device)
        ),
        'p': (
            torch.from_numpy(p_mean).float().to(device),
            torch.from_numpy(p_std).float().to(device)
        ),
        'vis': (
            torch.from_numpy(vis_mean).float().to(device),
            torch.from_numpy(vis_std).float().to(device)
        )
    }

    return scaler_stats


def normalize_batch(m, s, p, vis, scaler_stats):
    m = (m - scaler_stats['m'][0]) / scaler_stats['m'][1]
    s = (s - scaler_stats['s'][0]) / scaler_stats['s'][1]
    p = (p - scaler_stats['p'][0]) / scaler_stats['p'][1]
    vis = (vis - scaler_stats['vis'][0]) / scaler_stats['vis'][1]

    return m, s, p, vis


# ===============================
# 8. 测试函数
# 同时返回 segment-level 和 trial-level
# ===============================
@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    scaler_stats,
    device,
    amp_enabled=True
):
    model.eval()

    # segment-level
    seg_v_correct = 0
    seg_a_correct = 0
    seg_v_total = 0
    seg_a_total = 0

    # trial-level
    trial_v_logits = defaultdict(list)
    trial_a_logits = defaultdict(list)
    trial_v_label = {}
    trial_a_label = {}

    for m, s, p, vis, lv, la, tids in test_loader:
        m = m.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)

        lv = lv.to(device, non_blocking=True)
        la = la.to(device, non_blocking=True)

        m, s, p, vis = normalize_batch(m, s, p, vis, scaler_stats)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            ov, oa = model(m, s, p, vis)

        pred_v = ov.argmax(dim=1)
        pred_a = oa.argmax(dim=1)

        mask_v = lv != -1
        mask_a = la != -1

        if mask_v.sum().item() > 0:
            seg_v_correct += (pred_v[mask_v] == lv[mask_v]).sum().item()
            seg_v_total += mask_v.sum().item()

        if mask_a.sum().item() > 0:
            seg_a_correct += (pred_a[mask_a] == la[mask_a]).sum().item()
            seg_a_total += mask_a.sum().item()

        ov_cpu = ov.detach().float().cpu()
        oa_cpu = oa.detach().float().cpu()
        lv_cpu = lv.detach().cpu()
        la_cpu = la.detach().cpu()
        tids_cpu = tids.detach().cpu()

        for i in range(len(tids_cpu)):
            tid = int(tids_cpu[i].item())

            if int(lv_cpu[i].item()) != -1:
                trial_v_logits[tid].append(ov_cpu[i])
                trial_v_label[tid] = int(lv_cpu[i].item())

            if int(la_cpu[i].item()) != -1:
                trial_a_logits[tid].append(oa_cpu[i])
                trial_a_label[tid] = int(la_cpu[i].item())

    seg_v_acc = seg_v_correct / seg_v_total if seg_v_total > 0 else 0.0
    seg_a_acc = seg_a_correct / seg_a_total if seg_a_total > 0 else 0.0

    # trial-level 平均 logits
    trial_v_correct = 0
    trial_a_correct = 0
    trial_v_total = 0
    trial_a_total = 0

    for tid, logits_list in trial_v_logits.items():
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        pred = int(avg_logits.argmax().item())
        label = trial_v_label[tid]

        trial_v_correct += int(pred == label)
        trial_v_total += 1

    for tid, logits_list in trial_a_logits.items():
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        pred = int(avg_logits.argmax().item())
        label = trial_a_label[tid]

        trial_a_correct += int(pred == label)
        trial_a_total += 1

    trial_v_acc = trial_v_correct / trial_v_total if trial_v_total > 0 else 0.0
    trial_a_acc = trial_a_correct / trial_a_total if trial_a_total > 0 else 0.0

    return {
        "seg_v_acc": seg_v_acc,
        "seg_a_acc": seg_a_acc,
        "seg_v_total": seg_v_total,
        "seg_a_total": seg_a_total,

        "trial_v_acc": trial_v_acc,
        "trial_a_acc": trial_a_acc,
        "trial_v_total": trial_v_total,
        "trial_a_total": trial_a_total
    }


# ===============================
# 9. 主训练 + 测试流程
# ===============================
def train_rigorous():
    # ===========================
    # 修改成你自己的路径
    # ===========================
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VIS_DIR = r'D:\Users\cyz\dc\see'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = DEVICE.type == "cuda"

    print(f"使用设备：{DEVICE}")

    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])

    if len(files) == 0:
        raise FileNotFoundError(f"没有找到 npz 文件：{NPZ_DIR}")

    full_ds = DeapLoaderFixed(
        npz_path=NPZ_DIR,
        mat_path=MAT_DIR,
        visual_npy_path=VIS_DIR,
        files=files,
        use_all_segments=True
    )

    subjects = torch.unique(full_ds.sub_l).numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    seg_v_results = []
    seg_a_results = []
    trial_v_results = []
    trial_a_results = []

    for fold, (train_subs_idx, test_subs_idx) in enumerate(kf.split(subjects)):
        print("\n" + "=" * 60)
        print(f"Fold {fold + 1} / 5")
        print("=" * 60)

        train_subjects = subjects[train_subs_idx]
        test_subjects = subjects[test_subs_idx]

        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), train_subjects))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), test_subjects))[0]

        print(f"训练样本数：{len(train_idx)}")
        print(f"测试样本数：{len(test_idx)}")

        scaler_stats = compute_scaler_stats(full_ds, train_idx, DEVICE)

        v_weight = make_class_weight(
            full_ds.vl[train_idx].numpy(),
            DEVICE
        )

        a_weight = make_class_weight(
            full_ds.al[train_idx].numpy(),
            DEVICE
        )

        print(f"Valence 类别权重：{v_weight.detach().cpu().numpy()}")
        print(f"Arousal 类别权重：{a_weight.detach().cpu().numpy()}")

        train_loader = DataLoader(
            Subset(full_ds, train_idx),
            batch_size=128,
            shuffle=True,
            num_workers=0,
            pin_memory=amp_enabled,
            drop_last=False
        )

        test_loader = DataLoader(
            Subset(full_ds, test_idx),
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=amp_enabled,
            drop_last=False
        )

        model = MH_DGFNet_Full(
            hidden_dim=64,
            dropout=0.3
        ).to(DEVICE)

        ema = ModelEMA(model, decay=0.995)

        EPOCHS = 80
        warmup_epochs = 5
        base_lr = 3e-4

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS - warmup_epochs,
            eta_min=1e-6
        )

        scaler = torch.amp.GradScaler(
            'cuda',
            enabled=amp_enabled
        )

        # ===========================
        # 训练
        # ===========================
        for ep in range(EPOCHS):
            model.train()

            if ep < warmup_epochs:
                lr = base_lr * float(ep + 1) / float(warmup_epochs)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            total_loss = 0.0
            valid_batches = 0

            for m, s, p, vis, lv, la, tids in train_loader:
                m = m.to(DEVICE, non_blocking=True)
                s = s.to(DEVICE, non_blocking=True)
                p = p.to(DEVICE, non_blocking=True)
                vis = vis.to(DEVICE, non_blocking=True)

                lv = lv.to(DEVICE, non_blocking=True)
                la = la.to(DEVICE, non_blocking=True)

                m, s, p, vis = normalize_batch(m, s, p, vis, scaler_stats)

                # 数据增强，只在前半段训练使用
                if ep < 40:
                    s = add_gaussian_noise(s, std=0.03, prob=0.5)
                    p = add_gaussian_noise(p, std=0.03, prob=0.5)
                    m = add_gaussian_noise(m, std=0.02, prob=0.4)
                    vis = add_gaussian_noise(vis, std=0.01, prob=0.3)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type, enabled=amp_enabled):
                    ov, oa = model(m, s, p, vis)

                    mask_v = lv != -1
                    mask_a = la != -1

                    losses = []

                    if mask_v.sum().item() > 0:
                        loss_v = F.cross_entropy(
                            ov[mask_v],
                            lv[mask_v],
                            weight=v_weight,
                            label_smoothing=0.05
                        )
                        losses.append(loss_v)

                    if mask_a.sum().item() > 0:
                        loss_a = F.cross_entropy(
                            oa[mask_a],
                            la[mask_a],
                            weight=a_weight,
                            label_smoothing=0.05
                        )
                        losses.append(loss_a)

                    if len(losses) == 0:
                        continue

                    loss = sum(losses) / len(losses)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

                ema.update(model)

                total_loss += loss.item()
                valid_batches += 1

            if ep >= warmup_epochs:
                scheduler.step()

            if ep == 0 or (ep + 1) % 10 == 0:
                avg_loss = total_loss / max(valid_batches, 1)
                current_lr = optimizer.param_groups[0]['lr']

                print(
                    f"Epoch [{ep + 1:03d}/{EPOCHS}] "
                    f"Loss: {avg_loss:.4f} "
                    f"LR: {current_lr:.6f}"
                )

        # ===========================
        # 测试
        # 用 EMA 模型测试
        # ===========================
        print("\n开始测试当前 Fold...")

        metrics = evaluate_model(
            model=ema.ema,
            test_loader=test_loader,
            scaler_stats=scaler_stats,
            device=DEVICE,
            amp_enabled=amp_enabled
        )

        seg_v_results.append(metrics["seg_v_acc"])
        seg_a_results.append(metrics["seg_a_acc"])

        trial_v_results.append(metrics["trial_v_acc"])
        trial_a_results.append(metrics["trial_a_acc"])

        print(
            f"Fold {fold + 1} Segment-level -> "
            f"Valence Acc: {metrics['seg_v_acc']:.4f} "
            f"({metrics['seg_v_total']} samples), "
            f"Arousal Acc: {metrics['seg_a_acc']:.4f} "
            f"({metrics['seg_a_total']} samples)"
        )

        print(
            f"Fold {fold + 1} Trial-level   -> "
            f"Valence Acc: {metrics['trial_v_acc']:.4f} "
            f"({metrics['trial_v_total']} trials), "
            f"Arousal Acc: {metrics['trial_a_acc']:.4f} "
            f"({metrics['trial_a_total']} trials)"
        )

        del model
        del ema
        del optimizer
        del scheduler
        del scaler
        del train_loader
        del test_loader

        torch.cuda.empty_cache()
        gc.collect()

    # ===========================
    # 最终结果
    # ===========================
    print("\n" + "=" * 70)
    print("最终 5-Fold 交叉验证结果")
    print("=" * 70)

    print(
        f"Segment-level Valence Accuracy: "
        f"{np.mean(seg_v_results):.4f} ± {np.std(seg_v_results):.4f}"
    )

    print(
        f"Segment-level Arousal Accuracy: "
        f"{np.mean(seg_a_results):.4f} ± {np.std(seg_a_results):.4f}"
    )

    print("-" * 70)

    print(
        f"Trial-level Valence Accuracy: "
        f"{np.mean(trial_v_results):.4f} ± {np.std(trial_v_results):.4f}"
    )

    print(
        f"Trial-level Arousal Accuracy: "
        f"{np.mean(trial_a_results):.4f} ± {np.std(trial_a_results):.4f}"
    )

    print("=" * 70)


# ===============================
# 10. 程序入口
# ===============================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_rigorous()
