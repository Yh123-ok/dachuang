import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import scipy.io as sio
import gc
import warnings
from tqdm import tqdm

# ===============================
# 0. 全局计算加速配置
# ===============================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ===============================
# 1. 自适应图神经网络层 (脑电模态内特征提取)
# ===============================
class AdaptiveGNN(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0

        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        return self.norm(out + x)

# ===============================
# 2. 动态图融合层 (Dynamic Graph Fusion)
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
        k = min(self.k_neighbors, N)  # 防止 k > N

        theta_x = self.theta(x)
        phi_x = self.phi(x)
        sim_matrix = torch.matmul(theta_x, phi_x.transpose(1, 2)) / (C ** 0.5)

        # 稀疏化：保留 top-k 邻居
        if k < N:
            mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            topk_idx = torch.topk(sim_matrix, k=k, dim=-1).indices
            mask.scatter_(2, topk_idx, True)
            min_val = -1e4
            sim_matrix = sim_matrix.masked_fill(~mask, min_val)

        adj = F.softmax(sim_matrix, dim=-1)
        adj = self.dropout(adj)

        gcn_x = self.gcn_weight(x)
        out = torch.matmul(adj, gcn_x)
        return self.norm(x + self.act(out))

# ===============================
# 3. 主模型 MH-DGFNet (增强正则化)
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

        self.eeg_proj = nn.Sequential(nn.Linear(7, hidden_dim), nn.ELU(), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.randn(1, 32, hidden_dim))
        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim, dropout=dropout),
            AdaptiveGNN(hidden_dim, hidden_dim, dropout=dropout)
        )
        self.region_ops = nn.ModuleDict({
            r: nn.Linear(hidden_dim, hidden_dim) for r in self.regions.keys()
        })

        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 16, 8 * hidden_dim), nn.Dropout(dropout)
        )

        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(128, 8 * hidden_dim), nn.Dropout(dropout)
        )

        self.vis_enc = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(dropout),
            nn.Linear(512, 4 * hidden_dim), nn.Dropout(dropout)
        )

        self.modality_embed = nn.Parameter(torch.randn(4, hidden_dim))
        self.super_node = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.dgf_layers = nn.Sequential(
            DynamicGraphFusion(hidden_dim, k_neighbors=12, dropout=dropout),
            DynamicGraphFusion(hidden_dim, k_neighbors=12, dropout=dropout)
        )

        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )
        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )

    def forward(self, maps, stats, peri, visual):
        # stats: (B, 32, 7)
        h_eeg = self.eeg_proj(stats) + self.pos_embed
        h_eeg = self.eeg_gnn(h_eeg)

        # 区域聚合
        h_neuro = torch.stack([
            F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1)))
            for r, idx in self.regions.items()
        ], dim=1)  # (B, 5, hidden_dim)

        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)      # (B, 8, H)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)  # (B, 8, H)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)  # (B, 4, H)

        h_neuro = h_neuro + self.modality_embed[0]
        h_ms = h_ms + self.modality_embed[1]
        h_peri = h_peri + self.modality_embed[2]
        h_vis = h_vis + self.modality_embed[3]

        combined_nodes = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1)  # (B, 5+8+8+4=25, H)
        B = combined_nodes.size(0)
        super_n = self.super_node.expand(B, -1, -1)
        graph_nodes = torch.cat([super_n, combined_nodes], dim=1)  # (B, 26, H)

        fused_nodes = self.dgf_layers(graph_nodes)
        graph_repr = fused_nodes[:, 0] + fused_nodes[:, 1:].mean(dim=1)

        v_logits = self.v_head(graph_repr)
        a_logits = self.a_head(graph_repr)
        return v_logits, a_logits

# ===============================
# 4. 数据加载器 (修复标准化泄露 + 使用全部片段)
# ===============================
class DeapLoaderFixed(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files, 
                 scaler_stats=None, use_all_segments=True):
        """
        scaler_stats: tuple (mean, std) for standardization, if None return raw
        use_all_segments: if True, keep all 15 segments per trial; else keep last 8 (original)
        """
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        sub_ids = []
        self.segments_per_trial = 15
        self.keep_last = 15 if use_all_segments else 8

        print(f"加载数据：每个 Trial 使用 {self.keep_last} 个片段 (全部片段={use_all_segments})")
        print("✅ 启用极端情绪过滤：<3 为 Low(0)，>7 为 High(1)，3~7 之间标记为 -1 并在训练中忽略。")

        def filter_segments(arr):
            # arr shape: (40 * 15, ...) -> reshape to (40, 15, ...)
            reshaped = arr.reshape(40, self.segments_per_trial, *arr.shape[1:])
            return reshaped[:, -self.keep_last:, ...].reshape(-1, *arr.shape[1:])

        def encode_extreme_labels(scores, low_th=3.0, high_th=7.0):
            labels = np.full(scores.shape, -1, dtype=int)
            labels[scores < low_th] = 0
            labels[scores > high_th] = 1
            return labels

        for sub_idx, f in enumerate(tqdm(files, desc="Loading subjects")):
            sid = f[:3]
            lbl_path = os.path.join(mat_path, f"{sid}.mat")
            if not os.path.exists(lbl_path):
                warnings.warn(f"Label file {lbl_path} not found, skip subject {sid}")
                continue
            lbl = sio.loadmat(lbl_path)['labels']  # (40, 2)
            v_labels = encode_extreme_labels(lbl[:, 0])
            a_labels = encode_extreme_labels(lbl[:, 1])

            # 每个 trial 重复 keep_last 次
            v_list.append(np.repeat(v_labels, self.keep_last))
            a_list.append(np.repeat(a_labels, self.keep_last))

            npz_file = os.path.join(npz_path, f)
            if not os.path.exists(npz_file):
                warnings.warn(f"NPZ file {npz_file} not found")
                continue
            with np.load(npz_file) as d:
                m_list.append(filter_segments(d['eeg_allband_feature_map']))
                s_list.append(filter_segments(d['eeg_en_stat']))
                p_list.append(filter_segments(d['peri_feature']))

            # 视觉特征
            v_feats = []
            for t in range(1, 41):
                vis_path = os.path.join(visual_npy_path, f"{sid}_trial{t:02d}_features.npy")
                if os.path.exists(vis_path):
                    feat = np.load(vis_path)
                else:
                    warnings.warn(f"Missing {vis_path}, using zeros")
                    feat = np.zeros((15, 512))
                if feat.ndim == 1:
                    feat = np.tile(feat, (15, 1))
                elif feat.shape[0] != 15:
                    # 如果长度不是15，通过平均分段或插值调整为15
                    feat = np.vstack([np.mean(x, axis=0) for x in np.array_split(feat, 15)])
                v_feats.append(feat[-self.keep_last:, :])
            vis_list.append(np.concatenate(v_feats))

            num_samples = len(v_list[-1])
            sub_ids.extend([sub_idx] * num_samples)

        # 合并所有数据
        print("合并张量...")
        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float()
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()
        self.sub_l = torch.tensor(sub_ids).long()

        # 标准化参数 (如果需要)
        self.scaler_stats = scaler_stats  # (mean, std) for each feature dimension

        print(f"数据加载完成，总样本数: {len(self.vl)}")

    def __len__(self):
        return len(self.vl)

    def __getitem__(self, i):
        m = self.m[i]
        s = self.s[i]
        p = self.p[i]
        vis = self.vis[i]
        if self.scaler_stats is not None:
            mean, std = self.scaler_stats
            # 注意：m, s, p, vis 各自有不同的维度，需要分别标准化
            # 这里简化：仅对 EEG 特征 s (32,7) 和  peri (55,) 等做标准化，实际可扩展
            # 由于原代码中 norm 是对每个样本独立做，我们改为全局标准化
            # 这里我们只对 s 和 p 做标准化，m 和 vis 保持原样（或也做）
            s = (s - mean['s']) / (std['s'] + 1e-6)
            p = (p - mean['p']) / (std['p'] + 1e-6)
            # 对于 maps 和 vis，若需要也可添加
        return m, s, p, vis, self.vl[i], self.al[i]

# ===============================
# 5. 训练流程 (修复标准化泄露 + 增加数据增强)
# ===============================
def add_gaussian_noise(tensor, std=0.05):
    """添加高斯噪声增强"""
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(tensor) * std
        return tensor + noise
    return tensor

def train_rigorous():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VIS_DIR = r'D:\Users\cyz\dc\see'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {DEVICE}")

    # 获取所有被试文件
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found in {NPZ_DIR}")

    # 创建基础数据集（未标准化，原始数据）
    full_ds = DeapLoaderFixed(NPZ_DIR, MAT_DIR, VIS_DIR, files, scaler_stats=None, use_all_segments=True)

    subjects = np.arange(len(files))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    v_results, a_results = [], []

    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\n========== Fold {fold+1} ==========")

        # 获取训练和测试的索引
        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]

        # ----- 关键：在训练集上计算标准化参数 -----
        # 提取训练集的 EEG 统计特征 (s) 和 外周特征 (p)
        train_s = full_ds.s[train_idx].numpy()  # (N_train, 32, 7)
        train_p = full_ds.p[train_idx].numpy()  # (N_train, 55)
        # 展平后计算 mean/std
        s_mean = train_s.reshape(-1, train_s.shape[-1]).mean(axis=0)  # (7,)
        s_std = train_s.reshape(-1, train_s.shape[-1]).std(axis=0) + 1e-6
        p_mean = train_p.mean(axis=0)
        p_std = train_p.std(axis=0) + 1e-6
        scaler_stats = {'s': (torch.from_numpy(s_mean).float().to(DEVICE),
                              torch.from_numpy(s_std).float().to(DEVICE)),
                        'p': (torch.from_numpy(p_mean).float().to(DEVICE),
                              torch.from_numpy(p_std).float().to(DEVICE))}

        # 重新构建训练集和验证集 Dataset，传入标准化参数
        train_ds = DeapLoaderFixed(NPZ_DIR, MAT_DIR, VIS_DIR, files, scaler_stats=scaler_stats, use_all_segments=True)
        test_ds = DeapLoaderFixed(NPZ_DIR, MAT_DIR, VIS_DIR, files, scaler_stats=scaler_stats, use_all_segments=True)
        # 注意：上面重新加载了整个数据集，但我们需要只取对应索引的子集
        # 简便方法：使用 Subset 包装，但需要确保 __getitem__ 中的标准化使用的是训练集参数（已传入）
        # 由于 scaler_stats 相同，但索引需要对应正确的人。所以我们应该创建 Subset 而不是重新加载整个数据集。
        # 更高效：直接使用 full_ds 但动态应用标准化（需修改 __getitem__ 支持外部参数）。为了避免复杂，我们重新构建两个子数据集：
        # 但重新加载整个数据集开销大，这里采用另一种方式：直接在训练循环中标准化（推荐）
        # 为了代码清晰，我们选择在训练循环内手动标准化，而不修改 Dataset。
        # 以下采用更简洁的方法：在 DataLoader 的 collate_fn 中应用标准化（避免重复加载）
        # 但为了可读性，我们保留原始 full_ds 并在每次迭代时手动标准化（可能会稍慢但安全）

        # 我们改为：使用 Subset(full_ds, idx) 并在迭代时应用标准化参数
        # 定义标准化函数
        def apply_std(s_tensor, p_tensor):
            s_mean_cpu, s_std_cpu = scaler_stats['s'][0].cpu(), scaler_stats['s'][1].cpu()
            p_mean_cpu, p_std_cpu = scaler_stats['p'][0].cpu(), scaler_stats['p'][1].cpu()
            s_norm = (s_tensor - s_mean_cpu) / s_std_cpu
            p_norm = (p_tensor - p_mean_cpu) / p_std_cpu
            return s_norm, p_norm

        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=128, shuffle=True, 
                                  num_workers=0, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=128, shuffle=False,
                                 num_workers=0, pin_memory=True)

        model = MH_DGFNet_Full(hidden_dim=64, dropout=0.3).to(DEVICE)

        EPOCHS = 100
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)  # 降低lr，增加wd
        # 预热+余弦退火
        warmup_epochs = 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
        
        scaler = torch.amp.GradScaler('cuda')

        for ep in range(EPOCHS):
            model.train()
            total_loss = 0
            valid_batches = 0

            for m, s, p, v, lv, la in train_loader:
                m = m.to(DEVICE, non_blocking=True)
                s = s.to(DEVICE, non_blocking=True)
                p = p.to(DEVICE, non_blocking=True)
                v = v.to(DEVICE, non_blocking=True)
                lv = lv.to(DEVICE, non_blocking=True)
                la = la.to(DEVICE, non_blocking=True)

                # 应用标准化 (使用训练集统计量)
                s_mean_t = scaler_stats['s'][0].to(DEVICE)
                s_std_t = scaler_stats['s'][1].to(DEVICE)
                p_mean_t = scaler_stats['p'][0].to(DEVICE)
                p_std_t = scaler_stats['p'][1].to(DEVICE)
                s = (s - s_mean_t) / s_std_t
                p = (p - p_mean_t) / p_std_t

                # 数据增强 (仅训练时)
                if ep < 50:  # 前50个epoch增强，后面fine-tune
                    s = add_gaussian_noise(s, std=0.03)
                    p = add_gaussian_noise(p, std=0.03)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)
                    mask_v = lv != -1
                    mask_a = la != -1
                    loss = 0.0
                    if mask_v.sum() > 0:
                        l_v = F.cross_entropy(ov[mask_v], lv[mask_v], label_smoothing=0.1)
                        loss = loss + 0.5 * l_v
                    if mask_a.sum() > 0:
                        l_a = F.cross_entropy(oa[mask_a], la[mask_a], label_smoothing=0.1)
                        loss = loss + 0.5 * l_a

                if torch.is_tensor(loss) and loss > 0:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    valid_batches += 1

            # 学习率调度 (预热阶段手动线性增加)
            if ep < warmup_epochs:
                lr = 1e-4 * (ep + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()

            if (ep + 1) % 20 == 0 or ep == 0:
                avg_loss = total_loss / max(valid_batches, 1)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{ep+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # ===== 测试 =====
        model.eval()
        vc, ac = 0, 0
        total_v, total_a = 0, 0
        with torch.no_grad():
            for m, s, p, v, lv, la in test_loader:
                m = m.to(DEVICE, non_blocking=True)
                s = s.to(DEVICE, non_blocking=True)
                p = p.to(DEVICE, non_blocking=True)
                v = v.to(DEVICE, non_blocking=True)
                lv = lv.to(DEVICE, non_blocking=True)
                la = la.to(DEVICE, non_blocking=True)

                # 同样标准化 (使用训练集统计量)
                s = (s - scaler_stats['s'][0].to(DEVICE)) / scaler_stats['s'][1].to(DEVICE)
                p = (p - scaler_stats['p'][0].to(DEVICE)) / scaler_stats['p'][1].to(DEVICE)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)

                mask_v = lv != -1
                mask_a = la != -1
                if mask_v.sum() > 0:
                    vc += (ov[mask_v].argmax(1) == lv[mask_v]).sum().item()
                    total_v += mask_v.sum().item()
                if mask_a.sum() > 0:
                    ac += (oa[mask_a].argmax(1) == la[mask_a]).sum().item()
                    total_a += mask_a.sum().item()

        fold_v_acc = vc / total_v if total_v > 0 else 0
        fold_a_acc = ac / total_a if total_a > 0 else 0
        v_results.append(fold_v_acc)
        a_results.append(fold_a_acc)
        print(f"Fold {fold+1} -> Valence Acc: {fold_v_acc:.4f} ({total_v} samples), Arousal Acc: {fold_a_acc:.4f} ({total_a} samples)")

        # 清理
        del model, optimizer, scheduler, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*40)
    print("最终 5-Fold 交叉验证结果 (过滤 3~7 模糊评分，使用全部时间片段):")
    print(f"Valence Accuracy: {np.mean(v_results):.4f} ± {np.std(v_results):.4f}")
    print(f"Arousal Accuracy: {np.mean(a_results):.4f} ± {np.std(a_results):.4f}")
    print("="*40)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_rigorous()