import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
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
        k = min(self.k_neighbors, N)

        theta_x = self.theta(x)
        phi_x = self.phi(x)
        sim_matrix = torch.matmul(theta_x, phi_x.transpose(1, 2)) / (C ** 0.5)

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
# 3. 主模型 MH-DGFNet (无视觉模态)
# ===============================
class MH_DGFNet_NoVisual(nn.Module):
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

        # EEG 统计特征投影
        self.eeg_proj = nn.Sequential(nn.Linear(7, hidden_dim), nn.ELU(), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.randn(1, 32, hidden_dim))
        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim, dropout=dropout),
            AdaptiveGNN(hidden_dim, hidden_dim, dropout=dropout)
        )
        self.region_ops = nn.ModuleDict({
            r: nn.Linear(hidden_dim, hidden_dim) for r in self.regions.keys()
        })

        # 时频图编码器 (maps) -> 8 个节点
        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 16, 8 * hidden_dim), nn.Dropout(dropout)
        )

        # 外周生理信号编码器 -> 8 个节点
        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(128, 8 * hidden_dim), nn.Dropout(dropout)
        )

        # 模态嵌入 (3种模态: 脑区聚合, 时频图, 外周)
        self.modality_embed = nn.Parameter(torch.randn(3, hidden_dim))
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

    def forward(self, maps, stats, peri):
        # stats: (B, 32, 7)
        h_eeg = self.eeg_proj(stats) + self.pos_embed
        h_eeg = self.eeg_gnn(h_eeg)

        # 区域聚合 -> 5个节点
        h_neuro = torch.stack([
            F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1)))
            for r, idx in self.regions.items()
        ], dim=1)  # (B, 5, H)

        # 时频图编码 -> 8个节点
        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)      # (B, 8, H)

        # 外周生理信号编码 -> 8个节点
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)  # (B, 8, H)

        # 添加模态嵌入
        h_neuro = h_neuro + self.modality_embed[0]
        h_ms = h_ms + self.modality_embed[1]
        h_peri = h_peri + self.modality_embed[2]

        # 拼接所有节点: 5 + 8 + 8 = 21
        combined_nodes = torch.cat([h_neuro, h_ms, h_peri], dim=1)

        B = combined_nodes.size(0)
        super_n = self.super_node.expand(B, -1, -1)
        graph_nodes = torch.cat([super_n, combined_nodes], dim=1)  # (B, 22, H)

        fused_nodes = self.dgf_layers(graph_nodes)
        graph_repr = fused_nodes[:, 0] + fused_nodes[:, 1:].mean(dim=1)

        v_logits = self.v_head(graph_repr)
        a_logits = self.a_head(graph_repr)
        return v_logits, a_logits

# ===============================
# 4. 数据加载器 (无视觉特征)
# ===============================
class DeapLoaderNoVisual(Dataset):
    def __init__(self, npz_path, mat_path, files, use_all_segments=True):
        """
        只加载 EEG 时频图(maps)、EEG统计特征(stats)、外周生理信号(peri)
        """
        m_list, s_list, p_list, v_list, a_list = [], [], [], [], []
        sub_ids = []
        self.segments_per_trial = 15
        self.keep_last = 15 if use_all_segments else 8

        print(f"加载数据：每个 Trial 使用 {self.keep_last} 个片段 (全部片段={use_all_segments})")
        print("✅ 启用极端情绪过滤：<3 为 Low(0)，>7 为 High(1)，3~7 之间标记为 -1 并在训练中忽略。")

        def filter_segments(arr):
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

            num_samples = len(v_list[-1])
            sub_ids.extend([sub_idx] * num_samples)

        # 合并所有数据
        print("合并张量...")
        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()
        self.sub_l = torch.tensor(sub_ids).long()

        print(f"数据加载完成，总样本数: {len(self.vl)}")

    def __len__(self):
        return len(self.vl)

    def __getitem__(self, i):
        return self.m[i], self.s[i], self.p[i], self.vl[i], self.al[i]

# ===============================
# 5. 数据增强 (高斯噪声)
# ===============================
def add_gaussian_noise(tensor, std=0.05):
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(tensor) * std
        return tensor + noise
    return tensor

# ===============================
# 6. 训练流程 (无视觉)
# ===============================
def train_no_visual():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {DEVICE}")

    # 获取所有被试文件
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found in {NPZ_DIR}")

    # 创建基础数据集（原始数据，未标准化）
    full_ds = DeapLoaderNoVisual(NPZ_DIR, MAT_DIR, files, use_all_segments=True)

    subjects = np.arange(len(files))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    v_results, a_results = [], []

    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\n========== Fold {fold+1} ==========")

        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]

        # ----- 在训练集上计算标准化参数 -----
        train_s = full_ds.s[train_idx].numpy()  # (N_train, 32, 7)
        train_p = full_ds.p[train_idx].numpy()  # (N_train, 55)
        s_mean = train_s.reshape(-1, train_s.shape[-1]).mean(axis=0)  # (7,)
        s_std = train_s.reshape(-1, train_s.shape[-1]).std(axis=0) + 1e-6
        p_mean = train_p.mean(axis=0)
        p_std = train_p.std(axis=0) + 1e-6

        # 转换为 tensor (稍后移动到设备)
        s_mean_t = torch.from_numpy(s_mean).float()
        s_std_t = torch.from_numpy(s_std).float()
        p_mean_t = torch.from_numpy(p_mean).float()
        p_std_t = torch.from_numpy(p_std).float()

        # 定义 collate 中标准化的辅助函数（实际在训练循环中手动做）
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=128, shuffle=True,
                                  num_workers=0, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=128, shuffle=False,
                                 num_workers=0, pin_memory=True)

        model = MH_DGFNet_NoVisual(hidden_dim=64, dropout=0.3).to(DEVICE)

        EPOCHS = 100
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
        warmup_epochs = 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)

        scaler = torch.amp.GradScaler('cuda')

        for ep in range(EPOCHS):
            model.train()
            total_loss = 0
            valid_batches = 0

            for m, s, p, lv, la in train_loader:
                m = m.to(DEVICE, non_blocking=True)
                s = s.to(DEVICE, non_blocking=True)
                p = p.to(DEVICE, non_blocking=True)
                lv = lv.to(DEVICE, non_blocking=True)
                la = la.to(DEVICE, non_blocking=True)

                # 标准化
                s = (s - s_mean_t.to(DEVICE)) / s_std_t.to(DEVICE)
                p = (p - p_mean_t.to(DEVICE)) / p_std_t.to(DEVICE)

                # 数据增强 (前50个epoch)
                if ep < 50:
                    s = add_gaussian_noise(s, std=0.03)
                    p = add_gaussian_noise(p, std=0.03)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p)
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
            for m, s, p, lv, la in test_loader:
                m = m.to(DEVICE, non_blocking=True)
                s = s.to(DEVICE, non_blocking=True)
                p = p.to(DEVICE, non_blocking=True)
                lv = lv.to(DEVICE, non_blocking=True)
                la = la.to(DEVICE, non_blocking=True)

                s = (s - s_mean_t.to(DEVICE)) / s_std_t.to(DEVICE)
                p = (p - p_mean_t.to(DEVICE)) / p_std_t.to(DEVICE)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p)

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
    print("消融实验 (无视觉模态) 5-Fold 交叉验证结果:")
    print(f"Valence Accuracy: {np.mean(v_results):.4f} ± {np.std(v_results):.4f}")
    print(f"Arousal Accuracy: {np.mean(a_results):.4f} ± {np.std(a_results):.4f}")
    print("="*40)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_no_visual()