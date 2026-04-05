import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
import scipy.io as sio
import gc  # 引入垃圾回收机制

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

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            adj = F.softmax(attn, dim=-1)
            out = torch.matmul(adj, v)

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.norm(self.proj(out) + x)

# ===============================
# 2. 动态图融合层 (Dynamic Graph Fusion)
# ===============================
class DynamicGraphFusion(nn.Module):
    def __init__(self, dim, k_neighbors=10):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.theta = nn.Linear(dim, dim)
        self.phi = nn.Linear(dim, dim)
        self.gcn_weight = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, N, C = x.shape
        
        # 1. 动态拓扑图生成
        theta_x = self.theta(x)
        phi_x = self.phi(x)     
        
        sim_matrix = torch.matmul(theta_x, phi_x.transpose(1, 2)) / (C ** 0.5)
        
        # 稀疏化
        if self.k_neighbors < N:
            # ✅ 修复 Kernel 崩溃点 1: 改用 bool 类型的 mask 防止精度覆盖问题
            mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            topk_val, topk_idx = torch.topk(sim_matrix, k=self.k_neighbors, dim=-1)
            mask.scatter_(2, topk_idx, True)
            
            # ✅ 修复 Kernel 崩溃点 2: 改用 -1e4 而不是 float16.min，防止 AMP 下 softmax 下溢产生 NaN 引发底层断言失败
            min_val = -1e4 
            sim_matrix = sim_matrix.masked_fill(~mask, min_val)
            
        adj = F.softmax(sim_matrix, dim=-1)
        adj = self.dropout(adj)
        
        # 2. 消息传递与特征聚合
        gcn_x = self.gcn_weight(x)
        out = torch.matmul(adj, gcn_x) 
        
        # 3. 残差连接
        return self.norm(x + self.act(out))

# ===============================
# 3. 主模型 MH-DGFNet
# ===============================
class MH_DGFNet_Full(nn.Module):
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
        self.pos_embed = nn.Parameter(torch.randn(1, 32, hidden_dim))
        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim),
            AdaptiveGNN(hidden_dim, hidden_dim)
        )
        self.region_ops = nn.ModuleDict({
            r: nn.Linear(hidden_dim, hidden_dim) for r in self.regions.keys()
        })

        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 16, 8 * hidden_dim)
        )

        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128), nn.ELU(),
            nn.Linear(128, 8 * hidden_dim)
        )

        self.vis_enc = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(0.3),
            nn.Linear(512, 4 * hidden_dim)
        )

        self.modality_embed = nn.Parameter(torch.randn(4, hidden_dim))
        self.super_node = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.dgf_layers = nn.Sequential(
            DynamicGraphFusion(hidden_dim, k_neighbors=12),
            DynamicGraphFusion(hidden_dim, k_neighbors=12)
        )

        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )
        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, maps, stats, peri, visual):
        h_eeg = self.eeg_proj(stats) + self.pos_embed
        h_eeg = self.eeg_gnn(h_eeg)

        h_neuro = torch.stack([
            F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1)))
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
# 数据加载：优化内存占用，避免 System OOM
# ===============================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files):
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        sub_ids = []

        self.segments_per_trial = 15
        self.keep_last = 8  

        print(f"加载数据：启用时间截断，每个 Trial 仅保留后 {self.keep_last} 个高光情绪片段...")
        print("✅ 启用极端情绪过滤：<3 为 Low(0)，>7 为 High(1)，3~7 之间将被标记为 -1 并在训练中忽略。")

        def norm(x):
            return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        def filter_late_segments(arr):
            reshaped = arr.reshape(40, self.segments_per_trial, *arr.shape[1:])
            return reshaped[:, -self.keep_last:, ...].reshape(-1, *arr.shape[1:])

        def encode_extreme_labels(scores, low_th=3.0, high_th=7.0):
            labels = np.full(scores.shape, -1, dtype=int)
            labels[scores < low_th] = 0
            labels[scores > high_th] = 1
            return labels

        for sub_idx, f in enumerate(files):
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            
            v_labels = encode_extreme_labels(lbl[:, 0])
            a_labels = encode_extreme_labels(lbl[:, 1])
            
            v_list.append(np.repeat(v_labels, self.keep_last))
            a_list.append(np.repeat(a_labels, self.keep_last))

            with np.load(os.path.join(npz_path, f)) as d:
                m_list.append(filter_late_segments(norm(d['eeg_allband_feature_map'])))
                s_list.append(filter_late_segments(norm(d['eeg_en_stat'])))
                p_list.append(filter_late_segments(norm(d['peri_feature'])))

            v_feats = []
            for t in range(1, 41):
                path = os.path.join(visual_npy_path, f"{sid}_trial{t:02d}_features.npy")
                feat = np.load(path) if os.path.exists(path) else np.zeros((15, 512))

                if feat.ndim == 1:
                    feat = np.tile(feat, (15, 1))
                elif feat.shape[0] != 15:
                    feat = np.vstack([np.mean(x, 0) for x in np.array_split(feat, 15)])
                
                v_feats.append(feat[-self.keep_last:, :])

            vis_list.append(np.concatenate(v_feats))
            
            # 动态计算真实样本数量，防止数据越界引发崩溃
            num_samples = len(v_list[-1])
            sub_ids.extend([sub_idx] * num_samples)

        # ✅ 修复 Kernel 崩溃点 3: 分步执行 `concatenate` 并立刻清理原 list，防止 Jupyter 内存雪崩
        print("正在整理张量分配内存...")
        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        del m_list; gc.collect()
        
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        del s_list; gc.collect()
        
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        del p_list; gc.collect()
        
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float()
        del vis_list; gc.collect()
        
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()
        self.sub_l = torch.tensor(sub_ids).long()
        print("数据加载完毕。")

    def __len__(self):
        return len(self.vl)

    def __getitem__(self, i):
        return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i]

# ===============================
# 训练流程
# ===============================
def train_rigorous():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VIS_DIR = r'D:\Users\cyz\dc\see'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {DEVICE}")

    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    full_ds = DeapLoaderRAM(NPZ_DIR, MAT_DIR, VIS_DIR, files)

    subjects = np.arange(len(files))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    v_results, a_results = [], []

    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\n========== Fold {fold+1} ==========")

        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]

        train_loader = DataLoader(Subset(full_ds, train_idx), 128, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), 128, shuffle=False, num_workers=0, pin_memory=True)

        model = MH_DGFNet_Full().to(DEVICE)

        EPOCHS = 100
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        
        scaler = torch.amp.GradScaler('cuda')

        for ep in range(EPOCHS):
            model.train()
            total_loss = 0
            valid_batches = 0 

            for m, s, p, v, lv, la in train_loader:
                m, s, p, v = m.to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), v.to(DEVICE, non_blocking=True)
                lv, la = lv.to(DEVICE, non_blocking=True), la.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)
                    
                    mask_v = lv != -1
                    mask_a = la != -1
                    
                    loss = 0
                    if mask_v.sum() > 0:
                        l_v = F.cross_entropy(ov[mask_v], lv[mask_v], label_smoothing=0.1)
                        loss = loss + 0.6 * l_v
                    
                    if mask_a.sum() > 0:
                        l_a = F.cross_entropy(oa[mask_a], la[mask_a], label_smoothing=0.1)
                        loss = loss + 0.4 * l_a

                if torch.is_tensor(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    valid_batches += 1

            scheduler.step()
            
            if (ep + 1) % 20 == 0 or ep == 0:
                avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
                print(f"Epoch [{ep+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # ===== 测试验证 =====
        model.eval()
        vc, ac = 0, 0
        total_v, total_a = 0, 0 

        with torch.no_grad():
            for m, s, p, v, lv, la in test_loader:
                m, s, p, v = m.to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), v.to(DEVICE, non_blocking=True)
                lv_dev, la_dev = lv.to(DEVICE, non_blocking=True), la.to(DEVICE, non_blocking=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)

                mask_v = lv_dev != -1
                mask_a = la_dev != -1

                if mask_v.sum() > 0:
                    vc += (ov[mask_v].argmax(1) == lv_dev[mask_v]).sum().item()
                    total_v += mask_v.sum().item()
                
                if mask_a.sum() > 0:
                    ac += (oa[mask_a].argmax(1) == la_dev[mask_a]).sum().item()
                    total_a += mask_a.sum().item()

        fold_v_acc = vc / total_v if total_v > 0 else 0
        fold_a_acc = ac / total_a if total_a > 0 else 0
        v_results.append(fold_v_acc)
        a_results.append(fold_a_acc)

        print(f"Fold {fold+1} 结束 -> Valence 准确率: {fold_v_acc:.4f} (基于 {total_v} 个有效样本), Arousal 准确率: {fold_a_acc:.4f} (基于 {total_a} 个有效样本)")

        # ✅ 修复 Kernel 崩溃点 4: 每折训练结束后强制清理显存，防止 CUDA OOM 叠加引发崩溃
        del model, optimizer, scheduler, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*30)
    print("最终 5-Fold 交叉验证结果 (过滤 3~7 模糊评分后)：")
    print(f"Valence Accuracy: {np.mean(v_results):.4f} ± {np.std(v_results):.4f}")
    print(f"Arousal Accuracy: {np.mean(a_results):.4f} ± {np.std(a_results):.4f}")
    print("="*30)

if __name__ == "__main__":
    # ✅ Jupyter 中调用此项偶尔会导致线程递归死锁，但在 main 保护下是安全的
    torch.multiprocessing.freeze_support()
    train_rigorous()