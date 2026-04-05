# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
import scipy.io as sio

# ===============================
# 0. 全局计算加速配置
# ===============================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 针对 PyTorch 2.0+ 推荐的矩阵计算精度设置
torch.set_float32_matmul_precision('high')

# ===============================
# 1. 自适应图神经网络层 (保持不变，特征提取利器)
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
# 2. 主模型 MH-DGFNet（精简融合版）
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

        # ===== EEG =====
        self.eeg_proj = nn.Sequential(nn.Linear(7, hidden_dim), nn.ELU())

        # ✅ EEG位置编码（新增）
        self.pos_embed = nn.Parameter(torch.randn(1, 32, hidden_dim))

        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim),
            AdaptiveGNN(hidden_dim, hidden_dim)
        )

        self.region_ops = nn.ModuleDict({
            r: nn.Linear(hidden_dim, hidden_dim) for r in self.regions.keys()
        })

        # ===== EEG Map =====
        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 8 * hidden_dim)
        )

        # ===== Peri =====
        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128),
            nn.ELU(),
            nn.Linear(128, 8 * hidden_dim)
        )

        # ===== Visual =====
        self.vis_enc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 4 * hidden_dim)
        )

        # ===== ✅ CLS Token（新增）=====
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # ===== ✅ 模态权重 gating（新增）=====
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # ===== Transformer（优化）=====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
            dropout=0.1   # ✅ 降低dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ===== 输出 =====
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, maps, stats, peri, visual):
        # ===== EEG =====
        h_eeg = self.eeg_proj(stats)

        # ✅ 加位置编码
        h_eeg = h_eeg + self.pos_embed

        h_eeg = self.eeg_gnn(h_eeg)

        # 脑区聚合
        h_neuro = torch.stack([
            F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1)))
            for r, idx in self.regions.items()
        ], dim=1)

        # ===== 其他模态 =====
        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)

        # ===== 拼接 =====
        combined = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1)

        # ===== ✅ 模态加权 =====
        weights = self.modality_gate(combined)
        combined = combined * weights

        # ===== ✅ CLS Token =====
        B = combined.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        combined = torch.cat([cls, combined], dim=1)

        # ===== Transformer =====
        out = self.transformer(combined)

        # ✅ CLS输出 + 残差融合（增强稳定性）
        feat = out[:, 0] + combined[:, 1:].mean(dim=1)

        # ===== 输出 =====
        v_logits = self.v_head(feat)
        a_logits = self.a_head(feat)

        return v_logits, a_logits
# ===============================
# 3. 数据加载（情绪后置截断法）
# ===============================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files):
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        sub_ids = []

        # 核心逻辑：15个片段中，只取最后 8 个片段（情绪唤醒最强期）
        self.segments_per_trial = 15
        self.keep_last = 8  

        print(f"加载数据：启用时间截断，每个 Trial 仅保留后 {self.keep_last} 个高光情绪片段...")

        def norm(x):
            return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        def filter_late_segments(arr):
            # 将 (600, ...) 变为 (40, 15, ...) 然后切片取最后几个
            reshaped = arr.reshape(40, self.segments_per_trial, *arr.shape[1:])
            return reshaped[:, -self.keep_last:, ...].reshape(-1, *arr.shape[1:])

        for sub_idx, f in enumerate(files):
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            
            # 标签也只重复 keep_last 次
            v_list.append(np.repeat((lbl[:, 0] > 5).astype(int), self.keep_last))
            a_list.append(np.repeat((lbl[:, 1] > 5).astype(int), self.keep_last))

            with np.load(os.path.join(npz_path, f)) as d:
                # 提取并截取后半段特征
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
                
                # 视觉特征也只取最后 keep_last 个
                v_feats.append(feat[-self.keep_last:, :])

            vis_list.append(np.concatenate(v_feats))
            sub_ids.extend([sub_idx] * (40 * self.keep_last))

        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float()
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()
        self.sub_l = torch.tensor(sub_ids).long()

    def __len__(self):
        return len(self.vl)

    def __getitem__(self, i):
        return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i]

# ===============================
# 4. 训练主流程
# ===============================
def train_rigorous():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VIS_DIR = r'D:\Users\cyz\dc\see'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {DEVICE}")

    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    
    # 保证整个训练过程只加载一次数据
    full_ds = DeapLoaderRAM(NPZ_DIR, MAT_DIR, VIS_DIR, files)

    subjects = np.arange(len(files))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    v_results, a_results = [], []

    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\n========== Fold {fold+1} ==========")

        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]

        # num_workers=0 防止 Windows 多进程死锁
        train_loader = DataLoader(Subset(full_ds, train_idx), 128, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), 128, shuffle=False, num_workers=0, pin_memory=True)

        model = MH_DGFNet_Full().to(DEVICE)

        # 优化器与余弦退火学习率 (Epochs 提升至 100)
        EPOCHS = 100
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        
        # 修复了 Warning 的写法
        scaler = torch.amp.GradScaler('cuda')

        for ep in range(EPOCHS):
            model.train()
            total_loss = 0

            for m, s, p, v, lv, la in train_loader:
                m, s, p, v = m.to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), v.to(DEVICE, non_blocking=True)
                lv, la = lv.to(DEVICE, non_blocking=True), la.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)
                    
                    # 简化损失计算，去除域对抗
                    l_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                    l_a = F.cross_entropy(oa, la, label_smoothing=0.1)
                    loss = 0.7 * l_v + 0.3 * l_a

                scaler.scale(loss).backward()
                # 梯度裁剪，防止 Transformer 梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()

            scheduler.step()
            
            # 每 20 个 Epoch 打印一次训练状态
            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"Epoch [{ep+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # ===== 测试验证 =====
        model.eval()
        vc, ac, total = 0, 0, 0

        with torch.no_grad():
            for m, s, p, v, lv, la in test_loader:
                m, s, p, v = m.to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), v.to(DEVICE, non_blocking=True)
                lv_dev, la_dev = lv.to(DEVICE, non_blocking=True), la.to(DEVICE, non_blocking=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa = model(m, s, p, v)

                vc += (ov.argmax(1) == lv_dev).sum().item()
                ac += (oa.argmax(1) == la_dev).sum().item()
                total += lv.size(0)

        fold_v_acc = vc / total
        fold_a_acc = ac / total
        v_results.append(fold_v_acc)
        a_results.append(fold_a_acc)

        print(f"Fold {fold+1} 结束 -> Valence 准确率: {fold_v_acc:.4f}, Arousal 准确率: {fold_a_acc:.4f}")

    print("\n" + "="*30)
    print("最终 5-Fold 交叉验证结果：")
    print(f"Valence Accuracy: {np.mean(v_results):.4f} ± {np.std(v_results):.4f}")
    print(f"Arousal Accuracy: {np.mean(a_results):.4f} ± {np.std(a_results):.4f}")
    print("="*30)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_rigorous()