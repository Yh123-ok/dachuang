import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Function
from sklearn.model_selection import KFold
import numpy as np
import os
import scipy.io as sio

# ===============================
# 0. 全局计算加速配置
# ===============================
# 启用 cuDNN 自动优化卷积算法
torch.backends.cudnn.benchmark = True
# 允许 TF32 加速矩阵计算（适用于 Ampere GPU）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===============================
# 1. 梯度反转层（用于域对抗）
# ===============================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 前向传播不改变输入，仅记录系数 alpha
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时将梯度乘以 -alpha，实现对抗学习
        return grad_output.neg() * ctx.alpha, None

# ===============================
# 2. 自适应图神经网络层
# ===============================
class AdaptiveGNN(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        # 多头注意力参数定义
        self.heads = heads
        self.head_dim = out_dim // heads

        # QKV 线性映射
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

        # 输出映射与归一化
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, N, C]（batch, 节点数, 特征维度）
        B, N, C = x.shape

        # 构建多头 Q, K, V
        q = self.q(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        # 使用高效注意力（优先使用 PyTorch 内置优化）
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # 手动实现注意力机制
            attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            adj = F.softmax(attn, dim=-1)
            out = torch.matmul(adj, v)

        # 拼接多头输出
        out = out.transpose(1, 2).reshape(B, N, -1)

        # 残差连接 + 归一化
        return self.norm(self.proj(out) + x)

# ===============================
# 3. 主模型 MH-DGFNet（多模态融合）
# ===============================
class MH_DGFNet_Full(nn.Module):
    def __init__(self, hidden_dim=64, num_subjects=32):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 脑区划分（电极分组）
        self.regions = {
            'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],
            'Parietal': [6, 7, 8, 20, 21, 22],
            'Temporal': [9, 10, 24, 25],
            'Occipital': [28, 29, 30],
            'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]
        }

        # ===== EEG 分支 =====
        self.eeg_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ELU()
        )

        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim),
            AdaptiveGNN(hidden_dim, hidden_dim)
        )

        # 各脑区线性映射
        self.region_ops = nn.ModuleDict({
            r: nn.Linear(hidden_dim, hidden_dim)
            for r in self.regions.keys()
        })

        # ===== 多模态编码器 =====
        # EEG 频带图编码
        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 8 * hidden_dim)
        )

        # 外周生理信号编码
        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128),
            nn.ELU(),
            nn.Linear(128, 8 * hidden_dim)
        )

        # 视觉特征编码
        self.vis_enc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 4 * hidden_dim)
        )

        # ===== Transformer 融合模块 =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ===== 多任务输出头 =====
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        # 域分类器（用于跨被试对齐）
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_subjects)
        )

        # 不确定性权重（多任务 loss 自动调节）
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, maps, stats, peri, visual, alpha=1.0):
        # ===== EEG 图建模 =====
        h_eeg = self.eeg_gnn(self.eeg_proj(stats))

        # 按脑区聚合
        h_neuro = torch.stack([
            F.elu(self.region_ops[r](h_eeg[:, idx, :].mean(1)))
            for r, idx in self.regions.items()
        ], dim=1)

        # ===== 多模态特征对齐 =====
        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)

        # ===== Transformer 融合 =====
        combined = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1)
        feat = self.transformer(combined).mean(dim=1)

        # ===== 情感分类 =====
        v_logits = self.v_head(feat)
        a_logits = self.a_head(feat)

        # ===== 域对抗分支 =====
        feat_rev = GradientReversalLayer.apply(feat, alpha)
        d_logits = self.domain_head(feat_rev)

        return v_logits, a_logits, d_logits

# ===============================
# 4. 数据加载（内存 + 标准化）
# ===============================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files):

        # 初始化缓存列表
        m_list, s_list, p_list = [], [], []
        v_list, a_list, vis_list = [], [], []
        sub_ids = []

        print("加载数据并进行 Z-score 标准化...")

        for sub_idx, f in enumerate(files):
            sid = f[:3]

            # 标签处理（valence/arousal 二分类）
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            v_list.append(np.repeat((lbl[:, 0] > 5).astype(int), 15))
            a_list.append(np.repeat((lbl[:, 1] > 5).astype(int), 15))

            # Z-score 标准化函数
            def norm(x):
                return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

            # EEG/外周特征加载
            with np.load(os.path.join(npz_path, f)) as d:
                m_list.append(norm(d['eeg_allband_feature_map']))
                s_list.append(norm(d['eeg_en_stat']))
                p_list.append(norm(d['peri_feature']))

            # 视觉特征处理（统一为 15 帧）
            v_feats = []
            for t in range(1, 41):
                path = os.path.join(visual_npy_path, f"{sid}_trial{t:02d}_features.npy")
                feat = np.load(path) if os.path.exists(path) else np.zeros((15, 512))

                if feat.ndim == 1:
                    feat = np.tile(feat, (15, 1))
                elif feat.shape[0] != 15:
                    feat = np.vstack([np.mean(x, 0) for x in np.array_split(feat, 15)])

                v_feats.append(feat)

            vis_list.append(np.concatenate(v_feats))
            sub_ids.extend([sub_idx] * 600)

        # 转为 Tensor
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
        return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i], self.sub_l[i]

# ===============================
# 5. 训练主流程（5-Fold）
# ===============================
def train_rigorous():

    # 数据路径
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    MAT_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    VIS_DIR = r'D:\Users\cyz\dc\see'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    full_ds = DeapLoaderRAM(NPZ_DIR, MAT_DIR, VIS_DIR, files)

    subjects = np.arange(len(files))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    v_results, a_results = [], []

    # ===== 5 折交叉验证 =====
    for fold, (train_subs, test_subs) in enumerate(kf.split(subjects)):
        print(f"\nFold {fold+1}")

        # 构建索引（按被试划分）
        train_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[train_subs]))[0]
        test_idx = np.where(np.isin(full_ds.sub_l.numpy(), subjects[test_subs]))[0]

        train_loader = DataLoader(Subset(full_ds, train_idx), 128, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), 128, shuffle=False, num_workers=4, pin_memory=True)

        # 初始化模型
        model = MH_DGFNet_Full(num_subjects=len(subjects)).to(DEVICE)

        # 编译模型（Linux/Mac 推荐）
        if hasattr(torch, 'compile') and os.name != 'nt':
            model = torch.compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
        scaler = torch.cuda.amp.GradScaler()

        # ===== 训练 =====
        for ep in range(30):
            model.train()

            alpha = 2. / (1. + np.exp(-10 * (ep / 30))) - 1

            for m, s, p, v, lv, la, ld in train_loader:

                # 数据传输（非阻塞）
                m, s, p, v = m.to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), v.to(DEVICE, non_blocking=True)
                lv, la, ld = lv.to(DEVICE, non_blocking=True), la.to(DEVICE, non_blocking=True), ld.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # 自动混合精度
                with torch.autocast(device_type='cuda'):
                    ov, oa, od = model(m, s, p, v, alpha=alpha)

                    l_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                    l_a = F.cross_entropy(oa, la, label_smoothing=0.1)
                    l_d = F.cross_entropy(od, ld)

                    # 多任务不确定性加权
                    loss = (torch.exp(-model.log_vars[0]) * l_v + model.log_vars[0]) + \
                           (torch.exp(-model.log_vars[1]) * l_a + model.log_vars[1]) + \
                           (torch.exp(-model.log_vars[2]) * l_d + model.log_vars[2])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # ===== 测试 =====
        model.eval()
        vc, ac, total = 0, 0, 0

        with torch.no_grad():
            for m, s, p, v, lv, la, _ in test_loader:
                m, s, p, v = m.to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), v.to(DEVICE, non_blocking=True)
                lv_dev, la_dev = lv.to(DEVICE, non_blocking=True), la.to(DEVICE, non_blocking=True)

                with torch.autocast(device_type='cuda'):
                    ov, oa, _ = model(m, s, p, v)

                vc += (ov.argmax(1) == lv_dev).sum().item()
                ac += (oa.argmax(1) == la_dev).sum().item()
                total += lv.size(0)

        v_results.append(vc / total)
        a_results.append(ac / total)

        print(f"Valence: {vc/total:.4f}, Arousal: {ac/total:.4f}")

    # ===== 输出最终结果 =====
    print("\n最终结果：")
    print(f"Valence: {np.mean(v_results):.4f} ± {np.std(v_results):.4f}")
    print(f"Arousal: {np.mean(a_results):.4f} ± {np.std(a_results):.4f}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_rigorous()