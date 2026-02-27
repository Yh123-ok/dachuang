import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import scipy.io as sio

# 开启 CUDNN 加速
torch.backends.cudnn.benchmark = True

# ==========================================
# 1. 脑神经解剖学定义 (DEAP 32通道 10-20系统)
# ==========================================
BRAIN_REGIONS = {
    'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],  # 额叶：高级认知与情感
    'Parietal': [6, 7, 8, 20, 21, 22],              # 顶叶：躯体感觉
    'Temporal': [9, 10, 24, 25],                    # 颞叶：听觉与情绪加工
    'Occipital': [28, 29, 30],                      # 枕叶：视觉
    'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]  # 中央区
}

# ==========================================
# 2. 图神经网络基础模块 (GNN & DGF)
# ==========================================
class LocalGNNLayer(nn.Module):
    """局部自适应图卷积 (用于捕捉 32 个头皮电极之间的固有物理/神经连通性)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 可学习的 32x32 邻接矩阵
        self.adj = nn.Parameter(torch.randn(32, 32) * 0.01)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        # x: (Batch, 32, in_dim)
        adj_normalized = F.softmax(self.adj, dim=-1) 
        out = torch.matmul(adj_normalized, x)
        out = self.proj(out)
        return self.norm(self.act(out) + x)

class DGFLayer(nn.Module):
    """动态图融合网络 (根据模态节点的特征相似度实时动态建图)"""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ELU()

    def forward(self, x):
        # x: (Batch, Nodes, d_model)
        dist_sq = torch.cdist(x, x, p=2)**2
        sigmas = torch.exp(self.log_sigmas)
        
        adjs = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6))
        adj_final = adjs.mean(dim=1) 
        
        out = torch.matmul(adj_final, x)
        out = self.proj(out)
        return self.norm(self.act(out) + x)

# ==========================================
# 3. 双层图多模态主模型 (Dual-Graph Architecture)
# ==========================================
class Deep_MT_GNN_DGF(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # --- A. 脑电局部 GNN 分支 ---
        self.eeg_proj = nn.Linear(7, hidden_dim)
        self.eeg_gnn = nn.Sequential(
            LocalGNNLayer(hidden_dim, hidden_dim),
            LocalGNNLayer(hidden_dim, hidden_dim)
        )
        self.region_nets = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU()) 
            for r in BRAIN_REGIONS.keys()
        })

        # --- B. 空间 CNN 分支 (对抗个体差异) ---
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.InstanceNorm2d(16), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.cnn_node_gen = nn.Linear(128, 8 * hidden_dim)

        # --- C. 外周生理分支 ---
        self.peri_node_gen = nn.Sequential(
            nn.Linear(55, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, 8 * hidden_dim)
        )

        # --- D. 全局 DGF 跨模态融合 (21节点) ---
        self.modality_emb = nn.Parameter(torch.randn(1, 21, hidden_dim) * 0.02)
        self.dgf_fusion = nn.Sequential(
            DGFLayer(hidden_dim, num_heads=8),
            DGFLayer(hidden_dim, num_heads=8)
        )
        
        # --- E. 多任务决策头 ---
        self.dropout = nn.Dropout(0.4)
        self.v_head = nn.Sequential(nn.Linear(21 * hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(21 * hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.log_vars = nn.Parameter(torch.zeros(2)) # 自动平衡 V 和 A 的 Loss

    def forward(self, maps, stats, peri):
        # 1. GNN 处理 32 通道，并按脑区聚合为 5 个神经节点
        h_stats = self.eeg_gnn(self.eeg_proj(stats))
        region_nodes = []
        for r, idx in BRAIN_REGIONS.items():
            node_feat = self.region_nets[r](h_stats[:, idx, :].mean(dim=1))
            region_nodes.append(node_feat)
        h_neuro = torch.stack(region_nodes, dim=1) # (Batch, 5, H)

        # 2. 生成其他模态节点
        h_cnn = self.cnn_node_gen(self.cnn_extractor(maps)).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_node_gen(peri).view(-1, 8, self.hidden_dim)

        # 3. DGF 跨模态全连接图融合
        combined_nodes = torch.cat([h_neuro, h_cnn, h_peri], dim=1) + self.modality_emb
        fused_nodes = self.dgf_fusion(combined_nodes)
        
        # 4. 展平并输出
        flat_feat = self.dropout(fused_nodes.view(fused_nodes.size(0), -1))
        return self.v_head(flat_feat), self.a_head(flat_feat)

# ==========================================
# 4. 高速内存数据加载器
# ==========================================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, files):
        m_list, s_list, p_list, v_list, a_list = [], [], [], [], []
        
        for f in files:
            sid = f[:3]
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            v_list.append(np.repeat((lbl[:,0]>5).astype(int), 15))
            a_list.append(np.repeat((lbl[:,1]>5).astype(int), 15))
            
            with np.load(os.path.join(npz_path, f)) as d:
                m_list.append(d['eeg_allband_feature_map'])
                s_list.append(d['eeg_en_stat'])
                p_list.append(d['peri_feature'])
        
        # 一次性装载进内存
        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()

    def __len__(self): return len(self.vl)
    def __getitem__(self, i): return self.m[i], self.s[i], self.p[i], self.vl[i], self.al[i]

# ==========================================
# 5. LOSO 验证与训练引擎
# ==========================================
def run_loso_evaluation():
    # ==== 你的本地路径配置 ====
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    RAW_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    
    # ==== 训练超参数 ====
    BATCH_SIZE = 256
    EPOCHS = 30
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    loso_v_acc, loso_a_acc = [], []
    
    print(f"开始 LOSO 验证 | 总被试数: {len(all_files)} | Device: {DEVICE}")

    for i, test_f in enumerate(all_files):
        subj = test_f[:3]
        
        # 预加载数据
        train_ds = DeapLoaderRAM(NPZ_DIR, RAW_DIR, [f for f in all_files if f != test_f])
        test_ds = DeapLoaderRAM(NPZ_DIR, RAW_DIR, [test_f])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型与优化器
        model = Deep_MT_GNN_DGF().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.cuda.amp.GradScaler() # AMP 混合精度加速

        # 开始训练
        for ep in range(EPOCHS):
            model.train()
            for m, s, p, lv, la in train_loader:
                m, s, p, lv, la = m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), lv.to(DEVICE), la.to(DEVICE)
                optimizer.zero_grad()
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    ov, oa = model(m, s, p)
                    # 加入 Label Smoothing 缓解过拟合
                    loss_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                    loss_a = F.cross_entropy(oa, la, label_smoothing=0.1)
                    # 动态多任务权重
                    loss = (loss_v * torch.exp(-model.log_vars[0]) + model.log_vars[0]) + \
                           (loss_a * torch.exp(-model.log_vars[1]) + model.log_vars[1])
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            scheduler.step()

        # 模型评估
        model.eval()
        v_correct, a_correct, total = 0, 0, 0
        with torch.no_grad():
            for m, s, p, lv, la in test_loader:
                ov, oa = model(m.to(DEVICE), s.to(DEVICE), p.to(DEVICE))
                v_correct += (ov.argmax(1) == lv.to(DEVICE)).sum().item()
                a_correct += (oa.argmax(1) == la.to(DEVICE)).sum().item()
                total += lv.size(0)
        
        v_acc = v_correct / total
        a_acc = a_correct / total
        loso_v_acc.append(v_acc)
        loso_a_acc.append(a_acc)
        
        print(f"[{i+1}/{len(all_files)}] 被试: {subj} | Val Acc: {v_acc:.4f} | Aro Acc: {a_acc:.4f}")

    print("\n" + "="*50)
    print(" LOSO 最终评估报告 (双层图: Local GNN + Global DGF)")
    print("="*50)
    print(f"Valence 平均准确率: {np.mean(loso_v_acc):.4f}")
    print(f"Arousal 平均准确率: {np.mean(loso_a_acc):.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_loso_evaluation()