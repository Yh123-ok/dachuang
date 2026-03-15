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
# 2. 图神经网络基础模块 (GNN & 微状态 DGF)
# ==========================================
class LocalGNNLayer(nn.Module):
    """局部自适应图卷积 (用于捕捉 32 个头皮电极之间的固有物理/神经连通性)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.adj = nn.Parameter(torch.randn(32, 32) * 0.01)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        adj_normalized = F.softmax(self.adj, dim=-1) 
        out = torch.matmul(adj_normalized, x)
        out = self.proj(out)
        return self.norm(self.act(out) + x)

class MicrostateDGFLayer(nn.Module):
    """【修改点】引入微状态‘准稳态’(Quasi-stable)特性的动态图融合网络"""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        # 准稳态门控参数：学习大脑在状态演化时的平滑度
        self.transition_gate = nn.Parameter(torch.tensor([0.5]))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ELU()

    def forward(self, x, prev_adj=None):
        # 1. 计算当前状态的拓扑距离
        dist_sq = torch.cdist(x, x, p=2)**2
        sigmas = torch.exp(self.log_sigmas)
        curr_adj = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6)).mean(dim=1)
        
        # 2. 模拟微状态平滑约束 (如果存在前一层状态，则约束突变)
        if prev_adj is not None:
            gate = torch.sigmoid(self.transition_gate)
            adj_final = gate * curr_adj + (1 - gate) * prev_adj
        else:
            adj_final = curr_adj
        
        out = torch.matmul(adj_final, x)
        out = self.proj(out)
        return self.norm(self.act(out) + x), adj_final

# ==========================================
# 3. 双层图多模态主模型 (融入面部视觉特征版)
# ==========================================
class Microstate_MT_GNN_DGF(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # --- A. 脑电局部 GNN 分支 (5个节点) ---
        self.eeg_proj = nn.Linear(7, hidden_dim)
        self.eeg_gnn = nn.Sequential(
            LocalGNNLayer(hidden_dim, hidden_dim),
            LocalGNNLayer(hidden_dim, hidden_dim)
        )
        self.region_nets = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU()) 
            for r in BRAIN_REGIONS.keys()
        })

        # --- B. 脑微状态空间分支 (8个节点) ---
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 4, 3, padding=1), nn.InstanceNorm2d(4), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.ms_node_gen = nn.Linear(16, 8 * hidden_dim)

        # --- C. 外周生理分支 (8个节点) ---
        self.peri_node_gen = nn.Sequential(
            nn.Linear(55, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, 8 * hidden_dim)
        )

        # --- 【新增】F. 面部视觉分支 (将512维映射为4个图节点) ---
        self.vis_node_gen = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, 4 * hidden_dim)
        )

        # --- D. 全局 DGF 跨模态融合 ---
        # 【修改点】总节点数: 5(脑区) + 8(微状态) + 8(外周) + 4(视觉) = 25节点
        self.modality_emb = nn.Parameter(torch.randn(1, 25, hidden_dim) * 0.02)
        
        self.dgf_1 = MicrostateDGFLayer(hidden_dim, num_heads=8)
        self.dgf_2 = MicrostateDGFLayer(hidden_dim, num_heads=8)
        
        # --- E. 多任务决策头 ---
        self.dropout = nn.Dropout(0.4)
        # 【修改点】输入维度对应 25 个节点
        self.v_head = nn.Sequential(nn.Linear(25 * hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(25 * hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.log_vars = nn.Parameter(torch.zeros(2))

    # 【修改点】forward增加 visual 参数
    def forward(self, maps, stats, peri, visual):
        # 1. GNN 处理 32 通道并聚合为 5 个神经节点
        h_stats = self.eeg_gnn(self.eeg_proj(stats))
        region_nodes = []
        for r, idx in BRAIN_REGIONS.items():
            node_feat = self.region_nets[r](h_stats[:, idx, :].mean(dim=1))
            region_nodes.append(node_feat)
        h_neuro = torch.stack(region_nodes, dim=1) # (Batch, 5, H)

        # 2. 生成微状态、外周、以及视觉的图节点
        h_ms = self.ms_node_gen(self.ms_encoder(maps)).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_node_gen(peri).view(-1, 8, self.hidden_dim)
        
        # 【新增】将视觉特征转化为图节点
        h_vis = self.vis_node_gen(visual).view(-1, 4, self.hidden_dim)

        # 3. 跨模态微状态动态融合 (DGF)
        # 【修改点】拼接上视觉节点 h_vis
        combined_nodes = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1) + self.modality_emb
        
        fused_1, adj_1 = self.dgf_1(combined_nodes)
        fused_nodes, _ = self.dgf_2(fused_1, prev_adj=adj_1) 
        
        # 4. 展平并输出
        flat_feat = self.dropout(fused_nodes.view(fused_nodes.size(0), -1))
        return self.v_head(flat_feat), self.a_head(flat_feat)

# ==========================================
# 4. 高速内存数据加载器 (新增视觉特征读取)
# ==========================================
class DeapLoaderRAM(Dataset):
    def __init__(self, npz_path, mat_path, visual_npy_path, files):
        m_list, s_list, p_list, v_list, a_list, vis_list = [], [], [], [], [], []
        
        for f in files:
            sid = f[:3]
            # 标签
            lbl = sio.loadmat(os.path.join(mat_path, f"{sid}.mat"))['labels']
            v_list.append(np.repeat((lbl[:,0]>5).astype(int), 15))
            a_list.append(np.repeat((lbl[:,1]>5).astype(int), 15))
            
            # EEG及外周数据
            with np.load(os.path.join(npz_path, f)) as d:
                m_list.append(d['eeg_allband_feature_map'])
                s_list.append(d['eeg_en_stat'])
                p_list.append(d['peri_feature'])
                
            # 【新增】读取队友提取的对应受试者的视觉特征 .npy 文件
            # ⚠️ 注意：这里假设视觉特征文件名为 s01_visual.npy，并且样本数量已经与 EEG 对齐 (比如也是 40 trial * 15 = 600个样本)
            # 你需要根据你们实际保存的文件名格式修改下方路径
            vis_file = os.path.join(visual_npy_path, f"{sid}_features.npy")
            if os.path.exists(vis_file):
                vis_feat = np.load(vis_file)
                # 如果视觉特征没有分段(比如只有视频全长的特征)，你需要在这里用代码对齐到15个时间窗
                vis_list.append(vis_feat)
            else:
                # 容错：如果缺视频特征，用全0补全 (600, 512)
                print(f"警告：找不到视觉特征 {vis_file}，使用零矩阵填充。")
                vis_list.append(np.zeros((600, 512))) 
        
        self.m = torch.from_numpy(np.concatenate(m_list)).float()
        self.s = torch.from_numpy(np.concatenate(s_list)).view(-1, 32, 7).float()
        self.p = torch.from_numpy(np.concatenate(p_list)).float()
        # 【新增】视觉特征转 tensor
        self.vis = torch.from_numpy(np.concatenate(vis_list)).float() 
        self.vl = torch.from_numpy(np.concatenate(v_list)).long()
        self.al = torch.from_numpy(np.concatenate(a_list)).long()

    def __len__(self): return len(self.vl)
    # 【修改点】__getitem__ 多返回一个视觉特征
    def __getitem__(self, i): return self.m[i], self.s[i], self.p[i], self.vis[i], self.vl[i], self.al[i]

# ==========================================
# 5. LOSO 验证与训练引擎 (配合新参数)
# ==========================================
def run_loso_evaluation():
    NPZ_DIR = r'D:\Users\cyz\dc\222'
    RAW_DIR = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    # 【新增】视觉特征存放的文件夹路径 (对应你队友代码中的 OUTPUT_DIR)
    VISUAL_DIR = r'D:\大创\result' 
    
    BATCH_SIZE = 256
    EPOCHS = 30
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    loso_v_acc, loso_a_acc = [], []
    
    print(f"开始 LOSO 验证 | 总被试数: {len(all_files)} | Device: {DEVICE}")

    for i, test_f in enumerate(all_files):
        subj = test_f[:3]
        
        # 预加载数据，传入 VISUAL_DIR
        train_ds = DeapLoaderRAM(NPZ_DIR, RAW_DIR, VISUAL_DIR, [f for f in all_files if f != test_f])
        test_ds = DeapLoaderRAM(NPZ_DIR, RAW_DIR, VISUAL_DIR, [test_f])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = Microstate_MT_GNN_DGF().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler('cuda') 

        # 开始训练
        for ep in range(EPOCHS):
            model.train()
            # 【修改点】接收 v (视觉特征)
            for m, s, p, v, lv, la in train_loader:
                m, s, p, v = m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # 【修改点】传入 v
                    ov, oa = model(m, s, p, v)
                    loss_v = F.cross_entropy(ov, lv, label_smoothing=0.1)
                    loss_a = F.cross_entropy(oa, la, label_smoothing=0.1)
                    loss = (loss_v * torch.exp(-model.log_vars[0]) + model.log_vars[0]) + \
                           (loss_a * torch.exp(-model.log_vars[1]) + model.log_vars[1])
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            scheduler.step()

        # 模型评估
        model.eval()
        v_correct, a_correct, total = 0, 0, 0
        with torch.no_grad():
            for m, s, p, v, lv, la in test_loader:
                # 【修改点】传入 v
                ov, oa = model(m.to(DEVICE), s.to(DEVICE), p.to(DEVICE), v.to(DEVICE))
                v_correct += (ov.argmax(1) == lv.to(DEVICE)).sum().item()
                a_correct += (oa.argmax(1) == la.to(DEVICE)).sum().item()
                total += lv.size(0)
        
        v_acc = v_correct / total
        a_acc = a_correct / total
        loso_v_acc.append(v_acc)
        loso_a_acc.append(a_acc)
        
        print(f"[{i+1}/{len(all_files)}] 被试: {subj} | Val Acc: {v_acc:.4f} | Aro Acc: {a_acc:.4f}")

        del model, optimizer, train_ds, test_ds
        torch.cuda.empty_cache() 

    print("\n" + "="*50)
    print(" LOSO 最终评估报告 (微状态: Microstate CNN + Quasi-stable DGF + Visual)")
    print("="*50)
    print(f"Valence 平均准确率: {np.mean(loso_v_acc):.4f}")
    print(f"Arousal 平均准确率: {np.mean(loso_a_acc):.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_loso_evaluation()