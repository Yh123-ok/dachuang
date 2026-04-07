import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import os
import scipy.io as sio

# ==========================================
# 🛠️ 模块 1: 基础组件 - 残差动态图层
# ==========================================
class ResDGFLayer(nn.Module):
    """
    核心组件：动态图卷积 + 残差连接 + LayerNorm
    能够根据特征内容的相似性动态构建邻接矩阵。
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        # log_sigmas 控制高斯核的宽度，设为可学习参数
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x):
        # x shape: [Batch, Nodes, Dim]
        b, n, d = x.size()
        
        # 1. 计算节点间的成对欧氏距离
        dist_sq = torch.cdist(x, x, p=2)**2 # [B, N, N]
        
        # 2. 计算动态邻接矩阵 (RBF Kernel)
        sigmas = torch.exp(self.log_sigmas) # 确保 sigma > 0
        # 广播机制: [B, 1, N, N] / [H, 1, 1] -> [B, H, N, N]
        adjs = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6))
        
        # 3. 多头平均聚合
        adj_final = adjs.mean(dim=1) # [B, N, N]
        
        # 4. 图卷积操作
        out = torch.matmul(adj_final, x) # Aggregation
        out = self.proj(out)             # Update
        
        # 5. 残差连接与归一化 (关键步骤，防止深层网络退化)
        return self.norm(self.activation(out) + x)

# ==========================================
# 🛠️ 模块 2: 深层 GNN 堆叠块 (Deep Stack)
# ==========================================
class DeepGNNBlock(nn.Module):
    """
    堆叠多个 ResDGFLayer，增加感受野，捕捉高阶关系。
    """
    def __init__(self, d_model, layers=3, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            ResDGFLayer(d_model, num_heads) for _ in range(layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# 🧠 模块 3: Deep-Res-MT-DGF-GNN 模型主体
# ==========================================
class Deep_MT_DGF_GNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # --- 分支 A: CNN (处理脑电地形图 5x32x32) ---
        self.cnn_net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), 
            nn.Flatten() # 32*2*2 = 128
        )
        self.cnn_proj = nn.Linear(128, hidden_dim)

        # --- 分支 B: Deep GNN (处理电极统计特征 32x7) ---
        self.gnn_mapping = nn.Linear(7, hidden_dim)
        # 🌟 使用 3 层深度的 GNN 提取 32 个电极间的复杂空间关系
        self.deep_gnn = DeepGNNBlock(d_model=hidden_dim, layers=3, num_heads=4)

        # --- 分支 C: MLP (处理外周生理特征 55维) ---
        self.peri_net = nn.Sequential(
            nn.Linear(55, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ELU()
        )

        # --- 融合模块: 跨模态动态图 ---
        # 将 CNN, GNN, Peri 三个特征向量看作图中的 3 个节点进行融合
        self.fusion_layer = ResDGFLayer(hidden_dim, num_heads=8)
        
        # --- 多任务输出头 (MTL Heads) ---
        # 共享特征 -> 独立预测
        combined_dim = hidden_dim * 3
        self.v_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        
        # --- 自动权重参数 (Uncertainty Weights) ---
        # log_vars = log(sigma^2)，用于平衡多任务 Loss
        self.log_vars = nn.Parameter(torch.zeros(2)) 

    def forward(self, maps, stats, peri):
        # 1. 特征提取
        # CNN Branch
        h_cnn = self.cnn_proj(self.cnn_net(maps)).unsqueeze(1) # [B, 1, H]
        
        # Deep GNN Branch
        gnn_in = self.gnn_mapping(stats) 
        h_gnn_nodes = self.deep_gnn(gnn_in) # [B, 32, H] (经过3层交互)
        h_gnn = h_gnn_nodes.mean(dim=1, keepdim=True) # 全局聚合 [B, 1, H]
        
        # Peri Branch
        h_peri = self.peri_net(peri).unsqueeze(1) # [B, 1, H]

        # 2. 动态融合 (Heterogeneous Graph)
        # 拼接三个模态节点: [B, 3, H]
        combined = torch.cat([h_cnn, h_gnn, h_peri], dim=1) 
        # 学习模态间的注意力
        fused = self.fusion_layer(combined)
        
        # 3. 展平并分类
        flat_feat = fused.view(fused.size(0), -1) # [B, 3*H]
        return self.v_head(flat_feat), self.a_head(flat_feat)

# ==========================================
# 💾 模块 4: 数据加载与标签对齐
# ==========================================
class DeapMultiModalDataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir):
        self.npz_dir = npz_dir
        self.file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.samples_per_trial = 15 # 4s切片
        
        self.v_labels = []
        self.a_labels = []
        
        print(f"正在加载 {len(self.file_list)} 个被试的数据...")
        
        for f_name in self.file_list:
            # 文件名解析: 假设格式为 s01_features.npz 或 s01.npz
            subj_id = f_name[:3] # 取前三个字符，如 's01'
            mat_path = os.path.join(raw_mat_dir, f"{subj_id}.mat")
            
            if not os.path.exists(mat_path):
                print(f"警告: 找不到对应的标签文件 {mat_path}，跳过该文件。")
                continue
                
            # 加载原始 Label: [40, 4] -> Valence(0), Arousal(1)
            raw_labels = sio.loadmat(mat_path)['labels']
            
            # 提取 V 和 A 并二值化 (阈值 5)
            v_binary = (raw_labels[:, 0] > 5).astype(np.int64)
            a_binary = (raw_labels[:, 1] > 5).astype(np.int64)
            
            # 扩展标签: Trial级 -> Sample级 (40 -> 600)
            self.v_labels.append(np.repeat(v_binary, self.samples_per_trial))
            self.a_labels.append(np.repeat(a_binary, self.samples_per_trial))
            
        self.v_labels = np.concatenate(self.v_labels)
        self.a_labels = np.concatenate(self.a_labels)
        
        print(f"数据加载完成。总样本数: {len(self.v_labels)}")

    def __len__(self):
        return len(self.v_labels)

    def __getitem__(self, idx):
        # 计算该样本属于哪个文件
        file_idx = idx // 600 
        inner_idx = idx % 600
        
        file_path = os.path.join(self.npz_dir, self.file_list[file_idx])
        
        # 动态读取以节省内存
        with np.load(file_path) as data:
            # 地形图 [5, 32, 32]
            maps = torch.from_numpy(data['eeg_allband_feature_map'][inner_idx]).float()
            # 统计特征 [32, 7] (需 reshape)
            stats = torch.from_numpy(data['eeg_en_stat'][inner_idx]).view(32, 7).float()
            # 外周特征 [55]
            peri = torch.from_numpy(data['peri_feature'][inner_idx]).float()
            
        return maps, stats, peri, self.v_labels[idx], self.a_labels[idx]

# ==========================================
# 🚀 模块 5: 训练引擎与评估
# ==========================================
def train_deep_mt_dgf():
    # --- 配置区域 (请修改这里) ---
    NPZ_PATH = r'D:\Users\cyz\dc\222'                 # 特征文件夹路径
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'    # 原始 .mat 文件夹路径
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据
    dataset = DeapMultiModalDataset(NPZ_PATH, RAW_PATH)
    # 80% 训练, 20% 验证
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. 初始化模型
    model = Deep_MT_DGF_GNN(hidden_dim=64).to(DEVICE)
    # 注意: 优化器需要同时更新模型参数和自动权重参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    print(f"\n 开始训练 Deep-Res-MT-DGF-GNN (Device: {DEVICE})")
    print(f" GNN 深度: 3层 |  任务: Valence & Arousal 同时优化")
    
    best_avg_f1 = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        for maps, stats, peri, lv, la in train_loader:
            maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
            lv, la = lv.to(DEVICE), la.to(DEVICE)
            
            # 前向传播
            out_v, out_a = model(maps, stats, peri)
            
            # 计算独立 Loss
            loss_v = F.cross_entropy(out_v, lv)
            loss_a = F.cross_entropy(out_a, la)
            
            # --- 自动权重 Loss 计算 (Kendall et al.) ---
            # Loss = L * exp(-log_var) + log_var
            precision_v = torch.exp(-model.log_vars[0])
            precision_a = torch.exp(-model.log_vars[1])
            combined_loss = (loss_v * precision_v + model.log_vars[0]) + \
                            (loss_a * precision_a + model.log_vars[1])
            
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            total_loss += combined_loss.item()
            
        # --- 验证阶段 ---
        model.eval()
        v_preds, a_preds, v_gt, a_gt = [], [], [], []
        
        with torch.no_grad():
            for maps, stats, peri, lv, la in val_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                ov, oa = model(maps, stats, peri)
                
                v_preds.extend(torch.max(ov, 1)[1].cpu().numpy())
                a_preds.extend(torch.max(oa, 1)[1].cpu().numpy())
                v_gt.extend(lv.numpy())
                a_gt.extend(la.numpy())
        
        # 指标计算
        f1_v = f1_score(v_gt, v_preds, average='macro')
        f1_a = f1_score(a_gt, a_preds, average='macro')
        avg_f1 = (f1_v + f1_a) / 2
        
        # 获取当前任务权重 (用于观察模型侧重)
        w_v = torch.exp(-model.log_vars[0]).item()
        w_a = torch.exp(-model.log_vars[1]).item()
        
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val F1 -> V: {f1_v:.4f}, A: {f1_a:.4f} (Avg: {avg_f1:.4f}) | "
              f"Weights -> V: {w_v:.2f}, A: {w_a:.2f}")
        
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            torch.save(model.state_dict(), "best_deep_mt_model.pth")
            
    # --- 最终报告 ---
    print("\n🏆 --- 训练结束，最佳模型性能报告 ---")
    model.load_state_dict(torch.load("best_deep_mt_model.pth"))
    model.eval()
    # 这里省略再次跑一遍验证集的代码，直接使用最后一次结果或重新加载进行详细打印
    print(f"Best Average F1: {best_avg_f1:.4f}")
    # 你可以在这里加回 classification_report 打印详细分类表

if __name__ == "__main__":
    train_deep_mt_dgf()