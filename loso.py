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
class DeapLOSODataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir):
        self.npz_dir = npz_dir
        self.raw_mat_dir = raw_mat_dir
        self.file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.samples_per_subject = 600  # 每个被试固定的样本数 (40 trials * 15 samples)
        
        self.v_labels_list = [] # 按被试存放，方便提取
        self.a_labels_list = []
        
        print(f"📦 正在初始化数据管理器... 共检测到 {len(self.file_list)} 个被试")
        
        for f_name in self.file_list:
            subj_id = f_name[:3] 
            mat_path = os.path.join(raw_mat_dir, f"{subj_id}.mat")
            
            if not os.path.exists(mat_path):
                raise FileNotFoundError(f"未找到标签文件: {mat_path}")
                
            raw_labels = sio.loadmat(mat_path)['labels']
            # 二值化
            v_bin = (raw_labels[:, 0] > 5).astype(np.int64)
            a_bin = (raw_labels[:, 1] > 5).astype(np.int64)
            
            # 扩展并存入列表
            self.v_labels_list.append(np.repeat(v_bin, 15))
            self.a_labels_list.append(np.repeat(a_bin, 15))
            
        # 展平以便支持常规索引调用 __getitem__
        self.all_v = np.concatenate(self.v_labels_list)
        self.all_a = np.concatenate(self.a_labels_list)

    def __len__(self):
        return len(self.all_v)

    def get_train_indices(self, test_subj_idx):
        """核心：计算除了第 test_subj_idx 个被试外的所有索引"""
        all_indices = np.arange(len(self))
        test_start = test_subj_idx * self.samples_per_subject
        test_end = (test_subj_idx + 1) * self.samples_per_subject
        # 剔除测试集索引
        train_mask = np.ones(len(self), dtype=bool)
        train_mask[test_start:test_end] = False
        return all_indices[train_mask].tolist()

    def get_subject_data(self, subj_idx):
        """核心：一次性获取某个被试的全部 Tensor，避免测试时反复 IO"""
        file_path = os.path.join(self.npz_dir, self.file_list[subj_idx])
        with np.load(file_path) as data:
            m = torch.from_numpy(data['eeg_allband_feature_map']).float()
            s = torch.from_numpy(data['eeg_en_stat']).view(-1, 32, 7).float()
            p = torch.from_numpy(data['peri_feature']).float()
        
        v = torch.from_numpy(self.v_labels_list[subj_idx]).long()
        a = torch.from_numpy(self.a_labels_list[subj_idx]).long()
        return m, s, p, v, a

    def __getitem__(self, idx):
        subj_idx = idx // self.samples_per_subject
        inner_idx = idx % self.samples_per_subject
        file_path = os.path.join(self.npz_dir, self.file_list[subj_idx])
        
        with np.load(file_path) as data:
            maps = torch.from_numpy(data['eeg_allband_feature_map'][inner_idx]).float()
            stats = torch.from_numpy(data['eeg_en_stat'][inner_idx]).view(32, 7).float()
            peri = torch.from_numpy(data['peri_feature'][inner_idx]).float()
            
        return maps, stats, peri, self.all_v[idx], self.all_a[idx]
# ==========================================
# 🚀 模块 5: 训练引擎与评估
# ==========================================
def train_deep_mt_dgf_loso():
    # --- 配置区域 ---
    NPZ_PATH = r'D:\Users\cyz\dc\222'
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'
    BATCH_SIZE = 64
    EPOCHS = 10  # LOSO 建议 10-15 Epoch，保持被试间泛化性
    LR = 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据集
    dataset = DeapLOSODataset(NPZ_PATH, RAW_PATH)
    num_subjects = len(dataset.file_list)
    all_subject_f1 = []

    print(f"\n⚡ 启动 LOSO 验证流程 | 被试总数: {num_subjects} | 设备: {DEVICE}")

    # 2. 外部循环：Leave-One-Subject-Out
    for test_subj_idx in range(num_subjects):
        subj_name = dataset.file_list[test_subj_idx][:3]
        print(f"\n>>> [Fold {test_subj_idx+1}/{num_subjects}] 测试被试: {subj_name}")
        
        # 划分索引并创建训练 Loader
        train_idx = dataset.get_train_indices(test_subj_idx)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        
        # 预加载测试被试数据到显存 (加速验证)
        tm, ts, tp, tlv, tla = dataset.get_subject_data(test_subj_idx)
        tm, ts, tp = tm.to(DEVICE), ts.to(DEVICE), tp.to(DEVICE)

        # 初始化模型
        model = Deep_MT_DGF_GNN(hidden_dim=64).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

        # 3. 内部循环：训练
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for maps, stats, peri, lv, la in train_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)
                
                ov, oa = model(maps, stats, peri)
                
                # 多任务 Loss 计算
                loss_v = F.cross_entropy(ov, lv)
                loss_a = F.cross_entropy(oa, la)
                
                # 自动权重更新逻辑
                combined_loss = (loss_v * torch.exp(-model.log_vars[0]) + model.log_vars[0]) + \
                                (loss_a * torch.exp(-model.log_vars[1]) + model.log_vars[1])
                
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

        # 4. 评估阶段 (针对当前测试被试)
        model.eval()
        with torch.no_grad():
            # 预测并提取融合权重
            ov, oa, weights = model(tm, ts, tp, return_weights=True)
            
            # 计算 F1 指标
            pred_v = ov.argmax(dim=1).cpu().numpy()
            pred_a = oa.argmax(dim=1).cpu().numpy()
            f1_v = f1_score(tlv.numpy(), pred_v, average='macro')
            f1_a = f1_score(tla.numpy(), pred_a, average='macro')
            
            # 跨模态重要性分析
            # weights [Batch, 3, 3] -> 均值 -> 归一化
            imp = weights.mean(dim=0).sum(dim=0).cpu().numpy()
            imp /= imp.sum()
            
            print(f"   Done. Valence F1: {f1_v:.4f} | Arousal F1: {f1_a:.4f}")
            print(f"   🧠 模态贡献度: EEG-CNN: {imp[0]:.1%} | EEG-GNN: {imp[1]:.1%} | Peri: {imp[2]:.1%}")
            
            all_subject_f1.append((f1_v + f1_a) / 2)
            
        # 释放资源防止 OOM
        del model, optimizer
        torch.cuda.empty_cache()

    # 5. 打印全局实验结论
    print("\n" + "=".center(40, "="))
    print(f"🏆 LOSO 实验圆满结束")
    print(f"平均总体 F1: {np.mean(all_subject_f1):.4f}")
    print("=".center(40, "="))
if __name__ == "__main__":
    train_deep_mt_dgf_loso()