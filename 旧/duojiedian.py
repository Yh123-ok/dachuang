import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import scipy.io as sio

# ==========================================
#  模块 1: 基础组件 - 残差动态图层 (保持不变)
# ==========================================
class ResDGFLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.log_sigmas = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x):
        b, n, d = x.size()
        dist_sq = torch.cdist(x, x, p=2)**2 
        sigmas = torch.exp(self.log_sigmas) 
        adjs = torch.exp(-dist_sq.unsqueeze(1) / (2 * sigmas.unsqueeze(0)**2 + 1e-6))
        adj_final = adjs.mean(dim=1) 
        out = torch.matmul(adj_final, x) 
        out = self.proj(out)            
        return self.norm(self.activation(out) + x)

# ==========================================
#  模块 2: 深层 GNN 堆叠块 (保持不变)
# ==========================================
class DeepGNNBlock(nn.Module):
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
#  模块 3: Deep-Res-MT-DGF-GNN 模型主体 (重点修改)
# ==========================================
class Deep_MT_DGF_GNN(nn.Module):
    def __init__(self, hidden_dim=64, num_cnn_nodes=8, num_gnn_nodes=8, num_peri_nodes=8):
        """
        新增参数: num_cnn_nodes, num_gnn_nodes, num_peri_nodes 控制各模态生成的节点数量
        总节点数 = num_cnn_nodes + num_gnn_nodes + num_peri_nodes
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cnn_nodes = num_cnn_nodes
        self.num_gnn_nodes = num_gnn_nodes
        self.num_peri_nodes = num_peri_nodes
        
        # 总融合节点数 (本例中默认为 8+8+8 = 24个)
        self.total_nodes = self.num_cnn_nodes + self.num_gnn_nodes + self.num_peri_nodes
        
        # --- 分支 A: CNN ---
        self.cnn_net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), 
            nn.Flatten() # 输出 128 维
        )
        # 映射为 num_cnn_nodes 个 hidden_dim 节点
        self.cnn_proj = nn.Linear(128, self.num_cnn_nodes * hidden_dim)

        # --- 分支 B: Deep GNN ---
        self.gnn_mapping = nn.Linear(7, hidden_dim)
        self.deep_gnn = DeepGNNBlock(d_model=hidden_dim, layers=3, num_heads=4)
        # 用一个线性层将原始的 32 个通道(节点) 降维或升维到 num_gnn_nodes 个代表性节点
        self.gnn_node_pool = nn.Linear(32, self.num_gnn_nodes)

        # --- 分支 C: MLP (Peri) ---
        self.peri_net = nn.Sequential(
            nn.Linear(55, 128), 
            nn.LayerNorm(128), 
            nn.ELU(),
            # 映射为 num_peri_nodes 个 hidden_dim 节点
            nn.Linear(128, self.num_peri_nodes * hidden_dim)
        )

        # --- 融合模块 ---
        self.fusion_layer = ResDGFLayer(hidden_dim, num_heads=8)
        
        # --- 多任务输出头 ---
        # 展平后的总维度 = 总节点数 * hidden_dim
        combined_dim = self.total_nodes * hidden_dim
        
        self.v_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        
        # --- 自动权重参数 ---
        self.log_vars = nn.Parameter(torch.zeros(2)) 

    def forward(self, maps, stats, peri):
        # 1. CNN 提取特征并转化为多节点
        # shape: (batch, num_cnn_nodes * hidden_dim) -> (batch, num_cnn_nodes, hidden_dim)
        cnn_feat = self.cnn_proj(self.cnn_net(maps))
        h_cnn = cnn_feat.view(-1, self.num_cnn_nodes, self.hidden_dim)
        
        # 2. GNN 提取特征并池化为指定数量的节点
        gnn_in = self.gnn_mapping(stats) 
        h_gnn_nodes = self.deep_gnn(gnn_in) # shape: (batch, 32, hidden_dim)
        # 通过转置在序列维度(32)上做线性映射，然后再转置回来
        # (batch, hidden_dim, 32) -> (batch, hidden_dim, num_gnn_nodes) -> (batch, num_gnn_nodes, hidden_dim)
        h_gnn = self.gnn_node_pool(h_gnn_nodes.transpose(1, 2)).transpose(1, 2)
        
        # 3. 外周生理特征转化为多节点
        # shape: (batch, num_peri_nodes * hidden_dim) -> (batch, num_peri_nodes, hidden_dim)
        peri_feat = self.peri_net(peri)
        h_peri = peri_feat.view(-1, self.num_peri_nodes, self.hidden_dim)

        # 4. 拼接所有模态的节点
        # combined shape: (batch, 24, hidden_dim)
        combined = torch.cat([h_cnn, h_gnn, h_peri], dim=1) 
        
        # 5. 跨模态多节点图级融合
        fused = self.fusion_layer(combined)
        
        # 6. 展平并分类
        flat_feat = fused.view(fused.size(0), -1) 
        return self.v_head(flat_feat), self.a_head(flat_feat)

# ==========================================
#  模块 4: 数据加载 (保持不变)
# ==========================================
class DeapMultiModalDataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir, filenames=None):
        self.npz_dir = npz_dir
        if filenames is not None:
            self.file_list = filenames
        else:
            self.file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
            
        self.samples_per_trial = 15 
        self.v_labels = []
        self.a_labels = []
        
        for f_name in self.file_list:
            subj_id = f_name[:3] 
            mat_path = os.path.join(raw_mat_dir, f"{subj_id}.mat")
            
            if not os.path.exists(mat_path):
                print(f"警告: 找不到对应的标签文件 {mat_path}，跳过该文件。")
                continue
                
            raw_labels = sio.loadmat(mat_path)['labels']
            v_binary = (raw_labels[:, 0] > 5).astype(np.int64)
            a_binary = (raw_labels[:, 1] > 5).astype(np.int64)
            
            self.v_labels.append(np.repeat(v_binary, self.samples_per_trial))
            self.a_labels.append(np.repeat(a_binary, self.samples_per_trial))
            
        if len(self.v_labels) > 0:
            self.v_labels = np.concatenate(self.v_labels)
            self.a_labels = np.concatenate(self.a_labels)
        else:
            self.v_labels = np.array([])
            self.a_labels = np.array([])

    def __len__(self):
        return len(self.v_labels)

    def __getitem__(self, idx):
        file_idx = idx // 600 
        inner_idx = idx % 600
        file_path = os.path.join(self.npz_dir, self.file_list[file_idx])
        
        with np.load(file_path) as data:
            maps = torch.from_numpy(data['eeg_allband_feature_map'][inner_idx]).float()
            stats = torch.from_numpy(data['eeg_en_stat'][inner_idx]).view(32, 7).float()
            peri = torch.from_numpy(data['peri_feature'][inner_idx]).float()
            
        return maps, stats, peri, self.v_labels[idx], self.a_labels[idx]

# ==========================================
#  模块 5: 训练引擎 (LOSO 流程)
# ==========================================
def train_deep_mt_dgf_loso():
    # --- 配置区域 ---
    NPZ_PATH = r'D:\Users\cyz\dc\222'                 
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'    
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = sorted([f for f in os.listdir(NPZ_PATH) if f.endswith('.npz')])
    num_subjects = len(all_files)
    
    print(f"\n 开始 LOSO (Leave-One-Subject-Out) 验证")
    print(f"总被试数: {num_subjects} | Device: {DEVICE}")
    print(f"模型: Deep-Res-MT-DGF-GNN (多模态多节点融合) | Epochs per fold: {EPOCHS}")
    
    loso_results = {
        'subject': [], 'v_acc': [], 'v_f1': [], 'a_acc': [], 'a_f1': []
    }
    
    for i, test_file in enumerate(all_files):
        subj_name = test_file.split('.')[0]
        print(f"\n[{i+1}/{num_subjects}] 正在测试被试: {subj_name} ...")
        
        train_files = [f for f in all_files if f != test_file]
        test_files = [test_file]
        
        train_dataset = DeapMultiModalDataset(NPZ_PATH, RAW_PATH, filenames=train_files)
        test_dataset = DeapMultiModalDataset(NPZ_PATH, RAW_PATH, filenames=test_files)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # -----------------------------------------------------
        # 在这里实例化模型，按需配置节点数。当前配置: 8+8+8 = 24 个节点
        # -----------------------------------------------------
        model = Deep_MT_DGF_GNN(
            hidden_dim=64, 
            num_cnn_nodes=8, 
            num_gnn_nodes=8, 
            num_peri_nodes=8
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for maps, stats, peri, lv, la in train_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)
                
                out_v, out_a = model(maps, stats, peri)
                
                loss_v = F.cross_entropy(out_v, lv)
                loss_a = F.cross_entropy(out_a, la)
                
                precision_v = torch.exp(-model.log_vars[0])
                precision_a = torch.exp(-model.log_vars[1])
                combined_loss = (loss_v * precision_v + model.log_vars[0]) + \
                                (loss_a * precision_a + model.log_vars[1])
                
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
        
        model.eval()
        v_preds, a_preds, v_gt, a_gt = [], [], [], []
        
        with torch.no_grad():
            for maps, stats, peri, lv, la in test_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                ov, oa = model(maps, stats, peri)
                
                v_preds.extend(torch.max(ov, 1)[1].cpu().numpy())
                a_preds.extend(torch.max(oa, 1)[1].cpu().numpy())
                v_gt.extend(lv.numpy())
                a_gt.extend(la.numpy())
        
        curr_v_acc = accuracy_score(v_gt, v_preds)
        curr_a_acc = accuracy_score(a_gt, a_preds)
        curr_v_f1 = f1_score(v_gt, v_preds, average='macro')
        curr_a_f1 = f1_score(a_gt, a_preds, average='macro')
        
        print(f"   -> {subj_name} 结果: Val Acc={curr_v_acc:.4f}, Aro Acc={curr_a_acc:.4f}, Val F1={curr_v_f1:.4f}, Aro F1={curr_a_f1:.4f}")
        
        loso_results['subject'].append(subj_name)
        loso_results['v_acc'].append(curr_v_acc)
        loso_results['v_f1'].append(curr_v_f1)
        loso_results['a_acc'].append(curr_a_acc)
        loso_results['a_f1'].append(curr_a_f1)

    print("\n" + "="*50)
    print(" LOSO 最终评估报告 (多模态多节点 Deep-Res-MT-DGF-GNN)")
    print("="*50)
    
    avg_v_acc = np.mean(loso_results['v_acc'])
    avg_a_acc = np.mean(loso_results['a_acc'])
    avg_v_f1 = np.mean(loso_results['v_f1'])
    avg_a_f1 = np.mean(loso_results['a_f1'])
    
    print(f"Valence ➡️  Avg Acc: {avg_v_acc:.4f} | Avg F1: {avg_v_f1:.4f}")
    print(f"Arousal ➡️  Avg Acc: {avg_a_acc:.4f} | Avg F1: {avg_a_f1:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    train_deep_mt_dgf_loso()