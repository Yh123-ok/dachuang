import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import scipy.io as sio

# 开启 CuDNN Benchmark 提速卷积运算
torch.backends.cudnn.benchmark = True

# ==========================================
#  模块 1 & 2: 基础图组件 (保持前一版的原样)
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

class DeepGNNBlock(nn.Module):
    def __init__(self, d_model, layers=3, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([ResDGFLayer(d_model, num_heads) for _ in range(layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
#  模块 3: 模型主体 (新增: 模态嵌入)
# ==========================================
class Deep_MT_DGF_GNN(nn.Module):
    def __init__(self, hidden_dim=64, num_cnn_nodes=8, num_gnn_nodes=8, num_peri_nodes=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cnn_nodes = num_cnn_nodes
        self.num_gnn_nodes = num_gnn_nodes
        self.num_peri_nodes = num_peri_nodes
        self.total_nodes = num_cnn_nodes + num_gnn_nodes + num_peri_nodes
        
        # --- 新增: 模态位置嵌入 (Modality Embeddings) ---
        # 维度: (1, 总节点数, hidden_dim)，这会被加到融合前的节点特征上
        self.modality_emb = nn.Parameter(torch.randn(1, self.total_nodes, hidden_dim) * 0.02)
        
        self.cnn_net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.cnn_proj = nn.Linear(128, self.num_cnn_nodes * hidden_dim)

        self.gnn_mapping = nn.Linear(7, hidden_dim)
        self.deep_gnn = DeepGNNBlock(d_model=hidden_dim, layers=3, num_heads=4)
        self.gnn_node_pool = nn.Linear(32, self.num_gnn_nodes)

        self.peri_net = nn.Sequential(
            nn.Linear(55, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, self.num_peri_nodes * hidden_dim)
        )

        self.fusion_layer = ResDGFLayer(hidden_dim, num_heads=8)
        
        combined_dim = self.total_nodes * hidden_dim
        self.v_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.log_vars = nn.Parameter(torch.zeros(2)) 

    def forward(self, maps, stats, peri):
        # 1. CNN
        cnn_feat = self.cnn_proj(self.cnn_net(maps))
        h_cnn = cnn_feat.view(-1, self.num_cnn_nodes, self.hidden_dim)
        
        # 2. GNN
        gnn_in = self.gnn_mapping(stats) 
        h_gnn_nodes = self.deep_gnn(gnn_in) 
        h_gnn = self.gnn_node_pool(h_gnn_nodes.transpose(1, 2)).transpose(1, 2)
        
        # 3. Peri
        peri_feat = self.peri_net(peri)
        h_peri = peri_feat.view(-1, self.num_peri_nodes, self.hidden_dim)

        # 4. 拼接并加入模态嵌入
        combined = torch.cat([h_cnn, h_gnn, h_peri], dim=1) 
        combined = combined + self.modality_emb  # 广播机制自动应用到 batch 中每个样本
        
        # 5. 图融合与分类
        fused = self.fusion_layer(combined)
        flat_feat = fused.view(fused.size(0), -1) 
        return self.v_head(flat_feat), self.a_head(flat_feat)

# ==========================================
#  模块 4: 数据加载 (提速优化: 内存预加载)
# ==========================================
class DeapMultiModalDataset(Dataset):
    def __init__(self, npz_dir, raw_mat_dir, filenames=None):
        self.npz_dir = npz_dir
        self.file_list = filenames if filenames else sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.samples_per_trial = 15 
        
        self.v_labels, self.a_labels = [], []
        
        # 提速核心: 直接在这里把所有需要的数据读到内存列表里！
        # 避免 __getitem__ 中每秒成百上千次的磁盘 IO
        self.maps_data = []
        self.stats_data = []
        self.peri_data = []
        
        for f_name in self.file_list:
            # 1. 读取标签
            subj_id = f_name[:3] 
            mat_path = os.path.join(raw_mat_dir, f"{subj_id}.mat")
            if not os.path.exists(mat_path): continue
                
            raw_labels = sio.loadmat(mat_path)['labels']
            v_binary = (raw_labels[:, 0] > 5).astype(np.int64)
            a_binary = (raw_labels[:, 1] > 5).astype(np.int64)
            self.v_labels.append(np.repeat(v_binary, self.samples_per_trial))
            self.a_labels.append(np.repeat(a_binary, self.samples_per_trial))
            
            # 2. 读取特征放入内存
            file_path = os.path.join(self.npz_dir, f_name)
            with np.load(file_path) as data:
                self.maps_data.append(data['eeg_allband_feature_map'])
                self.stats_data.append(data['eeg_en_stat'])
                self.peri_data.append(data['peri_feature'])
            
        if len(self.v_labels) > 0:
            self.v_labels = np.concatenate(self.v_labels)
            self.a_labels = np.concatenate(self.a_labels)
            
            # 拼接并转换为 Tensor 保存在内存中
            self.maps_data = torch.from_numpy(np.concatenate(self.maps_data)).float()
            # 注意: stat 特征在这里直接 reshape 好，省去每次 getitem 计算
            self.stats_data = torch.from_numpy(np.concatenate(self.stats_data)).view(-1, 32, 7).float()
            self.peri_data = torch.from_numpy(np.concatenate(self.peri_data)).float()
        else:
            self.v_labels = np.array([])
            self.a_labels = np.array([])

    def __len__(self):
        return len(self.v_labels)

    def __getitem__(self, idx):
        # 告别 np.load，直接从内存索引，速度起飞 🚀
        return self.maps_data[idx], self.stats_data[idx], self.peri_data[idx], self.v_labels[idx], self.a_labels[idx]

# ==========================================
#  模块 5: 训练引擎 (提速优化: AMP + 多线程)
# ==========================================
def train_deep_mt_dgf_loso():
    NPZ_PATH = r'D:\Users\cyz\dc\222'                 
    RAW_PATH = r'E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab'    
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = sorted([f for f in os.listdir(NPZ_PATH) if f.endswith('.npz')])
    num_subjects = len(all_files)
    
    # 提速: 实例化 AMP 的 GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    print(f"\n 开始 LOSO 验证 | Device: {DEVICE} | 开启 AMP 加速")
    
    loso_results = {'subject': [], 'v_acc': [], 'v_f1': [], 'a_acc': [], 'a_f1': []}
    
    for i, test_file in enumerate(all_files):
        subj_name = test_file.split('.')[0]
        print(f"\n[{i+1}/{num_subjects}] 测试被试: {subj_name}")
        
        train_files = [f for f in all_files if f != test_file]
        test_files = [test_file]
        
        # 这里数据直接进内存，稍花几秒钟，但后续训练速度飞起
        train_dataset = DeapMultiModalDataset(NPZ_PATH, RAW_PATH, filenames=train_files)
        test_dataset = DeapMultiModalDataset(NPZ_PATH, RAW_PATH, filenames=test_files)
        
        # 提速: num_workers=4 (可根据CPU核心调整), pin_memory=True
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                 num_workers=4, pin_memory=True)
        
        model = Deep_MT_DGF_GNN(hidden_dim=64, num_cnn_nodes=8, num_gnn_nodes=8, num_peri_nodes=8).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for maps, stats, peri, lv, la in train_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)
                
                optimizer.zero_grad()
                
                # 提速: 自动混合精度上下文
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out_v, out_a = model(maps, stats, peri)
                    loss_v = F.cross_entropy(out_v, lv)
                    loss_a = F.cross_entropy(out_a, la)
                    
                    precision_v = torch.exp(-model.log_vars[0])
                    precision_a = torch.exp(-model.log_vars[1])
                    combined_loss = (loss_v * precision_v + model.log_vars[0]) + \
                                    (loss_a * precision_a + model.log_vars[1])
                
                # 提速: 反向传播和优化
                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
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
        
        print(f"   -> Val Acc={curr_v_acc:.4f}, Aro Acc={curr_a_acc:.4f}")
        
        loso_results['subject'].append(subj_name)
        loso_results['v_acc'].append(curr_v_acc)
        loso_results['a_acc'].append(curr_a_acc)
        # 省略部分保存代码...

if __name__ == "__main__":
    train_deep_mt_dgf_loso()