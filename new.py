import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import scipy.io as sio

torch.backends.cudnn.benchmark = True

# ==========================================
#  模块 1 & 2: 基础组件 (保持不变)
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
#  模块 3: 模型主体 (提分优化)
# ==========================================
class Deep_MT_DGF_GNN(nn.Module):
    def __init__(self, hidden_dim=64, num_cnn_nodes=8, num_gnn_nodes=8, num_peri_nodes=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cnn_nodes = num_cnn_nodes
        self.num_gnn_nodes = num_gnn_nodes
        self.num_peri_nodes = num_peri_nodes
        self.total_nodes = num_cnn_nodes + num_gnn_nodes + num_peri_nodes
        
        self.modality_emb = nn.Parameter(torch.randn(1, self.total_nodes, hidden_dim) * 0.02)
        
        # 提分点 1: 使用 InstanceNorm2d 替换 BatchNorm2d，消除被试间基线差异
        self.cnn_net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.InstanceNorm2d(16), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.ELU(),
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
        
        # 提分点 2: 增加强力 Dropout 防止过拟合
        self.dropout = nn.Dropout(p=0.4)
        
        combined_dim = self.total_nodes * hidden_dim
        self.v_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, 2))
        self.log_vars = nn.Parameter(torch.zeros(2)) 

    def forward(self, maps, stats, peri):
        h_cnn = self.cnn_proj(self.cnn_net(maps)).view(-1, self.num_cnn_nodes, self.hidden_dim)
        
        gnn_in = self.gnn_mapping(stats) 
        h_gnn_nodes = self.deep_gnn(gnn_in) 
        h_gnn = self.gnn_node_pool(h_gnn_nodes.transpose(1, 2)).transpose(1, 2)
        
        h_peri = self.peri_net(peri).view(-1, self.num_peri_nodes, self.hidden_dim)

        combined = torch.cat([h_cnn, h_gnn, h_peri], dim=1) 
        combined = combined + self.modality_emb 
        
        fused = self.fusion_layer(combined)
        
        # 应用 Dropout
        flat_feat = self.dropout(fused.view(fused.size(0), -1))
        return self.v_head(flat_feat), self.a_head(flat_feat)

# ... (Dataset 部分保留之前写在内存里的高速版本保持不变) ...

# ==========================================
#  模块 5: 训练引擎 (提分优化)
# ==========================================
def train_deep_mt_dgf_loso():
    # ... (前面的路径和变量配置不变) ...
    EPOCHS = 30
    LR = 0.001 # 配合余弦退火，初始学习率可以稍微给大一点
    
    # ... (外层文件遍历循环不变) ...
    
        model = Deep_MT_DGF_GNN(hidden_dim=64, num_cnn_nodes=8, num_gnn_nodes=8, num_peri_nodes=8).to(DEVICE)
        
        # 增加 weight_decay 实施 L2 正则化
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        
        # 提分点 3: 引入学习率调度器，后期平滑下降寻找更好落点
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for maps, stats, peri, lv, la in train_loader:
                maps, stats, peri = maps.to(DEVICE), stats.to(DEVICE), peri.to(DEVICE)
                lv, la = lv.to(DEVICE), la.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out_v, out_a = model(maps, stats, peri)
                    
                    # 提分点 4: 标签平滑 (Label Smoothing=0.1)
                    # 强迫模型不要输出绝对的1或0，缓解跨被试带来的“确信偏误”
                    loss_v = F.cross_entropy(out_v, lv, label_smoothing=0.1)
                    loss_a = F.cross_entropy(out_a, la, label_smoothing=0.1)
                    
                    precision_v = torch.exp(-model.log_vars[0])
                    precision_a = torch.exp(-model.log_vars[1])
                    combined_loss = (loss_v * precision_v + model.log_vars[0]) + \
                                    (loss_a * precision_a + model.log_vars[1])
                
                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # 每个 Epoch 结束后更新学习率
            scheduler.step()
        
        # ... (后续测试代码保持不变) ...