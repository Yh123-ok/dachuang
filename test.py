import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from typing import Tuple, List
import math
import os
from PIL import Image
import torchvision
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ==================== 1. 辅助函数 ====================
def knn(x: torch.Tensor, k: int, dilation: int = 1) -> torch.Tensor: 
    """
    计算膨胀k最近邻
    Args:
        x: 节点特征 (B, N, D)
        k: 邻居数
        dilation: 膨胀率
    Returns:
        idx: 邻居索引 (B, N, k)
    """
    B, N, D = x.shape   #B:图像数量 N:节点数量 D:特征维度
    device = x.device
  
    k_total = k * dilation

    # 正确计算欧式距离矩阵（正值）
    xx = torch.sum(x**2, dim=2, keepdim=True)  # (B, N, 1)
    xy = torch.matmul(x, x.transpose(2, 1))     # (B, N, N)
    pairwise_distance = xx + xx.transpose(2, 1) - 2 * xy  # (B, N, N)
    
    # 获取k*d个最近邻（排除自身）
    # 对每个节点，自身距离为0，所以从第2个开始取
    idx = pairwise_distance.topk(k=k_total+1, dim=-1, largest=False)[1][:, :, 1:]  # (B, N, k*d)
    
    # 膨胀采样
    if dilation > 1:
        idx = idx[:, :, ::dilation][:, :, :k]
    else:
        idx = idx[:, :, :k]
    
    return idx

# ==================== 2. MSF[2]PE模块（优化版） ====================
class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=1, pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # 加载预训练ResNet-18
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # 修改：移除更多下采样层，保持更大特征图
        # 原始：conv1, bn1, relu, maxpool (下采样4倍)
        # 修改：移除maxpool或减小其kernel_size
        self.backbone = nn.Sequential(
            resnet.conv1,      # 下采样2倍 (100->50)
            resnet.bn1,
            resnet.relu,
            # 修改：移除maxpool或使用较小的kernel
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),  # 保持尺寸或轻微下采样
            # 或者完全移除：nn.Identity(),
            resnet.layer1,     # 保持尺寸
            resnet.layer2,     # 下采样2倍 (50->25)
            resnet.layer3,     # 下采样2倍 (25->13)
            resnet.layer4,     # 下采样2倍 (13->7)
        )
        
        # 修改通道数以适应embed_dim=256
        self.channels = [64, 128, 256, 512]
        
        # 1x1卷积降维到更小的维度
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(self.channels[0], 32, kernel_size=1),  # 64->32
            nn.Conv2d(self.channels[1], 64, kernel_size=1),  # 128->64
            nn.Conv2d(self.channels[2], 128, kernel_size=1), # 256->128
            nn.Conv2d(self.channels[3], 256, kernel_size=1), # 512->256
        ])
        
        # 特征融合层
        total_channels = 32 + 64 + 128 + 256  # = 480
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 关键修改：确保有足够节点
        if patch_size > 1:
            # 使用更小的kernel和stride
            self.patch_conv = nn.Conv2d(
                embed_dim, embed_dim, 
                kernel_size=patch_size, 
                stride=1,  # 重要：stride=1 保持尺寸
                padding=patch_size//2
            )
        else:
            self.patch_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            patches: 补丁特征 (B, N, D)
        """
        B = x.shape[0]
        
        # 提取ResNet特征
        features = []
        x_temp = x
        for i, layer in enumerate(self.backbone):
            x_temp = layer(x_temp)
            if i >= 4:  # 从layer1开始记录特征
                features.append(x_temp)
                print(f"[DEBUG] Layer {i} output: {x_temp.shape}")
        
        # 处理各尺度特征 - 减小下采样率
        x1 = F.avg_pool2d(self.downsample_layers[0](features[0]), 2, 2)  # 50->25
        x2 = F.avg_pool2d(self.downsample_layers[1](features[1]), 2, 2)  # 25->13
        x3 = F.avg_pool2d(self.downsample_layers[2](features[2]), 2, 2)  # 13->7
        x4 = self.downsample_layers[3](features[3])  # 7x7
        
        print(f"[DEBUG] Multi-scale features: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}")
        
        # 统一到最大尺寸
        target_size = x1.shape[-2:]  # 使用最大的特征图尺寸
        print(f"[DEBUG] Target size: {target_size}")
        
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)
        
        # 拼接与融合
        concatenated = torch.cat([x1, x2, x3, x4], dim=1)
        fused = self.fusion(concatenated)
        
        print(f"[DEBUG] Fused feature shape: {fused.shape}")
        
        # 补丁嵌入
        if self.patch_conv is not None:
            patches = self.patch_conv(fused)
            B, D, H_p, W_p = patches.shape
            patches = patches.view(B, D, -1).transpose(1, 2)
        else:
            B, D, H, W = fused.shape
            patches = fused.view(B, D, -1).transpose(1, 2)
        
        print(f"[DEBUG] Final patches shape: {patches.shape}, N={H*W}")
        
        return patches
# ==================== 3. RG-Conv模块（修复版） ====================
class RGConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=9, dilation=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.dilation = dilation  # 修复：添加dilation属性
        
        # 关系感知聚合函数
        self.relation_weight = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 节点特征更新器
        self.node_updater = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
        # 特征变换层
        self.feature_transform = nn.Linear(in_dim, in_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        优化版：移除嵌套循环，使用向量化操作
        Args:
            x: 节点特征 (B, N, D)
        Returns:
            更新后的节点特征 (B, N, D')
        """
        B, N, D = x.shape
        device = x.device
        
        # 特征变换
        x_transformed = self.feature_transform(x)
        
        # 构建膨胀KNN图
        idx = knn(x_transformed, self.k, self.dilation)  # (B, N, k)
        
        # 收集邻居特征（向量化操作）
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x_reshaped = x.view(B * N, D)
        neighbors = x_reshaped[idx].view(B, N, self.k, D)
        
        # 中心特征扩展
        center = x.unsqueeze(2).expand(B, N, self.k, D)
        
        # 计算边缘特征
        edge_features = neighbors - center  # (B, N, k, D)
        
        # 计算边缘权重
        edge_features_flat = edge_features.view(-1, D)
        edge_weights = self.relation_weight(edge_features_flat).view(B, N, self.k, 1)
        
        # 聚合
        aggregated = torch.sum(edge_weights * edge_features, dim=2)  # (B, N, D)
        
        # 节点更新
        combined = torch.cat([x, aggregated], dim=2)  # (B, N, 2D)
        combined_flat = combined.view(-1, 2 * D)
        updated = self.node_updater(combined_flat).view(B, N, self.out_dim)
        
        return updated

# ==================== 4. RDGCN块（修复版） ====================
class RDGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=9, dilation=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.dilation = dilation
        
        # 输入变换层
        self.in_trans = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
        # RG-Conv层
        self.rg_conv = RGConv(out_dim, out_dim, k=k, dilation=dilation)
        
        # 输出变换层
        self.out_trans = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
        # 残差连接投影
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        identity = x
        
        # 输入变换
        x_transformed = self.in_trans(x.reshape(B*N, D)).reshape(B, N, -1)
        
        # RG-Conv操作
        x_conv = self.rg_conv(x_transformed)
        
        # 输出变换
        x_out = self.out_trans(x_conv.reshape(B*N, -1)).reshape(B, N, -1)
        
        # 残差连接
        identity = self.residual_proj(identity.reshape(B*N, D)).reshape(B, N, -1)
        x_out = x_out + identity
        
        return x_out

# ==================== 5. 识别头 ====================
class RecognitionHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局平均池化
        x_pooled = self.global_pool(x.transpose(1, 2)).squeeze(2)
        
        # 分类
        logits = self.classifier(x_pooled)
        
        return logits

# ==================== 6. 完整Face2Nodes模型（修复版） ====================
class Face2Nodes(nn.Module):
    def __init__(self,
                 num_classes: int = 5,
                 embed_dim: int = 256,
                 num_blocks: int = 4,
                 k: int = 8,
                 dilation: int = 2,
                 input_size: int = 100):
        super().__init__()
        
        self.input_size = input_size
        
        # 1. MSF[2]PE模块
        self.patch_embedding = MultiScalePatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=1
        )
        
        # 2. RDGCN模块
        self.rdgcn_blocks = nn.ModuleList()
        # 第一个块
        self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=1))
        # 中间块
        for i in range(1, num_blocks-1):
            current_dilation = 2 ** i
            self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=current_dilation))
        # 最后一个块
        self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=2**(num_blocks-1)))
        
        # 3. 识别头
        self.head = RecognitionHead(
            in_dim=embed_dim,
            num_classes=num_classes
        )
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 补丁嵌入
        patches = self.patch_embedding(x)
        
        # RDGCN块
        for block in self.rdgcn_blocks:
            patches = block(patches)
        
        # 识别头
        logits = self.head(patches)
        
        return logits

# ==================== 7. 带标签平滑的损失函数 ====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=1)
        
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==================== 8. 自定义数据集 ====================
class FERDataset(Dataset):
    """面部表情识别数据集"""
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.classes = ['anger', 'fear', 'happy', 'neutral', 'pain']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        data_path = os.path.join(root_dir, mode)
        if os.path.exists(data_path):
            for class_name in self.classes:
                class_path = os.path.join(data_path, class_name)
                if os.path.exists(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append(os.path.join(class_path, img_name))
                            self.labels.append(self.class_to_idx[class_name])
        
        print(f'Loaded {len(self.images)} images for {mode} mode')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个空图像和标签（应急处理）
            return torch.zeros(3, 100, 100), label

# ==================== 9. 训练和测试函数 ====================
def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                total_epochs: int):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: [{epoch}/{total_epochs}], '
                  f'Batch: [{batch_idx}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc

def test(model: nn.Module, 
         dataloader: DataLoader,
         device: torch.device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc

# ==================== 10. 主程序 ====================
def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数配置
    config = {
        'num_classes': 5,
        'embed_dim': 256,
        'num_blocks': 4,
        'k': 4,  # 减小k值，避免节点数不足
        'dilation': 2,
        'batch_size': 32,
        'num_epochs': 30, 
        'learning_rate': 0.001,
        'smoothing': 0.1,
        'data_path': r'/data/data'
    }
    
    # ========== 添加可视化数据存储 ==========
    train_loss_history = []
    train_acc_history = []
    test_acc_history = []
    epochs_list = []
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = FERDataset(
        root_dir=config['data_path'],
        transform=transform,
        mode='train'
    )
    
    test_dataset = FERDataset(
        root_dir=config['data_path'],
        transform=transform,
        mode='test'
    )
    
    # Windows下num_workers设为0避免报错
    num_workers = 0 if os.name == 'nt' else 2
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    model = Face2Nodes(
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_blocks=config['num_blocks'],
        k=config['k'],  # 使用减小后的k值
        dilation=config['dilation'],
        input_size=100
    ).to(device)
    
    # 损失函数和优化器
    criterion = LabelSmoothingCrossEntropy(smoothing=config['smoothing'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # 打印模型信息
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of testing samples: {len(test_dataset)}')
    
    # 训练循环
    best_acc = 0
    
    # ========== 添加可视化函数 ==========
    def plot_training_progress():
        """绘制训练进度图"""
        plt.figure(figsize=(12, 4))
        
        # 1. 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs_list, train_loss_history, 'b-', linewidth=2, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, train_acc_history, 'g-', linewidth=2, marker='s', label='训练准确率')
    
    # 确保测试数据与epochs对应
        if test_acc_history:
        # 如果测试数据较少，需要对齐
            if len(test_acc_history) < len(epochs_list):
            # 用最后一个值填充
                padded_test_acc = test_acc_history + [test_acc_history[-1]] * (len(epochs_list) - len(test_acc_history))
                test_epochs = epochs_list
            else:
                padded_test_acc = test_acc_history
                test_epochs = epochs_list[:len(test_acc_history)]
        
        plt.plot(test_epochs, padded_test_acc, 'r-', linewidth=2, marker='^', label='测试准确率')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('准确率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['num_epochs']
        )
        
        # 保存训练数据
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        epochs_list.append(epoch)
        
        print(f'Epoch {epoch}/{config["num_epochs"]} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # 测试
        if epoch % 2 == 0 or epoch == config['num_epochs']:  # 每2个epoch测试一次
            test_acc = test(model, test_loader, device)
            test_acc_history.append(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                # 保存最佳模型
                save_path = os.path.join(os.getcwd(), 'face2nodes_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                }, save_path)
                print(f'Saved best model to {save_path} with accuracy: {best_acc:.2f}%')
            
            # 每4个epoch显示一次训练进度图
            if epoch % 4 == 0:
                print("\n" + "="*50)
                print("当前训练进度可视化：")
                print("="*50)
                plot_training_progress()
        
        # 更新学习率
        scheduler.step()
    
    print(f'Training completed! Best test accuracy: {best_acc:.2f}%')
    
    # 最终可视化
    print("\n" + "="*50)
    print("最终训练结果可视化：")
    print("="*50)
    plot_training_progress()
    
    # 加载最佳模型
    if os.path.exists('face2nodes_best.pth'):
        checkpoint = torch.load('face2nodes_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_acc = test(model, test_loader, device)
        print(f'Final test accuracy with best model: {final_acc:.2f}%')
    else:
        print("未找到保存的最佳模型文件")

if __name__ == '__main__':
    main()