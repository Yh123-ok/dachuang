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
import matplotlib
matplotlib.use('Agg')  # 重要：使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
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
    B, N, D = x.shape
    device = x.device
  
    k_total = k * dilation

    xx = torch.sum(x**2, dim=2, keepdim=True)
    xy = torch.matmul(x, x.transpose(2, 1))
    pairwise_distance = xx + xx.transpose(2, 1) - 2 * xy
    
    idx = pairwise_distance.topk(k=k_total+1, dim=-1, largest=False)[1][:, :, 1:]
    
    if dilation > 1:
        idx = idx[:, :, ::dilation][:, :, :k]
    else:
        idx = idx[:, :, :k]
    
    return idx

# ==================== 2. MSF[2]PE模块 ====================
class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=1, pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        self.channels = [64, 128, 256, 512]
        
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(self.channels[0], 32, kernel_size=1),
            nn.Conv2d(self.channels[1], 64, kernel_size=1),
            nn.Conv2d(self.channels[2], 128, kernel_size=1),
            nn.Conv2d(self.channels[3], 256, kernel_size=1),
        ])
        
        total_channels = 32 + 64 + 128 + 256
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        if patch_size > 1:
            self.patch_conv = nn.Conv2d(
                embed_dim, embed_dim, 
                kernel_size=patch_size, 
                stride=1,
                padding=patch_size//2
            )
        else:
            self.patch_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        features = []
        x_temp = x
        for i, layer in enumerate(self.backbone):
            x_temp = layer(x_temp)
            if i >= 4:
                features.append(x_temp)
        
        x1 = F.avg_pool2d(self.downsample_layers[0](features[0]), 2, 2)
        x2 = F.avg_pool2d(self.downsample_layers[1](features[1]), 2, 2)
        x3 = F.avg_pool2d(self.downsample_layers[2](features[2]), 2, 2)
        x4 = self.downsample_layers[3](features[3])
        
        target_size = x1.shape[-2:]
        
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)
        
        concatenated = torch.cat([x1, x2, x3, x4], dim=1)
        fused = self.fusion(concatenated)
        
        if self.patch_conv is not None:
            patches = self.patch_conv(fused)
            B, D, H_p, W_p = patches.shape
            patches = patches.view(B, D, -1).transpose(1, 2)
        else:
            B, D, H, W = fused.shape
            patches = fused.view(B, D, -1).transpose(1, 2)
        
        return patches

# ==================== 3. RG-Conv模块 ====================
class RGConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=9, dilation=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.dilation = dilation
        
        self.relation_weight = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.node_updater = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
        self.feature_transform = nn.Linear(in_dim, in_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        device = x.device
        
        x_transformed = self.feature_transform(x)
        idx = knn(x_transformed, self.k, self.dilation)
        
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x_reshaped = x.view(B * N, D)
        neighbors = x_reshaped[idx].view(B, N, self.k, D)
        
        center = x.unsqueeze(2).expand(B, N, self.k, D)
        edge_features = neighbors - center
        
        edge_features_flat = edge_features.view(-1, D)
        edge_weights = self.relation_weight(edge_features_flat).view(B, N, self.k, 1)
        
        aggregated = torch.sum(edge_weights * edge_features, dim=2)
        combined = torch.cat([x, aggregated], dim=2)
        combined_flat = combined.view(-1, 2 * D)
        updated = self.node_updater(combined_flat).view(B, N, self.out_dim)
        
        return updated

# ==================== 4. RDGCN块 ====================
class RDGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=9, dilation=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.dilation = dilation
        
        self.in_trans = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
        self.rg_conv = RGConv(out_dim, out_dim, k=k, dilation=dilation)
        
        self.out_trans = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        identity = x
        
        x_transformed = self.in_trans(x.reshape(B*N, D)).reshape(B, N, -1)
        x_conv = self.rg_conv(x_transformed)
        x_out = self.out_trans(x_conv.reshape(B*N, -1)).reshape(B, N, -1)
        
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
        x_pooled = self.global_pool(x.transpose(1, 2)).squeeze(2)
        logits = self.classifier(x_pooled)
        return logits

# ==================== 6. 完整Face2Nodes模型 ====================
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
        
        self.patch_embedding = MultiScalePatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=1
        )
        
        self.rdgcn_blocks = nn.ModuleList()
        self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=1))
        for i in range(1, num_blocks-1):
            current_dilation = 2 ** i
            self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=current_dilation))
        self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=2**(num_blocks-1)))
        
        self.head = RecognitionHead(
            in_dim=embed_dim,
            num_classes=num_classes
        )
        
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
        patches = self.patch_embedding(x)
        for block in self.rdgcn_blocks:
            patches = block(patches)
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
            return torch.zeros(3, 100, 100), label

# ==================== 9. 可视化函数 ====================
def plot_training_progress(epochs_list, train_loss_history, train_acc_history, test_acc_history, output_dir):
    """绘制训练进度图并保存"""
    plt.figure(figsize=(15, 5))
    
    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs_list, train_loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 2. 训练准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs_list, train_acc_history, 'g-', linewidth=2, marker='s', markersize=4, label='Train')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 3. 测试准确率曲线
    plt.subplot(1, 3, 3)
    if test_acc_history:
        # 对齐测试数据的epoch
        test_epochs = epochs_list[:len(test_acc_history)]
        plt.plot(test_epochs, test_acc_history, 'r-', linewidth=2, marker='^', markersize=4, label='Test')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Test Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'training_progress_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 同时保存最新版本
    latest_path = os.path.join(output_dir, 'training_progress_latest.png')
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_acc_history, 'g-', linewidth=2, label='Train')
    if test_acc_history:
        test_epochs = epochs_list[:len(test_acc_history)]
        plt.plot(test_epochs, test_acc_history, 'r-', linewidth=2, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(latest_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training progress plots saved to:\n   {save_path}\n   {latest_path}")

# ==================== 10. 训练和测试函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc, all_preds, all_labels

# ==================== 11. 模型保存函数 ====================
def save_model(model, optimizer, epoch, metrics, config, save_dir, model_name='best'):
    """保存模型检查点"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 完整模型保存
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': timestamp,
        'model_architecture': str(model)
    }
    
    # 多个保存格式
    checkpoint_path = os.path.join(save_dir, f'{model_name}_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 仅保存模型权重（轻量版）
    weights_path = os.path.join(save_dir, f'{model_name}_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    # 保存整个模型（包含结构）
    full_model_path = os.path.join(save_dir, f'{model_name}_full_model.pth')
    torch.save(model, full_model_path)
    
    # 保存配置信息
    config_path = os.path.join(save_dir, f'{model_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'metrics': metrics,
            'config': config,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"✅ Model saved to {save_dir}/")
    print(f"   - Checkpoint: {checkpoint_path}")
    print(f"   - Weights only: {weights_path}")
    print(f"   - Full model: {full_model_path}")
    print(f"   - Config: {config_path}")

# ==================== 12. 主程序 ====================
def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"}')
    
    # ========== 创建输出目录 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'/root/autodl-tmp/results/experiment_{timestamp}'
    model_dir = os.path.join(output_dir, 'models')
    plot_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    print(f'Model directory: {model_dir}')
    print(f'Plot directory: {plot_dir}')
    
    # ========== 超参数配置 ==========
    config = {
        'num_classes': 5,
        'embed_dim': 256,
        'num_blocks': 4,
        'k': 4,
        'dilation': 2,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'smoothing': 0.1,
        'data_path': r'/root/autodl-tmp/dataset/',
        'device': str(device),
        'timestamp': timestamp
    }
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # ========== 数据加载 ==========
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
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
    
    if len(train_dataset) == 0:
        print(f"❌ No training data found at {config['data_path']}/train/")
        print("Please check if dataset is properly uploaded and extracted.")
        return
    
    num_workers = 0 if os.name == 'nt' else 4
    
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
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Testing samples: {len(test_dataset)}')
    
    # ========== 创建模型 ==========
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = Face2Nodes(
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_blocks=config['num_blocks'],
        k=config['k'],
        dilation=config['dilation'],
        input_size=100
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Model: Face2Nodes')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Model size: {total_params * 4 / (1024**2):.2f} MB')
    
    # ========== 训练设置 ==========
    criterion = LabelSmoothingCrossEntropy(smoothing=config['smoothing'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # ========== 训练历史记录 ==========
    train_loss_history = []
    train_acc_history = []
    test_acc_history = []
    epochs_list = []
    best_acc = 0
    
    # ========== 训练日志 ==========
    log_file = os.path.join(output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training Log - Experiment {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Testing samples: {len(test_dataset)}\n")
        f.write(f"Model parameters: {total_params:,}\n")
        f.write("="*50 + "\n\n")
    
    # ========== 训练循环 ==========
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['num_epochs']
        )
        
        # 记录训练历史
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        epochs_list.append(epoch)
        
        epoch_time = time.time() - epoch_start_time
        
        # 测试
        if epoch % 2 == 0 or epoch == config['num_epochs']:
            test_acc, all_preds, all_labels = test(model, test_loader, device)
            test_acc_history.append(test_acc)
        else:
            test_acc = None
        
        # 保存训练日志
        with open(log_file, 'a') as f:
            log_line = f'Epoch {epoch:03d}/{config["num_epochs"]:03d} | '
            log_line += f'Loss: {train_loss:.4f} | '
            log_line += f'Train Acc: {train_acc:.2f}% | '
            if test_acc:
                log_line += f'Test Acc: {test_acc:.2f}% | '
            log_line += f'Time: {epoch_time:.1f}s | '
            log_line += f'LR: {scheduler.get_last_lr()[0]:.6f}\n'
            f.write(log_line)
        
        # 打印进度
        print(f'Epoch {epoch:03d}/{config["num_epochs"]:03d} | '
              f'Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}%' if test_acc else '' + ' | '
              f'Time: {epoch_time:.1f}s')
        
        # 保存最佳模型
        if test_acc and test_acc > best_acc:
            best_acc = test_acc
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'epoch': epoch
            }
            
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                save_dir=model_dir,
                model_name='best'
            )
        
        # 定期保存检查点
        if epoch % 5 == 0 or epoch == config['num_epochs']:
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc if test_acc else 0,
                'epoch': epoch
            }
            
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                save_dir=model_dir,
                model_name=f'epoch_{epoch:03d}'
            )
            
            # 更新训练进度图
            plot_training_progress(
                epochs_list=epochs_list,
                train_loss_history=train_loss_history,
                train_acc_history=train_acc_history,
                test_acc_history=test_acc_history,
                output_dir=plot_dir
            )
            
            # 保存训练历史数据
            history_data = {
                'epochs': epochs_list,
                'train_loss': train_loss_history,
                'train_acc': train_acc_history,
                'test_acc': test_acc_history,
                'config': config
            }
            
            history_path = os.path.join(output_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
        
        # 更新学习率
        scheduler.step()
    
    # ========== 训练完成 ==========
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    print(f'Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
    print(f'Best test accuracy: {best_acc:.2f}%')
    
    # 最终可视化
    print("\nGenerating final visualizations...")
    plot_training_progress(
        epochs_list=epochs_list,
        train_loss_history=train_loss_history,
        train_acc_history=train_acc_history,
        test_acc_history=test_acc_history,
        output_dir=plot_dir
    )
    
    # 保存最终模型
    final_metrics = {
        'best_acc': best_acc,
        'final_train_loss': train_loss_history[-1],
        'final_train_acc': train_acc_history[-1],
        'final_test_acc': test_acc_history[-1] if test_acc_history else 0,
        'total_epochs': config['num_epochs'],
        'total_time': total_time
    }
    
    save_model(
        model=model,
        optimizer=optimizer,
        epoch=config['num_epochs'],
        metrics=final_metrics,
        config=config,
        save_dir=model_dir,
        model_name='final'
    )
    
    # 生成训练总结报告
    summary_report = os.path.join(output_dir, 'summary.txt')
    with open(summary_report, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FACE2NODES TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Experiment ID: {timestamp}\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nTRAINING RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Best test accuracy: {best_acc:.2f}%\n")
        f.write(f"Final train accuracy: {train_acc_history[-1]:.2f}%\n")
        if test_acc_history:
            f.write(f"Final test accuracy: {test_acc_history[-1]:.2f}%\n")
        f.write(f"Final train loss: {train_loss_history[-1]:.4f}\n")
        f.write(f"Total epochs: {config['num_epochs']}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Testing samples: {len(test_dataset)}\n\n")
        
        f.write("MODEL FILES SAVED:\n")
        f.write("-"*40 + "\n")
        for filename in os.listdir(model_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(model_dir, filename)
                size = os.path.getsize(filepath) / (1024*1024)
                f.write(f"{filename}: {size:.2f} MB\n")
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - Training logs: {log_file}")
    print(f"   - Model checkpoints: {model_dir}/")
    print(f"   - Training plots: {plot_dir}/")
    print(f"   - Config file: {config_path}")
    print(f"   - Summary report: {summary_report}")
    
    # 测试最终模型
    print("\n" + "="*60)
    print("Testing final model...")
    print("="*60)
    
    if os.path.exists(os.path.join(model_dir, 'best_checkpoint.pth')):
        checkpoint = torch.load(os.path.join(model_dir, 'best_checkpoint.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_acc, final_preds, final_labels = test(model, test_loader, device)
        print(f'Final test accuracy with best model: {final_acc:.2f}%')
    else:
        final_acc, final_preds, final_labels = test(model, test_loader, device)
        print(f'Final test accuracy: {final_acc:.2f}%')

if __name__ == '__main__':
    main()