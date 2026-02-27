import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models  # 这一行是缺少的！
import torchvision
from PIL import Image
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import glob
import pandas as pd
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

    # 超参数配
class Face2NodesFeatureExtractor(nn.Module):
    """
    面部表情识别模型的特征提取器
    支持输出任意维度的特征向量
    """
    def __init__(self, 
                 model_path=None,           # 预训练模型路径
                 embed_dim: int = 256,       # 原始特征维度
                 output_dim: int = 512,       # 输出特征维度（可自定义）
                 num_blocks: int = 4,         # RDGCN块数量
                 k: int = 8,                  # KNN邻居数
                 dilation: int = 2,            # 膨胀率
                 feature_level: str = 'global', # 特征类型
                 use_projection: bool = True,   # 是否使用特征投影
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        
        self.device = device
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.feature_level = feature_level
        self.use_projection = use_projection
        
        print(f"Initializing Face2Nodes Feature Extractor on {device}")
        print(f"Original feature dim: {embed_dim}, Output feature dim: {output_dim}")
        
        # 1. 构建模型结构（不包含分类头）
        self.patch_embedding = MultiScalePatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=1
        )
        
        # 创建RDGCN块
        self.rdgcn_blocks = nn.ModuleList()
        
        # 第一个块
        self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=1))
        
        # 中间块（膨胀率递增）
        for i in range(1, num_blocks-1):
            current_dilation = 2 ** i
            self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=current_dilation))
        
        # 最后一个块
        self.rdgcn_blocks.append(RDGCNBlock(embed_dim, embed_dim, k=k, dilation=2**(num_blocks-1)))
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. 特征投影层（将256维映射到任意维度）
        if use_projection and output_dim != embed_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            print(f"Added feature projection: {embed_dim} -> {output_dim}")
        else:
            self.feature_projection = nn.Identity()
            if output_dim != embed_dim:
                print(f"Note: output_dim ({output_dim}) != embed_dim ({embed_dim}) but projection is disabled")
        
        # 3. 如果有预训练模型，加载权重
        if model_path and os.path.exists(model_path):
            self.load_pretrained(model_path)
        else:
            print(f"Warning: Model path {model_path} not found. Using random initialization.")
        
        # 移到指定设备
        self.to(device)
        self.eval()
    
    def load_pretrained(self, model_path):
        """加载预训练模型权重"""
        print(f"Loading pretrained weights from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            
            # 移除分类头相关的权重
            keys_to_remove = [k for k in state_dict.keys() if 'head.' in k]
            for k in keys_to_remove:
                del state_dict[k]
            
            # 加载权重
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            print(f"Successfully loaded weights!")
            if missing_keys:
                print(f"Missing keys (should be only projection layer): {missing_keys[:5]}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量
        
        Returns:
            特征向量，shape (B, output_dim) 或 (B, N, output_dim)
        """
        x = x.to(self.device)
        
        # 1. 获取patch嵌入
        patches = self.patch_embedding(x)  # (B, N, embed_dim)
        
        if self.feature_level == 'patch':
            # 对patch特征进行投影
            if self.use_projection:
                B, N, D = patches.shape
                patches_flat = patches.view(-1, D)
                projected = self.feature_projection(patches_flat)
                return projected.view(B, N, -1)
            return patches
        
        # 2. 通过RDGCN块
        current = patches
        for block in self.rdgcn_blocks:
            current = block(current)
        
        if self.feature_level == 'last_block':
            # 对最后一个块的特征进行投影
            if self.use_projection:
                B, N, D = current.shape
                current_flat = current.view(-1, D)
                projected = self.feature_projection(current_flat)
                return projected.view(B, N, -1)
            return current
        
        elif self.feature_level == 'global':
            # 全局平均池化 (B, embed_dim)
            global_features = self.global_pool(current.transpose(1, 2)).squeeze(2)
            
            # 投影到目标维度
            if self.use_projection:
                global_features = self.feature_projection(global_features)
            
            return global_features
        
        else:
            # 默认返回全局特征
            global_features = self.global_pool(current.transpose(1, 2)).squeeze(2)
            if self.use_projection:
                global_features = self.feature_projection(global_features)
            return global_features
    
    @torch.no_grad()
    def extract_features(self, images, return_dict=False):
        """
        提取特征的便捷方法
        
        Args:
            images: 输入图像
            return_dict: 是否返回包含元数据的字典
            
        Returns:
            特征向量或特征字典
        """
        self.eval()
        
        # 处理不同类型的输入
        if isinstance(images, str):
            image = self.load_image(images)
            image_tensor = image.unsqueeze(0).to(self.device)
            features = self.forward(image_tensor)
            
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            features = self.forward(images.to(self.device))
            
        elif isinstance(images, list):
            tensors = []
            for img in images:
                if isinstance(img, str):
                    tensors.append(self.load_image(img))
                else:
                    tensors.append(img)
            batch = torch.stack(tensors).to(self.device)
            features = self.forward(batch)
        
        features = features.cpu()
        
        if return_dict:
            return {
                'features': features,
                'shape': features.shape,
                'dim': features.shape[-1],
                'norm': torch.norm(features, dim=1).mean().item(),
                'mean': features.mean().item(),
                'std': features.std().item()
            }
        return features
    
    def load_image(self, image_path, size=(100, 100)):
        """加载并预处理单张图像"""
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image)


# ==================== 增强版特征提取器 ====================

class EnhancedFeatureExtractor(Face2NodesFeatureExtractor):
    """增强版特征提取器，支持多维度输出和特征组合"""
    
    def __init__(self, *args, multi_scale_outputs=None, **kwargs):
        """
        multi_scale_outputs: 要输出的多个维度，例如 [128, 256, 512]
        """
        super().__init__(*args, **kwargs)
        
        self.multi_scale_outputs = multi_scale_outputs
        
        if multi_scale_outputs:
            self.multi_scale_projections = nn.ModuleDict()
            for dim in multi_scale_outputs:
                if dim != self.embed_dim:
                    self.multi_scale_projections[str(dim)] = nn.Sequential(
                        nn.Linear(self.embed_dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.GELU()
                    )
    
    def forward(self, x, output_dim=None):
        """
        指定输出维度
        
        Args:
            x: 输入图像
            output_dim: 指定的输出维度，如果为None则使用默认output_dim
        """
        x = x.to(self.device)
        
        # 基础特征提取
        patches = self.patch_embedding(x)
        current = patches
        for block in self.rdgcn_blocks:
            current = block(current)
        
        global_features = self.global_pool(current.transpose(1, 2)).squeeze(2)
        
        # 如果指定了特定的输出维度
        if output_dim is not None:
            if str(output_dim) in self.multi_scale_projections:
                return self.multi_scale_projections[str(output_dim)](global_features)
            elif output_dim == self.embed_dim:
                return global_features
            else:
                # 临时投影
                temp_proj = nn.Linear(self.embed_dim, output_dim).to(self.device)
                return temp_proj(global_features)
        
        # 默认行为
        if self.use_projection:
            return self.feature_projection(global_features)
        return global_features
    
    def extract_multi_scale(self, images):
        """提取多尺度特征"""
        features_dict = {}
        
        if isinstance(images, str):
            image = self.load_image(images).unsqueeze(0).to(self.device)
        else:
            image = images.to(self.device)
        
        # 基础特征
        patches = self.patch_embedding(image)
        current = patches
        for block in self.rdgcn_blocks:
            current = block(current)
        
        global_features = self.global_pool(current.transpose(1, 2)).squeeze(2)
        
        # 原始维度
        features_dict[f'dim_{self.embed_dim}'] = global_features.cpu()
        
        # 多尺度输出
        if self.multi_scale_outputs:
            for dim, proj in self.multi_scale_projections.items():
                features_dict[f'dim_{dim}'] = proj(global_features).cpu()
        
        return features_dict


# ==================== 使用示例 ====================

def extract_512_features():
    """提取512维特征向量 - 简化版"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Face2Nodes 特征提取器 - 输出512维特征向量")
    print("="*60)
    
    # 初始化特征提取器
    extractor = Face2NodesFeatureExtractor(
        model_path='/data/coding/face2nodes_best.pth',  # 您的模型路径
        embed_dim=256,                                    # 原始维度256
        output_dim=512,                                   # 输出维度512
        num_blocks=4,
        k=8,
        dilation=2,
        feature_level='global',
        use_projection=True,                              # 启用投影
        device=device
    )
    
    # ========== 方法1：处理单张图片 ==========
    print("\n" + "-"*40)
    print("1. 处理单张图片")
    print("-"*40)
    
    # 请修改为您的图片路径
    image_path = '/data/test/face_001.jpg'  # <--- 在这里修改您的图片路径
    
    if os.path.exists(image_path):
        # 提取真实图片的特征
        features = extractor.extract_features(image_path)
        
        print(f"✓ 图片路径: {image_path}")
        print(f"✓ 特征形状: {features.shape}")  # (1, 512)
        print(f"✓ 特征维度: {features.shape[1]} 维")
        print(f"\n特征向量 (前20个值):")
        print(features[0][:20])
        
        # 保存特征
        np.save('单张图片特征_512dim.npy', features.numpy())
        print(f"\n✓ 特征已保存到: 单张图片特征_512dim.npy")
    else:
        print(f"✗ 图片不存在: {image_path}")
        print("  使用随机数据演示...")
        
        # 使用随机数据演示
        dummy_input = torch.randn(1, 3, 100, 100)
        features = extractor(dummy_input)
        
        print(f"✓ 演示数据形状: {dummy_input.shape}")
        print(f"✓ 输出特征形状: {features.shape}")  # (1, 512)
        print(f"\n特征向量 (前20个值):")
        print(features[0][:20])
    
    # ========== 方法2：批量处理多张图片 ==========
    print("\n" + "-"*40)
    print("2. 批量处理多张图片")
    print("-"*40)
    
    # 假设有一个文件夹包含多张图片
    image_folder = '/data/coding/test'  # <--- 修改为您的图片文件夹路径
    
    if os.path.exists(image_folder):
        import glob
        
        # 获取文件夹中所有jpg和png图片
        image_paths = glob.glob(os.path.join(image_folder, '*.jpg')) + \
                      glob.glob(os.path.join(image_folder, '*.png'))
        
        if image_paths:
            print(f"找到 {len(image_paths)} 张图片")
            
            all_features = []
            for i, img_path in enumerate(image_paths[:5]):  # 先处理前5张演示
                feat = extractor.extract_features(img_path)
                all_features.append(feat.numpy())
                print(f"  处理第 {i+1} 张: {os.path.basename(img_path)} -> 特征形状 {feat.shape}")
            
            # 合并所有特征
            if all_features:
                batch_features = np.vstack(all_features)
                print(f"\n✓ 批量特征形状: {batch_features.shape}")  # (n, 512)
                
                # 保存批量特征
                np.save('批量图片特征_512dim.npy', batch_features)
                print(f"✓ 批量特征已保存到: 批量图片特征_512dim.npy")
        else:
            print(f"文件夹中没有图片: {image_folder}")
    else:
        print(f"文件夹不存在: {image_folder}")
        print("使用随机数据演示批量处理...")
        
        # 使用随机数据演示
        batch_size = 5
        batch_images = torch.randn(batch_size, 3, 100, 100)
        batch_features = extractor(batch_images)
        
        print(f"✓ 批量数据形状: {batch_images.shape}")
        print(f"✓ 批量特征形状: {batch_features.shape}")  # (5, 512)
    
    # ========== 方法3：特征相似度计算 ==========
    print("\n" + "-"*40)
    print("3. 特征相似度计算")
    print("-"*40)
    
    # 计算两张图片的相似度
    if os.path.exists(image_path):
        # 使用同一张图片的两个副本演示
        feat1 = extractor.extract_features(image_path)
        feat2 = extractor.extract_features(image_path)  # 相同图片
        
        # 计算余弦相似度
        cos_sim = torch.cosine_similarity(feat1, feat2)
        print(f"✓ 相同图片的余弦相似度: {cos_sim.item():.4f}")
        
        # 如果有第二张图片
        image_path2 = '/data/test/face_002.jpg'  # <--- 可选的第二张图片
        if os.path.exists(image_path2):
            feat2 = extractor.extract_features(image_path2)
            cos_sim = torch.cosine_similarity(feat1, feat2)
            print(f"✓ 不同图片的余弦相似度: {cos_sim.item():.4f}")
    else:
        # 使用随机数据演示
        feat1 = torch.randn(1, 512)
        feat2 = torch.randn(1, 512)
        cos_sim = torch.cosine_similarity(feat1, feat2)
        print(f"✓ 随机特征的余弦相似度: {cos_sim.item():.4f}")
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("特征提取完成！")
    print("="*60)
    print("✓ 输出特征格式: [图片数量 × 512]")
    print("✓ 每个图片用512维向量表示")
    print("✓ 特征文件已保存为 .npy 格式")
    print("\n使用方法:")
    print("  features = np.load('单张图片特征_512dim.npy')")
    print("  print(features.shape)  # (n, 512)")
    print("="*60)


# ==================== 主程序 ====================

if __name__ == '__main__':
    
    print("="*60)
    print("Face2Nodes 特征提取器 - 简化版")
    print("="*60)
    
    # 运行特征提取
    extract_512_features()