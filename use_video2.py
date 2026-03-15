#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from PIL import Image
import os
import warnings
import json
import time
from datetime import datetime
import pandas as pd
import cv2  # 用于视频处理
import glob

warnings.filterwarnings('ignore')

# ==================== 用户配置区域（只需要修改这里！）====================

# 1. 模型路径（必须修改）
MODEL_PATH = r'D:\大创\face2nodes_best.pth'  # 改为您的模型路径

# 2. 视频路径配置（二选一）
# 方式A：单个视频文件
# VIDEO_PATH = '/data/coding/071309_w_21-PA4-076.mp4'

# 方式B：视频目录（批量处理该目录下所有视频）
VIDEO_DIR = r'D:\大创\数据集\face_video\s10'  # 包含多个视频的目录

# 3. 输出目录
OUTPUT_DIR = r'D:\大创\result'  # 特征输出目录

# 4. 处理参数
TARGET_FEATURES = 15  # 每个视频提取的特征数量（固定15组）
OUTPUT_DIM = 512      # 输出特征维度

# 5. 文件命名选项
ADD_TIMESTAMP = True  # 是否在文件名添加时间戳

# ======================================================================

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

# ==================== MSF²PE模块 ====================
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

# ==================== RG-Conv模块 ====================
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

# ==================== RDGCN块 ====================
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

# ==================== 特征提取器 ====================
class Face2NodesFeatureExtractor(nn.Module):
    """
    面部表情识别模型的特征提取器
    支持输出任意维度的特征向量
    """
    def __init__(self, 
                 model_path=None,
                 embed_dim: int = 256,
                 output_dim: int = 512,
                 num_blocks: int = 4,
                 k: int = 8,
                 dilation: int = 2,
                 feature_level: str = 'global',
                 use_projection: bool = True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        
        self.device = device
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.feature_level = feature_level
        self.use_projection = use_projection
        
        print(f"初始化特征提取器，设备: {device}")
        print(f"原始特征维度: {embed_dim}, 输出特征维度: {output_dim}")
        
        # 构建模型结构
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
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征投影层
        if use_projection and output_dim != embed_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            print(f"添加特征投影层: {embed_dim} -> {output_dim}")
        else:
            self.feature_projection = nn.Identity()
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self.load_pretrained(model_path)
        else:
            print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化")
        
        self.to(device)
        self.eval()
    
    def load_pretrained(self, model_path):
        print(f"加载预训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        keys_to_remove = [k for k in state_dict.keys() if 'head.' in k]
        for k in keys_to_remove:
            del state_dict[k]
        
        self.load_state_dict(state_dict, strict=False)
        print(f"模型加载成功！")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        
        patches = self.patch_embedding(x)
        current = patches
        for block in self.rdgcn_blocks:
            current = block(current)
        
        global_features = self.global_pool(current.transpose(1, 2)).squeeze(2)
        
        if self.use_projection:
            global_features = self.feature_projection(global_features)
        
        return global_features
    
    @torch.no_grad()
    def extract_features(self, images):
        self.eval()
        
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
        else:
            features = self.forward(images)
        
        return features.cpu()
    
    def load_image(self, image_path, size=(100, 100)):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image)


# ==================== 视频处理器 ====================
class VideoFaceProcessor:
    """视频人脸特征提取器 - 固定提取指定数量的特征"""
    
    def __init__(self, extractor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.extractor = extractor
        self.device = device
        
        # 加载人脸检测器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("⚠️ 警告：人脸检测器加载失败，将使用整张图片")
        
        print("✓ 视频处理器初始化完成")
    
    def preprocess_frame(self, frame):
        """预处理整张图片（当人脸检测失败时使用）"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(pil_image).unsqueeze(0).to(self.device)
    
    def detect_faces(self, frame):
        """检测视频帧中的人脸 - 优化版"""
        if self.face_cascade.empty():
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 直方图均衡化，提高对比度
    
        # 使用更宽松的参数
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,        # 更小的缩放因子（更精细的搜索）
            minNeighbors=3,           # 降低邻居数要求（允许更多检测）
            minSize=(40, 40),         # 减小最小人脸尺寸
            maxSize=(300, 300),       # 设置最大人脸尺寸
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        # 如果还是没检测到，尝试不同的参数
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30)
            )
    
        return faces
    
    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(pil_image).unsqueeze(0).to(self.device)
    
    def process_video(self, video_path, output_dir, target_features=15, add_timestamp=True):
        """
        处理单个视频文件 - 严格提取指定数量的特征
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            target_features: 目标特征数量（默认15组）
            add_timestamp: 是否添加时间戳
        """
        start_time = time.time()
        
        # 检查视频文件                        
        if not os.path.exists(video_path):
            print(f"❌ 错误：视频文件不存在 - {video_path}")
            return None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
        # 生成输出文件名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{video_name}_features_{timestamp}.npy"
        else:
            output_filename = f"{video_name}_features.npy"
        
        feature_file = os.path.join(output_dir, output_filename)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 错误：无法打开视频文件 - {video_path}")
            return None

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0

        print(f"\n📹 处理视频: {os.path.basename(video_path)}")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps:.2f} fps")
        print(f"  视频时长: {video_duration:.2f} 秒")
        print(f"  目标特征数: {target_features}")
        
        # 计算均匀采样的时间点
        target_times = np.linspace(0, video_duration, target_features, endpoint=False)
        target_frames = [int(t * fps) for t in target_times]
        print(f"  采样时间点: {[f'{t:.1f}s' for t in target_times]}")

        # 准备保存结果
        all_features = []
        all_timestamps = []
        all_frame_ids = []

        frame_count = 0
        processed_count = 0
        
        # 创建要处理的帧集合
        frames_to_process = set(target_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检查当前帧是否需要处理
            if frame_count in frames_to_process:
                current_time = frame_count / fps
                print(f"  处理目标 {processed_count+1}/{target_features} - 帧 {frame_count} (时间 {current_time:.1f}s)")
                
                # 检测人脸
                faces = self.detect_faces(frame)
                feature_extracted = False
                
                # 如果有检测到人脸，使用最大的人脸
                if len(faces) > 0:
                    # 取最大的人脸（假设主要人物）
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    # 提取人脸区域
                    face_roi = frame[y:y+h, x:x+w]
                    
                    try:
                        input_tensor = self.preprocess_face(face_roi)
                        
                        with torch.no_grad():
                            features = self.extractor(input_tensor)
                        
                        all_features.append(features.cpu().numpy().squeeze())
                        all_timestamps.append(current_time)
                        all_frame_ids.append(frame_count)
                        
                        processed_count += 1
                        feature_extracted = True
                        print(f"    ✓ 成功提取人脸特征 {processed_count}/{target_features}")
                        
                    except Exception as e:
                        print(f"    ✗ 人脸特征提取失败: {e}")
                
                # 如果人脸检测失败或特征提取失败，使用整张图片
                if not feature_extracted:
                    try:
                        input_tensor = self.preprocess_frame(frame)
                        
                        with torch.no_grad():
                            features = self.extractor(input_tensor)
                        
                        all_features.append(features.cpu().numpy().squeeze())
                        all_timestamps.append(current_time)
                        all_frame_ids.append(frame_count)
                        
                        processed_count += 1
                        print(f"    ✓ 使用整张图片提取特征 {processed_count}/{target_features}")
                        
                    except Exception as e:
                        print(f"    ✗ 整张图片提取失败: {e}")
                        # 如果还是失败，用零向量填充
                        all_features.append(np.zeros(OUTPUT_DIM))
                        all_timestamps.append(current_time)
                        all_frame_ids.append(frame_count)
                        processed_count += 1
                        print(f"    ⚠️ 使用零向量填充 {processed_count}/{target_features}")
            
            frame_count += 1
            
            # 如果已经处理完所有目标帧，提前退出
            if processed_count >= target_features:
                break

        cap.release()

        # 确保正好有 target_features 个特征
        if len(all_features) < target_features:
            print(f"\n⚠️ 特征数不足 ({len(all_features)}/{target_features})，进行填充...")
            while len(all_features) < target_features:
                if len(all_features) > 0:
                    # 用最后一个特征填充
                    all_features.append(all_features[-1])
                    all_timestamps.append(all_timestamps[-1])
                    all_frame_ids.append(all_frame_ids[-1])
                else:
                    # 如果没有任何特征，用零向量
                    all_features.append(np.zeros(OUTPUT_DIM))
                    all_timestamps.append(0.0)
                    all_frame_ids.append(0)
                print(f"   填充第 {len(all_features)}/{target_features}")
        
        elif len(all_features) > target_features:
            print(f"\n⚠️ 特征数过多 ({len(all_features)}/{target_features})，截取前{target_features}个")
            all_features = all_features[:target_features]
            all_timestamps = all_timestamps[:target_features]
            all_frame_ids = all_frame_ids[:target_features]

        # 转换为numpy数组
        features_array = np.array(all_features)

        # 保存特征文件
        np.save(feature_file, features_array)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 视频处理完成！")
        print(f"  特征文件: {feature_file}")
        print(f"  特征矩阵形状: {features_array.shape}")
        print(f"  特征数: {len(all_features)}/{target_features}")
        print(f"  时间范围: {all_timestamps[0]:.2f}s - {all_timestamps[-1]:.2f}s")
        print(f"  用时: {elapsed_time:.1f}秒")

        return feature_file
    
    def process_video_batch(self, video_paths, output_dir, target_features=15, add_timestamp=True):
        """批量处理多个视频 - 每个视频固定提取target_features组特征"""
        print(f"\n{'='*60}")
        print(f"开始批量处理 {len(video_paths)} 个视频")
        print(f"每个视频固定提取 {target_features} 组特征")
        print(f"{'='*60}")
        
        results = []
        total_start_time = time.time()
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] 处理视频...")
            video_start_time = time.time()
            
            feature_file = self.process_video(
                video_path=video_path,
                output_dir=output_dir,
                target_features=target_features,
                add_timestamp=add_timestamp
            )
            
            video_elapsed = time.time() - video_start_time
            
            if feature_file:
                results.append({
                    'video': os.path.basename(video_path),
                    'feature_file': feature_file,
                    'status': 'success',
                    'time': f'{video_elapsed:.1f}s'
                })
            else:
                results.append({
                    'video': os.path.basename(video_path),
                    'feature_file': None,
                    'status': 'failed',
                    'time': f'{video_elapsed:.1f}s'
                })
        
        total_elapsed = time.time() - total_start_time
        
        # 打印汇总结果
        print(f"\n{'='*60}")
        print(f"批量处理完成！")
        print(f"总用时: {total_elapsed:.1f}秒")
        print(f"成功: {sum(1 for r in results if r['status'] == 'success')} 个")
        print(f"失败: {sum(1 for r in results if r['status'] == 'failed')} 个")
        print(f"{'='*60}")
        
        return results


# ==================== 主程序 ====================

def get_video_paths():
    """获取所有需要处理的视频路径"""
    video_paths = []
    
    # 检查是否定义了 VIDEO_DIR
    if 'VIDEO_DIR' in globals() and VIDEO_DIR and os.path.exists(VIDEO_DIR):
        print(f"使用视频目录模式: {VIDEO_DIR}")
        # 支持的视频格式
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
            video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext.upper())))
        video_paths = list(set(video_paths))  # 去重
        video_paths.sort()  # 排序
        print(f"找到 {len(video_paths)} 个视频文件")
    
    # 检查是否定义了 VIDEO_PATH（单个视频）
    elif 'VIDEO_PATH' in globals() and VIDEO_PATH:
        print("使用单个视频模式...")
        if os.path.exists(VIDEO_PATH):
            video_paths = [VIDEO_PATH]
        else:
            print(f"错误: 视频文件不存在 - {VIDEO_PATH}")
    
    return video_paths


def main():
    """主程序 - 批量视频处理，每个视频固定提取15组特征"""
    
    print("="*60)
    print("Face2Nodes 批量视频特征提取系统")
    print("="*60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标特征数: {TARGET_FEATURES} 组/视频")
    print("="*60)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：模型文件不存在 - {MODEL_PATH}")
        return
    
    # 获取所有需要处理的视频
    video_paths = get_video_paths()
    
    if not video_paths:
        print("❌ 错误：没有找到需要处理的视频文件")
        print("请配置 VIDEO_DIR 或 VIDEO_PATH")
        return
    
    print(f"\n找到 {len(video_paths)} 个待处理视频:")
    for i, v in enumerate(video_paths, 1):
        print(f"  {i}. {os.path.basename(v)}")
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 初始化特征提取器
    print("\n初始化特征提取器...")
    extractor = Face2NodesFeatureExtractor(
        model_path=MODEL_PATH,
        embed_dim=256,
        output_dim=OUTPUT_DIM,
        num_blocks=4,
        k=8,
        dilation=2,
        feature_level='global',
        use_projection=True,
        device=device
    )
    
    # 初始化视频处理器
    print("\n初始化视频处理器...")
    video_processor = VideoFaceProcessor(extractor, device=device)
    
    # 批量处理视频 - 固定15组特征
    print("\n开始批量处理视频...")
    
    results = video_processor.process_video_batch(
        video_paths=video_paths,
        output_dir=OUTPUT_DIR,
        target_features=TARGET_FEATURES,  # 固定15组
        add_timestamp=ADD_TIMESTAMP
    )
    
    # 保存处理结果汇总
    if results:
        summary_file = os.path.join(OUTPUT_DIR, f"batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': MODEL_PATH,
            'target_features': TARGET_FEATURES,
            'output_dim': OUTPUT_DIM,
            'total_videos': len(results),
            'success_count': sum(1 for r in results if r['status'] == 'success'),
            'failed_count': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理结果汇总已保存: {summary_file}")


if __name__ == '__main__':
    main()