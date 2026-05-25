#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import gc
import cv2
import glob
import json
import time
import copy
import random
import warnings
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

warnings.filterwarnings("ignore")


# ====================
# 用户配置区域
# ====================
MODEL_PATH = r"D:\Users\cyz\dc\moxing\face2nodes_best.pth"

# 可以填单个被试视频目录，例如：
VIDEO_DIR = r"E:\BaiduNetdiskDownload\DEAP\face_video\s22"

# 也可以填总目录，例如里面有 s01/s02/.../s32 子目录：
# VIDEO_ROOT = r"E:\BaiduNetdiskDownload\DEAP\face_video"
VIDEO_ROOT = None

OUTPUT_DIR = r"D:\Users\cyz\dc\see"

TARGET_FEATURES = 15
OUTPUT_DIM = 512
ADD_TIMESTAMP = False

VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv"]


# ====================
# 工具函数
# ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


def infer_subject_id(video_path):
    """
    从文件名或父目录中识别 s01/s02/.../s32。
    """
    base = os.path.basename(video_path)
    parent = os.path.basename(os.path.dirname(video_path))

    for text in [base, parent, video_path]:
        m = re.search(r"(s\d{2})", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()

    raise ValueError(f"无法从路径识别被试编号 sXX：{video_path}")


def infer_trial_id(video_path):
    """
    从文件名中识别 trial 编号。
    支持：
    s22_trial01.avi
    s22_trial_01.avi
    s22_trial-01.avi
    trial01.avi
    01.avi
    """
    name = os.path.splitext(os.path.basename(video_path))[0].lower()

    patterns = [
        r"trial[_-]?(\d{1,2})",
        r"video[_-]?(\d{1,2})",
        r"clip[_-]?(\d{1,2})",
        r"_(\d{1,2})$",
        r"^(\d{1,2})$",
        r"(\d{1,2})$"
    ]

    for pat in patterns:
        m = re.search(pat, name)
        if m:
            trial_id = int(m.group(1))
            if 1 <= trial_id <= 40:
                return trial_id

    raise ValueError(f"无法从文件名识别 trial 编号 1~40：{video_path}")


def make_feature_filename(video_path, add_timestamp=False):
    sid = infer_subject_id(video_path)
    trial_id = infer_trial_id(video_path)

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{sid}_trial{trial_id:02d}_features_{timestamp}.npy"

    return f"{sid}_trial{trial_id:02d}_features.npy"


def get_video_paths():
    video_paths = []

    if VIDEO_ROOT and os.path.exists(VIDEO_ROOT):
        print(f"使用总目录模式：{VIDEO_ROOT}")

        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(glob.glob(os.path.join(VIDEO_ROOT, "**", ext), recursive=True))
            video_paths.extend(glob.glob(os.path.join(VIDEO_ROOT, "**", ext.upper()), recursive=True))

    elif VIDEO_DIR and os.path.exists(VIDEO_DIR):
        print(f"使用单目录模式：{VIDEO_DIR}")

        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
            video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext.upper())))

    else:
        raise FileNotFoundError("请正确配置 VIDEO_DIR 或 VIDEO_ROOT")

    video_paths = sorted(list(set(video_paths)))

    valid_paths = []
    bad_paths = []

    for p in video_paths:
        try:
            infer_subject_id(p)
            infer_trial_id(p)
            valid_paths.append(p)
        except Exception as e:
            bad_paths.append((p, str(e)))

    if bad_paths:
        print("\n以下视频无法识别 sid 或 trial，将跳过：")
        for p, reason in bad_paths:
            print(f"  {os.path.basename(p)} -> {reason}")

    print(f"\n找到视频总数：{len(video_paths)}")
    print(f"可处理视频数：{len(valid_paths)}")

    return valid_paths


def verify_output_files(output_dir, video_paths):
    """
    检查输出文件能否被刚才训练代码读取。
    """
    print("\n" + "=" * 60)
    print("检查视觉特征输出")
    print("=" * 60)

    expected = []

    for p in video_paths:
        sid = infer_subject_id(p)
        trial_id = infer_trial_id(p)
        expected.append((sid, trial_id, os.path.join(output_dir, f"{sid}_trial{trial_id:02d}_features.npy")))

    ok = 0
    bad = 0

    for sid, trial_id, path in expected:
        if not os.path.exists(path):
            print(f"缺失：{os.path.basename(path)}")
            bad += 1
            continue

        arr = np.load(path)

        if arr.shape != (TARGET_FEATURES, OUTPUT_DIM):
            print(f"形状错误：{os.path.basename(path)} -> {arr.shape}，期望 {(TARGET_FEATURES, OUTPUT_DIM)}")
            bad += 1
            continue

        if not np.isfinite(arr).all():
            print(f"存在 NaN/Inf：{os.path.basename(path)}")
            bad += 1
            continue

        ok += 1

    print(f"\n检查完成：正确 {ok} 个，异常 {bad} 个")

    sid_to_trials = {}

    for sid, trial_id, _ in expected:
        sid_to_trials.setdefault(sid, set()).add(trial_id)

    for sid, trials in sorted(sid_to_trials.items()):
        missing = [i for i in range(1, 41) if i not in trials]
        if missing:
            print(f"{sid} 当前视频缺少 trial：{missing}")
        else:
            print(f"{sid} 已包含 40 个 trial")


# ====================
# KNN
# ====================
def knn(x: torch.Tensor, k: int, dilation: int = 1) -> torch.Tensor:
    B, N, D = x.shape

    k_total = min(k * dilation, max(N - 1, 1))

    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    xy = torch.matmul(x, x.transpose(2, 1))
    pairwise_distance = xx + xx.transpose(2, 1) - 2 * xy

    idx = pairwise_distance.topk(
        k=min(k_total + 1, N),
        dim=-1,
        largest=False
    )[1][:, :, 1:]

    if dilation > 1:
        idx = idx[:, :, ::dilation]

    idx = idx[:, :, :min(k, idx.shape[-1])]

    return idx


# ====================
# MSF2PE
# ====================
class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=1, pretrained=True):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained)

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
                embed_dim,
                embed_dim,
                kernel_size=patch_size,
                stride=1,
                padding=patch_size // 2
            )
        else:
            self.patch_conv = None

    def forward(self, x):
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

        x2 = F.interpolate(x2, size=target_size, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([x1, x2, x3, x4], dim=1))

        if self.patch_conv is not None:
            fused = self.patch_conv(fused)

        B, D, H, W = fused.shape
        patches = fused.view(B, D, -1).transpose(1, 2)

        return patches


# ====================
# RGConv / RDGCN
# ====================
class RGConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=9, dilation=1):
        super().__init__()

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

    def forward(self, x):
        B, N, D = x.shape
        device = x.device

        x_transformed = self.feature_transform(x)

        idx = knn(x_transformed, self.k, self.dilation)
        k_eff = idx.shape[-1]

        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = (idx + idx_base).reshape(-1)

        x_reshaped = x.reshape(B * N, D)
        neighbors = x_reshaped[idx].view(B, N, k_eff, D)

        center = x.unsqueeze(2).expand(B, N, k_eff, D)
        edge_features = neighbors - center

        edge_weights = self.relation_weight(edge_features.reshape(-1, D)).view(B, N, k_eff, 1)
        aggregated = torch.sum(edge_weights * edge_features, dim=2)

        combined = torch.cat([x, aggregated], dim=2)
        updated = self.node_updater(combined.reshape(-1, 2 * D)).view(B, N, -1)

        return updated


class RDGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=9, dilation=1):
        super().__init__()

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

        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        B, N, D = x.shape
        identity = x

        x = self.in_trans(x.reshape(B * N, D)).reshape(B, N, -1)
        x = self.rg_conv(x)
        x = self.out_trans(x.reshape(B * N, -1)).reshape(B, N, -1)

        identity = self.residual_proj(identity.reshape(B * N, D)).reshape(B, N, -1)

        return x + identity


# ====================
# 特征提取器
# ====================
class Face2NodesFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_path=None,
        embed_dim=256,
        output_dim=512,
        num_blocks=4,
        k=8,
        device=None
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim

        print(f"初始化特征提取器，设备：{self.device}")
        print(f"输出特征维度：{output_dim}")

        self.patch_embedding = MultiScalePatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=1,
            pretrained=True
        )

        self.rdgcn_blocks = nn.ModuleList()

        self.rdgcn_blocks.append(
            RDGCNBlock(embed_dim, embed_dim, k=k, dilation=1)
        )

        for i in range(1, num_blocks):
            self.rdgcn_blocks.append(
                RDGCNBlock(embed_dim, embed_dim, k=k, dilation=2 ** i)
            )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if output_dim != embed_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.feature_projection = nn.Identity()

        if model_path and os.path.exists(model_path):
            self.load_pretrained(model_path)
        else:
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        self.to(self.device)
        self.eval()

    def load_pretrained(self, model_path):
        print(f"加载预训练模型：{model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint.state_dict()

        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("head.") and ".head." not in k
        }

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        print("模型加载完成")
        print(f"missing keys 数量：{len(missing)}")
        print(f"unexpected keys 数量：{len(unexpected)}")

    def forward(self, x):
        x = x.to(self.device)

        patches = self.patch_embedding(x)

        current = patches
        for block in self.rdgcn_blocks:
            current = block(current)

        feat = self.global_pool(current.transpose(1, 2)).squeeze(2)
        feat = self.feature_projection(feat)

        return feat

    @torch.no_grad()
    def extract_features(self, image_batch):
        self.eval()

        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(0)

        feat = self.forward(image_batch.to(self.device))
        return feat.detach().cpu()


# ====================
# 视频处理器
# ====================
class VideoFaceProcessor:
    def __init__(self, extractor, device=None):
        self.extractor = extractor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            print("警告：人脸检测器加载失败，将使用整张图片")

        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("视频处理器初始化完成")

    def detect_faces(self, frame):
        if self.face_cascade.empty():
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30)
            )

        return faces

    def preprocess_bgr_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return self.transform(pil_img)

    def crop_main_face_or_full_frame(self, frame):
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return frame, False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        pad_x = int(w * 0.15)
        pad_y = int(h * 0.20)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            return frame, False

        return face, True

    def sample_frame_indices(self, total_frames, target_features):
        if total_frames <= 0:
            return [0] * target_features

        indices = np.linspace(0, total_frames - 1, target_features)
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, total_frames - 1)

        return indices.tolist()

    @torch.no_grad()
    def process_video(self, video_path, output_dir, target_features=15, add_timestamp=False):
        start_time = time.time()

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(video_path):
            print(f"错误：视频不存在：{video_path}")
            return None

        output_filename = make_feature_filename(video_path, add_timestamp=add_timestamp)
        output_path = os.path.join(output_dir, output_filename)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"错误：无法打开视频：{video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sid = infer_subject_id(video_path)
        trial_id = infer_trial_id(video_path)

        print("\n" + "-" * 60)
        print(f"处理视频：{os.path.basename(video_path)}")
        print(f"输出文件：{output_filename}")
        print(f"被试：{sid}，trial：{trial_id:02d}")
        print(f"总帧数：{total_frames}，FPS：{fps:.2f}")

        frame_indices = self.sample_frame_indices(total_frames, target_features)

        tensors = []
        used_face_count = 0

        for i, frame_idx in enumerate(frame_indices, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"  第 {i}/{target_features} 个采样帧读取失败，使用零图像")
                frame = np.zeros((100, 100, 3), dtype=np.uint8)

            crop, used_face = self.crop_main_face_or_full_frame(frame)

            if used_face:
                used_face_count += 1

            tensor = self.preprocess_bgr_image(crop)
            tensors.append(tensor)

        cap.release()

        batch = torch.stack(tensors, dim=0).to(self.device)

        try:
            features = self.extractor.extract_features(batch).numpy().astype(np.float32)
        except Exception as e:
            print(f"特征提取失败，使用零向量：{e}")
            features = np.zeros((target_features, OUTPUT_DIM), dtype=np.float32)

        if features.ndim == 1:
            features = np.tile(features[None, :], (target_features, 1))

        if features.shape[0] != target_features:
            fixed = np.zeros((target_features, OUTPUT_DIM), dtype=np.float32)
            n = min(target_features, features.shape[0])
            fixed[:n] = features[:n]

            if n > 0 and n < target_features:
                fixed[n:] = fixed[n - 1]

            features = fixed

        if features.shape[1] != OUTPUT_DIM:
            raise ValueError(f"输出维度错误：{features.shape}，期望第二维为 {OUTPUT_DIM}")

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        np.save(output_path, features)

        elapsed = time.time() - start_time

        print(f"保存成功：{output_path}")
        print(f"特征形状：{features.shape}")
        print(f"使用人脸裁剪帧数：{used_face_count}/{target_features}")
        print(f"用时：{elapsed:.1f} 秒")

        return output_path

    def process_video_batch(self, video_paths, output_dir, target_features=15, add_timestamp=False):
        print("\n" + "=" * 60)
        print(f"开始批量处理视频：{len(video_paths)} 个")
        print(f"每个视频输出：({target_features}, {OUTPUT_DIM})")
        print("=" * 60)

        results = []
        total_start = time.time()

        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}]")

            try:
                feature_file = self.process_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    target_features=target_features,
                    add_timestamp=add_timestamp
                )

                results.append({
                    "video": video_path,
                    "feature_file": feature_file,
                    "status": "success" if feature_file else "failed"
                })

            except Exception as e:
                print(f"处理失败：{video_path}")
                print(f"原因：{e}")

                results.append({
                    "video": video_path,
                    "feature_file": None,
                    "status": "failed",
                    "error": str(e)
                })

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_elapsed = time.time() - total_start

        print("\n" + "=" * 60)
        print("批量处理完成")
        print(f"总用时：{total_elapsed:.1f} 秒")
        print(f"成功：{sum(r['status'] == 'success' for r in results)}")
        print(f"失败：{sum(r['status'] == 'failed' for r in results)}")
        print("=" * 60)

        return results


# ====================
# 主程序
# ====================
def main():
    print("=" * 60)
    print("Face2Nodes DEAP 视觉特征提取")
    print("=" * 60)
    print(f"模型路径：{MODEL_PATH}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"目标特征数：{TARGET_FEATURES}")
    print(f"输出维度：{OUTPUT_DIM}")
    print("训练代码期望文件名：sXX_trialYY_features.npy")
    print("=" * 60)

    if ADD_TIMESTAMP:
        print("警告：ADD_TIMESTAMP=True 会导致训练代码找不到默认文件名，建议保持 False。")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    video_paths = get_video_paths()

    if len(video_paths) == 0:
        raise RuntimeError("没有找到可处理视频")

    print("\n待处理视频：")
    for i, p in enumerate(video_paths, 1):
        sid = infer_subject_id(p)
        trial_id = infer_trial_id(p)
        print(f"  {i:03d}. {os.path.basename(p)} -> {sid}_trial{trial_id:02d}_features.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")

    extractor = Face2NodesFeatureExtractor(
        model_path=MODEL_PATH,
        embed_dim=256,
        output_dim=OUTPUT_DIM,
        num_blocks=4,
        k=8,
        device=device
    )

    processor = VideoFaceProcessor(
        extractor=extractor,
        device=device
    )

    results = processor.process_video_batch(
        video_paths=video_paths,
        output_dir=OUTPUT_DIR,
        target_features=TARGET_FEATURES,
        add_timestamp=ADD_TIMESTAMP
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        f"visual_feature_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "video_dir": VIDEO_DIR,
        "video_root": VIDEO_ROOT,
        "output_dir": OUTPUT_DIR,
        "target_features": TARGET_FEATURES,
        "output_dim": OUTPUT_DIM,
        "total": len(results),
        "success": sum(r["status"] == "success" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "results": results
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n处理汇总已保存：{summary_file}")

    verify_output_files(OUTPUT_DIR, video_paths)


if __name__ == "__main__":
    main()
