import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from django.conf import settings


warnings.filterwarnings("ignore")


MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "face2nodes_best.pth")


CLASS_NAMES = {
    0: "anger",
    1: "fear",
    2: "happy",
    3: "neutral",
    4: "pain",
}


CLASS_NAMES_CN = {
    0: "愤怒",
    1: "恐惧",
    2: "高兴",
    3: "中性",
    4: "疼痛",
}


CLASS_DESCRIPTIONS = {
    0: "模型判断当前面部表情偏向愤怒状态。",
    1: "模型判断当前面部表情偏向恐惧或紧张状态。",
    2: "模型判断当前面部表情偏向高兴、积极状态。",
    3: "模型判断当前面部表情偏向中性、平静状态。",
    4: "模型判断当前面部表情偏向疼痛或不适状态。",
}


def knn(x: torch.Tensor, k: int, dilation: int = 1) -> torch.Tensor:
    B, N, D = x.shape

    k_total = k * dilation

    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    xy = torch.matmul(x, x.transpose(2, 1))
    pairwise_distance = xx + xx.transpose(2, 1) - 2 * xy

    idx = pairwise_distance.topk(
        k=k_total + 1,
        dim=-1,
        largest=False
    )[1][:, :, 1:]

    if dilation > 1:
        idx = idx[:, :, ::dilation][:, :, :k]
    else:
        idx = idx[:, :, :k]

    return idx


class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=1, pretrained=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 推理时不需要联网下载预训练权重，pth 里已经有训练好的权重
        resnet = models.resnet18(weights=None)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        x_transformed = self.in_trans(x.reshape(B * N, D)).reshape(B, N, -1)

        x_conv = self.rg_conv(x_transformed)

        x_out = self.out_trans(x_conv.reshape(B * N, -1)).reshape(B, N, -1)

        identity = self.residual_proj(identity.reshape(B * N, D)).reshape(B, N, -1)

        x_out = x_out + identity

        return x_out


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


class Face2Nodes(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 256,
        num_blocks: int = 4,
        k: int = 4,
        dilation: int = 2,
        input_size: int = 100
    ):
        super().__init__()

        self.input_size = input_size

        self.patch_embedding = MultiScalePatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=1,
            pretrained=False
        )

        self.rdgcn_blocks = nn.ModuleList()

        self.rdgcn_blocks.append(
            RDGCNBlock(embed_dim, embed_dim, k=k, dilation=1)
        )

        for i in range(1, num_blocks - 1):
            current_dilation = 2 ** i
            self.rdgcn_blocks.append(
                RDGCNBlock(embed_dim, embed_dim, k=k, dilation=current_dilation)
            )

        self.rdgcn_blocks.append(
            RDGCNBlock(embed_dim, embed_dim, k=k, dilation=2 ** (num_blocks - 1))
        )

        self.head = RecognitionHead(
            in_dim=embed_dim,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(x)

        for block in self.rdgcn_blocks:
            patches = block(patches)

        logits = self.head(patches)

        return logits


_model = None
_model_error = None


def get_transform():
    return transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model_once():
    global _model, _model_error

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        _model_error = f"模型文件不存在：{MODEL_PATH}"
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(MODEL_PATH, map_location=device)

        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        model = Face2Nodes(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 256),
            num_blocks=config.get("num_blocks", 4),
            k=config.get("k", 4),
            dilation=config.get("dilation", 2),
            input_size=100
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        _model = model
        _model_error = None

        return _model

    except Exception as e:
        _model_error = str(e)
        return None


def predict_image(image_path):
    model = load_model_once()

    if model is None:
        random_class = random.randint(0, 4)

        return {
            "model_loaded": False,
            "model_error": _model_error,
            "class_id": random_class,
            "class_name": CLASS_NAMES.get(random_class),
            "emotion": CLASS_NAMES_CN.get(random_class),
            "description": "模型未加载成功，当前为模拟结果。",
            "confidence": round(random.uniform(0.55, 0.95), 4),
            "all_probs": {},
        }

    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    x = get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1)[0]

        class_id = int(torch.argmax(probs).item())
        confidence = float(probs[class_id].item())

    return {
        "model_loaded": True,
        "model_error": None,
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "emotion": CLASS_NAMES_CN.get(class_id, f"类别{class_id}"),
        "description": CLASS_DESCRIPTIONS.get(class_id, ""),
        "confidence": confidence,
        "all_probs": {
            str(i): float(probs[i].item()) for i in range(len(probs))
        },
    }


def check_model_status():
    model = load_model_once()

    return {
        "model_loaded": model is not None,
        "model_error": _model_error,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
    }
