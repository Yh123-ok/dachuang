# -*- coding: utf-8 -*-

import os
import math
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEmotionNet(nn.Module):
    """
    视觉-only 情绪分类模型。

    输入:
        visual feature: [B, 512]

    输出:
        valence_logits: [B, 2]
        arousal_logits: [B, 2]

    标签含义:
        0 = low
        1 = high
    """

    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.valence_head = nn.Linear(hidden_dim, 2)
        self.arousal_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.backbone(x)
        valence_logits = self.valence_head(h)
        arousal_logits = self.arousal_head(h)
        return valence_logits, arousal_logits


def emotion_from_va(valence_label, arousal_label):
    """
    根据 Valence/Arousal 二分类结果映射成情绪文字。

    valence:
        0 = 低效价，负向
        1 = 高效价，正向

    arousal:
        0 = 低唤醒，平静/低能量
        1 = 高唤醒，激动/高能量
    """

    if valence_label == 1 and arousal_label == 1:
        return {
            "emotion": "兴奋/开心",
            "emotion_en": "excited_or_happy",
            "quadrant": "high_valence_high_arousal",
            "description": "正向高唤醒，可能处于开心、兴奋、积极状态",
        }

    if valence_label == 1 and arousal_label == 0:
        return {
            "emotion": "平静/放松",
            "emotion_en": "calm_or_relaxed",
            "quadrant": "high_valence_low_arousal",
            "description": "正向低唤醒，可能处于平静、放松、舒适状态",
        }

    if valence_label == 0 and arousal_label == 1:
        return {
            "emotion": "紧张/焦虑",
            "emotion_en": "tense_or_anxious",
            "quadrant": "low_valence_high_arousal",
            "description": "负向高唤醒，可能处于紧张、焦虑、压力状态",
        }

    return {
        "emotion": "低落/疲惫",
        "emotion_en": "sad_or_tired",
        "quadrant": "low_valence_low_arousal",
        "description": "负向低唤醒，可能处于低落、疲惫、无聊状态",
    }


class VisualEmotionPredictor:
    """
    视觉情绪预测器。

    支持两种模式：

    1. model 模式：
       如果 visual_emotion_best.pth 存在，就加载模型预测。

    2. demo 模式：
       如果模型不存在，就使用稳定的演示判断逻辑。
       这个模式不是科研结果，只用于后端和页面联调。
    """

    def __init__(
        self,
        model_path=None,
        input_dim=512,
        device=None,
        allow_demo_fallback=True,
    ):
        self.model_path = str(model_path) if model_path is not None else None
        self.input_dim = input_dim
        self.allow_demo_fallback = allow_demo_fallback

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = VisualEmotionNet(input_dim=input_dim)
        self.model.to(self.device)
        self.model.eval()

        self.mode = "demo"
        self.loaded = False
        self.load_message = ""

        self._try_load_model()

    def _try_load_model(self):
        """
        尝试加载训练好的视觉分类模型。
        """
        if not self.model_path:
            self.load_message = "未配置视觉情绪分类模型路径，使用 demo 模式。"
            print(self.load_message)
            return

        if not os.path.exists(self.model_path):
            self.load_message = (
                f"视觉情绪分类模型不存在：{self.model_path}，使用 demo 模式。"
            )
            print(self.load_message)
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                raise ValueError("checkpoint 格式不支持")

            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()

            self.mode = "model"
            self.loaded = True
            self.load_message = f"视觉情绪分类模型加载成功：{self.model_path}"
            print(self.load_message)

        except Exception as e:
            self.mode = "demo"
            self.loaded = False
            self.load_message = (
                f"视觉情绪分类模型加载失败：{e}，使用 demo 模式。"
            )
            print(self.load_message)

            if not self.allow_demo_fallback:
                raise

    def preprocess_feature(self, feature):
        """
        统一把输入转换成 torch.Tensor [1, 512]。

        支持:
            np.ndarray [512]
            np.ndarray [15, 512]
            torch.Tensor [512]
            torch.Tensor [15, 512]
            torch.Tensor [1, 512]
        """
        if isinstance(feature, torch.Tensor):
            x = feature.detach().float().cpu()
        else:
            x = torch.from_numpy(np.asarray(feature)).float()

        if x.ndim == 1:
            x = x.unsqueeze(0)

        elif x.ndim == 2:
            # 如果是 [15, 512]，取均值变成 [1, 512]
            if x.shape[1] == self.input_dim:
                x = x.mean(dim=0, keepdim=True)
            # 如果已经是 [1, 512]，保持
            elif x.shape[0] == 1 and x.shape[1] == self.input_dim:
                pass
            else:
                raise ValueError(f"视觉特征形状错误：{tuple(x.shape)}")

        elif x.ndim == 3:
            # [B, T, 512] -> [B, 512]
            if x.shape[-1] != self.input_dim:
                raise ValueError(f"视觉特征最后一维应该是 {self.input_dim}，实际 {x.shape[-1]}")
            x = x.mean(dim=1)

        else:
            raise ValueError(f"不支持的视觉特征维度：{tuple(x.shape)}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"视觉特征维度错误：{tuple(x.shape)}，期望最后一维 {self.input_dim}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x.to(self.device)

    @torch.no_grad()
    def predict(self, feature):
        """
        对视觉特征进行情绪预测。

        返回 dict，后面 views.py 可以直接 JsonResponse。
        """
        x = self.preprocess_feature(feature)

        if self.mode == "model":
            return self._predict_by_model(x)

        return self._predict_by_demo_rule(x)

    @torch.no_grad()
    def _predict_by_model(self, x):
        """
        用训练好的模型预测。
        """
        valence_logits, arousal_logits = self.model(x)

        valence_prob = F.softmax(valence_logits, dim=-1)
        arousal_prob = F.softmax(arousal_logits, dim=-1)

        valence_label = int(torch.argmax(valence_prob, dim=-1).item())
        arousal_label = int(torch.argmax(arousal_prob, dim=-1).item())

        valence_conf = float(valence_prob[0, valence_label].item())
        arousal_conf = float(arousal_prob[0, arousal_label].item())

        emotion_info = emotion_from_va(valence_label, arousal_label)

        return {
            "ok": True,
            "mode": "model",
            "model_loaded": True,

            "valence_label": valence_label,
            "arousal_label": arousal_label,

            "valence_text": "高效价/正向" if valence_label == 1 else "低效价/负向",
            "arousal_text": "高唤醒" if arousal_label == 1 else "低唤醒",

            "valence_confidence": round(valence_conf, 4),
            "arousal_confidence": round(arousal_conf, 4),

            "valence_probability": {
                "low": round(float(valence_prob[0, 0].item()), 4),
                "high": round(float(valence_prob[0, 1].item()), 4),
            },
            "arousal_probability": {
                "low": round(float(arousal_prob[0, 0].item()), 4),
                "high": round(float(arousal_prob[0, 1].item()), 4),
            },

            **emotion_info,
        }

    @torch.no_grad()
    def _predict_by_demo_rule(self, x):
        """
        demo 模式。

        说明：
        这不是严谨科研预测，只是为了在没有 visual_emotion_best.pth 的情况下，
        让后端、接口、页面可以先跑通。

        它会根据输入特征生成一个稳定的伪预测：
        同一个特征通常得到同一个结果。
        """

        x_cpu = x.detach().cpu().numpy().astype(np.float32)
        vec = x_cpu.reshape(-1)

        mean_val = float(np.mean(vec))
        std_val = float(np.std(vec))
        l2_val = float(np.linalg.norm(vec) / math.sqrt(len(vec)))

        # 用特征摘要生成稳定 hash，避免每次随机跳变
        digest = hashlib.md5(vec[: min(128, len(vec))].tobytes()).hexdigest()
        hash_int = int(digest[:8], 16)

        score_v = math.tanh(mean_val * 0.8 + std_val * 0.2 + ((hash_int % 100) - 50) / 250.0)
        score_a = math.tanh(l2_val * 0.5 + std_val * 0.3 + (((hash_int // 100) % 100) - 50) / 250.0)

        valence_high_prob = 1.0 / (1.0 + math.exp(-score_v * 2.0))
        arousal_high_prob = 1.0 / (1.0 + math.exp(-score_a * 2.0))

        valence_label = 1 if valence_high_prob >= 0.5 else 0
        arousal_label = 1 if arousal_high_prob >= 0.5 else 0

        emotion_info = emotion_from_va(valence_label, arousal_label)

        return {
            "ok": True,
            "mode": "demo",
            "model_loaded": False,
            "warning": "当前使用 demo 模式。请训练 visual_emotion_best.pth 后再用于真实预测。",

            "valence_label": valence_label,
            "arousal_label": arousal_label,

            "valence_text": "高效价/正向" if valence_label == 1 else "低效价/负向",
            "arousal_text": "高唤醒" if arousal_label == 1 else "低唤醒",

            "valence_confidence": round(max(valence_high_prob, 1 - valence_high_prob), 4),
            "arousal_confidence": round(max(arousal_high_prob, 1 - arousal_high_prob), 4),

            "valence_probability": {
                "low": round(1 - valence_high_prob, 4),
                "high": round(valence_high_prob, 4),
            },
            "arousal_probability": {
                "low": round(1 - arousal_high_prob, 4),
                "high": round(arousal_high_prob, 4),
            },

            "feature_summary": {
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "l2": round(l2_val, 6),
            },

            **emotion_info,
        }
