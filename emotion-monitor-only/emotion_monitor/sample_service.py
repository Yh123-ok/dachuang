# -*- coding: utf-8 -*-

import os
import re
import glob
import random

import numpy as np
from django.conf import settings

from .visual_classifier import VisualEmotionPredictor, emotion_from_va


def normalize_subject_id(subject):
    """
    统一被试编号格式。

    支持:
        22
        "22"
        "s22"
        "S22"

    返回:
        "s22"
    """
    if isinstance(subject, int):
        return f"s{subject:02d}"

    text = str(subject).strip().lower()

    if text.startswith("s"):
        num = int(text[1:])
    else:
        num = int(text)

    return f"s{num:02d}"


def normalize_trial_id(trial):
    """
    统一 trial 编号。

    支持:
        1
        "1"
        "01"
        "trial01"

    返回:
        1
    """
    text = str(trial).strip().lower()

    m = re.search(r"(\d+)", text)
    if not m:
        raise ValueError(f"无法识别 trial 编号：{trial}")

    trial_id = int(m.group(1))

    if not (1 <= trial_id <= 40):
        raise ValueError(f"trial 编号必须在 1~40 之间，当前：{trial_id}")

    return trial_id


def make_feature_filename(subject, trial):
    """
    生成视觉特征文件名。
    """
    sid = normalize_subject_id(subject)
    trial_id = normalize_trial_id(trial)
    return f"{sid}_trial{trial_id:02d}_features.npy"


def get_feature_path(subject, trial):
    """
    获取视觉特征文件完整路径。
    """
    filename = make_feature_filename(subject, trial)
    return os.path.join(str(settings.VISUAL_FEATURE_DIR), filename)


def list_available_samples():
    """
    扫描 VISUAL_FEATURE_DIR 下所有可用样本。

    文件名格式:
        s22_trial01_features.npy

    返回:
        [
            {
                "subject": "s22",
                "trial": 1,
                "filename": "...",
                "path": "..."
            }
        ]
    """
    feature_dir = str(settings.VISUAL_FEATURE_DIR)

    pattern = os.path.join(feature_dir, "s*_trial*_features.npy")
    paths = sorted(glob.glob(pattern))

    samples = []

    for path in paths:
        name = os.path.basename(path).lower()

        m = re.match(r"(s\d{2})_trial(\d{2})_features\.npy$", name)
        if not m:
            continue

        sid = m.group(1)
        trial_id = int(m.group(2))

        samples.append({
            "subject": sid,
            "trial": trial_id,
            "filename": os.path.basename(path),
            "path": path,
        })

    return samples


def load_visual_feature(subject, trial):
    """
    读取一个样本的视觉特征。

    期望 shape:
        (15, 512)

    也兼容:
        (512,)
        (1, 512)
    """
    path = get_feature_path(subject, trial)

    if not os.path.exists(path):
        raise FileNotFoundError(f"视觉特征文件不存在：{path}")

    arr = np.load(path).astype(np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != settings.VISUAL_OUTPUT_DIM:
            raise ValueError(f"视觉特征维度错误：{arr.shape}")
        arr = arr[None, :]

    elif arr.ndim == 2:
        if arr.shape[1] != settings.VISUAL_OUTPUT_DIM:
            raise ValueError(f"视觉特征维度错误：{arr.shape}")

    else:
        raise ValueError(f"不支持的视觉特征形状：{arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return arr, path


def try_load_deap_label(subject, trial):
    """
    尝试读取 DEAP matlab 标签。

    DEAP matlab 文件通常是:
        s01.mat
        s02.mat
        ...
        s32.mat

    里面 labels shape 通常是:
        (40, 4)

    4 个维度一般是:
        valence, arousal, dominance, liking

    注意:
        这个函数读取失败不会影响演示，只返回 None。
    """
    sid = normalize_subject_id(subject)
    trial_id = normalize_trial_id(trial)

    mat_dir = getattr(settings, "DEAP_MAT_DIR", None)

    if not mat_dir:
        return None

    mat_path = os.path.join(str(mat_dir), f"{sid}.mat")

    if not os.path.exists(mat_path):
        return None

    try:
        from scipy.io import loadmat
    except Exception:
        return {
            "available": False,
            "reason": "未安装 scipy，无法读取 .mat 标签文件。",
            "mat_path": mat_path,
        }

    try:
        mat = loadmat(mat_path)

        if "labels" not in mat:
            return {
                "available": False,
                "reason": ".mat 文件中没有 labels 字段。",
                "mat_path": mat_path,
            }

        labels = mat["labels"]

        if labels.shape[0] < trial_id:
            return {
                "available": False,
                "reason": f"labels 只有 {labels.shape[0]} 条，无法读取 trial {trial_id}。",
                "mat_path": mat_path,
            }

        row = labels[trial_id - 1]

        valence_score = float(row[0])
        arousal_score = float(row[1])

        valence_label = 1 if valence_score >= 5.0 else 0
        arousal_label = 1 if arousal_score >= 5.0 else 0

        emotion_info = emotion_from_va(valence_label, arousal_label)

        return {
            "available": True,
            "mat_path": mat_path,
            "valence_score": round(valence_score, 4),
            "arousal_score": round(arousal_score, 4),
            "valence_label": valence_label,
            "arousal_label": arousal_label,
            "valence_text": "高效价/正向" if valence_label == 1 else "低效价/负向",
            "arousal_text": "高唤醒" if arousal_label == 1 else "低唤醒",
            "emotion": emotion_info["emotion"],
            "emotion_en": emotion_info["emotion_en"],
            "quadrant": emotion_info["quadrant"],
        }

    except Exception as e:
        return {
            "available": False,
            "reason": str(e),
            "mat_path": mat_path,
        }


class SampleDemoService:
    """
    数据集样本演示服务。

    作用:
        1. 扫描有哪些 .npy 视觉样本
        2. 读取指定样本
        3. 调用视觉情绪分类器
        4. 返回预测结果和可选真实标签
    """

    def __init__(self):
        self.predictor = VisualEmotionPredictor(
            model_path=getattr(settings, "VISUAL_EMOTION_MODEL_PATH", None),
            input_dim=getattr(settings, "VISUAL_OUTPUT_DIM", 512),
            allow_demo_fallback=True,
        )

    def list_samples(self):
        """
        返回所有可用样本。
        """
        samples = list_available_samples()

        return {
            "ok": True,
            "feature_dir": str(settings.VISUAL_FEATURE_DIR),
            "count": len(samples),
            "samples": [
                {
                    "subject": item["subject"],
                    "trial": item["trial"],
                    "filename": item["filename"],
                }
                for item in samples
            ],
        }

    def predict_sample(self, subject=None, trial=None, random_sample=False):
        """
        预测指定样本。

        如果 random_sample=True，则随机选择一个可用样本。
        如果 subject/trial 为空，也会自动随机选择。
        """
        samples = list_available_samples()

        if len(samples) == 0:
            return {
                "ok": False,
                "error": "没有找到任何视觉特征样本。",
                "feature_dir": str(settings.VISUAL_FEATURE_DIR),
                "expected_filename": "s22_trial01_features.npy",
            }

        if random_sample or subject is None or trial is None:
            sample = random.choice(samples)
            sid = sample["subject"]
            trial_id = sample["trial"]
        else:
            sid = normalize_subject_id(subject)
            trial_id = normalize_trial_id(trial)

        feature, feature_path = load_visual_feature(sid, trial_id)

        prediction = self.predictor.predict(feature)

        label_info = try_load_deap_label(sid, trial_id)

        return {
            "ok": True,
            "demo_type": "dataset_sample",
            "subject": sid,
            "trial": trial_id,
            "feature_path": feature_path,
            "feature_shape": list(feature.shape),
            "prediction": prediction,
            "ground_truth": label_info,
        }
