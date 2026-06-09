import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from django.conf import settings

from .mh_dgfnet_model import MH_DGFNet_Full


_MODEL = None
_SCALER = None
_MODEL_ERROR = None
_LOCK = threading.Lock()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mh_dgfnet():
    global _MODEL, _SCALER, _MODEL_ERROR

    with _LOCK:
        if _MODEL is not None and _SCALER is not None:
            return _MODEL, _SCALER

        try:
            device = get_device()

            model_path = str(settings.MHDGFNET_MODEL_PATH)
            scaler_path = str(settings.MHDGFNET_SCALER_PATH)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在：{model_path}")

            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"标准化参数文件不存在：{scaler_path}")

            checkpoint = torch.load(model_path, map_location=device)
            scaler_checkpoint = torch.load(scaler_path, map_location=device)

            hidden_dim = checkpoint.get("hidden_dim", 64)
            dropout = checkpoint.get("dropout", 0.3)

            model = MH_DGFNet_Full(
                hidden_dim=hidden_dim,
                dropout=dropout
            ).to(device)

            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            scaler_stats = scaler_checkpoint["scaler_stats"]

            scaler_stats = {
                k: (
                    v[0].to(device).float(),
                    v[1].to(device).float()
                )
                for k, v in scaler_stats.items()
            }

            _MODEL = model
            _SCALER = scaler_stats
            _MODEL_ERROR = None

            return _MODEL, _SCALER

        except Exception as e:
            _MODEL = None
            _SCALER = None
            _MODEL_ERROR = str(e)
            raise


def check_mh_dgfnet_status():
    model_path = str(getattr(settings, "MHDGFNET_MODEL_PATH", ""))
    scaler_path = str(getattr(settings, "MHDGFNET_SCALER_PATH", ""))

    try:
        load_mh_dgfnet()
        loaded = True
        error = None
    except Exception as e:
        loaded = False
        error = str(e)

    return {
        "model_name": "MH_DGFNet_Full",
        "model_loaded": loaded,
        "model_error": error,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "scaler_path": scaler_path,
        "scaler_exists": os.path.exists(scaler_path),
        "device": str(get_device()),
    }


def normalize_batch(maps, stats, peri, visual, scaler_stats):
    maps = (maps - scaler_stats["m"][0]) / scaler_stats["m"][1]
    stats = (stats - scaler_stats["s"][0]) / scaler_stats["s"][1]
    peri = (peri - scaler_stats["p"][0]) / scaler_stats["p"][1]
    visual = (visual - scaler_stats["vis"][0]) / scaler_stats["vis"][1]

    return maps, stats, peri, visual


def predict_mh_dgfnet_features(maps, stats, peri, visual):
    """
    输入：
    maps   : N, 5, H, W
    stats  : N, 32, 7
    peri   : N, 55
    visual : N, 512

    N 通常是 15，即一个 trial 的 15 个 segment。
    """

    model, scaler_stats = load_mh_dgfnet()
    device = get_device()

    maps = np.asarray(maps, dtype=np.float32)
    stats = np.asarray(stats, dtype=np.float32)
    peri = np.asarray(peri, dtype=np.float32)
    visual = np.asarray(visual, dtype=np.float32)

    if maps.ndim != 4:
        raise ValueError(f"maps 期望 N,5,H,W，实际 shape={maps.shape}")

    if stats.ndim != 3:
        raise ValueError(f"stats 期望 N,32,7，实际 shape={stats.shape}")

    if peri.ndim != 2:
        raise ValueError(f"peri 期望 N,55，实际 shape={peri.shape}")

    if visual.ndim != 2:
        raise ValueError(f"visual 期望 N,512，实际 shape={visual.shape}")

    n = maps.shape[0]

    if stats.shape[0] != n:
        raise ValueError(f"stats 数量不一致：maps={n}, stats={stats.shape[0]}")

    if peri.shape[0] != n:
        raise ValueError(f"peri 数量不一致：maps={n}, peri={peri.shape[0]}")

    if visual.shape[0] != n:
        raise ValueError(f"visual 数量不一致：maps={n}, visual={visual.shape[0]}")

    if maps.shape[1] != 5:
        raise ValueError(f"maps 第二维必须是 5，实际 shape={maps.shape}")

    if stats.shape[1:] != (32, 7):
        raise ValueError(f"stats 必须是 N,32,7，实际 shape={stats.shape}")

    if peri.shape[1] != 55:
        raise ValueError(f"peri 必须是 N,55，实际 shape={peri.shape}")

    if visual.shape[1] != 512:
        raise ValueError(f"visual 必须是 N,512，实际 shape={visual.shape}")

    maps_t = torch.from_numpy(maps).float().to(device)
    stats_t = torch.from_numpy(stats).float().to(device)
    peri_t = torch.from_numpy(peri).float().to(device)
    visual_t = torch.from_numpy(visual).float().to(device)

    maps_t, stats_t, peri_t, visual_t = normalize_batch(
        maps_t,
        stats_t,
        peri_t,
        visual_t,
        scaler_stats
    )

    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            v_logits, a_logits = model(maps_t, stats_t, peri_t, visual_t)

        v_prob = F.softmax(v_logits.float(), dim=1)
        a_prob = F.softmax(a_logits.float(), dim=1)

    segment_results = []

    for i in range(n):
        v_low = float(v_prob[i, 0].detach().cpu().item())
        v_high = float(v_prob[i, 1].detach().cpu().item())

        a_low = float(a_prob[i, 0].detach().cpu().item())
        a_high = float(a_prob[i, 1].detach().cpu().item())

        v_pred = 1 if v_high >= v_low else 0
        a_pred = 1 if a_high >= a_low else 0

        segment_results.append({
            "segment_index": i,
            "valence_class": v_pred,
            "arousal_class": a_pred,
            "valence_label": "high" if v_pred == 1 else "low",
            "arousal_label": "high" if a_pred == 1 else "low",
            "valence_low_prob": round(v_low, 4),
            "valence_high_prob": round(v_high, 4),
            "arousal_low_prob": round(a_low, 4),
            "arousal_high_prob": round(a_high, 4),
            "valence_score": round(v_high * 2 - 1, 4),
            "arousal_score": round(a_high * 2 - 1, 4),
            "confidence": round(max(v_low, v_high) * max(a_low, a_high), 4),
        })

    summary = summarize_trial(segment_results)

    return {
        "ok": True,
        "model": "MH_DGFNet_Full",
        "num_segments": n,
        "segments": segment_results,
        "summary": summary,
    }


def summarize_trial(segment_results):
    """
    模仿 trial-level 思路：多个 segment 聚合。
    这里用概率/score 平均。
    """

    valence_score = float(np.mean([x["valence_score"] for x in segment_results]))
    arousal_score = float(np.mean([x["arousal_score"] for x in segment_results]))
    confidence = float(np.mean([x["confidence"] for x in segment_results]))

    valence_label = "high" if valence_score >= 0 else "low"
    arousal_label = "high" if arousal_score >= 0 else "low"

    emotion = classify_va(valence_score, arousal_score)

    return {
        "valence_score": round(valence_score, 4),
        "arousal_score": round(arousal_score, 4),
        "valence_label": valence_label,
        "arousal_label": arousal_label,
        "confidence": round(confidence, 4),
        "emotion_label": emotion["label"],
        "emotion_type": emotion["type"],
        "color": emotion["color"],
    }


def classify_va(valence, arousal):
    if valence >= 0 and arousal >= 0:
        return {
            "label": "Excited",
            "type": "高效价 / 高唤醒",
            "color": "#ffb020",
        }

    if valence >= 0 and arousal < 0:
        return {
            "label": "Calm",
            "type": "高效价 / 低唤醒",
            "color": "#28c76f",
        }

    if valence < 0 and arousal >= 0:
        return {
            "label": "Anxious",
            "type": "低效价 / 高唤醒",
            "color": "#ea5455",
        }

    return {
        "label": "Sad",
        "type": "低效价 / 低唤醒",
        "color": "#7367f0",
    }
def predict_multimodal_features(maps, stats, peri, visual):
    return predict_mh_dgfnet_features(
        maps=maps,
        stats=stats,
        peri=peri,
        visual=visual,
    )


def check_model_status():
    return check_mh_dgfnet_status()
