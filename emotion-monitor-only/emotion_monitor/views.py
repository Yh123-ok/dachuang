import os
import io
import time
import random
import pickle
import base64
import tempfile

import numpy as np


from scipy.signal import welch
from scipy.integrate import trapezoid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView

from .mh_dgfnet_infer import (
    check_mh_dgfnet_status,
    predict_mh_dgfnet_features,
)

from .feature_adapter import load_deap_style_trial_features


# ============================================================
# 页面视图
# ============================================================

class EmotionMonitorView(TemplateView):
    template_name = "emotion_monitor/emotion_monitor.html"


def model_status(request):
    return JsonResponse(
        check_mh_dgfnet_status(),
        json_dumps_params={"ensure_ascii": False}
    )


# ============================================================
# DEAP 原始数据信息
# ============================================================

DEAP_EEG_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"
]

DEAP_PERIPHERAL_CHANNELS = [
    "hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Resp", "Pleth", "Temp"
]

EEG_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45),
}


# ============================================================
# 通用工具
# ============================================================

def get_uploaded_file_flexible(request, field_names=None, allowed_exts=None):
    """
    更稳健地从 request.FILES 中获取上传文件。

    优先按 field_names 获取；
    如果取不到，则按文件扩展名 allowed_exts 自动搜索。

    用于避免前端 FormData 字段名不一致导致后端拿不到文件。
    """
    field_names = field_names or []
    allowed_exts = allowed_exts or []

    # 1. 优先按字段名查找
    for name in field_names:
        uploaded_file = request.FILES.get(name)
        if uploaded_file:
            return uploaded_file

    # 2. 如果按字段名找不到，则按扩展名兜底查找
    if allowed_exts:
        allowed_exts = [ext.lower() for ext in allowed_exts]

        for key in request.FILES.keys():
            uploaded_file = request.FILES.get(key)
            if not uploaded_file:
                continue

            ext = os.path.splitext(uploaded_file.name)[1].lower()

            if ext in allowed_exts:
                return uploaded_file

    return None


def build_upload_debug_info(request):
    """
    构建上传调试信息。
    当前端提示缺少文件时，可以看到实际收到的 FILES / POST。
    """
    return {
        "received_file_fields": list(request.FILES.keys()),
        "received_files": [
            {
                "field": key,
                "name": request.FILES[key].name,
                "size": request.FILES[key].size,
                "content_type": getattr(request.FILES[key], "content_type", "")
            }
            for key in request.FILES.keys()
        ],
        "received_post_fields": list(request.POST.keys()),
        "post_data": {
            key: request.POST.get(key)
            for key in request.POST.keys()
        },
        "content_type": request.META.get("CONTENT_TYPE", ""),
        "method": request.method,
    }


# ============================================================
# 原始 DEAP .dat 读取与可视化数据构建
# ============================================================

def load_deap_raw_dat(file_path):
    """
    读取 DEAP 原始 .dat 文件。

    DEAP 原始数据通常结构：
    data.shape   = [40, 40, 8064]
    labels.shape = [40, 4]

    其中：
    - 40 个 trial
    - 40 个通道
    - 前 32 个通道为 EEG
    - 后 8 个通道为外周生理信号
    - 采样率一般为 128 Hz
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if not isinstance(obj, dict):
        raise ValueError("DEAP .dat 文件格式错误：pickle 内容不是 dict")

    if "data" not in obj:
        raise ValueError("DEAP .dat 文件格式错误：缺少 data 字段")

    data = obj["data"]
    labels = obj.get("labels")

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(f"DEAP data 维度错误，期望 [40, 40, n_times]，实际 shape={data.shape}")

    if data.shape[0] < 1 or data.shape[1] < 40:
        raise ValueError(f"DEAP data shape 异常，期望至少 [trial, 40, n_times]，实际 shape={data.shape}")

    return data, labels


def downsample_1d_for_preview(x, target_len=300):
    """
    将一维原始信号压缩到前端适合展示的长度。
    这里只是为了显示，不改变原始数据含义。
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size == 0:
        return [0.0] * target_len

    if x.size == 1:
        return [round(float(x[0]), 4)] * target_len

    old_idx = np.linspace(0, 1, x.size)
    new_idx = np.linspace(0, 1, target_len)

    y = np.interp(new_idx, old_idx, x)

    return [round(float(v), 4) for v in y]


def normalize_for_display(x):
    """
    仅用于前端显示的标准化。

    注意：
    这不是模型预处理，只是为了让多个通道能在一张图里看清楚。
    如果你想展示真实幅值，可以去掉这个函数。
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(x)
    std = np.std(x)

    if std > 1e-6:
        x = (x - mean) / std
    else:
        x = x - mean

    x = np.clip(x, -5, 5)

    return x


def build_raw_signal_preview_from_deap_trial(trial_data, fs=128):
    """
    基于一个 DEAP trial 的原始数据，构建前端 EEG / 外周波形预览。

    trial_data.shape = [40, 8064]

    前 32 通道 EEG：
        trial_data[:32]

    后 8 通道外周：
        trial_data[32:]
    """
    trial_data = np.asarray(trial_data, dtype=np.float32)

    if trial_data.ndim != 2:
        raise ValueError(f"trial_data 维度错误，期望 [40, n_times]，实际 shape={trial_data.shape}")

    if trial_data.shape[0] < 40:
        raise ValueError(f"trial_data 通道数不足，期望至少 40，实际 shape={trial_data.shape}")

    eeg = trial_data[:32]
    peri = trial_data[32:40]

    # 选几个代表性 EEG 通道展示
    # 0: Fp1, 2: F3, 19: F4, 23: Cz
    eeg_channel_indices = [0, 2, 19, 23]

    eeg_preview = []

    for idx in eeg_channel_indices:
        ch_name = DEAP_EEG_CHANNELS[idx]

        # 用标准化后的原始信号做展示
        y = normalize_for_display(eeg[idx])

        eeg_preview.append({
            "name": ch_name,
            "data": downsample_1d_for_preview(y, 300)
        })

    # 外周信号展示：
    # DEAP 后 8 通道顺序：
    # hEOG, vEOG, zEMG, tEMG, GSR, Resp, Pleth, Temp
    peripheral_map = {
        "GSR": 4,
        "Resp": 5,
        "Pleth": 6,
    }

    peripheral_preview = []

    for name, idx in peripheral_map.items():
        y = normalize_for_display(peri[idx])

        peripheral_preview.append({
            "name": name,
            "data": downsample_1d_for_preview(y, 300)
        })

    return {
        "eeg_raw_preview": eeg_preview,
        "peripheral_raw_preview": peripheral_preview,
        "fs": fs,
        "source": "raw_deap",
        "display": "normalized_raw_signal"
    }


def compute_band_power_per_channel(eeg_data, fs=128, band=(8, 13)):
    """
    基于原始 EEG 计算每个通道在指定频段的功率。

    eeg_data.shape = [32, n_times]
    """
    eeg_data = np.asarray(eeg_data, dtype=np.float32)
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)

    low, high = band
    powers = []

    for ch in range(eeg_data.shape[0]):
        freqs, psd = welch(
            eeg_data[ch],
            fs=fs,
            nperseg=min(256, eeg_data.shape[1])
        )

        mask = (freqs >= low) & (freqs <= high)

        if np.any(mask):
            power = trapezoid(psd[mask], freqs[mask])
        else:
            power = 0.0

        powers.append(float(power))

    powers = np.asarray(powers, dtype=np.float32)

    # log 压缩，避免极端值导致脑地形图颜色过于集中
    powers = np.log1p(np.maximum(powers, 0))

    return powers

def topomap_png_base64_from_raw_eeg(eeg_data, fs=128):
    """
    基于原始 EEG 生成多个频段的脑地形图。

    返回格式：
    {
        "bands": [
            {
                "name": "Alpha",
                "image": "data:image/png;base64,..."
            },
            ...
        ],
        "source": "raw_eeg_bandpower"
    }
    """
    eeg_data = np.asarray(eeg_data, dtype=np.float32)

    if eeg_data.ndim != 2:
        raise ValueError(f"eeg_data 维度错误，期望 [32, n_times]，实际 shape={eeg_data.shape}")

    if eeg_data.shape[0] != 32:
        raise ValueError(f"eeg_data 通道数错误，期望 32，实际 shape={eeg_data.shape}")

    info = mne.create_info(
        ch_names=DEAP_EEG_CHANNELS,
        sfreq=fs,
        ch_types="eeg"
    )

    # 使用标准 10-20 系统电极位置
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")

    bands = []

    for band_name, band_range in EEG_BANDS.items():
        values = compute_band_power_per_channel(
            eeg_data=eeg_data,
            fs=fs,
            band=band_range
        )

        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)

        mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=6,
            sensors=True
        )

        ax.set_title(band_name, fontsize=10)

        buffer = io.BytesIO()

        fig.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.05
        )

        plt.close(fig)

        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        bands.append({
            "name": band_name,
            "image": "data:image/png;base64," + img_base64
        })

    return {
        "bands": bands,
        "source": "raw_eeg_bandpower",
        "method": "Welch PSD + MNE standard_1020 montage"
    }


def build_raw_dashboard_from_uploaded_file(raw_file, trial_index):
    """
    从上传的原始数据文件中构建原始信号可视化内容。

    当前支持：
    - DEAP 原始 .dat

    返回：
    {
        "raw_signals": {
            "eeg_raw_preview": [...],
            "peripheral_raw_preview": [...]
        },
        "raw_topomap": {
            "bands": [...]
        }
    }
    """
    if raw_file is None:
        return None

    ext = os.path.splitext(raw_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        for chunk in raw_file.chunks():
            tmp.write(chunk)

        tmp_path = tmp.name

    try:
        if ext != ".dat":
            raise ValueError(
                f"暂不支持该原始数据格式：{ext}，当前请上传 DEAP 原始 .dat 文件"
            )

        data, labels = load_deap_raw_dat(tmp_path)

        trial_zero_index = int(trial_index) - 1

        if trial_zero_index < 0 or trial_zero_index >= data.shape[0]:
            raise ValueError(
                f"trial_index 超出原始数据 trial 范围，当前 trial_index={trial_index}，原始数据 trial 数={data.shape[0]}"
            )

        trial_data = data[trial_zero_index]  # [40, 8064]

        raw_signals = build_raw_signal_preview_from_deap_trial(
            trial_data=trial_data,
            fs=128
        )

        eeg_raw = trial_data[:32]

        raw_topomap = topomap_png_base64_from_raw_eeg(
            eeg_data=eeg_raw,
            fs=128
        )

        return {
            "raw_signals": raw_signals,
            "raw_topomap": raw_topomap
        }

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ============================================================
# 预测结果转换为前端 dashboard
# ============================================================

def normalize_va_for_display(valence, arousal):
    """
    自动判断 Valence/Arousal 是 1~9 还是 -1~1。

    返回统一到 -1~1 的前端显示值。
    """
    valence = float(valence)
    arousal = float(arousal)

    if 1.0 <= valence <= 9.0 and 1.0 <= arousal <= 9.0:
        scale = "1-9"

        valence_display = (valence - 5.0) / 4.0
        arousal_display = (arousal - 5.0) / 4.0

    else:
        scale = "-1-1"

        valence_display = valence
        arousal_display = arousal

    valence_display = max(-1.0, min(1.0, valence_display))
    arousal_display = max(-1.0, min(1.0, arousal_display))

    return {
        "scale": scale,
        "valence_raw": valence,
        "arousal_raw": arousal,
        "valence_display": valence_display,
        "arousal_display": arousal_display,
    }


def get_emotion_from_va(valence_display, arousal_display):
    """
    基于 -1~1 的 VA 空间判断情绪象限。
    """
    if valence_display >= 0 and arousal_display >= 0:
        return {
            "emotion_label": "Excited",
            "emotion_type": "高效价 / 高唤醒",
            "color": "#ff9f43",
        }

    elif valence_display >= 0 and arousal_display < 0:
        return {
            "emotion_label": "Calm",
            "emotion_type": "高效价 / 低唤醒",
            "color": "#28c76f",
        }

    elif valence_display < 0 and arousal_display >= 0:
        return {
            "emotion_label": "Tense",
            "emotion_type": "低效价 / 高唤醒",
            "color": "#ea5455",
        }

    else:
        return {
            "emotion_label": "Sad",
            "emotion_type": "低效价 / 低唤醒",
            "color": "#7367f0",
        }


def build_dashboard_payload(prediction):
    """
    把 MH-DGFNet 预测结果转换成前端大屏展示字段。
    """
    summary = prediction.get("summary") or {}

    valence = float(summary.get("valence_score", 0.0))
    arousal = float(summary.get("arousal_score", 0.0))

    va_info = normalize_va_for_display(valence, arousal)

    valence_display = va_info["valence_display"]
    arousal_display = va_info["arousal_display"]

    emotion_info = get_emotion_from_va(valence_display, arousal_display)

    distance = (valence_display ** 2 + arousal_display ** 2) ** 0.5
    confidence = min(1.0, max(0.35, distance / 1.414))

    return {
        "current_emotion": {
            "emotion_label": emotion_info["emotion_label"],
            "emotion_type": emotion_info["emotion_type"],
            "valence": valence_display,
            "arousal": arousal_display,
            "valence_raw": va_info["valence_raw"],
            "arousal_raw": va_info["arousal_raw"],
            "va_scale": va_info["scale"],
            "confidence": confidence,
            "color": emotion_info["color"],
        },
        "face": {
            "expression": "Visual Feature",
            "quality": 95,
            "detected": True,
        },
        "peripheral": {
            "hr": "--",
            "hrv": "--",
            "gsr": "--",
            "resp": "--",
        },
        "model": {
            "version": "MH-DGFNet-Full",
        },
    }


def build_dashboard_from_prediction(prediction):
    """
    兼容旧接口 predict_deap_separated_upload 使用。
    """
    summary = prediction.get("summary") or {}

    valence = float(summary.get("valence_score", 0.0))
    arousal = float(summary.get("arousal_score", 0.0))
    confidence = float(summary.get("confidence", 0.0))

    return {
        "current_emotion": {
            "emotion_label": summary.get("emotion_label", "Unknown"),
            "emotion_type": summary.get("emotion_type", "--"),
            "valence": valence,
            "arousal": arousal,
            "confidence": confidence,
            "color": summary.get("color", "#7367f0"),
        },
        "face": {
            "expression": f"V:{summary.get('valence_label', '--')} / A:{summary.get('arousal_label', '--')}",
            "quality": random.randint(88, 98),
            "detected": True,
        },
        "peripheral": {
            "hr": random.randint(68, 92),
            "hrv": random.randint(38, 65),
            "gsr": round(random.uniform(3.5, 6.2), 1),
            "resp": random.randint(13, 19),
        },
        "model": {
            "version": "MH-DGFNet-Full",
        },
    }


# ============================================================
# 旧接口：subject_npz + visual_npy + trial_no
# ============================================================

@csrf_exempt
def predict_deap_separated_upload(request):
    """
    旧接口：
    1. subject_npz
    2. visual_npy
    3. trial_no
    """

    if request.method != "POST":
        return JsonResponse(
            {
                "ok": False,
                "error": "请使用 POST 请求",
            },
            json_dumps_params={"ensure_ascii": False}
        )

    subject_npz = request.FILES.get("subject_npz")
    visual_npy = request.FILES.get("visual_npy")
    trial_no = request.POST.get("trial_no", "1")

    if not subject_npz:
        return JsonResponse(
            {
                "ok": False,
                "error": "缺少 subject_npz 文件",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    if not visual_npy:
        return JsonResponse(
            {
                "ok": False,
                "error": "缺少 visual_npy 文件",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    upload_dir = os.path.join(settings.BASE_DIR, "media", "emotion_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    ts = int(time.time())

    subject_npz_path = os.path.join(
        upload_dir,
        f"{ts}_{subject_npz.name}"
    )

    visual_npy_path = os.path.join(
        upload_dir,
        f"{ts}_{visual_npy.name}"
    )

    with open(subject_npz_path, "wb+") as f:
        for chunk in subject_npz.chunks():
            f.write(chunk)

    with open(visual_npy_path, "wb+") as f:
        for chunk in visual_npy.chunks():
            f.write(chunk)

    try:
        features = load_deap_style_trial_features(
            npz_path=subject_npz_path,
            visual_npy_path=visual_npy_path,
            trial_no=trial_no,
        )

        prediction = predict_mh_dgfnet_features(
            maps=features["maps"],
            stats=features["stats"],
            peri=features["peri"],
            visual=features["visual"],
        )

        dashboard = build_dashboard_from_prediction(prediction)

        return JsonResponse(
            {
                "ok": True,
                "mode": "deap_separated_upload",
                "input_meta": features["meta"],
                "prediction": prediction,
                "dashboard": dashboard,
            },
            json_dumps_params={"ensure_ascii": False}
        )

    except Exception as e:
        return JsonResponse(
            {
                "ok": False,
                "error": str(e),
            },
            json_dumps_params={"ensure_ascii": False}
        )


# ============================================================
# 当前主接口：feature_npz + visual_npy + raw_data_file + trial_index
# ============================================================

@csrf_exempt
def predict_trial_separated(request):
    """
    当前主接口。

    上传字段：

    feature_npz:
        处理后的 EEG/外周特征，用于模型推理。
        包含：
        - eeg_allband_feature_map
        - eeg_en_stat
        - peri_feature

    visual_npy:
        当前 trial 对应的视觉特征文件，例如：
        s01_trial01_features.npy

    raw_data_file:
        原始 DEAP .dat 文件。
        用于：
        - 原始 EEG 动态预览
        - 原始外周生理动态预览
        - 原始 EEG 脑地形图

    trial_index:
        1~40
    """

    if request.method != "POST":
        return JsonResponse(
            {
                "ok": False,
                "error": "请使用 POST 请求",
            },
            json_dumps_params={"ensure_ascii": False}
        )

    # 调试输出：可以在 Django 控制台看到实际收到的上传字段
    print("========== predict_trial_separated upload debug ==========")
    print("FILES keys:", list(request.FILES.keys()))
    print("POST keys:", list(request.POST.keys()))
    print("CONTENT_TYPE:", request.META.get("CONTENT_TYPE", ""))
    for key in request.FILES.keys():
        f = request.FILES[key]
        print(
            f"FILE field={key}, name={f.name}, size={f.size}, content_type={getattr(f, 'content_type', '')}"
        )
    print("=========================================================")

    feature_npz = get_uploaded_file_flexible(
        request,
        field_names=[
            "feature_npz",
            "featureNpz",
            "feature_file",
            "subject_npz",
            "featureNpzInput",
        ],
        allowed_exts=[".npz"]
    )

    visual_npy = get_uploaded_file_flexible(
        request,
        field_names=[
            "visual_npy",
            "visualNpy",
            "visual_file",
            "visualNpyInput",
        ],
        allowed_exts=[".npy"]
    )

    raw_file = get_uploaded_file_flexible(
        request,
        field_names=[
            "raw_data_file",
            "raw_file",
            "rawDataFile",
            "raw_dat",
            "rawDataInput",
        ],
        allowed_exts=[".dat"]
    )

    trial_index = (
        request.POST.get("trial_index")
        or request.POST.get("trialIndex")
        or request.POST.get("trial_no")
        or request.POST.get("trialNo")
        or "1"
    )

    if not feature_npz:
        return JsonResponse(
            {
                "ok": False,
                "error": "缺少 feature_npz 文件",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    if not visual_npy:
        return JsonResponse(
            {
                "ok": False,
                "error": "缺少 visual_npy 文件",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    if not raw_file:
        return JsonResponse(
            {
                "ok": False,
                "error": "缺少 raw_data_file 原始数据文件。若要基于原始数据展示 EEG/外周/脑地形图，必须上传 DEAP 原始 .dat 文件。",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    raw_ext = os.path.splitext(raw_file.name)[1].lower()
    if raw_ext != ".dat":
        return JsonResponse(
            {
                "ok": False,
                "error": f"原始数据文件格式错误：当前为 {raw_ext}，必须上传 DEAP 原始 .dat 文件。",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    try:
        trial_index = int(trial_index)
    except ValueError:
        return JsonResponse(
            {
                "ok": False,
                "error": "trial_index 必须是整数，范围 1~40",
                "debug": build_upload_debug_info(request),
            },
            json_dumps_params={"ensure_ascii": False}
        )

    if trial_index < 1 or trial_index > 40:
        return JsonResponse(
            {
                "ok": False,
                "error": "trial_index 范围必须是 1~40",
                "debug": {
                    "trial_index": trial_index,
                    **build_upload_debug_info(request),
                },
            },
            json_dumps_params={"ensure_ascii": False}
        )

    upload_dir = os.path.join(settings.BASE_DIR, "media", "emotion_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    timestamp = int(time.time())

    npz_path = os.path.join(
        upload_dir,
        f"{timestamp}_{feature_npz.name}"
    )

    npy_path = os.path.join(
        upload_dir,
        f"{timestamp}_{visual_npy.name}"
    )

    with open(npz_path, "wb+") as f:
        for chunk in feature_npz.chunks():
            f.write(chunk)

    with open(npy_path, "wb+") as f:
        for chunk in visual_npy.chunks():
            f.write(chunk)

    try:
        # ----------------------------------------------------
        # 1. 读取处理后的特征，用于模型推理
        # ----------------------------------------------------
        with np.load(npz_path) as d:
            required_keys = [
                "eeg_allband_feature_map",
                "eeg_en_stat",
                "peri_feature",
            ]

            missing_keys = [k for k in required_keys if k not in d.files]

            if missing_keys:
                return JsonResponse(
                    {
                        "ok": False,
                        "error": f"npz 缺少字段：{missing_keys}",
                        "actual_keys": list(d.files),
                    },
                    json_dumps_params={"ensure_ascii": False}
                )

            maps_all = d["eeg_allband_feature_map"]
            stats_all = d["eeg_en_stat"]
            peri_all = d["peri_feature"]

        visual_all = np.load(npy_path)

        segments_per_trial = 15
        trial_idx = trial_index - 1

        total_segments = maps_all.shape[0]

        if total_segments % segments_per_trial != 0:
            return JsonResponse(
                {
                    "ok": False,
                    "error": (
                        f"特征段数无法按每 trial {segments_per_trial} 段整除，"
                        f"当前 eeg_allband_feature_map 第一维={total_segments}"
                    ),
                },
                json_dumps_params={"ensure_ascii": False}
            )

        inferred_trials = total_segments // segments_per_trial

        if trial_idx < 0 or trial_idx >= inferred_trials:
            return JsonResponse(
                {
                    "ok": False,
                    "error": (
                        f"trial_index 超出特征 NPZ trial 范围，"
                        f"当前 trial_index={trial_index}，NPZ 推断 trial 数={inferred_trials}"
                    ),
                },
                json_dumps_params={"ensure_ascii": False}
            )

        maps_all = maps_all.reshape(
            inferred_trials,
            segments_per_trial,
            *maps_all.shape[1:]
        )

        stats_all = stats_all.reshape(
            inferred_trials,
            segments_per_trial,
            *stats_all.shape[1:]
        )

        peri_all = peri_all.reshape(
            inferred_trials,
            segments_per_trial,
            *peri_all.shape[1:]
        )

        maps = maps_all[trial_idx, -segments_per_trial:, ...].astype(np.float32)
        stats = stats_all[trial_idx, -segments_per_trial:, ...].astype(np.float32)
        peri = peri_all[trial_idx, -segments_per_trial:, ...].astype(np.float32)

        stats = stats.reshape(segments_per_trial, 32, 7)

        visual = visual_all

        if visual.ndim == 1:
            visual = np.tile(visual, (segments_per_trial, 1))

        if visual.shape[-1] != 512:
            return JsonResponse(
                {
                    "ok": False,
                    "error": f"视觉特征维度错误，期望最后一维 512，实际 shape={visual.shape}",
                },
                json_dumps_params={"ensure_ascii": False}
            )

        if visual.shape[0] != segments_per_trial:
            visual = np.vstack([
                np.mean(x, axis=0)
                for x in np.array_split(visual, segments_per_trial)
            ])

        visual = visual[-segments_per_trial:, :].astype(np.float32)

        # ----------------------------------------------------
        # 2. 调用 MH-DGFNet 模型预测
        # ----------------------------------------------------
        prediction = predict_mh_dgfnet_features(
            maps=maps,
            stats=stats,
            peri=peri,
            visual=visual,
        )

        dashboard = build_dashboard_payload(prediction)

        # ----------------------------------------------------
        # 3. 基于原始 DEAP .dat 构建可视化数据
        # ----------------------------------------------------
        raw_dashboard = build_raw_dashboard_from_uploaded_file(
            raw_file=raw_file,
            trial_index=trial_index
        )

        if raw_dashboard:
            dashboard.update(raw_dashboard)

        # ----------------------------------------------------
        # 4. 返回前端
        # ----------------------------------------------------
        return JsonResponse(
            {
                "ok": True,
                "mode": "trial_separated",
                "trial_index": trial_index,
                "input_files": {
                    "feature_npz": feature_npz.name,
                    "visual_npy": visual_npy.name,
                    "raw_data_file": raw_file.name,
                },
                "input_shapes": {
                    "maps": list(maps.shape),
                    "stats": list(stats.shape),
                    "peri": list(peri.shape),
                    "visual": list(visual.shape),
                },
                "raw_data": {
                    "enabled": True,
                    "filename": raw_file.name,
                    "source": "DEAP .dat",
                    "visualization": [
                        "raw_eeg_preview",
                        "raw_peripheral_preview",
                        "raw_eeg_topomap"
                    ]
                },
                "prediction": prediction,
                "dashboard": dashboard,
            },
            json_dumps_params={"ensure_ascii": False}
        )

    except Exception as e:
        return JsonResponse(
            {
                "ok": False,
                "error": str(e),
                "debug": {
                    "feature_npz": feature_npz.name if feature_npz else None,
                    "visual_npy": visual_npy.name if visual_npy else None,
                    "raw_data_file": raw_file.name if raw_file else None,
                    "trial_index": trial_index,
                }
            },
            json_dumps_params={"ensure_ascii": False}
        )


# ============================================================
# 默认大屏数据
# ============================================================

def dashboard_data(request):
    """
    没上传数据时，大屏先用默认数据。
    """
    data = {
        "ok": True,
        "timestamp": int(time.time()),
        "current_emotion": {
            "emotion_label": "Waiting",
            "emotion_type": "等待三模态输入",
            "valence": 0.0,
            "arousal": 0.0,
            "confidence": 0.0,
            "color": "#7367f0",
        },
        "face": {
            "expression": "Waiting Visual Feature",
            "quality": 0,
            "detected": False,
        },
        "peripheral": {
            "hr": "--",
            "hrv": "--",
            "gsr": "--",
            "resp": "--",
        },
        "model": {
            "version": "MH-DGFNet-Full",
        },
    }

    return JsonResponse(data, json_dumps_params={"ensure_ascii": False})
