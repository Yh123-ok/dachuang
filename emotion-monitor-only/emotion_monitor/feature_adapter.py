import numpy as np
from django.conf import settings


def load_deap_style_trial_features(npz_path, visual_npy_path, trial_no):
    """
    按你的训练代码的数据格式解析输入。

    npz_path:
        一个 subject 的 npz 文件，里面包含：
        - eeg_allband_feature_map
        - eeg_en_stat
        - peri_feature

    visual_npy_path:
        一个 trial 的视觉特征 npy 文件，例如：
        s01_trial01_features.npy

    trial_no:
        1 到 40
    """

    trial_no = int(trial_no)

    if trial_no < 1 or trial_no > 40:
        raise ValueError(f"trial_no 必须在 1 到 40 之间，当前是 {trial_no}")

    segments_per_trial = int(getattr(settings, "MHDGFNET_SEGMENTS_PER_TRIAL", 15))
    keep_last = int(getattr(settings, "MHDGFNET_KEEP_LAST", 15))

    with np.load(npz_path) as d:
        required_keys = [
            "eeg_allband_feature_map",
            "eeg_en_stat",
            "peri_feature",
        ]

        missing = [k for k in required_keys if k not in d.files]

        if missing:
            raise ValueError(
                f"npz 缺少字段：{missing}，实际字段：{list(d.files)}"
            )

        maps_all = d["eeg_allband_feature_map"]
        stats_all = d["eeg_en_stat"]
        peri_all = d["peri_feature"]

    # 训练代码里是 40 个 trial，每个 trial 15 个 segment
    maps_all = maps_all.reshape(
        40,
        segments_per_trial,
        *maps_all.shape[1:]
    )

    stats_all = stats_all.reshape(
        40,
        segments_per_trial,
        *stats_all.shape[1:]
    )

    peri_all = peri_all.reshape(
        40,
        segments_per_trial,
        *peri_all.shape[1:]
    )

    trial_idx = trial_no - 1

    maps = maps_all[trial_idx, -keep_last:, ...].astype(np.float32)
    stats = stats_all[trial_idx, -keep_last:, ...].astype(np.float32)
    peri = peri_all[trial_idx, -keep_last:, ...].astype(np.float32)

    # stats 训练时最终 view 成 N,32,7
    stats = stats.reshape(-1, 32, 7)

    # peri 训练时是 N,55
    peri = peri.reshape(-1, 55)

    visual = np.load(visual_npy_path)

    # 训练代码逻辑：
    # 如果 visual 是 512，则复制成 15 个 segment
    if visual.ndim == 1:
        visual = np.tile(visual, (segments_per_trial, 1))

    if visual.shape[-1] != 512:
        raise ValueError(
            f"视觉特征最后一维必须是 512，当前 shape={visual.shape}"
        )

    # 如果视觉特征不是 15 段，就切成 15 段取均值
    if visual.shape[0] != segments_per_trial:
        visual = np.vstack([
            np.mean(x, axis=0)
            for x in np.array_split(visual, segments_per_trial)
        ])

    visual = visual[-keep_last:, :].astype(np.float32)

    # 最终四个输入必须数量一致
    n = maps.shape[0]

    if stats.shape[0] != n:
        raise ValueError(f"stats segment 数量不一致：maps={n}, stats={stats.shape}")

    if peri.shape[0] != n:
        raise ValueError(f"peri segment 数量不一致：maps={n}, peri={peri.shape}")

    if visual.shape[0] != n:
        raise ValueError(f"visual segment 数量不一致：maps={n}, visual={visual.shape}")

    return {
        "maps": maps,
        "stats": stats,
        "peri": peri,
        "visual": visual,
        "meta": {
            "trial_no": trial_no,
            "segments": n,
            "maps_shape": list(maps.shape),
            "stats_shape": list(stats.shape),
            "peri_shape": list(peri.shape),
            "visual_shape": list(visual.shape),
        }
    }
