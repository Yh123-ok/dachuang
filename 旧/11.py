import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置路径 ---
FEAT_DIR = r'D:\Users\cyz\dc\222' # 你保存 .npz 的路径
subject_id = 1  # 随便选一个受试者看看
sample_idx = 100 # 选第100个样本

# --- 加载数据 ---
file_path = os.path.join(FEAT_DIR, f's{subject_id:02d}_features.npz')
data = np.load(file_path)
maps = data['eeg_allband_feature_map'] # (600, 5, 32, 32)

# --- 绘图 ---
bands = ['Theta', 'Alpha', 'Slow-Alpha', 'Beta', 'Gamma']
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle(f'Subject {subject_id} - Sample {sample_idx} Topomaps (Optimized B)', fontsize=16)

for i in range(5):
    im = axes[i].imshow(maps[sample_idx, i, :, :], cmap='jet', origin='lower')
    axes[i].set_title(bands[i])
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()