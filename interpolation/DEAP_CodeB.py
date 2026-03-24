import numpy as np
import os
from scipy.interpolate import Rbf
from sklearn.preprocessing import MinMaxScaler
import mne

#  1. 配置与路径 
A_RESULT_DIR = r'D:\EEGLAB\Processed_Data'
SAVE_DIR = r'D:\EEGLAB\Processed_Data_Python'
samples_per_subject = 600 

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

#  2. 坐标获取与网格生成 
def get_coords_optimized():
    ch_names = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
        'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    ]
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    
    coords = []
    for ch in ch_names:
        name = ch if ch in pos_dict else next(k for k in pos_dict if k.upper() == ch.upper())
        coords.append(pos_dict[name][:2])
    
    coords = np.array(coords)
  
    scaler = MinMaxScaler(feature_range=(-0.48, 0.48))
    coords = scaler.fit_transform(coords)
    return coords[:, 0], coords[:, 1]

x_loc, y_loc = get_coords_optimized()
# 网格边界保持 0.5179
grid_edge = np.linspace(-0.5179, 0.5179, 32)
grid_x, grid_y = np.meshgrid(grid_edge, grid_edge)
mask = np.sqrt(grid_x**2 + grid_y**2) > 0.5179 

#  3. 加载数据 
print("Loading 1D features...")
all_psds = np.load(os.path.join(A_RESULT_DIR, 'final_psds.npy'))   
all_stats = np.load(os.path.join(A_RESULT_DIR, 'final_stats.npy')) 
all_peris = np.load(os.path.join(A_RESULT_DIR, 'final_peris.npy')) 

#  4. 插值循环 
PSD_features = np.zeros((19200, 5, 32, 32))

print("Starting Interpolation (Optimized v4 style)...")
for k in range(19200):
   
    psd_sample =all_psds[k].reshape(5, 32)
    
    for b in range(5):
        med = np.median(psd_sample[b])
        std = np.std(psd_sample[b])
        # 如果某个电极能量超过中位数 + 3倍标准差，通常就是肌电或坏道
        threshold = med + 3 * std
        psd_sample[b][psd_sample[b] > threshold] = med
        rbf = Rbf(x_loc, y_loc, psd_sample[b], function='thin_plate', smooth=0.05)
        zi = rbf(grid_x, grid_y)
        zi = np.flipud(zi)
        zi[mask] = np.nan # 暂时标记背景为 NaN
        PSD_features[k, b, :, :] = zi
    
    if k % 2000 == 0: print(f"Progress: {k}/19200 samples...")

#  5. 受试者内标准化 
print("Performing Subject-wise Normalization...")
for b in range(5):
    for s in range(32):
        start, end = s * 600, (s + 1) * 600
        sub_band_data = PSD_features[start:end, b, :, :]
        
        # 只在有电极信号的圆形区域内计算
        valid_idx = ~np.isnan(sub_band_data)
        if np.any(valid_idx):
            mean_val = np.mean(sub_band_data[valid_idx])
            std_val = np.std(sub_band_data[valid_idx])
            sub_band_data[valid_idx] = (sub_band_data[valid_idx] - mean_val) / (std_val + 1e-8)
        
        # 标准化后，背景填充为 0
        sub_band_data[np.isnan(sub_band_data)] = 0
        PSD_features[start:end, b, :, :] = sub_band_data

#  6. 保存 
for s in range(32):
    start, end = s * 600, (s + 1) * 600
    save_name = os.path.join(SAVE_DIR, f's{s+1:02d}_features.npz')
    np.savez(save_name, 
             eeg_allband_feature_map=PSD_features[start:end].astype(np.float32),
             eeg_en_stat=all_stats[start:end].astype(np.float32),
             peri_feature=all_peris[start:end].astype(np.float32))

print("Optimization Completed. Ready for Testing.")
