import numpy as np
import matplotlib.pyplot as plt

# --- 修改为你生成的任意一个文件 ---
CHECK_FILE = r'D:\Users\cyz\dc\222\s01_features.npz'

def check_data():
    data = np.load(CHECK_FILE)
    maps = data['eeg_allband_feature_map']
    stats = data['eeg_en_stat']
    peri = data['peri_feature']

    print(f"--- Data Integrity Report for {CHECK_FILE} ---")
    print(f"Maps Shape:  {maps.shape}  (Expect: [600, 5, 32, 32])")
    print(f"Stats Shape: {stats.shape} (Expect: [600, 224])")
    print(f"Peri Shape:  {peri.shape}  (Expect: [600, 55])")
    
    # 检查 NaN
    print(f"\nNaN Check: Maps: {np.isnan(maps).any()}, Stats: {np.isnan(stats).any()}, Peri: {np.isnan(peri).any()}")
    
    # 检查数值范围 (StandardScaler 之后应该在 -3 到 3 左右)
    print(f"Stats Mean: {stats.mean():.4f}, Std: {stats.std():.4f}")
    
    # --- 可视化第一段数据的 5 个频段拓扑图 ---
    plt.figure(figsize=(15, 3))
    bands = ['Theta', 'Alpha', 'Slow Alpha', 'Beta', 'Gamma']
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(maps[0, i, :, :], cmap='jet')
        plt.title(bands[i])
        plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_data()