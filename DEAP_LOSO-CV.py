import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import random
import numpy as np
import time
from datetime import timedelta
import os
import warnings

# 忽略 Flash Attention 等非致命警告
warnings.filterwarnings("ignore", category=UserWarning)

# 导入你自己的模块
from C2PCI_Net import Fusion_Model, weights_init, custom_weights_init
from DataLoader import dataset_loaders
from config import Config

# ==========================================
# 1. 监测工具：权重与数值检查
# ==========================================
def check_model_sanity(model):
    max_w = 0.0
    for param in model.parameters():
        if param.requires_grad:
            curr_max = param.data.abs().max().item()
            max_w = max(max_w, curr_max)
    return max_w

# ==========================================
# 2. 基础设置与种子初始化
# ==========================================
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
start_time = time.time()

# --- 初始化配置 ---
config = Config(dataset_name='DEAP') 
# 强制指定为 arousal（也可以在 config.py 里改）
config.label_type = 'arousal' 

device = torch.device(config.device if torch.cuda.is_available() else "cpu")
num_epochs = 50 

# --- 数据加载 ---
# 显式传递 config 中的 label_type
train_loaders, test_loaders = dataset_loaders(
    config.dataset_name, 
    batch_size=config.batch_size, 
    label_type=config.label_type  # 这一行至关重要！
)
print(f"LOSO 模式已就绪，受试者总数: {len(train_loaders)}")

best_result = []

# 确定标签索引：Arousal=1, Valence=0
# 逻辑：如果是 arousal 则取 index 1
label_idx = 1 if config.label_type.lower() == 'arousal' else 0
print(f"当前训练目标: {config.label_type.upper()} (Index: {label_idx})")

# ==========================================
# 3. 训练核心循环 (LOSO)
# ==========================================
for p in range(config.num_subjects):
    train_loader = train_loaders[p]
    test_loader = test_loaders[p]

    model = Fusion_Model(config).to(device)

    # 权重初始化
    model.apply(weights_init)
    model.apply(custom_weights_init)
    if hasattr(model, 'PSD_map_backbone'):
        model.PSD_map_backbone.init_weights()

    # --- 策略：Adam + Weight Decay ---
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_acc = 0.0
    best_f1_for_sub = 0.0
    patience_counter = 0
    early_stop_patience = 12 

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        
        for eeg_map, eeg_stat, peri, labels in train_loader:
            eeg_map = torch.clamp(torch.nan_to_num(eeg_map), -10, 10).to(device)
            eeg_stat = torch.clamp(torch.nan_to_num(eeg_stat), -10, 10).to(device)
            peri = torch.clamp(torch.nan_to_num(peri), -10, 10).to(device)
            
            # 注入噪声增强泛化
            eeg_map = eeg_map + torch.randn_like(eeg_map) * 0.01
            
            # 标签选择 (Arousal)
            y = labels[:, label_idx].long().to(device) if labels.dim() > 1 else labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(eeg_map, eeg_stat, peri)
            
            if isinstance(outputs, tuple):
                e = outputs[4] if len(outputs) >= 5 else outputs[-1]
            else:
                e = outputs
            
            loss = criterion(e, y)
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item() * eeg_map.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)
        current_max_w = check_model_sanity(model)

        # --- 测试阶段 ---
        model.eval()
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for eeg_map, eeg_stat, peri, labels in test_loader:
                eeg_map = torch.clamp(torch.nan_to_num(eeg_map), -10, 10).to(device)
                eeg_stat = torch.clamp(torch.nan_to_num(eeg_stat), -10, 10).to(device)
                peri = torch.clamp(torch.nan_to_num(peri), -10, 10).to(device)
                
                y = labels[:, label_idx].long().to(device) if labels.dim() > 1 else labels.long().to(device)

                outputs = model(eeg_map, eeg_stat, peri)
                if isinstance(outputs, tuple):
                    e = outputs[4] if len(outputs) >= 5 else outputs[-1]
                else:
                    e = outputs
                
                _, predictions = torch.max(e, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_predictions)
        test_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_f1_for_sub = test_f1
            patience_counter = 0 
        else:
            patience_counter += 1

        print(f"Sub {p+1:02d}, Ep [{epoch+1:02d}/{num_epochs}], Loss: {train_loss:.4f}, "
              f"Acc: {test_acc*100:.2f}% (Best: {best_acc*100:.2f}%), F1: {test_f1:.2f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}, MaxW: {current_max_w:.2f}")

        if patience_counter >= early_stop_patience:
            print(f"!!! Early Stopping triggered at Epoch {epoch+1} !!!")
            break

    # 记录该受试者结果
    best_result.append([p + 1, best_acc * 100, best_f1_for_sub])
    print(f">>> Subject {p+1} Finish. Best Acc: {best_acc*100:.2f}%")

    # --- 关键修改：实时保存 ---
    # 每跑完一个人就更新一次表格，防止断电丢失
    temp_df = pd.DataFrame(best_result, columns=['Subject ID', 'Best Accuracy', 'Best F1 Score'])
    temp_df.to_excel(f"{config.dataset_name}_{config.label_type.upper()}_LIVE_Results.xlsx", index=False)

# ==========================================
# 4. 统计与汇总
# ==========================================
print("\n" + "="*40)
print(f"FINAL LOSO SUMMARY FOR {config.label_type.upper()}")
print("="*40)
final_df = pd.DataFrame(best_result, columns=['Subject ID', 'Best Accuracy', 'Best F1 Score'])
print(final_df)
print(f"\nMean LOSO Accuracy: {final_df['Best Accuracy'].mean():.2f}%")

save_name = f"{config.dataset_name}_{config.label_type.upper()}_Final_Results.xlsx"
final_df.to_excel(save_name, index=False)
print(f"Final results saved to {save_name}")

end_time = time.time()
print(f"Total processing time: {str(timedelta(seconds=end_time - start_time))}")