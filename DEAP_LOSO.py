import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import accuracy_score, f1_score
import warnings

# 引入你的模块
from C2PCI_Net import Fusion_Model, weights_init, custom_weights_init
from DataLoader import dataset_loaders
from config import Config

warnings.filterwarnings("ignore")


# ==========================================
# 0. 核心工具函数
# ==========================================
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_training():

    # ================== 1. 基础配置 ==================
    start_time = time.time()
    set_seed(42)

    config = Config(dataset_name='DEAP')
    config.label_type = 'valence'
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = f"./results/{config.dataset_name}_{config.label_type}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 任务启动 | 数据集: {config.dataset_name} | 标签: {config.label_type.upper()}")
    print(f"🔧 设备: {config.device} | Batch: {config.batch_size}")

    # ================== 2. 加载数据 ==================
    train_loaders, test_loaders = dataset_loaders(
        config.dataset_name,
        batch_size=config.batch_size,
        label_type=config.label_type
    )

    print(f"✅ 数据加载完成: {len(train_loaders)} 个被试 (LOSO)")

    all_subject_metrics = []

    # ================== 3. LOSO ==================
    for subj_idx in range(len(train_loaders)):

        print(f"\nTraining Subject {subj_idx + 1:02d} ...")

        train_loader = train_loaders[subj_idx]
        test_loader = test_loaders[subj_idx]

        # -------- 模型初始化 --------
        model = Fusion_Model(config).to(config.device)
        model.apply(weights_init)
        model.apply(custom_weights_init)

        if hasattr(model, 'PSD_map_backbone') and hasattr(model.PSD_map_backbone, 'init_weights'):
            model.PSD_map_backbone.init_weights()
        if hasattr(model, 'hf_icma') and hasattr(model.hf_icma, 'init_weights'):
            model.hf_icma.init_weights()

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        history = {'train_loss': [], 'test_acc': [], 'test_f1': []}

        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 0

        # ================== Early Stopping ==================
        patience = 10
        min_delta = 1e-4
        early_stop_counter = 0
        best_state_dict = None

        # ================== 4. Epoch ==================
        for epoch in range(50):

             # -------- Train --------
            model.train()
            total_loss = 0.0
    
            for batch in train_loader:
                eeg_map, eeg_stat, peri, labels, trial_ids = batch
    
                eeg_map = eeg_map.to(config.device)
                eeg_stat = eeg_stat.to(config.device)
                peri = peri.to(config.device)
                labels = labels.to(config.device)
    
                optimizer.zero_grad()
                
                outputs = model(eeg_map, eeg_stat, peri)
                logits = outputs[4] if isinstance(outputs, (tuple, list)) else outputs
    
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            scheduler.step()

            # -------- Test --------
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in test_loader:
                    eeg_map, eeg_stat, peri, labels, trial_ids = batch

                    eeg_map = eeg_map.to(config.device)
                    eeg_stat = eeg_stat.to(config.device)
                    peri = peri.to(config.device)

                    outputs = model(eeg_map, eeg_stat, peri)
                    logits = outputs[4] if isinstance(outputs, (tuple, list)) else outputs

                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            curr_acc = accuracy_score(all_labels, all_preds)
            curr_f1 = f1_score(all_labels, all_preds, average='macro')

            # ================== Early Stopping 判断 ==================
            if curr_acc > best_acc + min_delta:
                best_acc = curr_acc
                best_f1 = curr_f1
                best_epoch = epoch + 1
                early_stop_counter = 0

                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                early_stop_counter += 1

            history['train_loss'].append(avg_loss)
            history['test_acc'].append(curr_acc)
            history['test_f1'].append(curr_f1)

            print(
                f"  Ep {epoch+1:02d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {curr_acc:.2%} | "
                f"F1: {curr_f1:.2%} | "
                f"Best: {best_acc:.2%}"
            )

            if early_stop_counter >= patience:
                print(f"  ⏹ Early Stopping triggered at Epoch {epoch+1}")
                break

        # ===== 恢复最佳模型 =====
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        print(f"Sub {subj_idx+1:02d} Done. Best Epoch: {best_epoch} | Acc: {best_acc:.2%}")

        # ================== 可视化 ==================
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'])
        plt.title('Train Loss')
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history['test_acc'], label='Acc')
        plt.plot(history['test_f1'], '--', label='F1')
        plt.axhline(best_acc, linestyle=':', color='r', label='Best')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/Sub_{subj_idx+1:02d}_curve.png")
        plt.close()

        all_subject_metrics.append({
            "Subject": subj_idx + 1,
            "Best_Acc": best_acc * 100,
            "Best_F1": best_f1 * 100,
            "Best_Epoch": best_epoch
        })

    # ================== 5. 汇总 ==================
    end_time = time.time()

    df = pd.DataFrame(all_subject_metrics)
    avg_row = df.mean(numeric_only=True).to_dict()
    avg_row['Subject'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    excel_path = f"{save_dir}/Final_Results.xlsx"
    df.to_excel(excel_path, index=False)

    print("\n" + "=" * 50)
    print(f"🎉 训练完成 | 总耗时: {timedelta(seconds=end_time - start_time)}")
    print(f"📊 平均准确率: {avg_row['Best_Acc']:.2f}%")
    print(f"📂 结果保存至: {os.path.abspath(excel_path)}")
    print("=" * 50)


if __name__ == "__main__":
    run_training()