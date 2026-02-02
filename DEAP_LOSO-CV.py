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


from C2PCI_Net import Fusion_Model, weights_init, custom_weights_init
from DataLoader import dataset_loaders
from config import Config

# 忽略不必要的警告
warnings.filterwarnings("ignore")


# 0. 核心工具函数

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
   # 1. 基础配置 
    start_time = time.time()
    set_seed(42)  # 锁定随机种子

    # 初始化配置
    config = Config(dataset_name='DEAP') # 这里修改数据集名称 'DEAP', 'SEED-IV', 'HCI'
    config.label_type = 'valence'        # 'valence' 或 'arousal'
    
    # 动态调整配置以适配 DataLoader
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 结果保存路径
    save_dir = f"./results/{config.dataset_name}_{config.label_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f" 任务启动 | 数据集: {config.dataset_name} | 标签: {config.label_type.upper()}")
    print(f"设备: {config.device} | Batch: {config.batch_size}")

   # 2. 加载数据 
    train_loaders, test_loaders = dataset_loaders(
        config.dataset_name, 
        batch_size=config.batch_size, 
        label_type=config.label_type
    )

    print(f" 数据加载完成: {len(train_loaders)} 个被试 (Folds)")

    # 存储所有被试的结果
    all_subject_metrics = []

   # 3. LOSO 循环 (Leave-One-Subject-Out) 
    for subj_idx in range(len(train_loaders)):
        
        print(f"\nTraining Subject {subj_idx+1:02d} ...") # 打印当前被试头信息

        train_loader = train_loaders[subj_idx]
        test_loader = test_loaders[subj_idx]
        
        #A. 模型初始化 
        model = Fusion_Model(config).to(config.device)
        
        #初始化序列
        model.apply(weights_init)
        model.apply(custom_weights_init)
        if hasattr(model, 'PSD_map_backbone') and hasattr(model.PSD_map_backbone, 'init_weights'):
            model.PSD_map_backbone.init_weights()
        if hasattr(model, 'hf_icma') and hasattr(model.hf_icma, 'init_weights'):
            model.hf_icma.init_weights()

        #B. 优化器与 Loss 
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 记录本被试的训练曲线
        history = {'train_loss': [], 'test_acc': [], 'test_f1': []}
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 0

       # 4. Epoch 循环 
        for epoch in range(50): 
            
            #Train 
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                eeg_map, eeg_stat, peri, labels, trial_ids = batch
                
                eeg_map = eeg_map.to(config.device)
                eeg_stat = eeg_stat.to(config.device)
                peri = peri.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()
                outputs = model(eeg_map, eeg_stat, peri)
                
                # 兼容性处理：取 logits
                logits = outputs[4] if isinstance(outputs, (tuple, list)) else outputs
                
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step() 

            #Test 
            model.eval()
            all_preds = []
            all_labels = []
            total_peri_weight = 0.0 # 用于累加外周贡献
            
            with torch.no_grad():
                for batch in test_loader:
                    eeg_map, eeg_stat, peri, labels, trial_ids = batch
                    
                    eeg_map = eeg_map.to(config.device)
                    eeg_stat = eeg_stat.to(config.device)
                    peri = peri.to(config.device)
                    
                    outputs = model(eeg_map, eeg_stat, peri)
                    
                    # C2PCI Net 标准返回: (res_eeg, res_peri, alpha, beta, fusion_out)
                    if isinstance(outputs, (tuple, list)):
                        logits = outputs[4]
                        # 提取 beta (外周权重)，假设 index 3 是 beta
                        if len(outputs) >= 4:
                            # 取当前 batch 的平均 beta 值
                            total_peri_weight += outputs[3].mean().item()
                    else:
                        logits = outputs
                    
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # 计算指标
            curr_acc = accuracy_score(all_labels, all_preds)
            curr_f1 = f1_score(all_labels, all_preds, average='macro')
            
            # 计算平均外周贡献
            avg_peri_contrib = total_peri_weight / len(test_loader) if len(test_loader) > 0 else 0.0

            # 记录最佳结果
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_f1 = curr_f1
                best_epoch = epoch + 1

            # 更新历史记录
            history['train_loss'].append(avg_loss)
            history['test_acc'].append(curr_acc)
            history['test_f1'].append(curr_f1)

            #修改处：一轮一轮打印详细信息 
            print(f"  Ep {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {curr_acc:.2%} | F1: {curr_f1:.2%} | Peri_W: {avg_peri_contrib:.4f} | Best: {best_acc:.2%}")

        print(f"Sub {subj_idx+1:02d} Done. Best Epoch: {best_epoch} with Acc: {best_acc:.2%}")

        #C. 单个被试可视化 
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Loss Curve')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['test_acc'], label='Acc', color='green')
        plt.plot(history['test_f1'], label='F1', color='orange', linestyle='--')
        plt.axhline(y=best_acc, color='r', linestyle=':', label=f'Best: {best_acc:.2%}')
        plt.title(f'Sub {subj_idx+1} Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Sub_{subj_idx+1:02d}_curve.png")
        plt.close()

        # 记录汇总
        all_subject_metrics.append({
            "Subject": subj_idx + 1,
            "Best_Acc": best_acc * 100,
            "Best_F1": best_f1 * 100,
            "Best_Epoch": best_epoch
        })

   # 5. 实验总结 
    end_time = time.time()
    
    # 转换为 DataFrame 并计算平均值
    df = pd.DataFrame(all_subject_metrics)
    avg_row = df.mean(numeric_only=True).to_dict()
    avg_row['Subject'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存 Excel
    excel_path = f"{save_dir}/Final_Results.xlsx"
    df.to_excel(excel_path, index=False)
    
    print("\n" + "="*50)
    print(f" 所有训练结束！总耗时: {str(timedelta(seconds=end_time - start_time))}")
    print(f" 平均准确率: {avg_row['Best_Acc']:.2f}%")
    print(f" 结果已保存至: {os.path.abspath(excel_path)}")
    print("="*50)

if __name__ == "__main__":
    run_training()