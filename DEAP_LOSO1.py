#这版是最近更改的，加入了一些别的控制和实验，但是效果和原来差不多，没提升，在模型的地方加了别的输出，需要我再发给您
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
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attn):

    attn = attn.detach().cpu().numpy()

    print("\n===== ATTENTION DEBUG =====")
    print("Shape:", attn.shape)
    print("Mean :", attn.mean())
    print("Std  :", attn.std())
    print("Max  :", attn.max())
    print("Min  :", attn.min())

    # 取第0个样本，第0个head
    sample = attn[0, 0]  # shape (32, 55)

    plt.figure(figsize=(12,5))
    plt.imshow(sample, aspect='auto')
    plt.colorbar()
    plt.xlabel("Key (55)")
    plt.ylabel("Query (32)")
    plt.title("Cross Attention Heatmap (Batch0, Head0)")
    plt.show()

    # 每个key的平均权重
    key_mean = attn.mean(axis=(0,1,2))
    plt.figure(figsize=(12,4))
    plt.plot(key_mean)
    plt.title("Mean Attention per Key")
    plt.show()

def orthogonality_loss(tokens):
    """
    tokens: (B, C, D)
    让 C 个 token 彼此正交
    """
    B, C, D = tokens.shape

    # L2 normalize
    tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)

    # Gram matrix
    gram = torch.matmul(tokens, tokens.transpose(1, 2))  # (B, C, C)

    # Identity
    I = torch.eye(C, device=tokens.device).unsqueeze(0)

    loss = ((gram - I) ** 2).mean()

    return loss

def run_training():

    # ================== 1. 基础配置 ==================
    start_time = time.time()
    set_seed(42)

    config = Config(dataset_name='DEAP')
    config.label_type = 'arousal'
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

        optimizer = optim.Adam(
    model.parameters(), 
    lr=2e-5,            # 从 1e-4 降到 2e-5，让模型学得更“细致”一些
    weight_decay=1e-2   # 从 5e-3 提高到 1e-2，加强 L2 正则化防止死记硬背
)
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
                # DataLoader 已经处理好二分类和列筛选，直接转 long
                labels = labels.to(config.device).long()
    
                optimizer.zero_grad()
                
                # 1. 获取五个输出
                # 按照你定义的顺序接住返回值
                tokens_stat,x_s1, z_t, x_s2, z_c, logits = model(eeg_map, eeg_stat, peri)
    
                # 2. 分别计算三个头的 CrossEntropy 损失
                loss_main = criterion(logits, labels)
                loss_aux_t = criterion(z_t, labels)
                loss_aux_c = criterion(z_c, labels)
    
                # 3. 联合损失计算 (给辅助损失分配 0.1 的权重)
                # 这里的 0.1 是超参数，可以根据收敛情况微调
                # ===== Orthogonality Loss =====
                loss_div = orthogonality_loss(tokens_stat)

                lambda_div = 0.005   # 建议从 0.01 开始

                loss = (
                    loss_main
                    + 0.1 * loss_aux_t
                    + 0.1 * loss_aux_c
                    + lambda_div * loss_div
                )
    
                # 4. 反向传播与优化 (只需一次！)
                loss.backward()
                
                # 梯度裁剪：防止 Transformer 层梯度爆炸，非常重要
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            scheduler.step()

            # -------- Test --------
            model.eval()
            all_preds, all_labels = [], []
            total_contrib = 0.0   # ★ 修复你原代码中的 bug

            with torch.no_grad():

                for batch_idx, batch in enumerate(test_loader):

                    eeg_map, eeg_stat, peri, labels, trial_ids = batch

                    eeg_map = eeg_map.to(config.device)
                    eeg_stat = eeg_stat.to(config.device)
                    peri = peri.to(config.device)

                    tokens_stat,x_s1, z_t, x_s2, z_c, logits = model(eeg_map, eeg_stat, peri)

        # ====== 只在第一个epoch的第一个batch可视化一次 ======
                    if epoch == 0 and subj_idx == 0:
                        print("Orth Loss:", loss_div.item())

                    if epoch == 0 and batch_idx == 0 and subj_idx == 0:
                    
            # 重新 forward 只拿 attention
                        _, cross_attn = model.hf_icma(
                            model.PSD_map_backbone(eeg_map),
                            model.eeg_backbone(eeg_stat),
                            model.peri_backbone(peri)
                        )

                        visualize_attention(cross_attn)

                    

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
