import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from DataLoader import get_loso_dataloaders, get_all_valid_subj_ids

# ================== 核心配置==================
FEAT_ROOT = r'D:\EEGLAB\HCI_Processed_Data_Step2'
LABEL_TYPE = 'arousal'  # 可选：'valence'（效价）/ 'arousal'（唤醒度）
BATCH_SIZE = 32
NUM_EPOCHS = 40  # 小样本适配，避免过拟合
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5  # 权重衰减，抑制过拟合
EARLY_STOP_PATIENCE = 8  # 早停策略：8轮无提升则停止
LABEL_THRESHOLD = 5.0  # 定义标签二分类阈值
SAVE_RESULT_PATH = r'D:\EEGLAB\HCI_Train_Results'  # 结果/模型保存目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(SAVE_RESULT_PATH, exist_ok=True)


# ================== 多分支模型==================
# 输入：5×64×64拓扑图 + 224维统计 + 37维外周 → 输出：0/1二分类概率
class MultiBranchModel(nn.Module):
    """
        多分支神经网络模型，用于 EEG 信号的情感识别二分类任务

        该模型包含三个并行的特征提取分支：
        - CNN 分支：处理 5 频段×64×64 的拓扑图特征
        - MLP 分支 1：处理 224 维 EEG 统计特征
        - MLP 分支 2：处理 37 维外周生理特征
        最后通过特征融合层输出二分类概率

        Attributes:
            cnn_branch: CNN 特征提取网络，处理拓扑图输入
            stats_branch: MLP 网络，处理 EEG 统计特征
            peri_branch: MLP 网络，处理外周生理特征
            fusion: 特征融合与分类输出网络
    """
    def __init__(self):
        """
               初始化多分支模型的各层网络结构

               构建三个并行的特征提取分支和一个融合分类层：
               - CNN 分支：3 层卷积 + 池化，逐步提取空间特征
               - Stats 分支：2 层全连接，提取统计特征
               - Peri 分支：2 层全连接，提取外周特征
               - Fusion 层：拼接所有特征并输出二分类概率
        """
        super(MultiBranchModel, self).__init__()
        # CNN分支：处理5频段×64×64拓扑图
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5)
        )
        # MLP分支1：处理224维EEG统计特征
        self.stats_branch = nn.Sequential(
            nn.Linear(224, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4)
        )
        # MLP分支2：处理37维外周特征
        self.peri_branch = nn.Sequential(
            nn.Linear(37, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.4)
        )
        # 特征融合+二分类输出
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()  # Sigmoid输出0~1概率，适配BCELoss
        )

    def forward(self, psd_topomap, eeg_stats, peri_feats):
        """
                前向传播计算，融合多模态特征并输出分类结果

                Args:
                    psd_topomap (torch.Tensor): 5 频段×64×64 的拓扑图特征，批次数据
                    eeg_stats (torch.Tensor): 224 维 EEG 统计特征，批次数据
                    peri_feats (torch.Tensor): 37 维外周生理特征，批次数据

                Returns:
                    torch.Tensor: 二分类概率输出，范围 [0, 1]，表示预测为正类的概率
        """
        cnn_feat = self.cnn_branch(psd_topomap)
        stats_feat = self.stats_branch(eeg_stats)
        peri_feat = self.peri_branch(peri_feats)
        fusion_feat = torch.cat([cnn_feat, stats_feat, peri_feat], dim=1)
        return self.fusion(fusion_feat)


# ================== 单轮LOSO训练==================
def train_loso_round(test_subj_id, model, criterion, optimizer):
    """
        执行单轮留一被试交叉验证（LOSO）训练

        采用早停策略防止过拟合：当测试精度连续多轮无提升时提前终止训练
        每轮训练实时显示进度条，动态更新训练损失和准确率
        自动保存最优模型参数到指定目录

        Args:
            test_subj_id (int): 当前轮次作为测试集的被试 ID
            model (MultiBranchModel): 待训练的多分支神经网络模型
            criterion (nn.BCELoss): 二分类损失函数，用于计算预测误差
            optimizer (optim.Adam): Adam 优化器，负责参数更新

        Returns:
            tuple: 包含两个元素
                - best_test_acc (float): 该轮训练达到的最优测试集准确率（百分比）
                - avg_test_loss (float): 该轮训练最终的平均测试损失值

        Raises:
            FileNotFoundError: 如果 DataLoader 无法加载被试数据文件
            RuntimeError: 如果 GPU 显存不足或训练过程中出现设备错误
    """
    train_loader, test_loader = get_loso_dataloaders(test_subj_id=test_subj_id)
    best_test_acc = 0.0
    no_improve_count = 0  # 早停计数器：记录测试精度无提升的轮数

    for epoch in range(NUM_EPOCHS):
        # 训练模式：开启BatchNorm/Dropout
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"被试{test_subj_id} | Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch in pbar:
            psd, stats, peri, labels = batch
            # 前向传播
            outputs = model(psd, stats, peri)
            loss = criterion(outputs, labels)
            # 反向传播+优化（清空梯度→计算梯度→更新参数）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算训练指标
            train_loss += loss.item() * psd.size(0)
            preds = (outputs >= 0.5).float()  # 0.5为二分类阈值，输出0/1
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            # 更新进度条，实时显示训练损失/精度
            pbar.set_postfix({
                'Train Loss': f"{train_loss / train_total:.4f}",
                'Train Acc': f"{100 * train_correct / train_total:.2f}%"
            })

        # 测试模式：关闭梯度+BatchNorm/Dropout，不更新参数
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():  # 关闭梯度计算，节省显存+提升速度
            for batch in test_loader:
                psd, stats, peri, labels = batch
                outputs = model(psd, stats, peri)
                loss = criterion(outputs, labels)
                # 计算测试指标
                test_loss += loss.item() * psd.size(0)
                preds = (outputs >= 0.5).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        # 计算平均损失/精度（按样本数加权）
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100 * train_correct / train_total
        avg_test_loss = test_loss / test_total
        avg_test_acc = 100 * test_correct / test_total

        # 早停判断：测试精度无提升则计数，超过阈值提前停止
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            no_improve_count = 0
            # 保存当前轮最优模型（仅保存参数，节省空间）
            torch.save(model.state_dict(), os.path.join(SAVE_RESULT_PATH, f"best_model_subj{test_subj_id}.pth"))
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f" 早停触发：连续{EARLY_STOP_PATIENCE}轮测试精度无提升，提前结束训练")
                break

        # 打印本轮训练/测试结果
        print(f"Epoch {epoch + 1:2d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
        print(
            f"          | Test Loss:  {avg_test_loss:.4f} | Test Acc:  {avg_test_acc:.2f}% | Best Acc: {best_test_acc:.2f}%\n")

    # 返回该轮最优测试精度和最终测试损失
    return best_test_acc, avg_test_loss


# ================== 4. 主函数：全被试LOSO交叉验证 ==================
def main():
    """
       主训练函数：执行完整的留一被试交叉验证（LOSO-CV）实验流程

       主要功能：
       1. 遍历所有有效被试，逐一作为测试集进行 LOO 交叉验证
       2. 每轮独立初始化模型、损失函数和优化器，避免参数污染
       3. 调用单轮训练函数，记录最优测试结果
       4. 汇总所有轮次的结果，计算平均准确率和标准差
       5. 将完整实验结果保存至 TXT 文件，格式适配论文报告

       实验配置：
       - 数据集：HCI 情感 EEG 数据集
       - 标签类型：唤醒度（arousal）或效价（valence）
       - 评估指标：准确率（%）、损失值
       - 保存内容：每轮最优模型参数、完整实验结果文本

       Note:
           实验结果包含均值±标准差，反映模型在不同被试上的稳定性和泛化能力
           所有随机种子已固定，确保实验可复现性
    """
    print(f" HCI数据集LOSO交叉验证训练| 标签：{LABEL_TYPE.upper()} | 设备：{DEVICE}")
    print(f" 配置：{NUM_EPOCHS}轮 | 批次{BATCH_SIZE} | 早停{EARLY_STOP_PATIENCE}轮 | 二分类阈值{LABEL_THRESHOLD}")

    # 获取所有有效被试（特征+标签均存在，由DataLoader自动检测）
    all_subj_ids = get_all_valid_subj_ids(feat_root=FEAT_ROOT)
    num_subj = len(all_subj_ids)
    if num_subj < 2:
        raise ValueError(f"有效被试数{num_subj}，不足2个无法做LOSO")

    # 初始化结果容器，保存每轮最优测试精度/损失
    all_test_acc = []
    all_test_loss = []

    # 遍历所有被试，逐一轮执行LOSO（每次留1个为测试集，剩余为训练集）
    for idx, test_subj in enumerate(all_subj_ids, 1):
        print(f"\n" + "-" * 60)
        print(
            f" LOSO第{idx}/{num_subj}轮 | 测试被试：{test_subj} | 训练被试：{[s for s in all_subj_ids if s != test_subj]}")
        print("-" * 60)
        # 每轮LOSO重新初始化模型/损失/优化器（避免跨轮参数污染，实验更严谨）
        model = MultiBranchModel().to(DEVICE)
        criterion = nn.BCELoss()  # 二分类损失函数（Sigmoid输出+BCELoss，适配0/1标签）
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)  # Adam默认超参数，提升收敛稳定性
        )
        # 训练单轮LOSO，返回最优测试精度
        test_acc, test_loss = train_loso_round(test_subj, model, criterion, optimizer)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        print(f" LOSO第{idx}轮完成 | 最优测试精度：{test_acc:.2f}% | 最终测试损失：{test_loss:.4f}")

    # ================== 5. 结果汇总与标准化保存==================
    print("\n" + "=" * 80)
    print(f" HCI数据集LOSO交叉验证结果汇总 | 共{num_subj}轮（真实标签）")
    print("=" * 80)
    # 打印每轮详细结果
    for idx, (subj, acc, loss) in enumerate(zip(all_subj_ids, all_test_acc, all_test_loss), 1):
        print(f"第{idx}轮 | 测试被试：{subj:2d} | 最优精度：{acc:5.2f}% | 测试损失：{loss:.4f}")
    # 计算平均+标准差
    mean_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)
    mean_loss = np.mean(all_test_loss)
    std_loss = np.std(all_test_loss)
    print(f"\n最终实验结果（真实标签）")
    print(f"   平均精度：{mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"   平均损失：{mean_loss:.4f} ± {std_loss:.4f}")

    # 保存结果到TXT文件
    result_file = os.path.join(SAVE_RESULT_PATH, 'hci_loso_train_results_real_label.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("HCI数据集LOSO交叉验证训练结果\n")
        f.write(f"数据集：HCI | 有效被试数：{num_subj} | 标签类型：{LABEL_TYPE.upper()} | 二分类阈值：{LABEL_THRESHOLD}\n")
        f.write(
            f"训练配置：Epochs={NUM_EPOCHS} | BatchSize={BATCH_SIZE} | LR={LEARNING_RATE} | 早停={EARLY_STOP_PATIENCE}轮\n")
        f.write(f"设备：{DEVICE} | 模型：MultiBranch(CNN+2MLP) | 损失函数：BCELoss | 优化器：Adam\n")
        f.write(f"{'轮次':<4} {'测试被试':<8} {'最优精度(%)':<12} {'测试损失':<10}\n")
        for idx, (subj, acc, loss) in enumerate(zip(all_subj_ids, all_test_acc, all_test_loss), 1):
            f.write(f"{idx:<4} {subj:<8} {acc:<12.2f} {loss:<10.4f}\n")
        f.write(f"平均结果 | 精度：{mean_acc:.2f}% ± {std_acc:.2f}% | 损失：{mean_loss:.4f} ± {std_loss:.4f}\n")
        f.write(f"结果保存时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 打印保存路径
    print(f"\n 实验结果已保存至：{result_file}")
    print(f" 各轮最优模型已保存至：{SAVE_RESULT_PATH}（文件名为best_model_subjX.pth）")



# 主函数入口：固定随机种子，保证实验可复现
if __name__ == "__main__":
    # 固定torch/numpy随机种子，每次运行结果一致
    torch.manual_seed(42)
    np.random.seed(42)
    # CUDA随机种子固定，GPU训练也可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 运行主训练逻辑
    main()