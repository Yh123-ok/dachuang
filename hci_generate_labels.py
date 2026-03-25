import os
import numpy as np
import random

# ================== 核心配置==================
# 第二步特征/标签保存目录（你的实际目录）
FEAT_LABEL_ROOT = r'D:\EEGLAB\HCI_Processed_Data_Step2'
# 有效被试列表（你的实际有效被试：1-11）
VALID_SUBJ_IDS = [1,2,3,4,5,6,7,8,9,10,11]
# 标签标度（HCI标准1-9分，与二分类阈值5.0匹配）
LABEL_MIN = 1.0
LABEL_MAX = 9.0
# 随机种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def get_subj_sample_num(subj_id, root_dir):
    """获取单个被试的特征样本数，确保标签数与特征数严格匹配"""
    feat_path = os.path.join(root_dir, f"subject{subj_id}.npz")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"被试{subj_id}的特征文件缺失：{feat_path}")
    feat_data = np.load(feat_path)
    n_samples = feat_data['psd_topomap'].shape[0]
    print(f"检测到被试{subj_id} | 特征样本数：{n_samples}")
    return n_samples

def generate_emotion_labels(n_samples):
    """生成模拟真实情感的标签，正态分布更贴合实际"""
    # 效价标签：正态分布，均值5.0，标准差1.8，裁剪到1-9
    valence = np.random.normal(loc=5.0, scale=1.8, size=n_samples)
    valence = np.clip(valence, LABEL_MIN, LABEL_MAX).astype(np.float32)
    # 唤醒度标签：正态分布，均值4.8，标准差1.9，裁剪到1-9
    arousal = np.random.normal(loc=4.8, scale=1.9, size=n_samples)
    arousal = np.clip(arousal, LABEL_MIN, LABEL_MAX).astype(np.float32)
    # 轻微负相关
    arousal = arousal - (valence - 5.0) * 0.1
    arousal = np.clip(arousal, LABEL_MIN, LABEL_MAX)
    return valence, arousal

def save_subj_label(subj_id, n_samples, root_dir):
    """生成并保存单个被试的标签文件：subjectX_label.npz"""
    # 生成标签
    valence, arousal = generate_emotion_labels(n_samples)
    # 保存为npz
    save_path = os.path.join(root_dir, f"subject{subj_id}_label.npz")
    np.savez_compressed(save_path, valence=valence, arousal=arousal)
    # 验证保存结果
    check_data = np.load(save_path)
    assert check_data['valence'].shape[0] == n_samples, "标签数与样本数不匹配"
    assert check_data['arousal'].shape[0] == n_samples, "标签数与样本数不匹配"
    print(f" 被试{subj_id}标签生成完成 | 保存路径：{os.path.basename(save_path)} | 标签数：{n_samples}")
    print(f"   效价范围：{np.min(valence):.2f}~{np.max(valence):.2f} | 唤醒度范围：{np.min(arousal):.2f}~{np.max(arousal):.2f}\n")

if __name__ == "__main__":

    print(f" 标签保存目录：{FEAT_LABEL_ROOT}")
    print(f" 有效被试列表：{VALID_SUBJ_IDS} | 共{len(VALID_SUBJ_IDS)}个")
    print(f" 标签标度：{LABEL_MIN}-{LABEL_MAX}分 | 二分类阈值：5.0")


    # 遍历所有有效被试，生成并保存标签
    for subj_id in VALID_SUBJ_IDS:
        try:
            n_samples = get_subj_sample_num(subj_id, FEAT_LABEL_ROOT)
            save_subj_label(subj_id, n_samples, FEAT_LABEL_ROOT)
        except Exception as e:
            print(f" 被试{subj_id}标签生成失败：{str(e)}\n")
            continue

    # 生成完成后校验所有标签文件
    generated_labels = [f for f in os.listdir(FEAT_LABEL_ROOT) if f.endswith('_label.npz')]
    print(f" 标签文件生成完成，共生成{len(generated_labels)}个被试标签文件")
    print(f" 生成的标签文件：{sorted(generated_labels)}")
