import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # 导入train_test_split

mark = '217'  # 标志物类型
NUM_SAMPLES = 6
BATH_SIZE =2
# 自定义数据集类
class ConcentrationDataset(Dataset):
    def __init__(self, data_files, input_dir):
        self.data_files = data_files
        self.input_dir = input_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 读取文件1的数据
        file_path1 = os.path.join(self.input_dir, self.data_files[idx])
        data1 = pd.read_csv(file_path1, header=None).values
        x1 = data1[:, 1].reshape(-1, 1)  # 只提取电流（第二列数据）

        # 对电流数据进行归一化
        x1_normalized = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))

        # 读取文件名中的浓度信息，作为标签
        file_name1 = self.data_files[idx]
        concentration1 = extract_concentration_from_filename(file_name1)
        label1 = torch.tensor(float(concentration1), dtype=torch.float32).view(1)

        # 随机选择另一个文件（x2），确保它和x1对应不同的样本
        idx2 = np.random.randint(0, len(self.data_files))
        file_path2 = os.path.join(self.input_dir, self.data_files[idx2])
        data2 = pd.read_csv(file_path2, header=None).values
        x2 = data2[:, 1].reshape(-1, 1)  # 只提取电流（第二列数据）

        # 对电流数据进行归一化
        x2_normalized = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))

        # 读取文件名中的浓度信息，作为标签
        file_name2 = self.data_files[idx2]
        concentration2 = extract_concentration_from_filename(file_name2)
        label2 = torch.tensor(float(concentration2), dtype=torch.float32).view(1)

        # 将电流数据转换为tensor
        x1_tensor = torch.tensor(x1_normalized, dtype=torch.float32)
        x2_tensor = torch.tensor(x2_normalized, dtype=torch.float32)

        # 返回两个电流数据（x1, x2）和它们对应的浓度标签
        return x1_tensor, x2_tensor, label1, label2


# 提取浓度信息的函数，并进行单位转换
def extract_concentration_from_filename(filename, log_transform=True):
    """
    从文件名中提取浓度值，并可选择进行对数变换。

    参数:
        filename (str): 文件名，格式为 'lab_[类别]_[浓度]_[编号]_[数据类型].csv'
        log_transform (bool): 是否对浓度值进行对数变换，默认为 False（不变换）

    返回:
        concentration_value (float): 浓度值，单位为摩尔
    """
    # 文件名格式为 lab_40_100pg_generated0.csv
    parts = filename.split('_')
    concentration = parts[2]  # 直接获取浓度字符串，如 100pg 或 100fg

    # 对浓度单位进行转换
    if 'fg' in concentration:
        concentration_value = float(concentration.replace('fg', '')) * 1e-15
    elif 'pg' in concentration:
        concentration_value = float(concentration.replace('pg', '')) * 1e-12
    else:
        concentration_value = float(concentration)  # 如果没有单位，直接取数值

    # 如果需要，对浓度值进行对数变换
    if log_transform:
        concentration_value = -np.log10(concentration_value)  # 使用自然对数
        # 若要使用以10为底的对数：concentration_value = np.log10(concentration_value)

    return concentration_value


# 配置参数
input_dir = f'C:\\Users\\30300\\Desktop\\Learn_pytorch\\FetProject\\Data\\lab_data\\generated_data_{NUM_SAMPLES}'  # 数据文件夹路径
data_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f.split('_')[1] == mark]  # 获取所有CSV文件

# 划分训练集和测试集
train_files, test_files = train_test_split(data_files, test_size=0.3, random_state=42)

# 数据加载器
train_dataset = ConcentrationDataset(train_files, input_dir)
test_dataset = ConcentrationDataset(test_files, input_dir)

train_loader = DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True)

# 验证加载的数据
# for batch_idx, (x1, x2, concentration1, concentration2) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}:")
#     print("x1 (参比样本电流):")
#     print(x1)
#     print("x2 (目标样本电流):")
#     print(x2)
#     print("Concentration1 (参比样本浓度):")
#     print(concentration1)
#     print("Concentration2 (目标样本浓度):")
#     print(concentration2)
#     print("-" * 50)
#     break
