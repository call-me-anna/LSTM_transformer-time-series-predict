import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_data(file_path='capacity.xlsx', sequence_length=10):  # 修改序列长度为10
    # 读取Excel数据
    try:
        df = pd.read_excel(file_path)
        # 假设数据在第一列，如果列名不同，请修改'capacity'为实际的列名
        data = df.iloc[:, 0].values
        print(f"原始数据点数: {len(data)}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None
    
    # 数据标准化
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    # 创建序列
    sequences = []
    targets = []
    
    for i in range(len(data_normalized) - sequence_length):
        seq = data_normalized[i:i+sequence_length]
        target = data_normalized[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # 转换为PyTorch张量
    sequences = torch.FloatTensor(np.array(sequences))
    targets = torch.FloatTensor(np.array(targets))
    
    # 划分训练集和测试集 (修改为70/30的比例)
    train_size = int(len(sequences) * 0.7)
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]
    
    print(f"数据集大小: 总计 {len(sequences)} 个样本")
    print(f"训练集: {len(train_sequences)} 个样本")
    print(f"测试集: {len(test_sequences)} 个样本")
    
    return (train_sequences, train_targets), (test_sequences, test_targets), scaler

class BatteryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
