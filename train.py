import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data_utils import load_data, BatteryDataset
from lstm_model import LSTMPredictor
from transformer_model import TransformerPredictor
from lstm_transformer_model import LSTMTransformerPredictor
import matplotlib.pyplot as plt
import os
import shutil
import json
import datetime


# train model . 
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, device='cuda'):
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(float(avg_train_loss))  # 转换为Python float
        
        # 评估
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(float(avg_test_loss))  # 转换为Python float
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    return [float(x) for x in train_losses], [float(x) for x in test_losses]

def evaluate_model(model, test_loader, scaler, device='cuda'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            
            # 转换回原始范围
            pred = scaler.inverse_transform(output.cpu().numpy())
            actual = scaler.inverse_transform(batch_y.cpu().numpy())
            
            # 转换为Python list
            predictions.extend([float(x) for x in pred.flatten()])
            actuals.extend([float(x) for x in actual.flatten()])
    
    # 计算指标
    mse = float(np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)]))
    mae = float(np.mean([abs(p - a) for p, a in zip(predictions, actuals)]))
    
    return predictions, actuals, mse, mae

def cleanup():
    """清理所有缓存和临时文件"""
    # 清理模型文件
    model_files = ['lstm_model.pth', 'transformer_model.pth', 'lstm_transformer_model.pth']
    for file in model_files:
        if os.path.exists(file):
            os.remove(file)
    
    # 清理图片文件
    if os.path.exists('results'):
        shutil.rmtree('results')
    os.makedirs('results', exist_ok=True)
    
    # 清理Python缓存
    for root, dirs, files in os.walk('.'):
        for dir in dirs:
            if dir == '__pycache__':
                shutil.rmtree(os.path.join(root, dir))
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("清理完成！")

def save_training_results(model_name, mse, mae, train_losses, test_losses, predictions, run_id):
    """保存每次训练的结果"""
    results = {
        'model': model_name,
        'mse': float(mse),
        'mae': float(mae),
        'train_losses': [float(x) for x in train_losses],
        'test_losses': [float(x) for x in test_losses],
        'predictions': [float(x) for x in predictions],
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存结果
    result_file = f'results/training_run_{model_name.lower().replace("-", "_")}_{run_id}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/loss_curves_{model_name.lower().replace("-", "_")}_{run_id}.png')
    plt.close()

def plot_predictions(predictions_dict, actuals, run_id):
    plt.figure(figsize=(15, 8))
    plt.plot(actuals, label='Actual', color='black', alpha=0.5)
    
    colors = {'LSTM': 'blue', 'Transformer': 'red', 'LSTM-Transformer': 'green'}
    for model_name, preds in predictions_dict.items():
        plt.plot(preds, label=f'{model_name}', color=colors.get(model_name, 'gray'), alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Predictions Comparison')
    plt.legend()
    plt.savefig(f'results/predictions_comparison_{run_id}.png')
    plt.close()

def main():
    # 在训练开始前清理
    cleanup()
    
    # 生成唯一的运行ID
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Starting training run: {run_id}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    (train_sequences, train_targets), (test_sequences, test_targets), scaler = load_data(sequence_length=10)
    
    # 创建数据加载器
    train_dataset = BatteryDataset(train_sequences, train_targets)
    test_dataset = BatteryDataset(test_sequences, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 定义模型
    models = {
        'LSTM': LSTMPredictor(),
        'Transformer': TransformerPredictor(),
        'LSTM-Transformer': LSTMTransformerPredictor()
    }
    
    # 存储所有模型的结果
    all_predictions = {}
    actuals = None
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        train_losses, test_losses = train_model(
            model, train_loader, test_loader, criterion, optimizer, device=device
        )
        
        # 评估模型
        predictions, current_actuals, mse, mae = evaluate_model(model, test_loader, scaler, device)
        if actuals is None:
            actuals = current_actuals
        
        # 存储预测结果
        all_predictions[model_name] = predictions
        
        # 保存模型
        torch.save(model.state_dict(), f'{model_name.lower().replace("-", "_")}_model.pth')
        
        # 打印结果
        print(f"\n{model_name} Results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # 保存训练结果
        save_training_results(model_name, mse, mae, train_losses, test_losses, predictions, run_id)
    
    # 绘制所有模型的预测对比图
    plot_predictions(all_predictions, actuals, run_id)
    
    print(f"\n训练完成！所有结果已保存到 results 目录")

if __name__ == '__main__':
    main()