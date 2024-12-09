import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
from data_utils import load_data, BatteryDataset
from lstm_model import LSTMPredictor
from transformer_model import TransformerPredictor
from lstm_transformer_model import LSTMTransformerPredictor

def load_results(results_dir='results'):
    """加载所有训练结果  """
    all_results = {}
    result_files = glob(os.path.join(results_dir, 'training_run_*.json'))
    
    if not result_files:
        print(f"Warning: No JSON files found in '{results_dir}'")
        return {}
    
    print(f"Found {len(result_files)} result files")
    for file in result_files:
        try:
            with open(file, 'r') as f:
                content = f.read()
                if len(content.strip()) < 10:
                    print(f"Warning: Skipping {file} due to insufficient content")
                    continue
                try:
                    results = json.loads(content)
                    model_name = results.get('model')
                    if not model_name:
                        print(f"Warning: Skipping {file} due to missing 'model' key")
                        continue
                    all_results[model_name] = {
                        'mse': float(results['mse']),
                        'mae': float(results['mae']),
                        'train_losses': [float(x) for x in results['train_losses']],
                        'test_losses': [float(x) for x in results['test_losses']],
                        'predictions': [float(x) for x in results['predictions']]
                    }
                    print(f"Loaded results for model: {model_name}")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error in {file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error reading file {file}: {str(e)}")
            continue
    
    if not all_results:
        print("Error: No valid result files were loaded")
        return {}
    
    print(f"Successfully loaded results for {len(all_results)} models: {list(all_results.keys())}")
    return all_results

def plot_losses(results):
    """绘制所有模型的损失曲线"""
    plt.figure(figsize=(12, 5))
    
    # 训练损失
    plt.subplot(1, 2, 1)
    for model_name, data in results.items():
        plt.plot(data['train_losses'], label=f'{model_name}')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 测试损失
    plt.subplot(1, 2, 2)
    for model_name, data in results.items():
        plt.plot(data['test_losses'], label=f'{model_name}')
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/loss_comparison.png')
    plt.close()

def plot_predictions(results):
    """绘制所有模型的预测结果对比"""
    plt.figure(figsize=(15, 8))
    
    # 获取任意一个模型的预测长度作为x轴
    x_axis = range(len(next(iter(results.values()))['predictions']))
    
    colors = {'LSTM': 'blue', 'Transformer': 'red', 'LSTM-Transformer': 'green'}
    for model_name, data in results.items():
        plt.plot(x_axis, data['predictions'], 
                label=f'{model_name}', 
                color=colors.get(model_name, 'gray'),
                alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Predictions Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/predictions_comparison.png')
    plt.close()

def plot_complete_data():
    """绘制完整数据集和所有模型的预测结果"""
    # 加载原始数据
    (train_sequences, train_targets), (test_sequences, test_targets), scaler = load_data(sequence_length=10)
    
    # 转换数据回原始范围
    train_data = scaler.inverse_transform(train_targets.reshape(-1, 1)).flatten()
    test_data = scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
    
    # 创建时间步
    train_steps = range(len(train_data))
    test_steps = range(len(train_data), len(train_data) + len(test_data))
    
    plt.figure(figsize=(15, 8))
    
    # 绘制训练数据
    plt.plot(train_steps, train_data, 'o-', color='black', label='Training Data', 
             markersize=2, alpha=0.5, linewidth=1)
    
    # 绘制测试数据
    plt.plot(test_steps, test_data, 's-', color='darkgray', label='Test Data', 
             markersize=2, alpha=0.5, linewidth=1)
    
    # 加载并绘制模型预测结果
    results = load_results()
    colors = {'LSTM': 'blue', 'Transformer': 'red', 'LSTM-Transformer': 'green'}
    
    for model_name, data in results.items():
        predictions = data['predictions']
        pred_steps = range(len(train_data), len(train_data) + len(predictions))
        plt.plot(pred_steps, predictions, 
                label=f'{model_name} Predictions',
                color=colors.get(model_name, 'gray'),
                alpha=0.7,
                linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Battery Capacity')
    plt.title('Complete Dataset and Model Predictions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围，确保数据显示合适
    plt.margins(x=0.02)
    
    plt.savefig('results/complete_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(results):
    """绘制模型性能指标对比"""
    models = list(results.keys())
    mse_values = [results[m]['mse'] for m in models]
    mae_values = [results[m]['mae'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, mse_values, width, label='MSE')
    plt.bar(x + width/2, mae_values, width, label='MAE')
    
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('results/metrics_comparison.png')
    plt.close()

def print_metrics(results):
    """打印所有模型的性能指标"""
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"{'Model':<20} {'MSE':<15} {'MAE':<15}")
    print("-" * 50)
    for model_name, data in results.items():
        print(f"{model_name:<20} {data['mse']:<15.6f} {data['mae']:<15.6f}")
    print("-" * 50)

def main():
    # 加载训练结果
    results = load_results()
    
    if not results:
        print("No training results found. Please run training first.")
        return
    
    # 创建可视化结果目录
    os.makedirs('results', exist_ok=True)
    
    # 绘制所有对比图
    print("\nGenerating visualizations...")
    plot_losses(results)
    plot_predictions(results)
    plot_metrics(results)
    
    # 绘制完整数据集可视化
    print("Generating complete dataset visualization...")
    plot_complete_data()
    
    # 打印性能指标
    print_metrics(results)
    
    print("\n可视化完成！所有图表已保存到 results 目录：")
    print("1. loss_comparison.png - 损失曲线对比")
    print("2. predictions_comparison.png - 预测结果对比")
    print("3. metrics_comparison.png - 性能指标对比")
    print("4. complete_visualization.png - 完整数据集可视化")

if __name__ == '__main__':
    main()