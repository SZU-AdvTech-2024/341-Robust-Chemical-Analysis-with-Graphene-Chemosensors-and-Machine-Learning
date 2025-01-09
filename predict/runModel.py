import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_pre import train_loader, test_loader, mark
from model import predict_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd  # 用于保存数据为CSV文件

# 设定全局字体和绘图风格（例如，Nature期刊风格）
plt.rcParams.update({
    'axes.titlesize': 16,  # 设置标题字体大小
    'axes.labelsize': 14,  # 设置坐标轴标签字体大小
    'xtick.labelsize': 12,  # 设置x轴刻度标签字体大小
    'ytick.labelsize': 12,  # 设置y轴刻度标签字体大小
    'legend.fontsize': 12,  # 设置图例字体大小
    'figure.figsize': (8, 6),  # 设置图形大小
    'savefig.dpi': 300,  # 保存图片时的分辨率
    'axes.linewidth': 1.5,  # 坐标轴线宽
})

# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
model = predict_model().to(device)  # 将模型移动到 GPU 或 CPU

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 配置学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# 训练过程
num_epochs = 200  # 训练的轮数

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0

    for batch_idx, (x1, x2, concentration1, concentration2) in enumerate(train_loader):
        # 将数据移动到 GPU 或 CPU
        x1, x2 = x1.to(device), x2.to(device)
        concentration1, concentration2 = concentration1.to(device), concentration2.to(device)

        # 将 x1 和 x2 输入模型，得到输出
        outputs = model(x1, x2)  # 获取模型输出

        if epoch % 100 == 0:
            print("--------预测浓度--------------------")
            print(outputs)
            print("--------浓度标签--------------------")
            print(concentration2)

        # 计算损失
        loss = criterion(outputs, concentration2.view(-1, 1))  # 目标是预测浓度
        running_loss += loss.item()  # 累加损失

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()
        optimizer.step()

    # 打印每个epoch的平均损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}")

    # 更新学习率
    scheduler.step(avg_loss)  # 使用训练损失来调整学习率


result_dir = f'..\\..\\Result\\lab_result\\mark{mark}'
os.makedirs(result_dir, exist_ok=True)

# 保存训练好的模型
torch.save(model.state_dict(), os.path.join(result_dir,f'concentration_predictor_{mark}.pth'))
print("Model saved as 'concentration_predictor.pth'")

# 测试过程
def save_prediction_results(all_labels, all_predictions, mark, result_dir):
    # 保存预测结果和真实值到 CSV 文件
    results_df = pd.DataFrame({
        'True Concentration': all_labels.flatten(),
        'Predicted Concentration': all_predictions.flatten(),
        'Residuals': all_labels.flatten() - all_predictions.flatten()
    })
    results_df.to_csv(os.path.join(result_dir, f'prediction_results_{mark}.csv'), index=False)
    print(f"Prediction results saved to 'prediction_results_{mark}.csv'")

def plot_scatter(all_labels, all_predictions, mark, result_dir):
    # 1. 预测值 vs 真实值的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_predictions, color='blue', alpha=0.5, edgecolor='k')  # 添加边框颜色
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red',
             linestyle='--')  # 理想预测线

    plt.xlabel('True Concentration', fontsize=14)
    plt.ylabel('Predicted Concentration', fontsize=14)
    plt.title(f'True vs Predicted Concentration {mark}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(result_dir, f'scatter_plot_{mark}.png'), dpi=300)  # 高分辨率保存图片
    plt.close()

def plot_residuals(all_labels, all_predictions, mark, result_dir):
    # 2. 残差图
    residuals = all_labels - all_predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(all_predictions, residuals, color='green', alpha=0.5, edgecolor='k')  # 添加边框颜色
    plt.hlines(0, min(all_predictions), max(all_predictions), colors='red', linestyles='--')  # 0 基准线

    plt.xlabel('Predicted Concentration', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.title(f'Residuals Plot {mark}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(result_dir, f'residual_plot_{mark}.png'), dpi=300)  # 高分辨率保存图片
    plt.close()

def test_model(test_loader, model, device, result_dir, mark):
    model.eval()  # 切换到评估模式
    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # 在测试时不需要计算梯度
        for batch_idx, (x1, x2, concentration1, concentration2) in enumerate(test_loader):
            # 将数据移动到 GPU 或 CPU
            x1, x2 = x1.to(device), x2.to(device)
            concentration1, concentration2 = concentration1.to(device), concentration2.to(device)

            # 将 x1 和 x2 输入模型，得到输出
            outputs = model(x1, x2)  # 使用 x1 和 x2 作为输入

            # 保存预测结果和真实标签
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(concentration2.cpu().numpy())

            # 计算损失
            loss = criterion(outputs, concentration2.view(-1, 1))  # 目标是预测浓度
            total_loss += loss.item()
            total_samples += 1

    # 将预测和标签合并成一维数组
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算评估指标
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)

    # 输出性能指标
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # 保存预测结果和真实值到 CSV 文件
    save_prediction_results(all_labels, all_predictions, mark, result_dir)

    # 绘制图形
    plot_scatter(all_labels, all_predictions, mark, result_dir)
    plot_residuals(all_labels, all_predictions, mark, result_dir)

    return mse, rmse, mae, r2


# 加载模型
model.load_state_dict(torch.load(os.path.join(result_dir,f'concentration_predictor_{mark}.pth')))
print("Model loaded.")

# 测试并输出评估指标
test_model(test_loader, model, device, result_dir, mark)
