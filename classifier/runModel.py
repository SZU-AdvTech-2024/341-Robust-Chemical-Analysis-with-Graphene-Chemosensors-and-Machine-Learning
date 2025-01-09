import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from Code.NNs.Classifier.Data_prep import train_dt, train_label, test_dt, test_label
from model import conv1d_model

''' Model Initialization
'''




# 定义批次大小
batch = 64
loader = DataLoader(list(zip(train_dt, train_label)), shuffle=True, batch_size=batch)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = conv1d_model().to(device)


# 定义训练参数
n_epochs = 800
loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数用于分类
loss_fn2 = nn.MSELoss()  # 可选的均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

''' Model Training
'''
for epoch in range(n_epochs):
    correct = 0
    sum = 0
    for dta, label in loader:
        dta, label = dta.to(device), label.to(device)  # 将数据和标签移到 GPU
        pred = model(dta)  # 获取模型预测结果
        loss1 = loss_fn(pred, torch.argmax(label, 1))  # 计算交叉熵损失
        loss2 = loss_fn2(pred, label.float())  # 计算均方误差损失
        loss = loss2  # 选择损失函数（这里使用均方误差损失）

        correct += (torch.argmax(pred, 1) == torch.argmax(label, 1)).float().sum()  # 计算正确预测数
        sum += dta.shape[0]

        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    accuracy = 100 * correct / sum
    print(f'Epoch: {epoch} \tTraining Loss: {loss:.6f} \tAccuracy: {accuracy:.6f}')

''' Model Testing
'''
batch_test = 1
loader_test = DataLoader(list(zip(test_dt, test_label)), shuffle=False, batch_size=batch_test)

correct = 0
sum = 0
pred_test = []
label_test = []
test_save = []

for dta, label in loader_test:
    dta, label = dta.to(device), label.to(device)
    pred = model(dta)  # 确保数据的形状符合模型要求
    correct += (torch.argmax(pred, 1) == torch.argmax(label, 1)).float().sum()  # 计算正确预测数
    sum += dta.shape[0]
    pred_test.append(np.array(pred.detach().cpu()).squeeze())
    label_test.append(np.array(label.cpu()).squeeze())
    test_save.append(np.array(dta.detach().cpu().squeeze()))

accuracy = 100 * correct / sum
print(f'Final Test Accuracy: {accuracy:.6f}')

''' Evaluation Metric
'''
torch.save(model.state_dict(), 'model.pth')
test_save = np.array(test_save)
pred_test = np.array(pred_test)
label_test = np.array(label_test)
save_path=''
with open(save_path + "test_save.pkl", "wb") as f:
    pickle.dump(test_save, f)
with open(save_path + "pred_test.pkl", "wb") as f:
    pickle.dump(pred_test, f)
with open(save_path + "label_test.pkl", "wb") as f:
    pickle.dump(label_test, f)
with open(save_path + "train_dt.pkl", "wb") as f:
        pickle.dump(train_dt, f)
with open(save_path + "test_dt.pkl", "wb") as f:
    pickle.dump(test_dt, f)

cm = confusion_matrix(np.argmax(pred_test, 1), np.argmax(label_test, 1))  # 计算混淆矩阵
print(f'Confusion Matrix:\n{cm}')

