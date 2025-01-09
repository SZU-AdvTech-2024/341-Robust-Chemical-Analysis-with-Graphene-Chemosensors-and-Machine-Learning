import torch
import torch.nn as nn
import numpy as np

# Custom LambdaLayer to implement Cosine similarity metric
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x1, x2):
        return self.lambd(x1, x2)

# Model Initialization
class predict_model(nn.Module):
    def __init__(self):
        super(predict_model, self).__init__()

        # 定义全连接层结构
        self.flatten = nn.Flatten()  # 用于展平输入数据

        self.linear_relu_stack_1 = nn.Sequential(
            nn.Linear(251, 512),  # 输入 250x2（电压 + 电流），输出 512
            nn.ReLU(),
            nn.Linear(512, 1024),  # 输出 1024
            nn.ReLU(),
        )

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(251, 512),  # 输入 250x2（电压 + 电流），输出 512
            nn.ReLU(),
            nn.Linear(512, 1024),  # 输出 1024
            nn.ReLU(),
        )

        # Cosine similarity 层
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # 最终输出层
        self.linear_relu_stack_3 = nn.Sequential(
            nn.Linear(1, 32),  # 输入 1 个值（cosine similarity），输出 32
            nn.ReLU(),
            nn.Linear(32, 1),  # 输出 1 个值（预测浓度）

        )

    def forward(self, x1, x2):
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        # 通过两个全连接网络进行处理
        x1 = self.linear_relu_stack_1(x1)
        x2 = self.linear_relu_stack_2(x2)

        # 计算 cosine similarity
        x3 = self.cos(x1, x2)
        x3 = x3.view(-1, 1)  # 展平成 1 列

        # 通过最终的全连接层获得输出
        x_out = self.linear_relu_stack_3(x3)

        return x_out
