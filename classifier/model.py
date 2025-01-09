
import torch.nn as nn


''' Model Initialization
'''


class conv1d_model(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改卷积层的输入通道数为 2（数据有 2 个特征通道），输出通道数为 4
        self.cnn = nn.Conv1d(2, 4, 2, stride=1)  # 输入通道为 2, 输出通道为 4
        self.relu = nn.ReLU()
        self.acti = nn.Sigmoid()
        self.flatten = nn.Flatten()
        # 修改全连接层的输入特征数（4 个通道，长度为 200）
        self.linear_pred = nn.Sequential(
            nn.Linear(4 * 201, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 7),  # 输出 7 个类别（根据你的数据标签的形状）
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}\n")  # 打印输入数据的形状
        # x 的形状是 (batch_size, 2, 201)
        # x.shape[0]=batch_size x.shape[2]=201
        x = x.view(x.shape[0], 2, x.shape[2])
        x1 = self.cnn(x)
        x1 = x1.view(x.shape[0], -1)
        x3 = self.linear_pred(x1)
        return x3


