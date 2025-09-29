import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        # Squeeze: 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: 使用两层全连接网络生成通道注意力
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        # Sigmoid 激活用于生成每个通道的权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入特征的批量大小
        batch_size, channels, _, _ = x.size()

        # Squeeze阶段，计算全局平均池化
        y = self.global_avg_pool(x)  # 输出尺寸: [batch_size, channels, 1, 1]
        y = y.view(batch_size, channels)  # 压缩为 [batch_size, channels]

        # Excitation阶段，利用两层全连接网络生成通道权重
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)  # 转回 [batch_size, channels, 1, 1]

        # 将生成的通道注意力权重应用到输入特征图
        return x * y.expand_as(x)  # 扩展成与输入特征图相同的尺寸
