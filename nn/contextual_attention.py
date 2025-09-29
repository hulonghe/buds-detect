import torch
from torch import nn


class ContextualAttention(nn.Module):
    def __init__(self, in_channels):
        super(ContextualAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 假设x为特征图
        context_map = torch.mean(x, dim=1, keepdim=True)  # 获取上下文信息
        context_map = self.conv(context_map)
        attention_map = self.softmax(context_map)
        return x * attention_map  # 在特征图中增强上下文信息
