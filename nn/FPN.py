import torch.nn as nn
import torch.nn.functional as F


class DirectionAwareConv(nn.Module):
    def __init__(self, in_c, out_c):
        """
            方向感知卷积
        """
        super().__init__()
        self.h = nn.Conv2d(in_c, out_c, kernel_size=(1, 3), padding=(0, 1))
        self.v = nn.Conv2d(in_c, out_c, kernel_size=(3, 1), padding=(1, 0))
        self.std = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        return self.h(x) + self.v(x) + self.std(x)


class DynamicFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, direction_aware=True):
        """
        :param in_channels_list: 输入特征通道数列表，例如 [32, 96, 320] or [24, 32, 96, 320]
        :param out_channels: 所有输出特征统一通道数
        """
        super().__init__()
        self.num_ins = len(in_channels_list)
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            # 横向 1x1 conv 压缩通道
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            # 输出卷积 3x3 conv 平滑
            if direction_aware:
                self.output_convs.append(
                    DirectionAwareConv(out_channels, out_channels)
                )
            else:
                self.output_convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                )

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        # Step 1: 横向通道对齐
        lateral_feats = [l_conv(feat) for l_conv, feat in zip(self.lateral_convs, inputs)]

        # Step 2: 自顶向下融合特征
        out_feats = [None] * self.num_ins
        out_feats[-1] = lateral_feats[-1]
        for i in reversed(range(self.num_ins - 1)):
            up = F.interpolate(out_feats[i + 1], size=lateral_feats[i].shape[2:], mode='nearest')
            out_feats[i] = lateral_feats[i] + up

        # Step 3: 每个融合后特征加 3x3 卷积平滑
        out_feats = [conv(feat) for conv, feat in zip(self.output_convs, out_feats)]

        return out_feats  # 返回 P2~P5 或 P3~P5，具体取决于输入
