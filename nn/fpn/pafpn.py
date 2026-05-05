"""
PAFPN: Path Aggregation Feature Pyramid Network
支持P2-P5多层输出，适合小目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conv + BN + Act"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPLayer(nn.Module):
    """CSP Layer for FPN"""
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=False):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvModule(in_channels, hidden_channels, 1, 1)
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
            for _ in range(num_blocks)
        ])
        self.conv3 = ConvModule(hidden_channels * 2, out_channels, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.blocks(x1)
        x2 = self.conv2(x)
        return self.conv3(torch.cat([x1, x2], dim=1))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = ConvModule(in_channels, out_channels, 1, 1)
        self.conv2 = ConvModule(out_channels, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class PAFPN(nn.Module):
    """
    Path Aggregation FPN (PAFPN)
    支持4层输出: P2, P3, P4, P5
    
    特征金字塔:
    - C2(stride=4) -> P2 (用于小目标)
    - C3(stride=8) -> P3
    - C4(stride=16) -> P4
    - C5(stride=32) -> P5
    """
    
    def __init__(self, 
                 in_channels_list,  # [C2, C3, C4, C5] 的通道数
                 out_channels=256,
                 num_blocks=3,
                 use_csp=True,
                 extra_p5=True):
        super().__init__()
        
        # in_channels_list: [C2, C3, C4, C5]
        c2, c3, c4, c5 = in_channels_list
        
        self.out_channels = out_channels
        self.extra_p5 = extra_p5
        
        # 1. Top-down pathway (FPN)
        # P5 -> P4 -> P3 -> P2
        self.fpn_conv5 = ConvModule(c5, out_channels, 1, 1)
        self.fpn_conv4 = ConvModule(c4, out_channels, 1, 1)
        self.fpn_conv3 = ConvModule(c3, out_channels, 1, 1)
        self.fpn_conv2 = ConvModule(c2, out_channels, 1, 1)
        
        # 融合卷积
        self.fpn_out5 = ConvModule(out_channels, out_channels, 3, 1)
        self.fpn_out4 = ConvModule(out_channels, out_channels, 3, 1)
        self.fpn_out3 = ConvModule(out_channels, out_channels, 3, 1)
        self.fpn_out2 = ConvModule(out_channels, out_channels, 3, 1)
        
        # 2. Bottom-up pathway (PAN)
        # P2 -> P3 -> P4 -> P5
        if use_csp:
            self.pan_conv2 = CSPLayer(out_channels * 2, out_channels, num_blocks)
            self.pan_conv3 = CSPLayer(out_channels * 2, out_channels, num_blocks)
            self.pan_conv4 = CSPLayer(out_channels * 2, out_channels, num_blocks)
            self.pan_conv5 = CSPLayer(out_channels * 2, out_channels, num_blocks)
        else:
            self.pan_conv2 = nn.Sequential(
                ConvModule(out_channels * 2, out_channels, 3, 1),
                ConvModule(out_channels, out_channels, 3, 1)
            )
            self.pan_conv3 = nn.Sequential(
                ConvModule(out_channels * 2, out_channels, 3, 1),
                ConvModule(out_channels, out_channels, 3, 1)
            )
            self.pan_conv4 = nn.Sequential(
                ConvModule(out_channels * 2, out_channels, 3, 1),
                ConvModule(out_channels, out_channels, 3, 1)
            )
            self.pan_conv5 = nn.Sequential(
                ConvModule(out_channels * 2, out_channels, 3, 1),
                ConvModule(out_channels, out_channels, 3, 1)
            )
        
        # 输出卷积
        self.pan_out2 = ConvModule(out_channels, out_channels, 3, 1)
        self.pan_out3 = ConvModule(out_channels, out_channels, 3, 1)
        self.pan_out4 = ConvModule(out_channels, out_channels, 3, 1)
        self.pan_out5 = ConvModule(out_channels, out_channels, 3, 1)
    
    def forward(self, features):
        """
        Args:
            features: [C2, C3, C4, C5] from backbone
        
        Returns:
            [P2, P3, P4, P5] 4层特征
        """
        c2, c3, c4, c5 = features
        
        # ===== Top-down FPN =====
        # P5
        p5 = self.fpn_conv5(c5)
        p5_out = self.fpn_out5(p5)
        
        # P4: C4 + upsampled P5
        p4 = self.fpn_conv4(c4)
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_out = self.fpn_out4(p4)
        
        # P3
        p3 = self.fpn_conv3(c3)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3_out = self.fpn_out3(p3)
        
        # P2 (新增：小目标)
        p2 = self.fpn_conv2(c2)
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        p2_out = self.fpn_out2(p2)
        
        # ===== Bottom-up PAN =====
        # PAN应该下采样：P2(大) -> P3(中) -> P4(小) -> P5(更小)
        # P2 -> P3: 下采样2倍
        p2_down = F.max_pool2d(p2_out, 2)
        # 确保尺寸匹配（处理奇数尺寸情况）
        if p2_down.shape[-1] != p3_out.shape[-1]:
            p2_down = F.interpolate(p2_down, size=p3_out.shape[-2:], mode='nearest')
        p3_in = torch.cat([p3_out, p2_down], dim=1)
        p3 = self.pan_conv2(p3_in)
        p3 = p3 + p3_out
        p3 = self.pan_out3(p3)
        
        # P3 -> P4: 下采样2倍
        p3_down = F.max_pool2d(p3, 2)
        if p3_down.shape[-1] != p4_out.shape[-1]:
            p3_down = F.interpolate(p3_down, size=p4_out.shape[-2:], mode='nearest')
        p4_in = torch.cat([p4_out, p3_down], dim=1)
        p4 = self.pan_conv3(p4_in)
        p4 = p4 + p4_out
        p4 = self.pan_out4(p4)
        
        # P4 -> P5: 下采样2倍
        p4_down = F.max_pool2d(p4, 2)
        if p4_down.shape[-1] != p5_out.shape[-1]:
            p4_down = F.interpolate(p4_down, size=p5_out.shape[-2:], mode='nearest')
        p5_in = torch.cat([p5_out, p4_down], dim=1)
        p5 = self.pan_conv4(p5_in)
        p5 = p5 + p5_out
        p5 = self.pan_out5(p5)
        
        return [p2_out, p3_out, p4_out, p5_out]


class SimpleFPN(nn.Module):
    """
    简化的FPN，只做top-down融合
    适合资源受限场景
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        c2, c3, c4, c5 = in_channels_list
        self.out_channels = out_channels
        
        # Lateral convs
        self.lateral_conv0 = ConvModule(c5, out_channels, 1, 1)
        self.lateral_conv1 = ConvModule(c4, out_channels, 1, 1)
        self.lateral_conv2 = ConvModule(c3, out_channels, 1, 1)
        self.lateral_conv3 = ConvModule(c2, out_channels, 1, 1)
        
        # Output convs
        self.fpn_out0 = ConvModule(out_channels, out_channels, 3, 1)
        self.fpn_out1 = ConvModule(out_channels, out_channels, 3, 1)
        self.fpn_out2 = ConvModule(out_channels, out_channels, 3, 1)
        self.fpn_out3 = ConvModule(out_channels, out_channels, 3, 1)
    
    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Top-down pathway
        p5 = self.lateral_conv0(c5)
        p4 = self.lateral_conv1(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_conv2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_conv3(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        
        # Output
        p5 = self.fpn_out0(p5)
        p4 = self.fpn_out1(p4)
        p3 = self.fpn_out2(p3)
        p2 = self.fpn_out3(p2)
        
        return [p2, p3, p4, p5]


# 测试
if __name__ == '__main__':
    fpn = PAFPN([64, 128, 256, 512], out_channels=256)
    features = [
        torch.randn(1, 64, 160, 160),  # C2
        torch.randn(1, 128, 80, 80),   # C3
        torch.randn(1, 256, 40, 40),   # C4
        torch.randn(1, 512, 20, 20),   # C5
    ]
    
    outputs = fpn(features)
    for i, out in enumerate(outputs):
        print(f'P{i+2}: {out.shape}')
