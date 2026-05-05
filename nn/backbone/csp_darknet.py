"""
CSPDarknet + SPPF backbone for Object Detection
优化版本：支持P2层输出，适合小目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BatchNorm + Activation"""
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


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNAct(hidden_channels * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPBlock(nn.Module):
    """CSP Bottleneck with optional residual"""
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5, 
                 shortcut=True, groups=1):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1, 1)
        
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, groups=groups, shortcut=shortcut)
            for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvBNAct(hidden_channels * 2, out_channels, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.blocks(x1)
        x2 = self.conv2(x)
        return self.conv3(torch.cat([x1, x2], dim=1))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone with SPPF
    输出4个尺度: C2, C3, C4, C5 (对应stride=4,8,16,32)
    """
    
    def __init__(self, 
                 depth_multiple=1.0,
                 width_multiple=1.0,
                 in_channels=3,
                 out_indices=(2, 3, 4, 5),
                 activation='SiLU'):
        super().__init__()
        
        # 计算实际的层数和通道数
        def make_divisible(x, divisor=8):
            return int((x + divisor / 2) // divisor * divisor)
        
        base_channels = 64
        base_depth = 3
        
        # 根据depth_multiple调整
        depth = lambda n: max(round(n * depth_multiple), 1)
        width = lambda n: make_divisible(n * width_multiple)
        
        self.out_indices = out_indices
        
        # Stem
        self.stem = ConvBNAct(in_channels, width(64), 6, 2, padding=2)
        
        # Stage 1 - stride=4
        self.stage1 = nn.Sequential(
            ConvBNAct(width(64), width(128), 3, 2),
            CSPBlock(width(128), width(128), depth(base_depth), expansion=0.5)
        )
        
        # Stage 2 - stride=8
        self.stage2 = nn.Sequential(
            ConvBNAct(width(128), width(256), 3, 2),
            CSPBlock(width(256), width(256), depth(base_depth * 3), expansion=0.5)
        )
        
        # Stage 3 - stride=16
        self.stage3 = nn.Sequential(
            ConvBNAct(width(256), width(512), 3, 2),
            CSPBlock(width(512), width(512), depth(base_depth * 3), expansion=0.5)
        )
        
        # Stage 4 - stride=32
        self.stage4 = nn.Sequential(
            ConvBNAct(width(512), width(1024), 3, 2),
            CSPBlock(width(1024), width(1024), depth(base_depth), expansion=0.5),
            SPPF(width(1024), width(1024), 5)
        )
        
        # 用于小目标：添加P2层输出
        # C2: stride=4, 用于检测极小目标
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Returns:
            outputs: [C2, C3, C4, C5] 对应stride=[4, 8, 16, 32]
        """
        outputs = []
        
        # Stem
        x = self.stem(x)  # stride=2
        
        # Stage 1
        x = self.stage1(x)  # stride=4 -> C2
        if 2 in self.out_indices:
            outputs.append(x)
        
        # Stage 2
        x = self.stage2(x)  # stride=8 -> C3
        if 3 in self.out_indices:
            outputs.append(x)
        
        # Stage 3
        x = self.stage3(x)  # stride=16 -> C4
        if 4 in self.out_indices:
            outputs.append(x)
        
        # Stage 4
        x = self.stage4(x)  # stride=32 -> C5
        if 5 in self.out_indices:
            outputs.append(x)
        
        return outputs


class CSPDarknetTiny(nn.Module):
    """
    轻量级CSPDarknet，用于移动端/小模型
    """
    
    def __init__(self, in_channels=3, out_indices=(2, 3, 4, 5)):
        super().__init__()
        
        base_channels = 32
        self.out_indices = out_indices
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels, 3, 2),
            ConvBNAct(base_channels, base_channels * 2, 3, 2),
        )
        
        # Stage 1 - C2
        self.stage1 = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 4, 3, 2),
            CSPBlock(base_channels * 4, base_channels * 4, 1, expansion=0.5)
        )
        
        # Stage 2 - C3
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels * 4, base_channels * 8, 3, 2),
            CSPBlock(base_channels * 8, base_channels * 8, 3, expansion=0.5)
        )
        
        # Stage 3 - C4
        self.stage3 = nn.Sequential(
            ConvBNAct(base_channels * 8, base_channels * 16, 3, 2),
            CSPBlock(base_channels * 16, base_channels * 16, 3, expansion=0.5)
        )
        
        # Stage 4 - C5
        self.stage4 = nn.Sequential(
            ConvBNAct(base_channels * 16, base_channels * 32, 3, 2),
            CSPBlock(base_channels * 32, base_channels * 32, 1, expansion=0.5),
            SPPF(base_channels * 32, base_channels * 32, 5)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        outputs = []
        
        x = self.stem(x)
        x = self.stage1(x)
        if 2 in self.out_indices:
            outputs.append(x)
        
        x = self.stage2(x)
        if 3 in self.out_indices:
            outputs.append(x)
        
        x = self.stage3(x)
        if 4 in self.out_indices:
            outputs.append(x)
        
        x = self.stage4(x)
        if 5 in self.out_indices:
            outputs.append(x)
        
        return outputs


def build_cspdarknet(arch='large', in_channels=3):
    """
    构建不同规模的CSPDarknet
    
    Args:
        arch: 'tiny', 'small', 'medium', 'large'
        in_channels: 输入通道数
    
    Returns:
        CSPDarknet模型
    """
    configs = {
        'tiny': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        'small': {'depth_multiple': 0.33, 'width_multiple': 0.5},
        'medium': {'depth_multiple': 0.67, 'width_multiple': 0.75},
        'large': {'depth_multiple': 1.0, 'width_multiple': 1.0},
    }
    
    cfg = configs.get(arch, configs['large'])
    return CSPDarknet(in_channels=in_channels, **cfg)


# 测试
if __name__ == '__main__':
    model = CSPDarknet()
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    for i, out in enumerate(outputs):
        print(f'C{i+2}: {out.shape}')
