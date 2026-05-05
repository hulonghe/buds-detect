"""
Decoupled Head for Object Detection
来自YOLOX的解耦头设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledHead(nn.Module):
    """
    Decoupled Classification and Regression Head
    
    结构:
    feat -> cls_conv -> cls_pred
          -> reg_conv -> reg_pred
    """
    
    def __init__(self, 
                 in_channels,
                 num_classes=80,
                 feat_channels=256,
                 num_layers=2,
                 use_dfl=True):  # DFL: Distribution Focal Loss
        super().__init__()
        
        self.num_classes = num_classes
        self.use_dfl = use_dfl
        
        # 分类分支
        cls_layers = []
        for i in range(num_layers):
            if i == 0:
                cls_layers.append(ConvModule(in_channels, feat_channels, 3, 1))
            else:
                cls_layers.append(ConvModule(feat_channels, feat_channels, 3, 1))
        self.cls_convs = nn.Sequential(*cls_layers)
        
        # 回归分支
        reg_layers = []
        for i in range(num_layers):
            if i == 0:
                reg_layers.append(ConvModule(in_channels, feat_channels, 3, 1))
            else:
                reg_layers.append(ConvModule(feat_channels, feat_channels, 3, 1))
        self.reg_convs = nn.Sequential(*reg_layers)
        
        # 分类输出
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, 1, 1)
        
        # 回归输出
        # DFL需要4*16=64通道输入，DFL内部会将其转换为4通道输出
        # 如果不用DFL，则直接输出4通道
        self.use_dfl = use_dfl
        if use_dfl:
            self.reg_pred = nn.Conv2d(feat_channels, 4 * 16, 1, 1)  # DFL模式: 64通道
            self.dfl = DFL()
        else:
            self.reg_pred = nn.Conv2d(feat_channels, 4, 1, 1)  # 普通模式: 4通道
            self.dfl = None
        
        # IoU质量分支 (可选，用于Quality Focal Loss)
        self.iou_pred = nn.Conv2d(feat_channels, 1, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        # 分类器初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        
        # 回归器初始化
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)
        
        # IoU预测初始化 (使初始输出接近0.5)
        nn.init.constant_(self.iou_pred.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征
        
        Returns:
            cls_score: [B, num_classes, H, W]
            reg_pred:  [B, 4, H, W] (可选经过DFL)
            iou_pred:  [B, 1, H, W]
        """
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        
        cls_score = self.cls_pred(cls_feat)
        
        reg_pred = self.reg_pred(reg_feat)
        if self.use_dfl and self.dfl is not None:
            reg_pred = self.dfl(reg_pred)
        
        iou_pred = self.iou_pred(reg_feat)
        
        return cls_score, reg_pred, iou_pred


class ConvModule(nn.Module):
    """Conv + BN + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL)
    将回归转换为离散分布学习
    """
    
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.register_buffer('project', torch.arange(c1).float().view(1, 1, c1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, 4*c1, H, W] 输入应该是4*c1通道
        
        Returns:
            [B, 4, H, W]
        """
        B, C, H, W = x.shape
        x = x.view(B, 4, self.c1, -1)  # [B, 4, c1, H*W]
        x = F.softmax(x, dim=2)
        x = (x * self.project).sum(2).view(B, 4, H, W)
        return x


class YOLOXHead(nn.Module):
    """
    完整的YOLOX Head (多层)
    支持P2-P5多尺度输出
    """
    
    def __init__(self,
                 num_classes=80,
                 in_channels_list=[256, 256, 256, 256],  # P2, P3, P4, P5
                 feat_channels=256,
                 use_dfl=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        
        # 每个尺度一个检测头
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        
        for in_channels in in_channels_list:
            head = DecoupledHead(
                in_channels,
                num_classes,
                feat_channels,
                use_dfl=use_dfl
            )
            self.cls_heads.append(head.cls_convs)
            self.reg_heads.append(head)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: [P2, P3, P4, P5]
        
        Returns:
            cls_scores: [B, num_classes, H, W] 列表
            reg_preds:  [B, 4, H, W] 列表
            iou_preds:  [B, 1, H, W] 列表
        """
        cls_scores = []
        reg_preds = []
        iou_preds = []
        
        for i, feat in enumerate(features):
            cls_score, reg_pred, iou_pred = self.reg_heads[i](feat)
            cls_scores.append(cls_score)
            reg_preds.append(reg_pred)
            iou_preds.append(iou_pred)
        
        return cls_scores, reg_preds, iou_preds


# 简化的检测头
class LiteDecoupledHead(nn.Module):
    """轻量级解耦头"""
    
    def __init__(self, in_channels, num_classes=80):
        super().__init__()
        
        # 分类分支
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1, 1)
        )
        
        # 回归分支
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 分类器初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_conv[-1].bias, bias_value)
    
    def forward(self, x):
        cls_score = self.cls_conv(x)
        reg_pred = self.reg_conv(x)
        return cls_score, reg_pred


import math

# 测试
if __name__ == '__main__':
    head = YOLOXHead(num_classes=80)
    features = [
        torch.randn(1, 256, 160, 160),
        torch.randn(1, 256, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 256, 20, 20),
    ]
    
    cls_scores, reg_preds, iou_preds = head(features)
    
    for i, (cls, reg, iou) in enumerate(zip(cls_scores, reg_preds, iou_preds)):
        print(f'Level {i}: cls={cls.shape}, reg={reg.shape}, iou={iou.shape}')
