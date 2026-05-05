"""
DynamicBoxDetectorV2 - 极简版目标检测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nn.backbone.csp_darknet import CSPDarknet
from nn.fpn.pafpn import PAFPN
from nn.head.decoupled_head import YOLOXHead


class DynamicBoxDetectorV2(nn.Module):
    
    def __init__(self,
                 img_size=320,
                 num_classes=1,
                 backbone_type='cspdarknet',
                 fpn_type='pafpn',
                 use_dfl=False,
                 hidden_dim=128,
                 dropout=0.0,
                 device='cuda',
                 fpn_out_channels=128,
                 extra_p2=True):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = CSPDarknet(out_indices=(2, 3, 4, 5))
        
        # FPN
        self.fpn = PAFPN([128, 256, 512, 1024], fpn_out_channels, use_csp=False)
        
        # Head
        self.det_head = YOLOXHead(
            num_classes=num_classes,
            in_channels_list=[fpn_out_channels] * 4,
            feat_channels=fpn_out_channels,
            use_dfl=False
        )
        
        self.strides = [8, 16, 32, 64]
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        if hasattr(self.det_head, 'reg_heads'):
            for head in self.det_head.reg_heads:
                if hasattr(head, 'cls_pred'):
                    nn.init.constant_(head.cls_pred.bias, bias_value)
    
    def decode_boxes(self, reg_preds, img_size):
        """
        解码: 原始预测 -> xyxy 归一化坐标
        """
        B, N, _ = reg_preds.shape
        
        decoded = []
        start_idx = 0
        
        for stride in self.strides:
            feat_size = img_size // stride
            num_anchors = feat_size * feat_size
            end_idx = start_idx + num_anchors
            
            reg_level = reg_preds[:, start_idx:end_idx]
            start_idx = end_idx
            
            pred = torch.sigmoid(reg_level)
            
            x1 = pred[..., 0:1] * stride / img_size
            y1 = pred[..., 1:2] * stride / img_size
            x2 = pred[..., 2:3] * stride / img_size
            y2 = pred[..., 3:4] * stride / img_size
            
            decoded.append(torch.cat([x1, y1, x2, y2], dim=-1))
        
        result = torch.cat(decoded, dim=1)
        return result.clamp(0, 1)
    
    def forward(self, x):
        B = x.shape[0]
        
        features = self.backbone(x)
        p_features = self.fpn(features)
        
        cls_scores, reg_preds, iou_preds = self.det_head(p_features)
        
        cls_list = []
        reg_list = []
        iou_list = []
        
        for cls, reg, iou in zip(cls_scores, reg_preds, iou_preds):
            B, C, H, W = cls.shape
            cls_list.append(cls.view(B, C, -1).permute(0, 2, 1))
            reg_list.append(reg.view(B, 4, -1).permute(0, 2, 1))
            iou_list.append(iou.view(B, 1, -1).permute(0, 2, 1))
        
        cls_scores = torch.cat(cls_list, dim=1)
        reg_preds = torch.cat(reg_list, dim=1)
        iou_preds = torch.cat(iou_list, dim=1)
        
        return cls_scores, reg_preds, iou_preds, p_features
    
    def inference(self, x, conf_thresh=0.25, iou_thresh=0.45):
        cls_scores, reg_preds, iou_preds, _ = self(x)
        
        decoded_boxes = self.decode_boxes(reg_preds, self.img_size)
        
        cls_scores_sigmoid = cls_scores.sigmoid()
        iou_scores_sigmoid = iou_preds.sigmoid()
        quality_scores = cls_scores_sigmoid * iou_scores_sigmoid
        
        from torchvision.ops import nms
        
        B = quality_scores.shape[0]
        results = []
        
        for b in range(B):
            scores = quality_scores[b].max(dim=1)[0]
            labels = quality_scores[b].argmax(dim=1)
            boxes = decoded_boxes[b]
            
            keep = scores > conf_thresh
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            if len(boxes) == 0:
                results.append((torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
                continue
            
            keep = nms(boxes, scores, iou_thresh)
            results.append((boxes[keep], scores[keep], labels[keep]))
        
        return results


def build_detector_v2(config):
    return DynamicBoxDetectorV2(
        img_size=config.get('img_size', 320),
        num_classes=config.get('num_classes', 1),
    )


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DynamicBoxDetectorV2(
        img_size=320,
        num_classes=1,
    ).to(device)
    
    x = torch.randn(2, 3, 320, 320).to(device)
    
    cls_scores, reg_preds, iou_preds, features = model(x)
    
    print(f"输入: {x.shape}")
    print(f"分类: {cls_scores.shape}")
    print(f"回归: {reg_preds.shape}")
    print(f"IoU: {iou_preds.shape}")
    
    decoded = model.decode_boxes(reg_preds, 320)
    print(f"解码后: {decoded.shape}, range: [{decoded.min():.3f}, {decoded.max():.3f}]")
