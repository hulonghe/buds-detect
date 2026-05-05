"""
TTA: Test Time Augmentation
推理时数据增强，提升检测精度
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms


class TTA:
    """
    Test Time Augmentation
    
    支持的增强方式:
    - 水平翻转
    - 多尺度推理
    - 多尺度NMS融合
    """
    
    def __init__(self, 
                 scales=[0.75, 1.0, 1.25],
                 flip=True,
                 flip_direction='horizontal',
                 nms_thresh=0.5,
                 score_thresh=0.25):
        
        self.scales = scales
        self.flip = flip
        self.flip_direction = flip_direction
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
    
    def __call__(self, model, images, img_size=640):
        """
        执行TTA推理
        
        Args:
            model: 检测模型
            images: [B, C, H, W] 输入图像
            img_size: 目标尺寸
        
        Returns:
            boxes: [B, N, 4] 归一化框坐标
            scores: [B, N] 置信度
            labels: [B, N] 类别
        """
        device = images.device
        B = images.shape[0]
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # 原始尺度
        boxes_s, scores_s, labels_s = self._inference_single(
            model, images, img_size
        )
        all_boxes.append(boxes_s)
        all_scores.append(scores_s)
        all_labels.append(labels_s)
        
        # 多尺度推理
        for scale in self.scales:
            scaled_images = F.interpolate(
                images, 
                scale_factor=scale, 
                mode='bilinear', 
                align_corners=False
            )
            boxes_s, scores_s, labels_s = self._inference_single(
                model, scaled_images, img_size
            )
            
            # 还原坐标
            if scale != 1.0:
                boxes_s = boxes_s / scale
            
            all_boxes.append(boxes_s)
            all_scores.append(scores_s)
            all_labels.append(labels_s)
        
        # 水平翻转
        if self.flip:
            flipped_images = torch.flip(images, dims=[3])
            boxes_f, scores_f, labels_f = self._inference_single(
                model, flipped_images, img_size
            )
            
            # 翻转坐标
            boxes_f[..., [0, 2]] = 1.0 - boxes_f[..., [2, 0]]
            
            all_boxes.append(boxes_f)
            all_scores.append(scores_f)
            all_labels.append(labels_f)
        
        # 融合所有预测
        final_boxes, final_scores, final_labels = self._merge_results(
            all_boxes, all_scores, all_labels, B
        )
        
        return final_boxes, final_scores, final_labels
    
    def _inference_single(self, model, images, img_size):
        """单次推理"""
        with torch.no_grad():
            cls_scores, reg_preds, iou_preds, _ = model(images)
        
        # 简化的后处理（实际使用dynamic_postprocess）
        from utils.dynamic_postprocess import dynamic_postprocess
        
        cls_scores = cls_scores.sigmoid()
        iou_preds = iou_preds.sigmoid()
        quality_scores = (cls_scores * iou_preds).sqrt()
        
        boxes, scores, labels = dynamic_postprocess(
            quality_scores, reg_preds,
            img_size=(img_size, img_size),
            score_thresh=self.score_thresh,
            iou_thresh=self.nms_thresh,
            upscale=False,
            method='nms'
        )
        
        return boxes, scores, labels
    
    def _merge_results(self, all_boxes, all_scores, all_labels, B):
        """融合多增强结果"""
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for b in range(B):
            # 收集该batch的所有预测
            boxes_list = [all_boxes[i][b] for i in range(len(all_boxes))]
            scores_list = [all_scores[i][b] for i in range(len(all_scores))]
            labels_list = [all_labels[i][b] for i in range(len(all_labels))]
            
            # 拼接
            boxes = torch.cat(boxes_list, dim=0) if boxes_list[0].numel() > 0 else torch.zeros(0, 4)
            scores = torch.cat(scores_list, dim=0) if scores_list[0].numel() > 0 else torch.zeros(0)
            labels = torch.cat(labels_list, dim=0) if labels_list[0].numel() > 0 else torch.zeros(0, dtype=torch.long)
            
            if boxes.numel() > 0:
                # 加权融合 (根据置信度)
                boxes, scores, labels = self._weighted_fusion(
                    boxes, scores, labels
                )
                
                # NMS
                keep = nms(boxes, scores, self.nms_thresh)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            
            final_boxes.append(boxes)
            final_scores.append(scores)
            final_labels.append(labels)
        
        return final_boxes, final_scores, final_labels
    
    def _weighted_fusion(self, boxes, scores, labels):
        """加权融合重复检测"""
        if boxes.shape[0] <= 1:
            return boxes, scores, labels
        
        # 按类别分别处理
        unique_labels = torch.unique(labels)
        
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for label in unique_labels:
            mask = labels == label
            boxes_c = boxes[mask]
            scores_c = scores[mask]
            
            if boxes_c.shape[0] <= 1:
                final_boxes.append(boxes_c)
                final_scores.append(scores_c)
                final_labels.append(label.expand(boxes_c.shape[0]))
                continue
            
            # IoU聚类
            iou_matrix = self._box_iou(boxes_c, boxes_c)
            
            # 简单的加权融合
            for i in range(boxes_c.shape[0]):
                if scores_c[i] < 0.1:
                    continue
                
                # 找高IoU的其他框
                ious = iou_matrix[i]
                overlap_mask = ious > 0.8
                
                if overlap_mask.sum() > 1:
                    # 加权平均
                    weights = scores_c[overlap_mask]
                    weights = weights / weights.sum()
                    
                    boxes_c[i] = (boxes_c[overlap_mask] * weights.unsqueeze(1)).sum(dim=0)
                    scores_c[i] = scores_c[overlap_mask].max()
            
            final_boxes.append(boxes_c)
            final_scores.append(scores_c)
            final_labels.append(label.expand(boxes_c.shape[0]))
        
        return (
            torch.cat(final_boxes, dim=0),
            torch.cat(final_scores, dim=0),
            torch.cat(final_labels, dim=0)
        )
    
    def _box_iou(self, box1, box2):
        """计算IoU矩阵"""
        x1 = torch.maximum(box1[:, None, 0], box2[:, 0])
        y1 = torch.maximum(box1[:, None, 1], box2[:, 1])
        x2 = torch.minimum(box1[:, None, 2], box2[:, 2])
        y2 = torch.minimum(box1[:, None, 3], box2[:, 3])
        
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union = area1[:, None] + area2 - inter
        
        return inter / (union + 1e-6)


class MultiScaleInference:
    """
    多尺度推理 (不带翻转)
    """
    
    def __init__(self, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
        self.scales = scales
    
    def __call__(self, model, images, img_size=640):
        """多尺度推理"""
        device = images.device
        
        all_results = []
        
        for scale in self.scales:
            if scale == 1.0:
                scaled = images
            else:
                scaled = F.interpolate(
                    images,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                )
            
            with torch.no_grad():
                cls_scores, reg_preds, iou_preds, _ = model(scaled)
            
            cls_scores = cls_scores.sigmoid()
            iou_preds = iou_preds.sigmoid()
            quality_scores = (cls_scores * iou_preds).sqrt()
            
            from utils.dynamic_postprocess import dynamic_postprocess
            
            boxes, scores, labels = dynamic_postprocess(
                quality_scores, reg_preds,
                img_size=(img_size, img_size),
                score_thresh=0.25,
                iou_thresh=0.5,
                upscale=False,
                method='nms'
            )
            
            # 还原坐标
            if scale != 1.0:
                boxes = [b / scale for b in boxes]
            
            all_results.append((boxes, scores, labels))
        
        # 融合
        return self._merge(all_results)
    
    def _merge(self, results):
        """融合多尺度结果"""
        B = len(results[0][0])
        
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for b in range(B):
            boxes_list = [r[0][b] for r in results]
            scores_list = [r[1][b] for r in results]
            labels_list = [r[2][b] for r in results]
            
            boxes = torch.cat(boxes_list, dim=0) if boxes_list[0].numel() > 0 else torch.zeros(0, 4)
            scores = torch.cat(scores_list, dim=0) if scores_list[0].numel() > 0 else torch.zeros(0)
            labels = torch.cat(labels_list, dim=0) if labels_list[0].numel() > 0 else torch.zeros(0, dtype=torch.long)
            
            # NMS
            if boxes.numel() > 0:
                keep = nms(boxes, scores, 0.5)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            
            final_boxes.append(boxes)
            final_scores.append(scores)
            final_labels.append(labels)
        
        return final_boxes, final_scores, final_labels


def build_tta(scales=[0.75, 1.0, 1.25], flip=True):
    """构建TTA"""
    return TTA(scales=scales, flip=flip)


if __name__ == '__main__':
    # 测试
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 256, 3, 1, 1)
            self.cls_head = nn.Conv2d(256, 80, 1)
            self.reg_head = nn.Conv2d(256, 4, 1)
            self.iou_head = nn.Conv2d(256, 1, 1)
        
        def forward(self, x):
            f = self.conv(x)
            cls = self.cls_head(f)
            reg = self.reg_head(f)
            iou = self.iou_head(f)
            
            B, C, H, W = cls.shape
            cls = cls.view(B, C, -1).permute(0, 2, 1)
            reg = reg.view(B, 4, -1).permute(0, 2, 1)
            iou = iou.view(B, 1, -1).permute(0, 2, 1)
            
            return cls, reg, iou, None
    
    model = SimpleModel()
    tta = build_tta(scales=[1.0], flip=False)
    
    images = torch.randn(2, 3, 640, 640)
    
    boxes, scores, labels = tta(model, images)
    print(f"Boxes: {[b.shape for b in boxes]}")
    print(f"Scores: {[s.shape for s in scores]}")
    print(f"Labels: {[l.shape for l in labels]}")
