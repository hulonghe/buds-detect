import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss
    来自FCOS和ATSS的改进，将分类和IoU质量结合
    
    核心思想：
    - 使用IoU作为分类目标的权重
    - 对正样本进行质量感知训练
    - 对负样本使用focal weight
    """
    
    def __init__(self, beta=2.0, alpha=0.25, loss_weight=1.0):
        super().__init__()
        self.beta = beta  # focal gamma
        self.alpha = alpha  # focal alpha
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, target_iou=None):
        """
        Args:
            pred: [B, N, C] 预测分数 (logits)
            target: [B, N, C] one-hot目标
            target_iou: [B, N] IoU质量分数（可选）
        
        Returns:
            loss: scalar
        """
        pred_sigmoid = pred.sigmoid()
        
        if target_iou is not None:
            # Quality Focal Loss: 使用IoU作为权重
            target_weight = target * target_iou.unsqueeze(-1).clamp(0, 1)
        else:
            target_weight = target
        
        # Focal weight
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * \
                       ((1 - pt) ** self.beta)
        
        # BCE loss with focal weight
        bce = -(target * torch.log(pred_sigmoid + 1e-7) + 
                 (1 - target) * torch.log(1 - pred_sigmoid + 1e-7))
        
        loss = focal_weight * bce
        return loss.sum() / (target.sum() + 1e-6)


class FocalIoULoss(nn.Module):
    """
    Focal IoU Loss
    来自FocalNet/IOS等工作的改进
    
    核心思想：
    - 对高质量IoU样本(接近1)降低权重
    - 对低质量IoU样本(接近0)增加权重
    - 类似于Focal Loss的思想
    """
    
    def __init__(self, gamma=1.5, alpha=0.75, loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight
    
    def forward(self, pred_iou, target_iou):
        """
        Args:
            pred_iou: [N] 预测IoU
            target_iou: [N] 目标IoU
        
        Returns:
            loss: scalar
        """
        # 防止梯度爆炸
        pred_iou = pred_iou.clamp(1e-4, 1.0 - 1e-4)
        target_iou = target_iou.clamp(1e-4, 1.0 - 1e-4)
        
        # Focal weight
        weight = self.alpha * (1 - target_iou) ** self.gamma
        
        # BCE-style IoU loss - 使用更稳定的实现
        log_pred = torch.log(pred_iou.clamp(min=1e-7))
        log_1_pred = torch.log((1 - pred_iou).clamp(min=1e-7))
        
        bce = -(target_iou * log_pred + (1 - target_iou) * log_1_pred)
        
        # 处理 inf 值
        bce = torch.where(torch.isfinite(bce), bce, torch.zeros_like(bce))
        loss = weight * bce
        
        return loss.mean()


class GIoUFocalLoss(nn.Module):
    """
    GIoU + Focal组合损失
    对小目标的IoU损失增加权重
    """
    
    def __init__(self, gamma=1.0, loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1,y1,x2,y2)
            target_boxes: [N, 4]
        
        Returns:
            loss: scalar
        """
        # 计算GIoU
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_boxes.unbind(-1)
        
        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
        union_area = pred_area + tgt_area - inter_area + 1e-7
        
        # IoU
        iou = inter_area / union_area
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, tgt_x1)
        enclose_y1 = torch.min(pred_y1, tgt_y1)
        enclose_x2 = torch.max(pred_x2, tgt_x2)
        enclose_y2 = torch.max(pred_y2, tgt_y2)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7
        
        # GIoU
        giou = iou - (enclose_area - union_area) / enclose_area
        
        # Focal weight based on IoU quality
        weight = (1 - giou) ** self.gamma
        
        loss = weight * (1 - giou)
        return loss.mean()


class ComboLoss(nn.Module):
    """
    组合损失: Focal Loss + GIoU Loss + IoU Loss
    适用于YOLO系列检测器
    """
    
    def __init__(self, cls_weight=1.0, box_weight=1.0, iou_weight=1.0,
                 cls_beta=2.0, cls_alpha=0.25, box_gamma=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.iou_weight = iou_weight
        
        self.focal_loss = QualityFocalLoss(beta=cls_beta, alpha=cls_alpha)
        self.giou_loss = GIoUFocalLoss(gamma=box_gamma)
        self.focal_iou_loss = FocalIoULoss(gamma=1.5, alpha=0.75)
    
    def forward(self, pred_cls, pred_boxes, target_cls, target_boxes, target_iou=None):
        """
        Args:
            pred_cls: [B, N, C]
            pred_boxes: [B, N, 4]
            target_cls: [B, N, C] one-hot
            target_boxes: [B, N, 4]
            target_iou: [B, N] (可选)
        
        Returns:
            total_loss, cls_loss, box_loss, iou_loss
        """
        # 分类损失
        cls_loss = self.focal_loss(pred_cls, target_cls, target_iou)
        
        # 过滤正样本
        pos_mask = target_cls.sum(dim=-1) > 0
        
        if pos_mask.sum() > 0:
            pred_pos = pred_boxes[pos_mask]
            tgt_pos = target_boxes[pos_mask]
            
            # GIoU损失
            box_loss = self.giou_loss(pred_pos, tgt_pos)
            
            # Focal IoU损失
            from torchvision.ops import box_iou
            ious = box_iou(pred_pos, tgt_pos).diagonal()
            iou_loss = self.focal_iou_loss(ious, ious.detach())
        else:
            box_loss = pred_boxes.sum() * 0
            iou_loss = pred_boxes.sum() * 0
        
        total_loss = (
            self.cls_weight * cls_loss + 
            self.box_weight * box_loss + 
            self.iou_weight * iou_loss
        )
        
        return total_loss, cls_loss, box_loss, iou_loss
