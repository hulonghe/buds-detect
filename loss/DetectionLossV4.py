"""
DetectionLossV4 - 简化版检测损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class DetectionLossV4(nn.Module):
    
    def __init__(self,
                 num_classes=1,
                 cls_weight=1.0,
                 box_weight=2.0,
                 fg_iou_threshold=0.5,
                 img_size=320,
                 device='cuda'):
        super().__init__()

        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.fg_iou_threshold = fg_iou_threshold
        self.img_size = img_size
        self.device = device
        self.strides = [8, 16, 32, 64]

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def decode_boxes(self, reg_preds):
        """解码原始预测到xyxy"""
        B, N, _ = reg_preds.shape
        
        decoded = []
        start_idx = 0
        
        for stride in self.strides:
            feat_size = self.img_size // stride
            num_anchors = feat_size * feat_size
            end_idx = start_idx + num_anchors
            
            reg_level = reg_preds[:, start_idx:end_idx]
            start_idx = end_idx
            
            pred = torch.sigmoid(reg_level)
            
            x1 = pred[..., 0:1] * stride / self.img_size
            y1 = pred[..., 1:2] * stride / self.img_size
            x2 = pred[..., 2:3] * stride / self.img_size
            y2 = pred[..., 3:4] * stride / self.img_size
            
            decoded.append(torch.cat([x1, y1, x2, y2], dim=-1))
        
        result = torch.cat(decoded, dim=1)
        return result.clamp(0, 1)

    def forward(self,
                pred_cls, pred_boxes, pred_iou, features,
                target_boxes, target_labels, aux_targets,
                epoch=0, epochs=100, warmup_epochs=0):
        """
        pred_boxes: [B, N, 4] 原始预测
        target_boxes: [B, M, 4] xyxy标签
        """
        B, N, C = pred_cls.shape
        device = pred_cls.device

        pred_boxes_decoded = self.decode_boxes(pred_boxes)
        
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        
        num_pos_total = 0
        num_valid_batch = 0

        for b in range(B):
            gt_boxes = target_boxes[b]
            gt_labels = target_labels[b]
            M = gt_boxes.shape[0]
            
            if M == 0:
                neg_target = torch.zeros((N, C), device=device, dtype=torch.float32)
                loss = self.bce_loss(pred_cls[b], neg_target)
                total_cls_loss += loss
                continue
            
            num_valid_batch += 1
            
            ious = box_iou(pred_boxes_decoded[b], gt_boxes)
            
            max_ious, matched_gt_idx = ious.max(dim=1)
            
            pos_mask = max_ious > self.fg_iou_threshold
            num_pos = pos_mask.sum().item()
            num_pos_total += num_pos
            
            if num_pos > 0:
                target_cls = torch.zeros((N, C), device=device, dtype=torch.float32)
                
                pos_pred_boxes = pred_boxes_decoded[b][pos_mask]
                pos_matched_gt_idx = matched_gt_idx[pos_mask]
                
                valid = pos_matched_gt_idx < M
                pos_pred_boxes = pos_pred_boxes[valid]
                pos_matched_gt_idx = pos_matched_gt_idx[valid]
                
                if len(pos_matched_gt_idx) == 0:
                    continue
                
                pos_gt_labels = gt_labels[pos_matched_gt_idx]
                target_cls[pos_mask, :] = 0
                target_cls[pos_mask][valid, pos_gt_labels] = 1.0
                
                loss_cls = self.bce_loss(pred_cls[b], target_cls)
                total_cls_loss += loss_cls
                
                pos_gt_boxes = gt_boxes[pos_matched_gt_idx]
                
                box_loss = F.l1_loss(pos_pred_boxes, pos_gt_boxes, reduction='mean')
                total_box_loss += box_loss
            else:
                neg_target = torch.zeros((N, C), device=device, dtype=torch.float32)
                loss = self.bce_loss(pred_cls[b], neg_target)
                total_cls_loss += loss

        if num_valid_batch == 0:
            num_valid_batch = 1

        total_cls_loss = total_cls_loss / num_valid_batch * self.cls_weight
        total_box_loss = total_box_loss / max(1, num_pos_total) * self.box_weight

        total_loss = total_cls_loss + total_box_loss
        
        if not torch.isfinite(total_loss):
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)
            total_cls_loss = torch.tensor(0.0, device=device)
            total_box_loss = torch.tensor(0.0, device=device)

        return total_loss, total_cls_loss, total_box_loss, torch.tensor(0.0, device=device)


def build_detection_loss_v4(config):
    return DetectionLossV4(
        num_classes=config.get('num_classes', 1),
        cls_weight=config.get('cls_weight', 1.0),
        box_weight=config.get('box_weight', 2.0),
    )


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = DetectionLossV4(num_classes=1, device=device, img_size=320)

    B, N, C = 2, 8400, 1

    pred_cls = torch.randn(B, N, C).to(device)
    pred_boxes = torch.randn(B, N, 4).to(device)
    pred_iou = torch.randn(B, N, 1).to(device)

    gt_boxes = [torch.rand(5, 4).to(device).clamp(0.01, 0.99), torch.rand(3, 4).to(device).clamp(0.01, 0.99)]
    gt_labels = [torch.zeros(5, dtype=torch.long).to(device), torch.zeros(3, dtype=torch.long).to(device)]

    total_loss, cls_loss, box_loss, iou_loss = criterion(
        pred_cls, pred_boxes, pred_iou, None,
        gt_boxes, gt_labels, None,
        epoch=0, epochs=100, warmup_epochs=5
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Cls Loss: {cls_loss.item():.4f}")
    print(f"Box Loss: {box_loss.item():.4f}")
