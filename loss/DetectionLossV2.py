import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou


class DetectionLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 reg_type='giou',  # 'giou', 'diou', 'ciou'
                 reg_weight=0.5,
                 obj_weight=2.0,
                 cls_weight=1.0,
                 use_focal_loss=False,
                 img_size=(640, 640),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss

    def forward(self, preds, targets, strides):
        total_cls_loss = 0.
        total_obj_loss = 0.
        total_reg_loss = 0.

        for i, (obj_logits, reg_preds, cls_logits) in enumerate(preds):
            B, _, H, W = obj_logits.shape
            device = obj_logits.device
            stride = strides[i]

            # 生成网格中心坐标
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            coords = torch.stack((xv, yv), dim=-1).to(device)  # [H,W,2]
            coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]
            coords = coords * stride + stride // 2  # 映射到像素空间

            # 初始化目标
            cls_target = torch.zeros_like(cls_logits)
            obj_target = torch.zeros_like(obj_logits)
            reg_target = torch.zeros_like(reg_preds)
            reg_mask = torch.zeros_like(obj_logits).bool()

            for b in range(B):
                gt = targets[b]
                boxes = gt['boxes'].clone().to(device)
                labels = gt['labels'].to(device)

                for j, (x1, y1, x2, y2) in enumerate(boxes):
                    gt_center = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2], device=device)
                    cx = int(gt_center[0] / stride)
                    cy = int(gt_center[1] / stride)

                    if 0 <= cx < W and 0 <= cy < H:
                        reg_target[b, :, cy, cx] = torch.tensor([
                            gt_center[0] - x1,
                            gt_center[1] - y1,
                            x2 - gt_center[0],
                            y2 - gt_center[1]
                        ])
                        cls_target[b, labels[j], cy, cx] = 1
                        obj_target[b, 0, cy, cx] = 1
                        reg_mask[b, 0, cy, cx] = True

            # 分类损失
            if self.use_focal_loss:
                cls_loss = self.focal_loss(cls_logits, cls_target, alpha=0.25, gamma=2.0)
                obj_loss = self.focal_loss(obj_logits, obj_target, alpha=0.25, gamma=2.0)
            else:
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction='sum') / B
                obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction='sum') / B

            # 回归损失
            if reg_mask.sum() > 0:
                pred_boxes = self.decode_ltrb(reg_preds, coords)
                target_boxes = self.decode_ltrb(reg_target, coords)

                pred_flat = pred_boxes[reg_mask.squeeze()]
                target_flat = target_boxes[reg_mask.squeeze()]

                iou = self.compute_iou(pred_flat, target_flat, self.reg_type).clamp(min=1e-6)
                reg_loss = (1.0 - iou).mean()
            else:
                reg_loss = torch.tensor(0.0, device=device)

            total_cls_loss += cls_loss * self.cls_weight
            total_obj_loss += obj_loss * self.obj_weight
            total_reg_loss += reg_loss * self.reg_weight

        return total_cls_loss + total_obj_loss + total_reg_loss, total_cls_loss, total_obj_loss, total_reg_loss

    def decode_ltrb(self, ltrb, center):
        l = ltrb[:, 0, :, :].clamp(min=0)
        t = ltrb[:, 1, :, :].clamp(min=0)
        r = ltrb[:, 2, :, :].clamp(min=0)
        b = ltrb[:, 3, :, :].clamp(min=0)
        x1 = center[..., 0] - l
        y1 = center[..., 1] - t
        x2 = center[..., 0] + r
        y2 = center[..., 1] + b
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def compute_iou(self, boxes1, boxes2, mode='giou'):
        if mode == 'giou':
            return torch.diag(generalized_box_iou(boxes1, boxes2))
        elif mode in ['diou', 'ciou']:
            return self.diou_ciou(boxes1, boxes2, mode)
        else:
            raise ValueError(f"Unknown IoU type: {mode}")

    def diou_ciou(self, boxes1, boxes2, mode='diou', eps=1e-7):
        # boxes: [N, 4] in xyxy

        x1, y1, x2, y2 = boxes1.unbind(1)
        x1g, y1g, x2g, y2g = boxes2.unbind(1)

        # areas
        area1 = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area2 = (x2g - x1g).clamp(0) * (y2g - y1g).clamp(0)

        # intersection
        inter_x1 = torch.max(x1, x1g)
        inter_y1 = torch.max(y1, y1g)
        inter_x2 = torch.min(x2, x2g)
        inter_y2 = torch.min(y2, y2g)
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        union = area1 + area2 - inter + eps
        iou = inter / union

        # center distance
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        cx2 = (x1g + x2g) / 2
        cy2 = (y1g + y2g) / 2
        center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # enclosing box
        enclose_x1 = torch.min(x1, x1g)
        enclose_y1 = torch.min(y1, y1g)
        enclose_x2 = torch.max(x2, x2g)
        enclose_y2 = torch.max(y2, y2g)
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

        if mode == 'diou':
            return iou - center_dist / enclose_diag
        else:  # ciou
            v = (4 / (3.141592653589793 ** 2)) * (torch.atan((x2 - x1) / (y2 - y1 + eps)) -
                                                  torch.atan((x2g - x1g) / (y2g - y1g + eps))) ** 2
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            return iou - center_dist / enclose_diag - alpha * v

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        modulating_factor = (1.0 - p_t) ** gamma
        loss = alpha_factor * modulating_factor * ce_loss
        return loss.sum() / pred.size(0)
