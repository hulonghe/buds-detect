import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou

# 假设这些自定义模块已存在
from loss.SigmoidFocalLoss import SigmoidFocalLoss
from utils.ciou import complete_ciou
from utils.eiou import complete_eiou
from utils.diou import complete_diou
from utils.bbox import xyxy2xywh
from loss.balanced_l1_loss import BalancedL1Loss
from utils.gwd import gwd_loss


class DetectionLossV3(nn.Module):
    """
    ✅ 支持多类别 (Multi-class)
    ✅ 分离"是否前景"(cls)和"匹配质量"(quality)
    ✅ 避免 NxM 内存爆炸
    """

    def __init__(self, score_thresh=0.25, iou_thresh=0.25,
                 iou_type="ciou", box_type="gwd", cls_type='bce',
                 device_type="cpu", sum_weighted=False,
                 gamma=1.0, alpha=0.25, beta=2.0, cls_num=1):
        """
        Args:
            cls_num (int): 类别总数。类别索引假设为 [0, cls_num-1]。
        """
        super().__init__()

        self.cls_num = cls_num
        self.log_vars = nn.Parameter(torch.zeros(4))  # cls, box, iou, quality
        self.aux_weight = 0.5

        # 分类损失：二分类前景/背景 (SigmoidFocalLoss)
        self.cls_loss_fn = SigmoidFocalLoss(gamma=gamma, alpha=alpha, reduction='none')

        # 质量损失：IoU 回归 (BCE)
        self.quality_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # 边框损失函数
        if box_type == 'balanced':
            self.balanced_l1_loss = BalancedL1Loss(beta=beta, alpha=alpha, gamma=gamma, reduction='none')

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.iou_type = iou_type
        self.box_type = box_type
        self.sum_weighted = sum_weighted

    # --------------------------
    # weighted loss
    # --------------------------
    def compute_weighted_loss(self, losses):
        if self.sum_weighted:
            return sum(losses)

        out = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            out += precision * loss + self.log_vars[i]
        return out

    # --------------------------
    # forward
    # --------------------------
    def compute_sosa_loss(self, cls_pred, pred_boxes, gt_boxes, alpha=1.0, beta=1.0, gamma=2.0):
        device = cls_pred.device
        num_pos = cls_pred.size(0)
        if num_pos == 0:
            return torch.tensor(0.0, device=device)

        cls_pred = torch.sigmoid(cls_pred)
        cls_score, _ = cls_pred.max(dim=1)

        with torch.no_grad():
            ious = box_iou(pred_boxes, gt_boxes)
            ious = torch.clamp(ious, min=1e-6, max=1.0)
            q = torch.clamp((cls_score ** alpha) * (ious ** beta), min=1e-6, max=1.0)

        weights = (1.0 - q) ** gamma
        sosa_loss = - (weights * torch.log(q + 1e-6))
        if torch.isnan(sosa_loss).any():
            sosa_loss = torch.zeros_like(q).mean()
        else:
            sosa_loss = sosa_loss.mean()
            norm_factor = sosa_loss.detach() + 1.0
            sosa_loss = sosa_loss / norm_factor
        return sosa_loss

    def forward(self, pred_classes, pred_boxes, pred_quality, aux_logits,
                target_boxes, target_labels, aux_targets,
                epoch, epochs, warmup_epoch, weights=None):

        device = pred_classes.device
        B, N, C = pred_classes.shape
        
        # 安全检查：确保预测的类别通道数与定义的 cls_num 一致
        # 如果 pred_classes 的 C 维度与 cls_num 不一致，通常意味着模型输出层配置错误
        # 这里我们信任传入的 pred_classes 形状，但使用 self.cls_num 来初始化 target

        # --------------------------
        # 1. Assign（多类别匹配）
        # --------------------------
        pos_masks = []
        matched_indices = []

        for b in range(B):
            tgt_boxes = target_boxes[b].to(device)
            tgt_labels = target_labels[b]

            if tgt_boxes.numel() == 0:
                pos_masks.append(torch.zeros(N, dtype=torch.bool, device=device))
                matched_indices.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            pos_mask, matched_gt_idx = self.dynamic_assign(
                pred_boxes[b],
                pred_classes[b],
                tgt_boxes,
                tgt_labels,
                epoch, epochs, warmup_epoch
            )

            pos_masks.append(pos_mask)
            matched_indices.append(matched_gt_idx)

        pos_mask = torch.stack(pos_masks)  # [B,N]

        # --------------------------
        # 2. Classification（二分类前景/背景） + Quality（IoU 回归）
        # --------------------------
        # cls_target: binary 1/0 — 该位置是否为前景
        target_cls = torch.zeros((B, N, self.cls_num), dtype=pred_classes.dtype, device=device)

        for b in range(B):
            if pos_masks[b].sum() == 0:
                continue

            tgt_labels = target_labels[b]
            matched_gt_idx = matched_indices[b]
            pos_idx = pos_masks[b]

            matched_labels = tgt_labels[matched_gt_idx]

            # 分类目标：binary 1（该位置是前景）
            target_cls[b, pos_idx, matched_labels] = 1.0

        # 分类损失
        cls_loss_all = self.cls_loss_fn(pred_classes, target_cls)  # [B,N,C]

        num_pos_total = pos_mask.sum()
        if num_pos_total > 0:
            loss_cls = cls_loss_all.sum() / num_pos_total
        else:
            cls_loss_all = cls_loss_all.view(B, -1)
            loss_cls = cls_loss_all.mean(dim=1).mean()

        # 质量损失：用 centerness 代替 IoU，提供尖锐的正负区分度
        # centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)) 在偏离中心时急剧衰减
        if num_pos_total > 0:
            pos_flat_q = pos_mask.view(-1)
            pred_pos_boxes_q = pred_boxes.view(-1, 4)[pos_flat_q]
            gt_list_q = []
            for b in range(B):
                if pos_masks[b].sum() == 0:
                    continue
                gt_list_q.append(target_boxes[b][matched_indices[b]])
            matched_gt_boxes_q = torch.cat(gt_list_q, dim=0)

            pred_cx = (pred_pos_boxes_q[:, 0] + pred_pos_boxes_q[:, 2]) / 2
            pred_cy = (pred_pos_boxes_q[:, 1] + pred_pos_boxes_q[:, 3]) / 2
            gt_cx = (matched_gt_boxes_q[:, 0] + matched_gt_boxes_q[:, 2]) / 2
            gt_cy = (matched_gt_boxes_q[:, 1] + matched_gt_boxes_q[:, 3]) / 2

            l = pred_cx - matched_gt_boxes_q[:, 0]
            r = matched_gt_boxes_q[:, 2] - pred_cx
            t = pred_cy - matched_gt_boxes_q[:, 1]
            b = matched_gt_boxes_q[:, 3] - pred_cy

            # centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
            # clamp ratio 到 [0,1]：预测中心在框外时 ratio 为负 → clamp 到 0 → centerness=0
            # lr_ratio = (torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)).clamp(min=0.0, max=1.0)
            # tb_ratio = (torch.min(t, b) / torch.max(t, b).clamp(min=1e-6)).clamp(min=0.0, max=1.0)
            # centerness = torch.sqrt(lr_ratio * tb_ratio)

            # target_quality = torch.zeros((B, N, 1), dtype=pred_classes.dtype, device=device)
            # target_quality.view(B, N)[pos_mask] = centerness.detach().to(target_quality.dtype)

            with torch.no_grad():
                # matched_gt_boxes_q 和 pred_pos_boxes_q 已经在你的代码中提取出来了
                ious_for_quality = box_iou(pred_pos_boxes_q, matched_gt_boxes_q).diag().clamp(0, 1.0)
            target_quality = torch.zeros((B, N, 1), dtype=pred_classes.dtype, device=device)
            target_quality.view(B, N)[pos_mask] = ious_for_quality.to(target_quality.dtype)

            quality_loss_all = self.quality_loss_fn(pred_quality, target_quality)  # [B,N,1]
            quality_loss_all = quality_loss_all.squeeze(-1)
            loss_quality = (quality_loss_all * pos_mask.float()).sum() / num_pos_total
            if torch.isnan(loss_quality) or torch.isinf(loss_quality):
                loss_quality = torch.tensor(0.0, device=device)
        else:
            loss_quality = torch.tensor(0.0, device=device)

        # --------------------------
        # 3. Box + IoU（向量化）
        # --------------------------
        pos_flat = pos_mask.view(-1)

        if pos_flat.sum() > 0:
            pred_pos_boxes = pred_boxes.view(-1, 4)[pos_flat]

            gt_list = []
            for b in range(B):
                if pos_masks[b].sum() == 0:
                    continue
                gt_list.append(target_boxes[b][matched_indices[b]])

            matched_gt_boxes = torch.cat(gt_list, dim=0)

            pred_xywh = xyxy2xywh(pred_pos_boxes)
            gt_xywh = xyxy2xywh(matched_gt_boxes)

            # box loss
            if self.box_type == "gwd":
                loss_box = gwd_loss(pred_xywh, gt_xywh)
            elif self.box_type == "balanced":
                loss_box = self.balanced_l1_loss(pred_xywh, gt_xywh).mean()
            elif self.box_type == "sosa":
                pred_pos_cls = pred_classes.view(-1, C)[pos_flat]
                loss_box = self.compute_sosa_loss(pred_pos_cls, pred_pos_boxes, matched_gt_boxes)
            else:
                loss_box = F.smooth_l1_loss(pred_xywh, gt_xywh)

            # IoU loss
            if self.iou_type == "ciou":
                ciou = complete_ciou(pred_pos_boxes, matched_gt_boxes)
                loss_iou = (1 - ciou).mean()
            elif self.iou_type == "giou":
                giou = generalized_box_iou(pred_pos_boxes, matched_gt_boxes)
                loss_iou = (1 - giou).mean()
            elif self.iou_type == "diou":
                diou = complete_diou(pred_pos_boxes, matched_gt_boxes)
                loss_iou = (1 - diou).mean()
            else:
                eiou = complete_eiou(pred_pos_boxes, matched_gt_boxes)
                loss_iou = (1 - eiou).mean()
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_iou = torch.tensor(0.0, device=device)

        # --------------------------
        # 4. Aux loss
        # --------------------------
        loss_aux = torch.tensor(0.0, device=device)

        if aux_logits is not None and aux_targets is not None:
            for b in range(B):
                tgt_c2 = torch.tensor(aux_targets[b]['small'], device=device).float()
                tgt_c3 = torch.tensor(aux_targets[b]['normal'], device=device).float()

                loss_aux += F.binary_cross_entropy_with_logits(
                    aux_logits['small'][b], tgt_c2.unsqueeze(0)
                )
                loss_aux += F.binary_cross_entropy_with_logits(
                    aux_logits['normal'][b], tgt_c3.unsqueeze(0)
                )

            loss_aux /= B

        # --------------------------
        # 5. 权重融合
        # --------------------------
        # 始终应用手动权重，作为 Loss 计算的先验基准
        if weights is not None:
            cls_w, box_w, iou_w = weights[:3]
            quality_w = weights[3] if len(weights) > 3 else 1.0
            if cls_w is not None:
                loss_cls = loss_cls * float(cls_w)
            if box_w is not None:
                loss_box = loss_box * float(box_w)
            if iou_w is not None:
                loss_iou = loss_iou * float(iou_w)
            loss_quality = loss_quality * float(quality_w)

        total_loss = self.compute_weighted_loss([loss_cls, loss_box, loss_iou, loss_quality])
        total_loss += self.get_aux_weight(epoch, epochs * 0.3) * loss_aux

        return total_loss, loss_cls.detach(), loss_box.detach(), loss_iou.detach(), loss_quality.detach()

    # --------------------------
    # dynamic assign（多类别版本）
    # --------------------------
    def dynamic_assign(self, pred_boxes, pred_cls_scores, gt_boxes, tgt_labels,
                       epoch, epochs, warmup_epoch):
        import math

        N, M = pred_boxes.size(0), gt_boxes.size(0)
        device = pred_boxes.device

        if M == 0 or N == 0:
            return torch.zeros(N, dtype=torch.bool, device=device), torch.zeros(0, dtype=torch.long, device=device)

        progress = epoch / max(1, epochs - 1)
        cosine = (1 - math.cos(math.pi * progress)) / 2

        iou_thresh = 0.2 + (self.iou_thresh - 0.2) * cosine
        score_thresh = 0.05 + (self.score_thresh - 0.05) * cosine
        topk = max(4, int(10 - 6 * cosine))
        iou_cost_weight = 0.5 + 1.5 * cosine

        ious = box_iou(pred_boxes, gt_boxes)
        pred_probs = torch.sigmoid(pred_cls_scores)

        max_ious_per_pred = ious.max(dim=1).values
        max_scores = pred_probs.max(dim=1).values
        # 使用固定阈值进行初步过滤，避免自适应阈值方向反导致的优化错误
        score_mask = max_scores > 0.01
        valid_mask = (ious > iou_thresh) & score_mask.unsqueeze(1)

        tgt_labels_exp = tgt_labels.view(1, -1).expand(N, -1)
        prob_pos = torch.gather(pred_probs, 1, tgt_labels_exp)
        cls_cost = -torch.log(prob_pos.clamp(min=1e-7))

        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        center_scale = 0.05
        dist_x = (pred_cx.unsqueeze(1) - gt_cx.unsqueeze(0)) / center_scale
        dist_y = (pred_cy.unsqueeze(1) - gt_cy.unsqueeze(0)) / center_scale
        center_dist = torch.sqrt(dist_x ** 2 + dist_y ** 2)
        center_cost = torch.clamp(center_dist, 0, 10) * 0.15

        cost = cls_cost - ious * iou_cost_weight + center_cost
        cost[~valid_mask] = 1e5

        # 防止候选框不足 topk 导致 torch.topk 报错
        k = min(topk, cost.size(1))
        _, topk_idx = torch.topk(cost.T, k=k, largest=False)

        matching_matrix = torch.zeros_like(cost)
        for gt_idx in range(M):
            matching_matrix[topk_idx[gt_idx], gt_idx] = 1

        pos_mask = matching_matrix.sum(dim=1) > 0
        if pos_mask.sum() == 0:
            matched_gt_idx = torch.zeros(0, dtype=torch.long, device=device)
        else:
            matched_gt_idx = matching_matrix[pos_mask].argmax(dim=1)

        return pos_mask, matched_gt_idx

    def get_aux_weight(self, cur_epoch, max_epoch):
        if cur_epoch >= max_epoch:
            return 0.0
        return self.aux_weight * (1 - cur_epoch / max_epoch)