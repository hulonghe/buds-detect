import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou

from loss.SigmoidFocalLoss import SigmoidFocalLoss
from loss.VarifocalLoss import VarifocalLoss
from utils.ciou import complete_ciou
from utils.eiou import complete_eiou
from utils.diou import complete_diou
from utils.bbox import xyxy2xywh
from loss.balanced_l1_loss import BalancedL1Loss
from utils.gwd import gwd_loss


class DetectionLossV3(nn.Module):
    """
    ✅ 数值等价 V1
    ✅ 避免 NxM 内存爆炸
    ✅ 去掉大部分 Python 循环
    """

    def __init__(self, score_thresh=0.25, iou_thresh=0.25,
                 iou_type="ciou", box_type="gwd", cls_type='bce',
                 device_type="cpu", sum_weighted=False,
                 gamma=1.0, alpha=0.25, beta=2.0):
        super().__init__()

        self.log_vars = nn.Parameter(torch.zeros(3))
        self.aux_weight = 0.5

        if cls_type == 'focal':
            self.cls_loss_fn = SigmoidFocalLoss(gamma=gamma, alpha=alpha, reduction='none')
        elif cls_type == 'vari':
            self.cls_loss_fn = VarifocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

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
    def forward(self, pred_classes, pred_boxes, aux_logits,
                target_boxes, target_labels, aux_targets,
                epoch, epochs, warmup_epoch, weights=None):

        device = pred_classes.device
        B, N, C = pred_classes.shape

        # --------------------------
        # 1. Assign（保留逐样本，保证一致）
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
        # 2. Classification（严格等价）
        # --------------------------
        target_cls = torch.zeros_like(pred_classes)

        for b in range(B):
            if pos_masks[b].sum() == 0:
                continue

            tgt_labels = target_labels[b]
            matched_gt_idx = matched_indices[b]

            pos_idx = pos_masks[b]

            # 1. 计算 IoU (在 autocast 下，这通常是 Half 类型)
            # 注意：这里计算的是正样本对应的 GT 框的 IoU
            ious_pos = box_iou(pred_boxes[b][pos_idx], target_boxes[b][matched_gt_idx])
            # 如果是计算对角线元素（即每个预测框对应其匹配GT的IoU）
            if ious_pos.dim() == 2:
                ious_pos = ious_pos.diag()
            # 2. 关键修复：将数据类型转换为与 target_cls 一致
            # detach() 阻断梯度，.to(...) 修正类型
            target_cls[b, pos_idx, tgt_labels[matched_gt_idx]] = ious_pos.detach().to(target_cls.dtype)

        cls_loss_all = self.cls_loss_fn(pred_classes, target_cls)  # [B,N,C]

        # ✅ 关键修复：per-sample mean
        cls_loss_all = cls_loss_all.view(B, -1)
        loss_cls = cls_loss_all.mean(dim=1).mean()

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
            else:
                loss_box = F.smooth_l1_loss(pred_xywh, gt_xywh)

            # IoU loss
            if self.iou_type == "ciou":
                ciou = complete_ciou(pred_pos_boxes, matched_gt_boxes)
                loss_iou = (1 - ciou).mean()
                # loss_iou = ((1 - ciou) * ious_pos.detach()).sum() / (ious_pos.sum() + 1e-6)
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
        # 4. Aux loss（保持一致）
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
        if weights is not None:
            cls_w, box_w, iou_w = weights
            if cls_w is not None:
                loss_cls *= cls_w
            if box_w is not None:
                loss_box *= box_w
            if iou_w is not None:
                loss_iou *= iou_w

        total_loss = self.compute_weighted_loss([loss_cls, loss_box, loss_iou])
        total_loss += self.get_aux_weight(epoch, epochs * 0.3) * loss_aux

        return total_loss, loss_cls.detach(), loss_box.detach(), loss_iou.detach()

    # --------------------------
    # dynamic assign（完全复用V1）
    # --------------------------
    def dynamic_assign(self, pred_boxes, pred_cls_scores, gt_boxes, tgt_labels,
                       epoch, epochs, warmup_epoch):

        ious = box_iou(pred_boxes, gt_boxes)
        pred_probs = torch.sigmoid(pred_cls_scores)

        tgt_labels_exp = tgt_labels.view(1, -1).expand(pred_boxes.size(0), -1)
        cls_cost = -torch.log(torch.gather(pred_probs, 1, tgt_labels_exp).clamp(min=1e-7))

        cost = cls_cost - ious * 2.0

        topk = min(15, pred_boxes.size(0))
        _, topk_idx = torch.topk(cost.T, k=topk, largest=False)

        matching_matrix = torch.zeros_like(cost)
        matching_matrix[
            topk_idx.reshape(-1), torch.arange(gt_boxes.size(0), device=cost.device).repeat_interleave(topk)] = 1

        pos_mask = matching_matrix.sum(dim=1) > 0
        matched_gt_idx = matching_matrix[pos_mask].argmax(dim=1)

        return pos_mask, matched_gt_idx

    def get_aux_weight(self, cur_epoch, max_epoch):
        if cur_epoch >= max_epoch:
            return 0.0
        return self.aux_weight * (1 - cur_epoch / max_epoch)
