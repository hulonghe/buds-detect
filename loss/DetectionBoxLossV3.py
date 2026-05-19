import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou

# 假设这些自定义模块已存在
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
    ✅ 支持多类别 (Multi-class)
    ✅ 数值等价 V1
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
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.aux_weight = 0.5

        # 分类损失函数
        if cls_type == 'focal':
            self.cls_loss_fn = SigmoidFocalLoss(gamma=gamma, alpha=alpha, reduction='none')
        elif cls_type == 'vari':
            self.cls_loss_fn = VarifocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

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

    def forward(self, pred_classes, pred_boxes, aux_logits,
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
        # 2. Classification（多类别目标构建）
        # --------------------------
        # 使用 self.cls_num 确保维度正确。
        # 注意：这里假设 pred_classes 的最后一维大小等于 self.cls_num
        target_cls = torch.zeros((B, N, self.cls_num), dtype=pred_classes.dtype, device=device)

        for b in range(B):
            if pos_masks[b].sum() == 0:
                continue

            tgt_labels = target_labels[b]
            matched_gt_idx = matched_indices[b]
            pos_idx = pos_masks[b]

            # 1. 计算正样本匹配到的 GT 框的 IoU
            # pred_boxes[b][pos_idx] 形状: [num_pos, 4]
            # target_boxes[b][matched_gt_idx] 形状: [num_pos, 4]
            ious_pos = box_iou(pred_boxes[b][pos_idx], target_boxes[b][matched_gt_idx])
            
            # box_iou 返回对角线元素，如果是单框则是标量，多框则是向量
            if ious_pos.dim() == 2:
                ious_pos = ious_pos.diag()
            
            # 2. 获取匹配到的 GT 的类别标签
            # matched_gt_idx 是 GT 的索引，我们需要从 tgt_labels 中取出对应的类别 ID
            matched_labels = tgt_labels[matched_gt_idx]

            # 3. 填充目标张量
            # 这是一个高级索引操作：
            # target_cls[b, pos_idx, matched_labels] = ious_pos
            # 这将把对应位置、对应类别的值设为 IoU (用于 Varifocal 等)
            target_cls[b, pos_idx, matched_labels] = ious_pos.detach().to(target_cls.dtype)

        # 计算分类损失
        cls_loss_all = self.cls_loss_fn(pred_classes, target_cls)  # [B,N,C]

        # per-sample mean
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

        iou_thresh = 0.05 + (self.iou_thresh - 0.05) * cosine
        score_thresh = 0.05 + (self.score_thresh - 0.05) * cosine
        topk = max(15, int(25 - 10 * cosine))
        iou_cost_weight = 0.5 + 1.5 * cosine

        ious = box_iou(pred_boxes, gt_boxes)
        pred_probs = torch.sigmoid(pred_cls_scores)

        max_ious_per_pred = ious.max(dim=1).values
        max_scores = pred_probs.max(dim=1).values
        adaptive_thresh = score_thresh * (1 - max_ious_per_pred)
        score_mask = max_scores > adaptive_thresh
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

        _, topk_idx = torch.topk(cost.T, k=topk, largest=False)

        matching_matrix = torch.zeros_like(cost)
        matching_matrix[
            topk_idx.reshape(-1),
            torch.arange(M, device=device).repeat_interleave(topk)
        ] = 1

        pos_mask = matching_matrix.sum(dim=1) > 0
        matched_gt_idx = matching_matrix[pos_mask].argmax(dim=1)

        return pos_mask, matched_gt_idx

    def get_aux_weight(self, cur_epoch, max_epoch):
        if cur_epoch >= max_epoch:
            return 0.0
        return self.aux_weight * (1 - cur_epoch / max_epoch)