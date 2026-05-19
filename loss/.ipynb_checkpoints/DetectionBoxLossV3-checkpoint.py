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

        # 1. 计算所有预测框和所有GT框的 IoU
        # ious shape: [N, M] (N: anchors, M: gt_boxes)
        ious = box_iou(pred_boxes, gt_boxes)
        
        pred_probs = torch.sigmoid(pred_cls_scores) # [N, C]

        # 2. 构建类别成本矩阵
        # 我们需要知道每个预测框对于每个GT类别的预测概率
        # tgt_labels shape: [M]
        # 我们需要从 pred_probs 中取出对应类别的概率
        
        # 方法：将 tgt_labels 扩展为 [1, M]，然后 expand 到 [N, M]
        # 这样我们就有了每个 GT 对应的类别 ID
        tgt_labels_exp = tgt_labels.view(1, -1).expand(pred_boxes.size(0), -1) # [N, M]
        
        # 使用 gather 获取每个 GT 对应类别的预测概率
        # pred_probs 是 [N, C], tgt_labels_exp 是 [N, M]
        # 结果 prob_pos 是 [N, M]，表示预测框 i 预测为 GT j 类别的概率
        prob_pos = torch.gather(pred_probs, 1, tgt_labels_exp)
        
        # 3. 计算分类成本
        # 概率越高，成本越低 (-log(p))
        cls_cost = -torch.log(prob_pos.clamp(min=1e-7))

        # 4. 总成本
        # 我们想要最小化分类成本，最大化 IoU
        # cost = cls_cost - ious * factor
        cost = cls_cost - ious * 2.0

        # 5. TopK 匹配
        topk = min(15, pred_boxes.size(0))
        # 对每个 GT (列)，找出成本最小的 TopK 个预测框 (行)
        # cost.T shape: [M, N]
        _, topk_idx = torch.topk(cost.T, k=topk, largest=False)

        # 6. 构建匹配矩阵
        matching_matrix = torch.zeros_like(cost) # [N, M]
        
        # 将 topk_idx 展平，作为行索引
        # 列索引是每个 GT 重复 topk 次
        matching_matrix[
            topk_idx.reshape(-1), 
            torch.arange(gt_boxes.size(0), device=cost.device).repeat_interleave(topk)
        ] = 1

        # 7. 确定正样本
        # 如果某行（预测框）被分配给了至少一个 GT，则为正样本
        pos_mask = matching_matrix.sum(dim=1) > 0
        
        # 对于正样本，找到它匹配的是哪个 GT (取最大 IoU 或最小成本的那个)
        # 这里简单地取匹配矩阵中为 1 的索引
        # 注意：如果一个预测框匹配了多个 GT，通常取 IoU 最大的那个
        # 但在这种 TopK 策略中，通常一个预测框只匹配一个 GT，或者取第一个
        matched_gt_idx = matching_matrix[pos_mask].argmax(dim=1)

        return pos_mask, matched_gt_idx

    def get_aux_weight(self, cur_epoch, max_epoch):
        if cur_epoch >= max_epoch:
            return 0.0
        return self.aux_weight * (1 - cur_epoch / max_epoch)