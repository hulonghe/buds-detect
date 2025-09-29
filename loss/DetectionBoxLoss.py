import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou

from loss.SigmoidFocalLoss import SigmoidFocalLoss
from loss.VarifocalLoss import VarifocalLoss
from utils.ciou import complete_ciou
from utils.eiou import complete_eiou
from utils.gwd import gwd_loss
from utils.diou import complete_diou
from utils.bbox import xyxy2xywh
from loss.balanced_l1_loss import BalancedL1Loss
from utils.dynamic_assign_ciou import da_complete_ciou


class DetectionLoss(nn.Module):

    def __init__(self, score_thresh=0.25, iou_thresh=0.25,
                 iou_type="ciou", box_type="gwd", cls_type='bce',
                 device_type="cpu", sum_weighted=False,
                 gamma=1.0, alpha=0.25, beta=2.0):
        super().__init__()
        # 每项任务引入一个可学习的 log sigma^2
        self.log_vars = torch.nn.Parameter(torch.zeros(3))
        self.aux_weight = 0.5

        if cls_type == 'focal':
            self.cls_loss_fn = SigmoidFocalLoss(gamma=gamma, alpha=alpha, reduction='mean', device_type=device_type)
        elif cls_type == 'vari':
            self.cls_loss_fn = VarifocalLoss(alpha=alpha, gamma=gamma, from_logits=True, reduction='mean',
                                             device_type=device_type)
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        if box_type == 'balanced':
            self.balanced_l1_loss = BalancedL1Loss(beta=beta, alpha=alpha, gamma=gamma, reduction='mean')

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.iou_type = iou_type
        self.box_type = box_type
        self.device_type = device_type
        self.sum_weighted = sum_weighted
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_weighted_loss(self, losses):
        if self.sum_weighted:
            return sum(losses)

        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted = precision * loss + self.log_vars[i]  # precision-loss + log σ
            weighted_losses.append(weighted)
        return sum(weighted_losses)

    def compute_sosa_loss(self, cls_pred, pred_boxes, gt_boxes, alpha=1.0, beta=1.0, gamma=2.0):
        """
        cls_pred: [num_pos, C] → 未sigmoid
        pred_boxes: [num_pos, 4]
        gt_boxes: [num_pos, 4]
        返回：L_sosa
        """
        device = cls_pred.device
        num_pos = cls_pred.size(0)
        if num_pos == 0:
            return torch.tensor(0.0, device=device)

        # 获取每个预测框对应的最大类概率
        cls_pred = torch.sigmoid(cls_pred)  # 确保在 [0,1]
        cls_score, _ = cls_pred.max(dim=1)  # [num_pos]

        # 计算 CIoU 作为回归质量
        with torch.no_grad():
            ious = box_iou(pred_boxes, gt_boxes)  # [num_pos]
            ious = torch.clamp(ious, min=1e-6, max=1.0)
            # 计算融合质量分 q
            q = torch.clamp((cls_score ** alpha) * (ious ** beta), min=1e-6, max=1.0)

        # 计算 SOSA 损失
        weights = (1.0 - q) ** gamma
        sosa_loss = - (weights * torch.log(q + 1e-6))
        if torch.isnan(sosa_loss).any():
            sosa_loss = torch.zeros_like(q).mean()
        else:
            sosa_loss = sosa_loss.mean()
            norm_factor = (sosa_loss.detach() + 1.0)  # 从图外计算，不参与梯度
            sosa_loss = sosa_loss / norm_factor * 1.0  # 结果在 [0,2]

        return sosa_loss

    def forward(self, pred_classes, pred_boxes, aux_logits,
                target_boxes, target_labels, aux_targets,
                epoch, epochs, warmup_epoch,
                weights=None):
        """
            pred_classes: 预测分类 shape [B,N,C] ，把分类得分同时作为置信度，分类ID为C的下标，从0开始
            pred_boxes: 预测边框回归 shape [B,N,4] 边框为[x1,y1,x2,y2]
            target_boxes: 真实框 shape [B, N, 4] 同上
            target_labels: 框标签 shape [B,N] 每个框的类别ID，从0开始
        """

        device = pred_classes.device
        B, N, _ = pred_classes.shape

        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_iou = torch.tensor(0.0, device=device)
        loss_aux = torch.tensor(0.0, device=device)

        for b in range(B):
            pred_logits = pred_classes[b]  # [N, C]，已经是 sigmoid 后的结果（多标签置信度）
            tgt_boxes = target_boxes[b].to(device)
            tgt_labels = target_labels[b]
            num_gt = tgt_boxes.size(0)

            if aux_logits is not None and aux_targets is not None:
                tgt_c2 = torch.tensor(aux_targets[b]['small'], device=device, dtype=torch.float32)
                tgt_c3 = torch.tensor(aux_targets[b]['normal'], device=device, dtype=torch.float32)
                loss_aux += F.binary_cross_entropy_with_logits(aux_logits['small'][b], tgt_c2.unsqueeze(0))
                loss_aux += F.binary_cross_entropy_with_logits(aux_logits['normal'][b], tgt_c3.unsqueeze(0))

            if num_gt == 0:
                # 没有 GT 时，只计算背景分类 loss
                loss_cls += self.cls_loss_fn(pred_logits, None)
                continue

            pos_mask, matched_gt_idx = self.dynamic_assign(pred_boxes[b], pred_logits,
                                                           tgt_boxes, tgt_labels,
                                                           epoch=epoch, epochs=epochs, warmup_epoch=warmup_epoch)
            # 如果没有正样本，只计算背景 loss
            if pos_mask.sum() == 0:
                loss_cls += self.cls_loss_fn(pred_logits, targets=None, no_positive=True)
                continue

            # 构造 one-hot 形式标签
            target_cls = torch.zeros_like(pred_logits)  # [N, C]
            target_label = tgt_labels[matched_gt_idx]  # [num_pos]
            target_cls[pos_mask, target_label] = 1.0  # one-hot 编码目标类别
            # 调用 focal loss（输入是 sigmoid 输出的置信度）
            loss_cls += self.cls_loss_fn(pred_logits, target_cls)

            if pos_mask.sum() > 0:
                pred_pos_boxes = pred_boxes[b][pos_mask]
                matched_gt = tgt_boxes[matched_gt_idx]

                # 把 box 分解为坐标
                gt_xywh = xyxy2xywh(matched_gt)
                pred_xywh = xyxy2xywh(pred_pos_boxes)
                if self.box_type == "gwd":
                    loss_box += gwd_loss(pred_xywh, gt_xywh)
                elif self.box_type == 'balanced':
                    loss_box += self.balanced_l1_loss(pred_xywh, gt_xywh)
                elif self.box_type == 'sosa':
                    loss_box += self.compute_sosa_loss(pred_logits[pos_mask], pred_pos_boxes, matched_gt)
                elif self.box_type == 'l1':
                    loss_box += F.smooth_l1_loss(pred_xywh, gt_xywh)

                # [iou_type] Loss
                if self.iou_type == "ciou":
                    ciou = complete_ciou(pred_pos_boxes, matched_gt)
                    loss_iou += (1.0 - ciou).mean()
                if self.iou_type == "giou":
                    giou = generalized_box_iou(pred_pos_boxes, matched_gt)
                    loss_iou += (1.0 - giou).mean()
                if self.iou_type == "eiou":
                    eiou = complete_eiou(pred_pos_boxes, matched_gt)
                    loss_iou += (1.0 - eiou).mean()
                if self.iou_type == 'diou':
                    diou = complete_diou(pred_pos_boxes, matched_gt)
                    loss_iou += (1.0 - diou).mean()
        # 平均到 batch
        loss_cls /= B
        loss_box /= B
        loss_iou /= B
        loss_aux /= B

        if weights is not None:
            cls_w, box_w, iou_w = weights
            if cls_w is not None:
                loss_cls *= cls_w
            if box_w is not None:
                loss_box *= box_w
            if iou_w is not None:
                loss_iou *= iou_w

        weighted_loss = self.compute_weighted_loss([loss_cls, loss_box, loss_iou])
        weighted_loss += self.get_aux_weight(epoch, epochs * 0.3) * loss_aux

        return weighted_loss, loss_cls.detach(), loss_box.detach(), loss_iou.detach()

    def dynamic_assign(self, pred_boxes, pred_cls_scores, gt_boxes, tgt_labels,
                       topk=10, epoch=0, epochs=100, warmup_epoch=5, decay_epochs=5):
        """
        Args:
            pred_boxes: Tensor[N, 4] 已经Sigmoid，边框为[x1,y1,x2,y2]
            pred_cls_scores: Tensor[N, C] 未进行sigmoid
            gt_boxes: Tensor[M, 4] 同上
            tgt_labels: Tensor[M] 框所属类别ID，从0开始
            topk: 初始每个 GT 最多选取的预测框数
            epoch: 当前训练轮数
            warmup_epoch: 仅 IoU 阈值 / topk_ratio 预热的轮数
            decay_epochs: 在 warmup_epoch+10 后，做 topk_ratio/iou_thresh 过渡的轮数
        Returns:
            pos_mask: BoolTensor[N]，正样本掩码
            matched_gt_idx: LongTensor[#pos]，每个正样本对应的 GT 索引
        """
        N, M = pred_boxes.size(0), gt_boxes.size(0)
        device = pred_boxes.device

        if M == 0 or N == 0:
            return torch.zeros(N, dtype=torch.bool, device=device), torch.zeros(0, dtype=torch.long, device=device)

        # 1) 调用调度器，获取当前阈值
        iou_thresh, cur_score_thresh, cur_topk = self.get_thresholds(
            epoch=epoch, total_epochs=epochs, N=N, M=M, topk=topk, warmup_epoch=warmup_epoch, decay_epochs=decay_epochs
        )
        # 2) 计算 IoU 矩阵
        ious = box_iou(pred_boxes, gt_boxes)
        # 3) 计算每个预测框的最大 IoU，用于 adaptive score filter
        max_ious_per_pred = ious.max(dim=1).values  # [N]
        # 4) 计算每个预测框的分类最大概率
        pred_probs = torch.sigmoid(pred_cls_scores)  # [N, C]
        max_scores = pred_probs.max(dim=1).values  # [N]
        # 5) IoU-aware score filter
        adaptive_thresh = cur_score_thresh * (1 - max_ious_per_pred)  # IoU 越大，允许 score 越低
        score_mask = max_scores > adaptive_thresh  # [N]
        # 6) valid_mask，用于后续匹配
        valid_mask = (ious > iou_thresh) & score_mask.unsqueeze(1)  # [N, M]

        # === 分类代价（交叉熵）===
        tgt_labels_exp = tgt_labels.view(1, M).expand(N, M)  # [N, M]
        cls_cost = -torch.log(torch.gather(pred_probs, 1, tgt_labels_exp))
        cls_cost[~valid_mask] = 1e5  # 屏蔽无效位置

        # === 综合代价：分类 cost - IoU 奖励 ===
        # cost = cls_cost - 1.5 * ious  # [N, M]
        cost = cls_cost * 1.0 - ious * 1.5
        # === TopK 匈牙利分配 ===
        cur_topk_clamped = min(cur_topk, N)
        topk_cost, topk_idx = torch.topk(cost.T, k=min(cur_topk_clamped, N), largest=False)
        matching_matrix = torch.zeros((N, M), device=device)
        arange_M = torch.arange(M, device=device).unsqueeze(1).expand(-1, cur_topk_clamped)
        matching_matrix[topk_idx.reshape(-1), arange_M.reshape(-1)] = 1

        pos_mask = matching_matrix.sum(dim=1) > 0
        matched_gt_idx = matching_matrix[pos_mask].argmax(dim=1)

        return pos_mask, matched_gt_idx

    def get_thresholds(self, epoch, N, M, total_epochs, topk=10, warmup_epoch=5, decay_epochs=5):
        """
        返回当前 epoch 下的动态调度参数：
        - iou_thresh
        - score_thresh
        - cur_topk

        Args:
            epoch (int): 当前epoch
            N (int): 当前batch大小
            M (int): 最大值
            total_epochs (int): 总的epoch数
            topk (int): topk值，默认10
            warmup_epoch (int): warmup阶段的epoch数，默认5
            decay_epochs (int): decay阶段的epoch数，默认5
        """

        # 计算过渡阶段的epoch数
        transition_epoch = int(total_epochs * 0.7)

        # ---------------- iou_thresh 调度 ----------------
        if epoch < warmup_epoch:
            iou_thresh = self.iou_thresh * 0.1
        elif epoch < warmup_epoch + transition_epoch:
            # 过渡阶段，逐渐增加到最终值
            alpha = (epoch - warmup_epoch) / transition_epoch
            iou_thresh = self.iou_thresh * (0.1 + 0.9 * alpha)  # 0.1x → 1.0x
        elif epoch < warmup_epoch + transition_epoch + decay_epochs:
            # Decay阶段
            beta = (epoch - (warmup_epoch + transition_epoch)) / decay_epochs
            iou_thresh = self.iou_thresh * (1.0 + 0.0 * beta)
        else:
            iou_thresh = self.iou_thresh

        # ---------------- score_thresh 调度 ----------------
        if epoch < warmup_epoch:
            score_thresh = 0.05
        elif epoch < warmup_epoch + transition_epoch:
            # 过渡阶段，逐渐增加到最终值
            alpha = (epoch - warmup_epoch) / transition_epoch
            score_thresh = 0.05 + alpha * (self.score_thresh - 0.05)  # 0.05 → self.score_thresh
        else:
            score_thresh = self.score_thresh

        # ---------------- topk 调度 ----------------
        warmup_topk = topk * 5
        if epoch < warmup_epoch:
            cur_topk = warmup_topk
        elif epoch < warmup_epoch + transition_epoch:
            # 过渡阶段，逐渐增加topk
            alpha = (epoch - warmup_epoch) / transition_epoch
            cur_topk = max(1, int(warmup_topk + (topk - warmup_topk) * alpha))
        elif epoch < warmup_epoch + transition_epoch + decay_epochs:
            decay_step = epoch - (warmup_epoch + transition_epoch)
            decay_alpha = decay_step / decay_epochs
            start_ratio, end_ratio = 1.0, 0.2
            ratio = start_ratio + (end_ratio - start_ratio) * decay_alpha
            cur_topk = max(10, min(int(N * ratio), M * topk))
        else:
            cur_topk = max(10, min(int(N * 0.2), M * topk))

        # 保证最小正样本数
        min_pos = max(10, int(N * 0.05))
        cur_topk = max(cur_topk, min_pos)

        return iou_thresh, score_thresh, cur_topk

    def get_aux_weight(self, cur_epoch, max_epoch):
        """
        cur_epoch: 当前 epoch (从0开始)
        max_epoch: 总 epoch 数
        start: 初始权重
        end: 最终权重
        return: 当前 aux_loss 权重
        """
        if cur_epoch >= max_epoch:
            return 0.0
        return self.aux_weight + (0.0 - self.aux_weight) * (cur_epoch / max_epoch)
