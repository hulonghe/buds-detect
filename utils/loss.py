import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou as box_giou, box_iou
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress


def simple_heatmap_mse(heatmap_pred, heatmap_gt, epoch=None, warmup_epochs=5, history_status=None, mask=None,
                       gt_coords=None, stride=None, **kwargs):
    return F.mse_loss(heatmap_pred, heatmap_gt)


def naive_bbox_loss(pred, target, epoch=None, warmup_epochs=5, history_status=None, mask=None,
                    gt_coords=None, stride=None, **kwargs):
    if epoch < warmup_epochs:
        return F.l1_loss(pred, target)
    else:
        return simple_bbox_loss(pred, target, mask=mask, gt_coords=gt_coords, stride=stride)


def simple_bbox_loss(pred, target, mask=None,
                     gt_coords=None, stride=None, history_status=None, **kwargs):
    """
    Args:
        pred:   Tensor[B, 4, H, W]  —— 模型预测的中心点偏移 (l, t, r, b)
        target: Tensor[B, 4, H, W]  —— GT 编码后的偏移目标
        mask:   Tensor[B, 1, H, W]  —— 有效监督区域
        gt_coords: List[Tensor[B_i, 2]]  —— 每张图 GT 中心点坐标 (cx, cy)，单位为原图像素
        stride: int，特征图下采样比例，用于坐标映射
        history_status: 用于动态 mask 策略，可以忽略
    """
    device = pred.device

    # -------- 生成 mask --------
    mask_supervise = get_mask_valid_mask(history_status, gt_coords, stride, mask, device)
    mask_expanded = mask_supervise.expand_as(pred) if mask_supervise is not None else torch.ones_like(pred)

    # -------- 简单 L1 损失 --------
    l1_loss = F.l1_loss(pred, target, reduction='none')  # [B, 4, H, W]
    loss = l1_loss * mask_expanded

    denom = mask_expanded.sum().clamp(min=1.0)
    total_loss = loss.sum() / denom

    return total_loss


def heatmap_loss(heatmap_pred, heatmap_gt, gt_coords=None,
                 alpha=1, beta=2, eps=1e-6, threshold=1.0, stride=None,
                 normalize=True, history_status=None,
                 epoch=None, warmup_epochs=5,
                 mask=None):
    if epoch < warmup_epochs:
        return simple_heatmap_mse(heatmap_pred, heatmap_gt)

    device = heatmap_pred.device
    dynamic_threshold = 0.8 if (epoch is not None and epoch < warmup_epochs) else threshold
    pred = torch.clamp(heatmap_pred, eps, 1 - eps)

    pos_mask = (heatmap_gt >= dynamic_threshold).float()
    strict_neg_mask = 1 - pos_mask

    pos_loss = -torch.log(pred) * ((1 - pred) ** alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * (pred ** alpha) * ((1 - heatmap_gt) ** beta) * strict_neg_mask
    loss = pos_loss + neg_loss

    mask_supervise = get_mask_valid_mask(history_status, gt_coords, stride, mask, device)
    if mask_supervise is not None:
        loss = loss * mask_supervise
        if normalize:
            num_pos = (pos_mask * mask_supervise).sum().clamp(min=1.0)
            loss = loss.sum() / num_pos
        else:
            loss = loss.sum() / mask_supervise.sum().clamp(min=1.0)
    else:
        if normalize:
            num_pos = pos_mask.sum().clamp(min=1.0)
            loss = loss.sum() / num_pos
        else:
            loss = loss.mean()

    return loss


def bbox_loss(pred, target, mask, heatmap_pred=None,
              epoch=None, warmup_epochs=5,
              boxes=None, labels=None, stride=None, history_status=None,
              gt_coords=None):
    """
    Args:
        gt_coords: List[Tensor]  # 每个 batch 元素形如  [num_gt, 2]，值为 (cx, cy) 绝对像素坐标
        stride:    int           # 当前特征层的下采样步长
    """
    device = pred.device
    trend = history_status

    # ---------- 1. 生成用于监督的 mask ----------
    mask_supervise = get_mask_valid_mask(trend, gt_coords, stride, mask, device)
    mask_expanded = mask_supervise.expand_as(pred)  # [B, 4, H, W]

    # ---------- 2. L1 / Smooth‑L1 ----------
    if epoch is not None and epoch < warmup_epochs:
        elementwise = F.smooth_l1_loss(pred, target, reduction='none')
    else:
        elementwise = F.l1_loss(pred, target, reduction='none')

    masked_l1 = elementwise * mask_expanded

    # ---------- 3. 动态权重 ----------
    loss_weight = torch.tensor(1.0, device=device)
    # if trend == 'increasing':
    #     loss_weight *= 1.2
    # elif trend == 'stable':
    #     # 只监督中心点后正样本急剧减少，放大权重防止梯度过小
    #     loss_weight *= 1.5
    weighted_loss = masked_l1 * loss_weight

    # 避免除零
    denom = mask_expanded.sum().clamp(min=1.0)
    total_loss = weighted_loss.sum() / denom

    # ---------- 4. GIoU ----------
    # if epoch is not None and epoch > warmup_epochs:
    #     giou_loss = compute_giou_loss_with_matching(
    #         pred_bbox=pred, gt_boxes_list=boxes,
    #         mask=mask_supervise, stride=stride
    #     )
    #
    #     # 从每个类别中选出 topk 点，分别监督 bbox，对应的 gt_box 是该类目标。
    #     loss_bbox_topk = compute_topk_bbox_loss_multiclass(
    #         pred, boxes, heatmap_pred, labels, topk=50, stride=stride
    #     )
    #
    #     total_loss = 0.4 * total_loss + 0.25 * loss_bbox_topk + 0.35 * giou_loss

    return total_loss


def compute_topk_bbox_loss_multiclass(pred_bbox, gt_boxes_list, heatmap_pred,
                                      gt_classes, topk=100, stride=1):
    """
    每类选 topk 点，结合其对应 pred_bbox 与同类 gt 匹配做 loss
    Args:
        pred_bbox: [B, 4, H, W]
        heatmap_pred: [B, C, H, W]
        gt_boxes_list: List[Tensor], 每个元素为 [num_gt, 4]
        gt_classes: List[Tensor], 每个元素为 [num_gt]
    Returns:
        topk giou loss
    """
    device = pred_bbox.device
    B, C, H, W = heatmap_pred.shape
    total_loss = []

    for b in range(B):
        gt_boxes = gt_boxes_list[b].to(device)
        gt_cls = gt_classes[b].to(device)

        if gt_boxes.numel() == 0:
            continue

        for c in range(C):
            class_scores = heatmap_pred[b, c]  # [H, W]
            gt_c_mask = (gt_cls == c)
            if gt_c_mask.sum() == 0:
                continue

            topk_scores, topk_inds = torch.topk(class_scores.view(-1), k=topk)
            py = topk_inds // W
            px = topk_inds % W

            dx = pred_bbox[b, 0, py, px]
            dy = pred_bbox[b, 1, py, px]
            dw = pred_bbox[b, 2, py, px]
            dh = pred_bbox[b, 3, py, px]

            pred_cx = px.float() * stride + dx * stride
            pred_cy = py.float() * stride + dy * stride
            pred_w = torch.exp(dw) * stride
            pred_h = torch.exp(dh) * stride

            pred_boxes = torch.stack([
                pred_cx - pred_w / 2,
                pred_cy - pred_h / 2,
                pred_cx + pred_w / 2,
                pred_cy + pred_h / 2,
            ], dim=1)

            gt_c_boxes = gt_boxes[gt_c_mask]
            if pred_boxes.shape[0] == 0 or gt_c_boxes.shape[0] == 0:
                continue

            with torch.no_grad():
                cost_matrix = -box_iou(pred_boxes, gt_c_boxes)
                cost_matrix[~torch.isfinite(cost_matrix)] = 1e6
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())

            matched_preds = pred_boxes[row_ind]
            matched_gts = gt_c_boxes[col_ind]

            giou = box_giou(matched_preds, matched_gts)
            loss = 1.0 - giou
            total_loss.append(loss.mean())

    if not total_loss:
        return torch.tensor(0., device=device, requires_grad=True)
    return torch.stack(total_loss).mean()


def get_mask_valid_mask(trend, gt_coords, stride, mask, device):
    # ---------- 1. 生成用于监督的 mask ----------
    if trend == 'stable' and gt_coords is not None and stride is not None:
        # 只保留中心点：shape == (B, 1, H, W)
        center_mask = torch.zeros_like(mask, dtype=torch.float32, device=device)

        # 将 GT 中心点投射到特征图索引
        # 先拼 tensor -> (sum_gt, 3) : [batch_index, iy, ix]
        batch_idx, iy, ix = [], [], []
        for b, coords in enumerate(gt_coords):
            if coords.numel() == 0:
                continue
            idx = coords.long()  # (num_gt, 2) -> (ix, iy)
            ix.append(idx[:, 0].clamp(0, mask.shape[-1] - 1))
            iy.append(idx[:, 1].clamp(0, mask.shape[-2] - 1))
            batch_idx.append(torch.full((idx.shape[0],), b, device=device, dtype=torch.long))

        if batch_idx:  # 可能没有正样本
            batch_idx = torch.cat(batch_idx)
            ix = torch.cat(ix)
            iy = torch.cat(iy)
            center_mask[batch_idx, 0, iy, ix] = 1.0
        # 膨胀中心点mask，radius可调
        mask_supervise = expand_center_mask(center_mask, radius=1)
    else:
        mask_supervise = mask.float()
    return mask_supervise


def expand_center_mask(center_mask, radius=1):
    """
    对中心点 mask 做邻域膨胀，radius=1 表示膨胀到3x3范围。
    Args:
        center_mask: Tensor, shape [B, 1, H, W], dtype float32，中心点标记为1，其他为0
        radius: int, 膨胀半径，邻域大小为 (2*radius+1)
    Returns:
        expanded_mask: Tensor, shape [B, 1, H, W], 1表示膨胀区域
    """
    kernel_size = 2 * radius + 1
    # 构造全1卷积核
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=center_mask.device, dtype=torch.float32)
    # 卷积实现膨胀，padding 保持尺寸不变
    expanded = F.conv2d(center_mask, kernel, padding=radius)
    # 只要邻域内有点为1，则置为1
    expanded_mask = (expanded > 0).float()
    return expanded_mask


def compute_giou_loss_with_matching(pred_bbox, gt_boxes_list, mask, stride):
    device = pred_bbox.device
    B, _, H, W = pred_bbox.shape

    shift_x = torch.arange(W, device=device) * stride
    shift_y = torch.arange(H, device=device) * stride
    grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    centers = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    total_loss = []
    for b in range(B):
        pos_mask = mask[b, 0] > 0
        if pos_mask.sum() == 0 or len(gt_boxes_list[b]) == 0:
            continue

        pos_indices = pos_mask.nonzero(as_tuple=False)
        py, px = pos_indices[:, 0], pos_indices[:, 1]
        centers_b = centers[py, px]

        dx = pred_bbox[b, 0, py, px]
        dy = pred_bbox[b, 1, py, px]
        dw = pred_bbox[b, 2, py, px]
        dh = pred_bbox[b, 3, py, px]

        pred_cx = centers_b[:, 0] * stride + dx * stride
        pred_cy = centers_b[:, 1] * stride + dy * stride
        pred_w = torch.exp(dw) * stride
        pred_h = torch.exp(dh) * stride

        pred_boxes = torch.stack([
            pred_cx - pred_w / 2,
            pred_cy - pred_h / 2,
            pred_cx + pred_w / 2,
            pred_cy + pred_h / 2
        ], dim=1)  # [num_pos, 4]

        gt_boxes = gt_boxes_list[b].to(device)

        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            continue

        with torch.no_grad():
            cost_matrix = -box_iou(pred_boxes, gt_boxes)
            if not torch.isfinite(cost_matrix).all():
                cost_matrix[~torch.isfinite(cost_matrix)] = 1e6
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())

        matched_preds = pred_boxes[row_ind]
        matched_gts = gt_boxes[col_ind]

        giou = box_giou(matched_preds, matched_gts)
        loss = 1.0 - giou

        total_loss.append(loss.mean())

    if not total_loss:
        return torch.tensor(0., device=device, requires_grad=True)
    return torch.stack(total_loss).mean()


def adaptive_focal_loss(pred, gt, alpha=1, beta=2, eps=1e-6, epoch=0, warmup_epochs=5):
    """带 warmup 的 focal loss，逐渐加强焦点效应"""
    pred = torch.clamp(pred, eps, 1 - eps)

    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()

    # 动态 alpha，前 warmup_epochs 内从 0 线性增长到指定 alpha
    current_alpha = alpha * min(1.0, epoch / warmup_epochs)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, current_alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, current_alpha) * torch.pow(1 - gt, beta) * neg_mask

    num_pos = pos_mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    loss = (pos_loss + neg_loss).sum(dim=(1, 2, 3)) / num_pos.squeeze()

    return loss.mean()


def reward_focal_loss(pred, gt, alpha=1, beta=2, eps=1e-6, reward_ratio=0.05):
    pred = torch.clamp(pred, eps, 1 - eps)
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()

    log_pred = torch.log(pred)
    log_1_minus_pred = torch.log(1 - pred)

    pos_loss = -log_pred * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = -log_1_minus_pred * torch.pow(pred, alpha) * torch.pow(1 - gt, beta) * neg_mask

    reward_mask = ((pred > 0.5).float() * neg_mask)
    reward_loss = reward_ratio * pred * reward_mask

    total_loss = pos_loss + neg_loss - reward_loss

    # Flatten for batch-wise sum
    total_loss = total_loss.flatten(1)
    pos_counts = pos_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    loss = total_loss.sum(1) / pos_counts

    # 加强 debug 可视化
    if torch.isnan(loss).any():
        print("🚨 NaN in loss detected")
        print("Pred min/max:", pred.min().item(), pred.max().item())
        print("Reward loss max:", reward_loss.max().item())
        print("Pos count:", pos_counts)
        print("Sample loss:", loss)

    return loss.mean()


def analyze_loss_trend(loss_list, window_size=10, threshold=0.001):
    """
    分析最近 window_size 个 epoch 的损失趋势。

    参数:
        loss_list: List[float] 历史损失
        window_size: int，滑动窗口大小
        threshold: float，斜率变化的敏感度阈值

    返回:
        trend: 'decreasing', 'increasing', 'stable', 或 'unknown' | 下降 上升 平稳 未知
        avg_change_rate: 平均变化率（线性拟合斜率）
        std_dev: 标准差
    """

    if len(loss_list) < window_size:
        return "unknown", 0.0, 0.0

    recent_losses = loss_list[-window_size:]
    x = np.arange(window_size)
    slope, _, _, _, _ = linregress(x, recent_losses)

    avg_change_rate = slope
    std_dev = np.std(recent_losses)

    # 趋势判断修正
    if slope < -threshold:
        trend = 'decreasing'
    elif slope > threshold:
        trend = 'increasing'
    else:
        trend = 'stable'

    return trend, avg_change_rate, std_dev
