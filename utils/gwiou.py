import torch
import torch.nn as nn

def gwd_loss(pred_boxes, target_boxes, alpha=1.0, tau=1.0, reduction='mean', eps=1e-9):
    """
    pred_boxes, target_boxes: [N, 4] in (x1, y1, x2, y2) format
    实现基于 Gaussian Wasserstein Distance，适用于旋转框任务（或通用框）
    """
    # center and size
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    pw = pred_boxes[:, 2] - pred_boxes[:, 0]
    ph = pred_boxes[:, 3] - pred_boxes[:, 1]

    gx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    gy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    gw = target_boxes[:, 2] - target_boxes[:, 0]
    gh = target_boxes[:, 3] - target_boxes[:, 1]

    # center distance
    dx = px - gx
    dy = py - gy
    center_dist = dx ** 2 + dy ** 2

    # shape distance（高斯分布的协方差矩阵 trace 距离）
    wh_diff = (pw - gw) ** 2 + (ph - gh) ** 2
    wh_sum = pw * pw + ph * ph + gw * gw + gh * gh + eps

    shape_dist = 1.0 - torch.exp(-wh_diff / (tau * wh_sum))  # 可调参数 tau 控制梯度

    # 总损失（高斯 Wasserstein 距离）
    loss = center_dist + alpha * shape_dist

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # [N]
