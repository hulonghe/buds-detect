def gwd_loss(pred_boxes, target_boxes, tau=1.0, reduction='mean'):
    """
    Gaussian Wasserstein Distance Loss
    :param pred_boxes: [N, 4] in (x1, y1, x2, y2)
    :param target_boxes: [N, 4] in (x1, y1, x2, y2)
    :param tau: scaling factor
    :param reduction: 'mean' or 'sum'
    """
    # 中心点和宽高
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    pw = pred_boxes[:, 2] - pred_boxes[:, 0]
    ph = pred_boxes[:, 3] - pred_boxes[:, 1]

    tx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    ty = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    tw = target_boxes[:, 2] - target_boxes[:, 0]
    th = target_boxes[:, 3] - target_boxes[:, 1]

    # L2 中心点距离
    delta_xy = (px - tx) ** 2 + (py - ty) ** 2

    # 协方差矩阵的 trace
    delta_wh = (pw - tw) ** 2 + (ph - th) ** 2

    gwd = (delta_xy + delta_wh) / tau

    if reduction == 'mean':
        return gwd.mean()
    elif reduction == 'sum':
        return gwd.sum()
    else:
        return gwd  # [N]
