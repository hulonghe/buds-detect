import torch


# def eiou_loss(pred_boxes, target_boxes, reduction='mean'):
#     """
#     Extended IoU (EIoU) Loss
#     :param pred_boxes: [N, 4] format (x1, y1, x2, y2)
#     :param target_boxes: [N, 4] format (x1, y1, x2, y2)
#     """
#     # 预测框中心、宽高
#     px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
#     py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
#     pw = pred_boxes[:, 2] - pred_boxes[:, 0]
#     ph = pred_boxes[:, 3] - pred_boxes[:, 1]

#     tx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
#     ty = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
#     tw = target_boxes[:, 2] - target_boxes[:, 0]
#     th = target_boxes[:, 3] - target_boxes[:, 1]

#     # 相交区域
#     x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
#     y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
#     x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
#     y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

#     inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
#     area_p = pw * ph
#     area_t = tw * th
#     union = area_p + area_t - inter
#     iou = inter / (union + 1e-7)

#     # 最小闭包框
#     cw = torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0])
#     ch = torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1])
#     c2 = cw ** 2 + ch ** 2 + 1e-7

#     # 中心点距离
#     center_dist = (px - tx) ** 2 + (py - ty) ** 2

#     # 宽高差异
#     w_dist = (pw - tw) ** 2
#     h_dist = (ph - th) ** 2

#     eiou = 1 - iou + center_dist / c2 + w_dist / (cw ** 2 + 1e-7) + h_dist / (ch ** 2 + 1e-7)

#     if reduction == 'mean':
#         return eiou.mean()
#     elif reduction == 'sum':
#         return eiou.sum()
#     else:
#         return eiou

    
def complete_eiou(pred_boxes, target_boxes):
    # 计算 IoU
    x1, y1, x2, y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x1g, y1g, x2g, y2g = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
    wp, hp = x2 - x1, y2 - y1
    wg, hg = x2g - x1g, y2g - y1g
    inter = torch.clamp(torch.min(x2, x2g) - torch.max(x1, x1g), 0) * \
            torch.clamp(torch.min(y2, y2g) - torch.max(y1, y1g), 0)
    union = wp * hp + wg * hg - inter + 1e-8
    iou = inter / union
    
    # 最小外接矩形
    xc1, yc1 = torch.max(x1, x1g), torch.max(y1, y1g)
    xc2, yc2 = torch.min(x2, x2g), torch.min(y2, y2g)
    wc, hc = xc2 - xc1, yc2 - yc1
    
    # 中心点距离
    cx_p, cy_p = (x1 + x2) / 2, (y1 + y2) / 2
    cx_g, cy_g = (x1g + x2g) / 2, (y1g + y2g) / 2
    d2 = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2
    c2 = wc ** 2 + hc ** 2 + 1e-8
    
    # 宽高差
    w_diff = (wg - wp) ** 2 / (wc ** 2 + 1e-8)
    h_diff = (hg - hp) ** 2 / (hc ** 2 + 1e-8)
    
    eiou = iou - (d2 / c2 + w_diff + h_diff)
    return eiou
