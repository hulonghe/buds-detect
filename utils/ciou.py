import torch


def complete_ciou(pred_boxes, gt_boxes, eps=1e-7):
    """
    计算 Complete IoU (CIoU) - PyTorch 版本
    pred_boxes: (N, 4) - 预测框 [x1, y1, x2, y2]
    gt_boxes:   (N, 4) - 真实框 [x1, y1, x2, y2]
    return: (N,) CIoU，范围 [-1,1]，越大越好
    """

    # 宽高
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    # 交集
    inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # 各自面积
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h

    # IoU
    union_area = pred_area + gt_area - inter_area + eps
    iou = inter_area / union_area

    # 外包框对角线平方
    enclose_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    c2 = enclose_w ** 2 + enclose_h ** 2 + eps

    # 中心点距离平方
    pred_ctr_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_ctr_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_ctr_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    rho2 = (pred_ctr_x - gt_ctr_x) ** 2 + (pred_ctr_y - gt_ctr_y) ** 2

    # 长宽比一致性项 v & alpha
    atan_gt = torch.atan(gt_w / (gt_h + eps))
    atan_pred = torch.atan(pred_w / (pred_h + eps))
    v = (4 / (torch.pi ** 2)) * (atan_gt - atan_pred) ** 2
    alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - (rho2 / c2 + alpha * v)
    # ciou = torch.clamp(ciou, min=-1.0, max=1.0)
    ciou01 = ((ciou + 1.0) * 0.5).clamp(eps, 1.0 - eps)
    return ciou01
