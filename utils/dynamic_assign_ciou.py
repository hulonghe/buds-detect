import torch

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    returns: iou [N, M], inter [N, M], union [N, M]
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)
    return iou, inter, union


def da_complete_ciou(pred_boxes, gt_boxes, eps=1e-7):
    """
    pred_boxes: [N, 4]
    gt_boxes: [M, 4]
    returns: ciou [N, M]
    """
    # 中心点
    pred_center = (pred_boxes[:, None, :2] + pred_boxes[:, None, 2:]) / 2
    gt_center = (gt_boxes[None, :, :2] + gt_boxes[None, :, 2:]) / 2

    # 宽高
    pred_wh = (pred_boxes[:, None, 2:] - pred_boxes[:, None, :2]).clamp(min=eps)
    gt_wh = (gt_boxes[None, :, 2:] - gt_boxes[None, :, :2]).clamp(min=eps)

    # IoU
    iou, _, _ = box_iou(pred_boxes, gt_boxes)
    iou = iou.clamp(min=eps, max=1.0)

    # 中心点距离
    center_dist = ((pred_center - gt_center) ** 2).sum(dim=-1)

    # 最小闭包框对角线
    enclose_lt = torch.min(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    enclose_rb = torch.max(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    enclose_diag = ((enclose_rb - enclose_lt) ** 2).sum(dim=-1) + eps

    # 宽高比惩罚项
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(gt_wh[..., 0] / (gt_wh[..., 1] + eps)) -
        torch.atan(pred_wh[..., 0] / (pred_wh[..., 1] + eps)), 2
    )

    # Alpha 权重
    with torch.no_grad():
        alpha = v / (1 - iou.detach() + v + eps)

    # CIoU
    ciou = iou - center_dist / enclose_diag - alpha * v
    return ciou