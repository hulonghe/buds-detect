import torch
import torch.nn as nn

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def complete_diou(pred_boxes, target_boxes, eps=1e-7):
    """
    pred_boxes, target_boxes: [N, 4] in (x1, y1, x2, y2) format
    """
    # Intersection
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Areas
    pred_area = box_area(pred_boxes)
    target_area = box_area(target_boxes)

    # Union
    union = pred_area + target_area - inter + eps
    iou = inter / union

    # Center distance
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    gy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_dist = (px - gx) ** 2 + (py - gy) ** 2

    # Enclosing box diagonal
    cx1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    cy1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    cx2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    cy2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    diag = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) + eps

    diou = iou - center_dist / diag
    return diou
