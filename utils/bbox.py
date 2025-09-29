import math
import torch
import torchvision.ops as ops


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    将 [x1, y1, x2, y2] 坐标格式转换为 [cx, cy, w, h] 格式。

    参数:
        boxes: Tensor，形状为 [..., 4]，每行是 [x1, y1, x2, y2]

    返回:
        Tensor，形状与输入相同，为 [cx, cy, w, h]
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, h, w), dim=-1)


def decode_bbox_tensor(bbox_tensor):
    """
        [cx, cy, w, h] → [x1, y1, x2, y2]
    """
    cx, cy, w, h = bbox_tensor.unbind(dim=1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1).to(bbox_tensor.device)


def cxywh_to_xyxy(x_c, y_c, w, h, img_size=None):
    """
        转换回像素坐标（相对值 → 绝对值），注意：这里统一了 img_size×img_size
    """

    assert img_size is not None, "img_size must be provided"
    W, H = img_size
    x_c *= W
    y_c *= H
    w *= W
    h *= H
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2
    # 修正边缘：确保坐标在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W - 1, x_max)
    y_max = min(H - 1, y_max)
    return x_min, y_min, x_max, y_max


def xyxy_to_cxywh(xyxy, source_size, target_size=None):
    """
        将xyxy格式的坐标转为中心点坐标
    """
    source_w, source_h = source_size
    target_w, target_h = target_size if target_size else source_size

    x_min, y_min, x_max, y_max = xyxy
    # 先归一化 → 再映射到特征图尺寸
    x_min_feat = x_min / source_w * target_w
    x_max_feat = x_max / source_w * target_w
    y_min_feat = y_min / source_h * target_h
    y_max_feat = y_max / source_h * target_h
    # 中心点和宽高
    cx = ((x_min_feat + x_max_feat) / 2).clamp(0, target_w - 1)
    cy = ((y_min_feat + y_max_feat) / 2).clamp(0, target_h - 1)
    w = (x_max_feat - x_min_feat).clamp(1e-3)
    h = (y_max_feat - y_min_feat).clamp(1e-3)
    return cx, cy, w, h


def box_diou(pred, target, eps=1e-7):
    # pred, target: [N, 4] (x1, y1, x2, y2)
    pred = pred.float()
    target = target.float()

    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx1, gy1, gx2, gy2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # 交集区域
    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_pred = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_gt = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union_area = area_pred + area_gt - inter_area + eps

    iou = inter_area / union_area

    # 中心点距离平方
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    center_dist = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # 包围盒对角线距离平方
    enclose_x1 = torch.min(px1, gx1)
    enclose_y1 = torch.min(py1, gy1)
    enclose_x2 = torch.max(px2, gx2)
    enclose_y2 = torch.max(py2, gy2)
    diag_dist = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=eps)

    diou = iou - center_dist / diag_dist
    return diou.clamp(min=-1.0, max=1.0)


def box_ciou(pred, target, eps=1e-7):
    pred = pred.float()
    target = target.float()

    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx1, gy1, gx2, gy2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_pred = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_gt = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union_area = area_pred + area_gt - inter_area + eps

    iou = inter_area / union_area

    # 中心点距离
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    center_dist = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    enclose_x1 = torch.min(px1, gx1)
    enclose_y1 = torch.min(py1, gy1)
    enclose_x2 = torch.max(px2, gx2)
    enclose_y2 = torch.max(py2, gy2)
    diag_dist = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=eps)

    # 尺寸相似度项
    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0)
    gh = (gy2 - gy1).clamp(min=0)

    v = (4 / (math.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / diag_dist + alpha * v)
    return ciou.clamp(min=-1.0, max=1.0)


def dynamic_postprocess(score_map, box_map, img_size, score_thresh=0.25, iou_thresh=0.25, nms=True, upscale=True,use_softnms=False):
    """
    score_map: [B, N, 1] -> 置信度
    box_map:   [B, N, 4] -> 归一化框 [x_min, y_min, x_max, y_max]，范围在 [0, 1]
    img_size:  原图尺寸 (W, H)
    nms:  是否启用
    upscale:  边框是否还原到原始大小
    """
    B, N, _ = score_map.shape
    boxes_all, scores_all, labels_all = [], [], []

    for b in range(B):
        scores = score_map[b]  # [N]
        boxes = box_map[b]  # [N, 4]

        boxes_res, scores_res, labels_res = nms_postprocess(scores, boxes, img_size, score_thresh, iou_thresh, nms,
                                                            upscale,use_softnms)
        boxes_all.append(boxes_res)
        scores_all.append(scores_res)
        labels_all.append(labels_res)

    return boxes_all, scores_all, labels_all


def nms_postprocess(scores, boxes, img_size,
                    score_thresh=0.25, iou_thresh=0.25,
                    nms=True, upscale=True, use_softnms=False):
    """
    Args:
        scores (Tensor): [N, C]，每个框对每个类别的置信度（sigmoid 输出）
        boxes (Tensor): [N, 4]，归一化坐标 [x_min, y_min, x_max, y_max]
        img_size (tuple): (W, H)
        score_thresh: 分数阈值
        iou_thresh: NMS 阈值
        nms: 是否执行 NMS
        upscale: 是否恢复像素坐标
    Returns:
        boxes_all: [M, 4]，像素坐标
        scores_all: [M]
        labels_all: [M]，类别索引
    """
    N, C = scores.shape
    W, H = img_size
    boxes_all = []
    scores_all = []
    labels_all = []

    for cls in range(C):
        cls_scores = scores[:, cls]  # [N]
        keep_mask = cls_scores > score_thresh
        if keep_mask.sum() == 0:
            continue

        filtered_scores = cls_scores[keep_mask]
        filtered_boxes = boxes[keep_mask]
        filtered_scores = filtered_scores.to(filtered_boxes)

        # 合法性检查
        x_min, y_min, x_max, y_max = filtered_boxes.unbind(dim=1)
        valid_mask = (
                (x_min >= 0) & (y_min >= 0) &
                (x_max <= 1) & (y_max <= 1) &
                (x_max > x_min) & (y_max > y_min)
        )
        filtered_boxes = filtered_boxes[valid_mask]
        filtered_scores = filtered_scores[valid_mask]

        if filtered_scores.numel() == 0:
            continue

        # 坐标还原
        boxes_pix = filtered_boxes.clone()
        if upscale:
            boxes_pix[:, [0, 2]] *= W
            boxes_pix[:, [1, 3]] *= H

        # NMS
        if nms:
            if use_softnms:
                boxes_pix, filtered_scores, keep_indices = soft_nms(
                    boxes_pix, filtered_scores, iou_thresh=iou_thresh, sigma=0.5, score_thresh=score_thresh
                )
            else:
                keep = ops.nms(boxes_pix, filtered_scores, iou_thresh)
                boxes_pix = boxes_pix[keep]
                filtered_scores = filtered_scores[keep]

        # 添加结果
        boxes_all.append(boxes_pix)
        scores_all.append(filtered_scores)
        labels_all.append(torch.full((filtered_scores.size(0),), cls, dtype=torch.long, device=scores.device))

    if not boxes_all:
        return (
            torch.zeros((0, 4), device=scores.device),
            torch.zeros((0,), device=scores.device),
            torch.zeros((0,), dtype=torch.long, device=scores.device),
        )

    return (
        torch.cat(boxes_all, dim=0),
        torch.cat(scores_all, dim=0),
        torch.cat(labels_all, dim=0),
    )



def soft_nms(boxes, scores, iou_thresh=0.25, sigma=0.5, score_thresh=0.001):
    """
    Gaussian Soft-NMS
    Args:
        boxes: [N, 4] in pixels
        scores: [N]
        iou_thresh: IoU阈值
        sigma: 衰减系数
        score_thresh: 置信度阈值（低于则移除）
    Returns:
        keep_boxes: [M, 4]
        keep_scores: [M]
        keep_indices: [M]
    """
    boxes = boxes.clone()
    scores = scores.clone()
    N = boxes.size(0)
    indices = torch.arange(N, device=boxes.device)

    keep_boxes = []
    keep_scores = []
    keep_indices = []

    while boxes.size(0) > 0:
        max_score_idx = torch.argmax(scores)
        max_box = boxes[max_score_idx]
        max_score = scores[max_score_idx]
        max_idx = indices[max_score_idx]

        keep_boxes.append(max_box)
        keep_scores.append(max_score)
        keep_indices.append(max_idx)

        if boxes.size(0) == 1:
            break

        boxes = torch.cat([boxes[:max_score_idx], boxes[max_score_idx + 1:]], dim=0)
        scores = torch.cat([scores[:max_score_idx], scores[max_score_idx + 1:]], dim=0)
        indices = torch.cat([indices[:max_score_idx], indices[max_score_idx + 1:]], dim=0)

        ious = ops.box_iou(max_box.unsqueeze(0), boxes).squeeze(0)
        decay = torch.exp(- (ious ** 2) / sigma)
        scores *= decay

        keep_mask = scores > score_thresh
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        indices = indices[keep_mask]

    return torch.stack(keep_boxes), torch.tensor(keep_scores, device=boxes.device), torch.tensor(keep_indices, device=boxes.device)