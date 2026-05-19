import torch
import torchvision.ops as ops
import numpy as np
from utils.bbox import dynamic_postprocess as dyp

def iou_numpy(box1, box2):
    """计算两个框的 IoU (numpy)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def weighted_boxes_fusion_single(boxes, scores, labels, iou_thresh=0.55, skip_box_thr=0.0):
    """
    单图的 Weighted Boxes Fusion (WBF)
    Args:
        boxes:  [N,4] numpy
        scores: [N]   numpy
        labels: [N]   numpy
    Returns:
        boxes_fused, scores_fused, labels_fused
    """
    boxes = boxes.astype(float)
    scores = scores.astype(float)
    labels = labels.astype(int)

    picked_boxes, picked_scores, picked_labels = [], [], []

    for cls in np.unique(labels):
        inds = np.where(labels == cls)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        fused = []
        for i, box in enumerate(cls_boxes):
            if cls_scores[i] < skip_box_thr:
                continue
            matched = False
            for f in fused:
                if iou_numpy(box, f["box"]) > iou_thresh:
                    f["box"] = (f["box"] * f["weight"] + box * cls_scores[i]) / (f["weight"] + cls_scores[i])
                    f["score"] = max(f["score"], cls_scores[i])
                    f["weight"] += cls_scores[i]
                    matched = True
                    break
            if not matched:
                fused.append({"box": box, "score": cls_scores[i], "weight": cls_scores[i]})

        for f in fused:
            picked_boxes.append(f["box"])
            picked_scores.append(f["score"])
            picked_labels.append(cls)

    return np.array(picked_boxes), np.array(picked_scores), np.array(picked_labels)



def dynamic_postprocess(score_map, box_map, img_size, score_thresh=0.25, iou_thresh=0.25,
                        upscale=True, method="nms"):
    """
    目标检测预测结果后处理 (支持 NMS、Soft-NMS 和 WBF)

    Args:
        score_map: [B, N, C] -> 分类分数
        box_map:   [B, N, 4] -> 归一化坐标 [x_min, y_min, x_max, y_max]
        img_size:  (W, H)，原图尺寸
        score_thresh: float，置信度阈值
        iou_thresh: float，NMS 或 WBF 的 IoU 阈值
        upscale: bool，是否还原到像素级坐标
        method: str，"nms" 或 "wbf" 或 "soft-nms"

    Returns:
        boxes_all:  [B, M, 4]
        scores_all: [B, M]
        labels_all: [B, M]
    """

    # === 使用原来的 NMS/Soft-NMS 分支 ===
    if method == "soft-nms" or method == "nms":
        # dyp 是你原来的 nms/soft-nms 函数
        return dyp(score_map, box_map, img_size,
                   score_thresh=score_thresh, iou_thresh=iou_thresh,
                   nms=True, upscale=upscale, use_softnms=(method == "soft-nms"))

    B, N, C = score_map.shape
    W, H = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)

    boxes_all, scores_all, labels_all = [], [], []

    for b in range(B):
        scores_per_img = score_map[b]  # [N, C]
        boxes_per_img = box_map[b]     # [N, 4]

        final_boxes, final_scores, final_labels = [], [], []

        # === 每个类别单独处理 ===
        for c in range(C):
            scores_c = scores_per_img[:, c]
            keep = scores_c > score_thresh
            if keep.sum() == 0:
                continue

            boxes_c = boxes_per_img[keep]
            scores_c = scores_c[keep]

            # === 像素级还原 ===
            if upscale:
                boxes_c = boxes_c.clone()
                boxes_c[:, [0, 2]] *= W
                boxes_c[:, [1, 3]] *= H

            if method == "nms":
                keep_inds = ops.nms(boxes_c, scores_c, iou_thresh)
                boxes_c = boxes_c[keep_inds]
                scores_c = scores_c[keep_inds]
                labels_c = torch.full((len(keep_inds),), c, dtype=torch.long)

            elif method == "wbf":
                # WBF 处理，需要调用你原来的 weighted_boxes_fusion_single
                import numpy as np
                boxes_np = boxes_c.cpu().numpy()
                scores_np = scores_c.cpu().numpy()
                labels_np = np.full(len(scores_np), c)
                boxes_np, scores_np, labels_np = weighted_boxes_fusion_single(
                    [boxes_np], [scores_np], [labels_np],
                    iou_thresh=iou_thresh, skip_box_thr=score_thresh
                )
                boxes_c = torch.tensor(boxes_np, dtype=torch.float32)
                scores_c = torch.tensor(scores_np, dtype=torch.float32)
                labels_c = torch.tensor(labels_np, dtype=torch.long)
            else:
                raise ValueError(f"Unsupported method: {method}")

            final_boxes.append(boxes_c)
            final_scores.append(scores_c)
            final_labels.append(labels_c)

        if final_boxes:
            boxes_all.append(torch.cat(final_boxes, dim=0))
            scores_all.append(torch.cat(final_scores, dim=0))
            labels_all.append(torch.cat(final_labels, dim=0))
        else:
            boxes_all.append(torch.zeros((0, 4)))
            scores_all.append(torch.zeros((0,)))
            labels_all.append(torch.zeros((0,), dtype=torch.long))

    return boxes_all, scores_all, labels_all
