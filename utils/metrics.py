import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import json
import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.interpolate import interp1d

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def categorize_fp(ious, matched_flags, iou_thresh):
    if len(ious) == 0:
        return 'bg'  # 背景误检（无 GT）
    max_iou = np.max(ious)
    idx = np.argmax(ious)
    if matched_flags[idx]:
        return 'dup'  # 重复预测
    elif max_iou < iou_thresh:
        return 'offset'  # 偏移预测
    return 'unknown'


def compute_ap_metrics(pred_list, gt_list, iou_thresholds=None, plot_iou_dist=False, plot_pr=False,
                       model_name="Model", save_path=None):
    if iou_thresholds is None:
        # 和 YOLO 一致：mAP@0.5:0.95, 步长 0.05
        iou_thresholds = [0.5, 0.95 + 1e-6, 0.05]

    results = {}
    ap_values = []
    iou_distributions = []
    fp_types_counter = {'dup': 0, 'offset': 0, 'bg': 0}

    min_iou, max_iou, step = iou_thresholds
    ious_ = np.arange(min_iou, max_iou, step)

    pr_curves = {}

    for i, iou_thresh in enumerate(ious_):
        scores, labels = [], []
        total_gt = 0
        miss_gt = 0

        for preds, gts in zip(pred_list, gt_list):
            pred_boxes, pred_scores = preds
            matched = np.zeros(len(gts), dtype=bool)
            total_gt += len(gts)

            for box, score in zip(pred_boxes, pred_scores):
                ious = np.array([compute_iou(box, gt) for gt in gts])
                max_iou_val = ious.max() if len(ious) > 0 else 0
                idx = ious.argmax() if len(ious) > 0 else -1

                if max_iou_val >= iou_thresh and idx != -1 and not matched[idx]:
                    scores.append(score)
                    labels.append(1)
                    matched[idx] = True
                    if plot_iou_dist:
                        iou_distributions.append(max_iou_val)
                else:
                    scores.append(score)
                    labels.append(0)
                    fp_type = categorize_fp(ious, matched, iou_thresh)
                    fp_types_counter[fp_type] += 1
                    if plot_iou_dist:
                        iou_distributions.append(max_iou_val if len(ious) > 0 else 0.0)

            miss_gt += np.sum(~matched)

        if total_gt == 0 or len(labels) == 0:
            results[f'AP@{iou_thresh:.2f}'] = 0.0
            if i == 0:
                results.update({'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'MR': 1.0})
            ap_values.append(0.0)
            continue

        scores = np.array(scores)
        labels = np.array(labels)
        sorted_idxs = np.argsort(-scores)
        labels = labels[sorted_idxs]

        tp = np.cumsum(labels)
        fp = np.cumsum(1 - labels)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (total_gt + 1e-6)

        # === YOLO风格 AP 计算 ===
        precision_mono = precision.copy()
        for j in range(len(precision_mono) - 2, -1, -1):
            precision_mono[j] = max(precision_mono[j], precision_mono[j + 1])

        recall2 = np.concatenate(([0.0], recall, [1.0]))
        precision2 = np.concatenate(([1.0], precision_mono, [0.0]))

        recall_interp = np.linspace(0, 1, 101)
        f_interp = interp1d(recall2, precision2, kind='linear')
        precision_interp = f_interp(recall_interp)
        ap = np.mean(precision_interp)

        results[f'AP@{iou_thresh:.2f}'] = ap
        ap_values.append(ap)

        if i == 0:  # IoU=0.5 的指标
            p, r = precision[-1], recall[-1]
            f1 = 2 * p * r / (p + r + 1e-6)
            results.update({
                'Precision': p,
                'Recall': r,
                'F1': f1,
                'MR': miss_gt / (total_gt + 1e-6)
            })

            # === 保存 PR 曲线 ===
            pr_curves['recall_smooth'] = recall_interp
            pr_curves['precision_smooth'] = precision_interp
            pr_curves['ap@0.5'] = ap

    if len(ious_) > 1:
        results['mAP'] = np.mean(ap_values)

    # 加入 FP 分类
    results.update({
        'FPDup': fp_types_counter['dup'],
        'FPOffset': fp_types_counter['offset'],
        'FPBg': fp_types_counter['bg'],
    })

    # === 关键扩展：PR 曲线返回 ===
    if 'recall_smooth' in pr_curves:
        results['PR'] = {
            'recall': pr_curves['recall_smooth'].copy(),
            'precision': pr_curves['precision_smooth'].copy(),
            'AP@0.5': float(pr_curves['ap@0.5']),
        }

    # IoU 分布图
    if plot_iou_dist and len(iou_distributions) > 0:
        plt.hist(iou_distributions, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
        plt.title("IoU Distribution of Predictions")
        plt.xlabel("IoU")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    # === PR 曲线绘制（平滑版） ===
    if plot_pr and 'recall_smooth' in pr_curves:
        recall = pr_curves['recall_smooth']
        precision = pr_curves['precision_smooth']
        ap = pr_curves['ap@0.5']

        # 1. 插值使 precision 单调递减（VOC-style）
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # 2. 补充起点 (0,1) 和终点 (1,0)
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([1.0], precision, [0.0]))

        recall_smooth = np.linspace(0, 1, 500)  # 500点平滑
        f_interp = interp1d(recall, precision, kind='linear')
        precision_smooth = f_interp(recall_smooth)

        # 4. 绘制平滑 PR 曲线
        plt.figure(figsize=(7, 7))
        plt.plot(recall_smooth, precision_smooth, color='blue', linewidth=2.5,
                 label=f"{model_name} (AP@0.5={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Smoothed)")
        plt.xticks(np.arange(0, 1.01, 0.2))
        plt.yticks(np.arange(0, 1.01, 0.2))
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="upper right")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"平滑 PR 曲线已保存到: {save_path}")
        else:
            plt.show()

    return results


def evaluate_with_coco(batch_pred_boxes, batch_pred_scores, batch_pred_labels,
                       batch_gt_boxes, batch_gt_labels, img_size=640):
    """
    基于 pycocotools 的评估函数（单类或多类都可用）
    Args:
        batch_pred_boxes: List of [M, 4], 预测框 (x1,y1,x2,y2)，归一化到[0,1]
        batch_pred_scores: List of [M], 预测置信度
        batch_pred_labels: List of [M], 预测类别
        batch_gt_boxes: List of [N, 4], GT 框 (x1,y1,x2,y2)，归一化到[0,1]
        batch_gt_labels: List of [N], GT 标签
        img_size: int, 假设输入图像尺寸 (用于反归一化坐标)
    Returns:
        results: dict, 包含 mAP, AP50, Precision, Recall, F1, FP, MissRate
    """

    # ===== 1. 构造 COCO 格式的 GT =====
    images = []
    annotations = []

    # 自动获取所有类别 ID
    unique_labels = set()
    for labels in batch_gt_labels:
        unique_labels.update(labels)
    categories = [{"id": 0, "name": f"tea-bud"}]

    ann_id = 1
    for img_id, (gt_boxes, gt_labels) in enumerate(zip(batch_gt_boxes, batch_gt_labels)):
        images.append({"id": img_id})
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = np.array(box) * img_size
            w, h = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt_dict = {
        "info": {
            "description": "Custom dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025-09-02"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    gt_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
    json.dump(coco_gt_dict, gt_file)
    gt_file.flush()
    coco_gt = COCO(gt_file.name)

    # ===== 2. 构造预测结果 =====
    predictions = []
    for img_id, (pred_boxes, pred_scores, pred_labels) in enumerate(
            zip(batch_pred_boxes, batch_pred_scores, batch_pred_labels)):
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = np.array(box) * img_size
            w, h = x2 - x1, y2 - y1
            predictions.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    if len(predictions) == 0:
        return {
            "mAP": 0.0,
            "AP@0.50": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "FP": int(0),
            "MR": 0.0
        }

    dt_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
    json.dump(predictions, dt_file)
    dt_file.flush()
    coco_dt = coco_gt.loadRes(dt_file.name)

    # ===== 3. 使用 COCOeval =====
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ===== 4. 从 COCOeval.stats 提取指标 =====
    mAP = float(coco_eval.stats[0])  # AP@[0.5:0.95]
    AP50 = float(coco_eval.stats[1])  # AP@0.5
    precision = float(coco_eval.stats[0])  # 使用 AP@[0.5:0.95] 近似 precision
    recall = float(coco_eval.stats[8])  # recall@100 detections

    # ===== 5. 手工计算 TP/FP/MissRate =====
    total_gt = sum(len(b) for b in batch_gt_boxes)
    total_pred = sum(len(b) for b in batch_pred_boxes)
    FN = total_gt * (1 - recall)
    TP = total_gt - FN
    FP = max(total_pred - TP, 0)
    miss_rate = FN / total_gt if total_gt > 0 else 0.0

    # ===== 6. 计算 F1 =====
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ===== 7. 返回结果 =====
    results = {
        "mAP": mAP,
        "AP@0.50": AP50,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP": int(FP),
        "MR": miss_rate
    }

    return results


def evaluate_yolo_style(batch_pred_boxes, batch_pred_scores, batch_pred_labels,
                        batch_gt_boxes, batch_gt_labels, iou_thresholds=None, img_size=640):
    """
    YOLO风格评估函数，输出指标与训练日志一致
    Args:
        batch_pred_boxes: List of [M,4], 预测框归一化到[0,1]
        batch_pred_scores: List of [M], 预测置信度
        batch_pred_labels: List of [M], 预测类别
        batch_gt_boxes: List of [N,4], GT框归一化到[0,1]
        batch_gt_labels: List of [N], GT标签
        iou_thresholds: list, IoU阈值列表，用于计算 AP50/AP75
        img_size: int, 图像大小
    Returns:
        results: dict, 包含 Precision, Recall, F1, mAP, AP50, FP, MissRate
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    def xyxy_iou(box1, box2):
        """
        box1: [4]  一维数组
        box2: [M,4] 二维数组
        """
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - inter
        iou = inter / (union + 1e-16)
        return iou

    all_TP, all_FP, all_FN = 0, 0, 0
    aps = []

    for iou_thr in iou_thresholds:
        TP, FP, FN = 0, 0, 0

        for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
                batch_pred_boxes, batch_pred_scores, batch_pred_labels, batch_gt_boxes, batch_gt_labels):

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            elif len(gt_boxes) == 0:
                FP += len(pred_boxes)
                continue
            elif len(pred_boxes) == 0:
                FN += len(gt_boxes)
                continue

            gt_boxes = np.array(gt_boxes) * img_size
            pred_boxes = np.array(pred_boxes) * img_size
            matched_gt = set()

            # 按分数排序
            if len(pred_scores) > 0:
                idxs = np.argsort(-np.array(pred_scores))
                pred_boxes = pred_boxes[idxs]
                pred_labels = np.array(pred_labels)[idxs]

            for pb, pl in zip(pred_boxes, pred_labels):
                # 匹配同类 GT
                mask = np.array(gt_labels) == pl
                if mask.sum() == 0:
                    FP += 1
                    continue
                ious = xyxy_iou(pb, gt_boxes[mask])
                max_iou_idx = np.argmax(ious)
                if ious[max_iou_idx] >= iou_thr and max_iou_idx not in matched_gt:
                    TP += 1
                    matched_gt.add(max_iou_idx)
                else:
                    FP += 1

            FN += len(gt_boxes) - len(matched_gt)

        all_TP += TP
        all_FP += FP
        all_FN += FN
        aps.append(TP / (TP + FP + FN + 1e-16))  # 近似AP

    precision = all_TP / (all_TP + all_FP + 1e-16)
    recall = all_TP / (all_TP + all_FN + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    mAP = np.mean(aps)
    AP50 = aps[0] if len(aps) > 0 else 0.0
    AP75 = aps[1] if len(aps) > 1 else 0.0
    FP = all_FP
    MR = all_FN / (all_TP + all_FN + 1e-16)

    results = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP": mAP,
        "AP@0.50": AP50,
        "AP75": AP75,
        "FP": FP,
        "MR": MR
    }
    return results
