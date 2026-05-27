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


def compute_ap_metrics(pred_list, gt_list, iou_thresholds=None,
                       plot_iou_dist=False, plot_pr=False,
                       model_name="Model", save_path=None,
                       cls_num=1):

    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.95 + 1e-6, 0.05]

    # === 固定类别范围 ===
    all_classes = list(range(cls_num))

    results = {}
    all_aps = []
    pr_curves = {}

    # 用 dict 防止重复
    per_class_stats = {}

    min_iou, max_iou, step = iou_thresholds
    ious_ = np.arange(min_iou, max_iou, step)

    # === 遍历 IoU ===
    for i, iou_thresh in enumerate(ious_):
        class_aps = []
        all_class_pr_data = []

        fp_types_counter = {'dup': 0, 'offset': 0, 'bg': 0}
        iou_distributions = []

        total_tp_05 = 0
        total_fp_05 = 0
        total_gt_05 = 0
        total_miss_05 = 0

        # === 遍历类别 ===
        for cls in all_classes:
            scores, labels = [], []
            total_gt = 0

            for preds, gts in zip(pred_list, gt_list):

                # === 解析预测 ===
                if len(preds) == 3:
                    pred_boxes, pred_scores, pred_labels = preds
                else:
                    pred_boxes, pred_scores = preds
                    pred_labels = np.zeros(len(pred_boxes), dtype=int)

                # === 解析 GT ===
                if isinstance(gts, tuple):
                    gt_boxes, gt_labels = gts
                else:
                    gt_boxes = gts
                    gt_labels = np.zeros(len(gt_boxes), dtype=int)

                # === 当前类别筛选 ===
                cls_pred_inds = np.where(pred_labels == cls)[0]
                cls_gt_inds = np.where(gt_labels == cls)[0]

                cls_pred_boxes = pred_boxes[cls_pred_inds]
                cls_pred_scores = pred_scores[cls_pred_inds]
                cls_gt_boxes = gt_boxes[cls_gt_inds]

                total_gt += len(cls_gt_boxes)
                matched = np.zeros(len(cls_gt_boxes), dtype=bool)

                # === 匹配 ===
                for box, score in zip(cls_pred_boxes, cls_pred_scores):

                    if len(cls_gt_boxes) == 0:
                        scores.append(score)
                        labels.append(0)
                        fp_types_counter['bg'] += 1
                        continue

                    ious = np.array([compute_iou(box, gt) for gt in cls_gt_boxes])
                    max_iou_val = ious.max()
                    idx = ious.argmax()

                    if max_iou_val >= iou_thresh and not matched[idx]:
                        scores.append(score)
                        labels.append(1)
                        matched[idx] = True
                    else:
                        scores.append(score)
                        labels.append(0)

                        if i == 0:
                            fp_type = categorize_fp(ious, matched, iou_thresh)
                            fp_types_counter[fp_type] += 1

                    if plot_iou_dist:
                        iou_distributions.append(max_iou_val)

            # === 计算 AP ===
            if total_gt == 0 or len(labels) == 0:
                ap = 0.0
                if i == 0:
                    total_gt_05 += total_gt
                    total_miss_05 += total_gt
                    per_class_stats[cls] = {
                        'class_id': cls,
                        'AP': 0.0,
                        'Precision': 0.0,
                        'Recall': 0.0,
                        'F1': 0.0,
                        'Total_GT': int(total_gt)
                    }
            else:
                scores = np.array(scores)
                labels = np.array(labels)

                sorted_idxs = np.argsort(-scores)
                labels = labels[sorted_idxs]

                tp = np.cumsum(labels)
                fp = np.cumsum(1 - labels)

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (total_gt + 1e-6)

                # 单调化
                precision_mono = precision.copy()
                for j in range(len(precision_mono) - 2, -1, -1):
                    precision_mono[j] = max(precision_mono[j], precision_mono[j + 1])

                recall2 = np.concatenate(([0.0], recall, [1.0]))
                precision2 = np.concatenate(([1.0], precision_mono, [0.0]))

                recall_interp = np.linspace(0, 1, 101)
                f_interp = interp1d(recall2, precision2, kind='linear')
                precision_interp = f_interp(recall_interp)

                ap = np.mean(precision_interp)

                all_class_pr_data.append({
                    'recall': recall_interp,
                    'precision': precision_interp,
                    'ap': ap
                })

                # === 仅 IoU=0.5 统计详细指标 ===
                if i == 0:
                    total_tp_05 += tp[-1]
                    total_fp_05 += fp[-1]
                    total_gt_05 += total_gt
                    total_miss_05 += (total_gt - tp[-1])

                    p_cls = tp[-1] / (tp[-1] + fp[-1] + 1e-6)
                    r_cls = tp[-1] / (total_gt + 1e-6)
                    f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls + 1e-6)

                    per_class_stats[cls] = {
                        'class_id': cls,
                        'AP': float(ap),
                        'Precision': float(p_cls),
                        'Recall': float(r_cls),
                        'F1': float(f1_cls),
                        'Total_GT': int(total_gt)
                    }

            class_aps.append(ap)

        # === 当前 IoU 的 mAP ===
        mAP = np.mean(class_aps)
        results[f'mAP@{iou_thresh:.2f}'] = mAP
        all_aps.append(mAP)

        # === PR 曲线（IoU=0.5）===
        if i == 0 and len(all_class_pr_data) > 0:
            all_precisions = np.array([x['precision'] for x in all_class_pr_data])
            mean_precision = np.mean(all_precisions, axis=0)
            mean_recall = all_class_pr_data[0]['recall']
            mean_ap = np.mean([x['ap'] for x in all_class_pr_data])

            pr_curves['recall_smooth'] = mean_recall
            pr_curves['precision_smooth'] = mean_precision
            pr_curves['ap@0.5'] = mean_ap

            p = total_tp_05 / (total_tp_05 + total_fp_05 + 1e-6)
            r = total_tp_05 / (total_gt_05 + 1e-6)
            f1 = 2 * p * r / (p + r + 1e-6)
            mr = total_miss_05 / (total_gt_05 + 1e-6)

            results.update({
                'Precision': p,
                'Recall': r,
                'F1': f1,
                'MR': mr
            })

            # === FP 分类（仅 IoU=0.5）===
            results.update({
                'FPDup': fp_types_counter['dup'],
                'FPOffset': fp_types_counter['offset'],
                'FPBg': fp_types_counter['bg'],
            })

    # === 最终 mAP ===
    results['mAP'] = np.mean(all_aps)

    # === 补全类别 ===
    final_stats = []
    for cls in all_classes:
        if cls in per_class_stats:
            final_stats.append(per_class_stats[cls])
        else:
            final_stats.append({
                'class_id': cls,
                'AP': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'Total_GT': 0
            })

    # 可选排序（推荐）
    final_stats = sorted(final_stats, key=lambda x: x['AP'], reverse=True)

    results['per_class_stats'] = final_stats

    # === PR 输出 ===
    if 'recall_smooth' in pr_curves:
        results['PR'] = {
            'recall': pr_curves['recall_smooth'].copy(),
            'precision': pr_curves['precision_smooth'].copy(),
            'AP@0.5': float(pr_curves['ap@0.5']),
        }

    # === IoU 分布 ===
    if plot_iou_dist and len(iou_distributions) > 0:
        plt.hist(iou_distributions, bins=20, range=(0, 1))
        plt.title("IoU Distribution")
        plt.show()

    # === PR 曲线 ===
    if plot_pr and 'recall_smooth' in pr_curves:
        recall = pr_curves['recall_smooth']
        precision = pr_curves['precision_smooth']

        plt.figure(figsize=(7, 7))
        plt.plot(recall, precision, linewidth=2.5,
                 label=f"{model_name} (AP@0.5={pr_curves['ap@0.5']:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
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
