import os.path
import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from matplotlib import patches
from scipy.ndimage import maximum_filter

from utils.helper import get_topk_boxes


def draw_rectangle(draw, bbox, color="green", width=2):
    """在图像上画矩形"""
    draw.rectangle(bbox, outline=color, width=width)


def visualize_predictions(imgs_np,
                          pred_hm_list,
                          pred_bbox_list,
                          gt_hm_list,
                          gt_boxes,
                          gt_labels,
                          strides=None,
                          save_path=None,
                          save_file=None,
                          k=50,
                          score_thresh=0.25):
    if strides is None:
        print("strides is None, please check your code.")
        return

    num_samples = len(imgs_np)
    num_features = len(pred_hm_list)

    pred_final_boxes, pred_final_scores, pred_final_classes, pred_final_coords = [], [], [], []
    for f in range(num_features):
        final_boxes, final_scores, final_classes, final_coords = get_topk_boxes(pred_hm_list[f], pred_bbox_list[f],
                                                                                K=k,
                                                                                score_thresh=score_thresh)
        pred_final_boxes.append(final_boxes)
        pred_final_scores.append(final_scores)
        pred_final_classes.append(final_classes)
        pred_final_coords.append(final_coords)

    for b in range(0, num_samples, 2):
        fig, axes = plt.subplots(num_features, 3, figsize=(12, 4 * num_features))
        if num_features == 1:
            axes = axes[np.newaxis, ...]

        img = imgs_np[b]
        img_h, img_w, img_c = img.shape
        unique_labels = np.unique(gt_labels[b])
        if len(unique_labels) == 0:
            continue

        for f_idx in range(num_features):
            stride = strides[f_idx]
            pred_hm = pred_hm_list[f_idx][b]  # shape: [C, H, W]
            gt_hm = gt_hm_list[f_idx][b]  # shape: [C, H, W]

            # === 合并热图 ===
            combined_pred_hm = np.mean(pred_hm[unique_labels], axis=0)
            combined_gt_hm = np.mean(gt_hm[unique_labels], axis=0)

            # === 获取目标点坐标 ===
            draw_img = img.copy()
            fig_ax_img, fig_ax_gt, fig_ax_pred = axes[f_idx]

            fig_ax_img.imshow(draw_img)
            fig_ax_img.set_title(f'Image {b} - FM {f_idx}')
            fig_ax_gt.imshow(combined_gt_hm, cmap='viridis', alpha=0.6)
            fig_ax_gt.set_title(f'GT Heatmap')
            fig_ax_pred.imshow(combined_pred_hm, cmap='viridis', alpha=0.6)
            fig_ax_pred.set_title(f'Pred Heatmap')

            # === 画 GT 框 ===
            for label in unique_labels:
                mask = gt_labels[b] == label
                for box in gt_boxes[b][mask]:
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=1.2, edgecolor='green', facecolor='none')
                    fig_ax_img.add_patch(rect)

            # === 画 Pred 框 ===
            final_boxes = pred_final_boxes[f_idx][b]
            final_scores = pred_final_scores[f_idx][b]
            final_classes = pred_final_classes[f_idx][b]
            final_coords = pred_final_coords[f_idx][b]
            for label in unique_labels:
                indices = np.where(final_classes == label)[0]
                for i in indices:
                    pred_box = final_boxes[i]
                    pred_score = final_scores[i]
                    pred_classes = final_classes[i]
                    xs, ys = final_coords[i]

                    dx, dy, dh, dw = pred_box
                    # 使用指数变换恢复宽高
                    h = np.exp(dh) * stride
                    w = np.exp(dw) * stride

                    cx = (xs + dx) * stride
                    cy = (ys + dy) * stride

                    x1 = np.clip(cx - w / 2, 0, img_w - 1)
                    y1 = np.clip(cy - h / 2, 0, img_h - 1)
                    x2 = np.clip(cx + w / 2, 0, img_w - 1)
                    y2 = np.clip(cy + h / 2, 0, img_h - 1)

                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=1.2, edgecolor='red', facecolor='none')
                    fig_ax_img.add_patch(rect)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            if save_file:
                plt.savefig(f"{save_path}/{save_file}", dpi=150, bbox_inches='tight')
            else:
                plt.savefig(f"{save_path}/img_{b}.png", dpi=150, bbox_inches='tight')
        plt.close()


def visualize_single_prediction(img, gt_heatmaps, gt_bboxes, gt_masks, strides, save_path=None):
    """
    可视化单张图像的真实 heatmap 和 bbox。

    参数:
        img: PIL.Image 图像对象 (H, W, 3)
        gt_heatmaps: list of [C, H, W] 每层特征图对应的 ground truth heatmap（numpy array）
        gt_bboxes: list of [4, H, W] 每层特征图对应的 ground truth bbox（numpy array）
        gt_masks: list of [C, H, W] 每层特征图对应的 mask（numpy array）
        strides: list of int，每层特征图对应的 stride
        save_path: str or None，保存路径，若为 None 则直接显示图像
    """
    num_features = len(gt_heatmaps)
    fig, axes = plt.subplots(num_features, 2, figsize=(12, 6 * num_features))

    if num_features == 1:
        axes = [axes]

    img = Image.fromarray(img)  # 从NumPy数组创建PIL图像

    for f_idx in range(num_features):
        # 获取当前层数据
        gt_hm = gt_heatmaps[f_idx]  # [C, H, W]
        gt_bbox = gt_bboxes[f_idx]  # [4, H, W]
        mask = gt_masks[f_idx]  # [C, H, W]
        stride = strides[f_idx]

        # 使用类别最大值作为 heatmap 显示
        gt_hm_show = gt_hm.max(axis=0)  # [H, W]

        # 创建图像副本用于绘制
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        # 获取 mask 中为 True 的位置（即有目标的位置）
        pos_mask = mask.any(axis=0)  # [H, W]
        ys, xs = np.where(pos_mask)  # 获取所有非零坐标

        # 批量解码边界框,还原到图像空间
        dxs = gt_bbox[0, ys, xs]
        dys = gt_bbox[1, ys, xs]
        bhs = gt_bbox[2, ys, xs] * stride
        bws = gt_bbox[3, ys, xs] * stride
        center_x = (xs + dxs) * stride
        center_y = (ys + dys) * stride

        x_mins = np.clip(center_x - bws / 2, 0, img.width)
        y_mins = np.clip(center_y - bhs / 2, 0, img.height)
        x_maxs = np.clip(center_x + bws / 2, 0, img.width)
        y_maxs = np.clip(center_y + bhs / 2, 0, img.height)

        # 绘制每个框
        for i in range(len(xs)):
            draw_rectangle(draw, [(x_mins[i], y_mins[i]), (x_maxs[i], y_maxs[i])], color="green")

        # 显示图像和热图
        ax_img, ax_hm = axes[f_idx]

        ax_img.imshow(img_with_boxes)
        ax_img.set_title(f"Feature Map {f_idx} with Ground Truth BBoxes")
        ax_img.axis("off")

        ax_hm.imshow(gt_hm_show, cmap='viridis', alpha=0.6)
        ax_hm.set_title(f"Feature Map {f_idx} Heatmap")
        ax_hm.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def draw_boxes(img, boxes, labels=None, scores=None, color=(0, 255, 0),
               label_text=None, thickness=1, normalized=True):
    """
    img: numpy array [H, W, 3]，值范围 [0, 255]
    boxes: tensor [N, 4]，格式为 [x1, y1, x2, y2]，可以是归一化的或像素坐标
    labels: optional, tensor [N]
    scores: optional, tensor [N]
    color: box color (B, G, R)
    label_text: list of str，长度为 num_classes，用于显示类别名
    thickness: 边框线粗细
    normalized: 是否是归一化坐标（True 表示 [0,1]，False 表示像素坐标）
    """
    img = img.copy()
    h, w = img.shape[:2]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        scores = scores.cpu().numpy()

    # 将归一化坐标转换为像素坐标
    if normalized:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * w  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * h  # y1, y2

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and label_text is not None:
            cls_id = int(labels[i])
            score_id = f'[{float(scores[i]):.2f}]' if scores is not None else None
            if 0 <= cls_id < len(label_text):
                label_str = label_text[cls_id] + (score_id if score_id is not None else "")
                cv2.putText(img, label_str, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, lineType=cv2.LINE_AA)
    return img
