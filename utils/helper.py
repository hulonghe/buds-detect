import math
import random

import pynvml
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

import os

from utils.mean_std import get_mean_std
import psutil

# 设置环境变量禁止更新检查
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "true"

seed_base = 2025


# 获取 GPU 显存信息（MB）
def print_gpu_usage(gpu_id=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    pynvml.nvmlShutdown()
    return f"{util.gpu}%|{meminfo.used / 1024 ** 2:.0f}/{meminfo.total / 1024 ** 2:.0f}MB"


# 获取系统内存使用情况（GB）
def get_cpu_memory_info():
    mem = psutil.virtual_memory()
    used = (mem.total - mem.available) / 1024 ** 3
    total = mem.total / 1024 ** 3
    return f"{used:.1f}/{total:.1f}GB ({mem.percent}%)"


# 写入日志函数
def write_train_log(epoch, epochs,
                    epoch_loss, epoch_iou_loss, epoch_cls_loss, epoch_reg_loss,
                    current_lr, log_file="runs/train_log.txt", mode='Train'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    line = (f"{mode} Epoch {epoch + 1}/{epochs}, "
            f"Total Loss: {epoch_loss:.8f}, "
            f"Iou Loss: {epoch_iou_loss:.8f}, "
            f"Cls Loss: {epoch_cls_loss:.8f}, "
            f"Box Loss: {epoch_reg_loss:.8f}, "
            f"lr: {current_lr:.8f}\n")

    with open(log_file, 'a') as f:
        f.write(line)

    return line


# 写入验证日志函数
def write_val_log(epoch, epochs, metrics, log_file="runs/val_log.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 生成日志行
    metric_str = ", ".join([
        f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in metrics.items()
    ])

    line = f"Val Epoch {epoch + 1}/{epochs}, {metric_str}\n"

    # 写入文件
    with open(log_file, 'a') as f:
        f.write(line)

    return line


def generate_targets(targets, feature_sizes, strides, img_size=640, num_classes=None, device="cpu"):
    """
    Anchor-Free 动态中心采样目标生成器（改进版 FCOS 风格）。

    返回：
        List[Dict]，每层输出：
            - obj_target: [B, 1, H, W]
            - cls_target: [B, H, W]
            - reg_target: [B, 4, H, W]
            - reg_mask:   [B, 1, H, W] 表示该位置是否计算回归 loss
    """
    batch_size = len(targets)
    num_levels = len(feature_sizes)
    outputs = []

    for lvl in range(num_levels):
        H, W = feature_sizes[lvl]
        stride = strides[lvl]

        obj_target = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
        cls_target = torch.full((batch_size, H, W), -1, dtype=torch.long, device=device)
        reg_target = torch.zeros((batch_size, 4, H, W), dtype=torch.float32, device=device)
        reg_mask = torch.zeros((batch_size, 1, H, W), dtype=torch.bool, device=device)

        for b in range(batch_size):
            boxes = targets[b]['boxes'].to(device)  # [N, 4]
            labels = targets[b]['labels'].to(device)  # [N]
            if boxes.numel() == 0:
                continue

            x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            for i in range(boxes.size(0)):
                box = boxes[i]
                label = labels[i]
                center_x, center_y = cx[i], cy[i]
                bw, bh = w[i], h[i]

                # 动态匹配半径：控制在当前层特征图上的匹配范围
                radius = max(2, int(0.5 * min(bw, bh) / stride))
                grid_cx = int(center_x / stride)
                grid_cy = int(center_y / stride)

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        gy = grid_cy + dy
                        gx = grid_cx + dx
                        if 0 <= gx < W and 0 <= gy < H:
                            obj_target[b, 0, gy, gx] = 1.0
                            cls_target[b, gy, gx] = label
                            reg_mask[b, 0, gy, gx] = True
                            reg_target[b, :, gy, gx] = torch.tensor([
                                center_x / img_size,
                                center_y / img_size,
                                bw / img_size,
                                bh / img_size
                            ], device=device)

        outputs.append({
            'obj_target': obj_target,
            'cls_target': cls_target,
            'reg_target': reg_target,
            'reg_mask': reg_mask
        })

    return outputs


def draw_boxes_and_save(image_path, results, save_path, class_names=None):
    """
    在图像上绘制检测框并保存结果，自动修正越界框
    :param image_path: 图像地址
    :param results: 检测结果列表，每个元素是包含bbox,score,class的字典
    :param save_path: 结果保存路径
    :param class_names: 类别名称列表
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    width, height = image.size
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for result in results:
        bbox = result.get('bbox')
        score = result.get('score', 1.0)
        class_id = result.get('class', -1)

        # 输入检查
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if class_names is not None and (class_id < 0 or class_id >= len(class_names)):
            continue

        x1, y1, x2, y2 = map(float, bbox)

        # Step 1: 纠正越界坐标，限制在图像范围内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width - 1))  # 确保宽度 > 0
        y2 = max(y1 + 1, min(y2, height - 1))  # 确保高度 > 0

        corrected_bbox = [x1, y1, x2, y2]

        # Step 2: 绘制矩形框
        draw.rectangle(corrected_bbox, outline="red", width=2)

        # Step 3: 构建标签文本
        if class_names:
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"{class_id}: {score:.2f}"

        # Step 4: 获取文本大小
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        tx, ty = x1, y1 - th

        # Step 5: 如果标签位置在图像外，调整到下方或内部
        if ty < 0:
            ty = y2  # 放在框下方
        if tx + tw > width:
            tx = width - tw  # 向左对齐

        # Step 6: 绘制文本背景和文字
        draw.rectangle([tx, ty, tx + tw, ty + th], fill="red")
        draw.text((tx, ty), label, fill="white", font=font)

    # Step 7: 保存图像
    try:
        image.save(save_path)
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")


def get_train_transform(img_size=640, path=None, mean=None, std=None, val=False):
    if path is not None and (mean is None or std is None):
        mean, std = get_mean_std(path, img_size)

    if val:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return A.Compose([
        # 色彩扰动
        A.OneOf([
            A.ColorJitter(0.1, 0.1, 0.1, 0.05),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
        ], p=0.15),

        # 局部对比度增强
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.2),

        # 噪声、压缩
        A.OneOf([
            A.ISONoise(p=0.2),
            A.ImageCompression(quality_range=(40, 70), p=0.1)
        ], p=0.3),

        # 几何变化
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               p=0.4),
            A.Affine(shear=3, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=0.3)
        ], p=0.3),

        # 翻转与旋转
        A.OneOf([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.2)
        ], p=0.25),

        # 模糊扰动
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 5)),
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3)
        ], p=0.3),

        # 遮挡模拟
        A.OneOf([
            CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.03, 0.1), hole_width_range=(0.03, 0.1),
                          fill="random_uniform", p=0.2),
            A.GridDropout(ratio=0.05, p=0.15)
        ], p=0.2),

        # 尺寸标准化
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),

        # 归一化 + Tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def clear_gpu_cache():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def collate_fn(batch):
    # Step 1: 筛除无效样本（如 target 中无 boxes）
    valid_batch = []
    for item in batch:
        if item is None:
            continue
        img_tensor, boxes, labels, gt_heatmaps, gt_bboxes, gt_masks, gt_coords, gt_sigma_map = item
        valid_batch.append((img_tensor, boxes, labels, gt_heatmaps, gt_bboxes, gt_masks, gt_coords, gt_sigma_map))

    # Step 2: 判断是否为空 batch
    if len(valid_batch) == 0:
        return None

    return valid_batch


def make_aux_targets(boxes, img_size, small_thr=32):
    """
    boxes: Tensor/ndarray [N,4], 归一化到 [0,1]
    img_size: (H, W)
    small_thr: 小目标阈值(像素)
    """
    H, W = img_size
    has_obj = int(len(boxes) > 0)
    has_small = 0
    for (x1, y1, x2, y2) in boxes:
        w = (x2 - x1) * W
        h = (y2 - y1) * H
        if max(w, h) < small_thr:
            has_small = 1
            break
    return {'small': has_small, 'normal': has_obj}


def custom_collate_fn(batch):
    """
    batch: list of (img, boxes, labels)
    img: Tensor [C,H,W]
    boxes: Tensor/ndarray [N,4] (归一化 xyxy)
    labels: Tensor/ndarray [N]
    """
    imgs, boxes, labels, img_paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # [B,C,H,W]
    B, C, H, W = imgs.shape

    aux_targets = []
    for b in range(B):
        aux_targets.append(make_aux_targets(boxes[b], (H, W)))

    return imgs, boxes, labels, aux_targets, img_paths


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1, b1, c1 = 1, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap, center, radius, k=1):
    """
    在 heatmap 中绘制单个高斯圆，并返回中心坐标与半径。
    输入：
        heatmap: Tensor[(H, W)]
        center: (x, y)
        radius: int
    输出：
        center: Tensor[2]
        radius: float
    """
    h, w = heatmap.shape[-2:]
    x, y = int(center[0]), int(center[1])

    if radius <= 0 or not (0 <= x < w and 0 <= y < h):
        return None, None  # 返回空，调用者判断

    diameter = 2 * radius + 1
    device = heatmap.device
    gaussian = torch.exp(-((torch.arange(diameter, device=device).float() - radius) ** 2) / (2 * radius ** 2))
    gaussian = gaussian[:, None] * gaussian[None, :]

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if masked_gaussian.shape == masked_heatmap.shape:
        heatmap[y - top:y + bottom, x - left:x + right] = torch.maximum(masked_heatmap, masked_gaussian * k)

    return torch.tensor([x, y], dtype=torch.float32, device=device), float(radius)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def get_topk_boxes(pred_heatmap, pred_bbox, K=100, score_thresh=0.3):
    """
    Args:
        pred_heatmap: [B, C, H, W]，np.float32，已 sigmoid
        pred_bbox:    [B, 4, H, W]，np.float32，如 [cx, cy, w, h] 或 [l, t, r, b]

    Returns:
        final_boxes:   [B, K, 4]
        final_scores:  [B, K]
        final_classes: [B, K]
        final_coords:  [B, K, 2]
    """
    B, C, H, W = pred_heatmap.shape

    # 展平 heatmap 到 [B, C, H*W]
    heatmap = pred_heatmap.reshape(B, C, -1)

    # 1. 每类内部 topk（取冗余点）
    topk_scores, topk_inds = [], []
    for hm in heatmap:
        scores, inds = [], []
        for c in range(C):
            flat = hm[c]
            idx = np.argsort(-flat)
            topk_idx = idx[:K]
            topk_score = flat[topk_idx]
            scores.append(topk_score)
            inds.append(topk_idx)
        topk_scores.append(scores)
        topk_inds.append(inds)

    topk_scores = np.array(topk_scores)  # [B, C, K]
    topk_inds = np.array(topk_inds)  # [B, C, K]

    # 2. 构造类别索引
    topk_classes = np.tile(np.arange(C).reshape(1, C, 1), (B, 1, K))  # [B, C, K]

    # 3. 展平成 [B, C*K]
    topk_scores_flat = topk_scores.reshape(B, -1)  # [B, C*K]
    topk_inds_flat = topk_inds.reshape(B, -1)  # [B, C*K]
    topk_classes_flat = topk_classes.reshape(B, -1)  # [B, C*K]

    # 4. 在所有类别中再 topk，选最终 K 个点
    topk_idx_all = np.argsort(-topk_scores_flat, axis=1)[:, :K]  # [B, K]
    topk_scores_all = np.take_along_axis(topk_scores_flat, topk_idx_all, axis=1)  # [B, K]
    topk_inds_all = np.take_along_axis(topk_inds_flat, topk_idx_all, axis=1)  # [B, K]
    topk_classes_all = np.take_along_axis(topk_classes_flat, topk_idx_all, axis=1)  # [B, K]

    # 5. 计算 x, y 坐标
    topk_ys = topk_inds_all // W  # [B, K]
    topk_xs = topk_inds_all % W  # [B, K]
    final_coords = np.stack([topk_xs, topk_ys], axis=-1).astype(np.float32)  # [B, K, 2]

    # 6. 取出 bbox
    pred_bbox = np.transpose(pred_bbox, (0, 2, 3, 1))  # [B, H, W, 4]
    final_boxes = []
    for b in range(B):
        boxes = pred_bbox[b][topk_ys[b], topk_xs[b]]  # [K, 4]
        final_boxes.append(boxes)
    final_boxes = np.array(final_boxes)  # [B, K, 4]

    return final_boxes, topk_scores_all, topk_classes_all, final_coords


def set_seed(seed: int = 42, deterministic: bool = False, workers: bool = True):
    """
    固定所有可控随机种子，确保训练过程可复现。
    
    参数：
        seed (int): 种子值
        deterministic (bool): 是否设置为确定性（强一致但可能影响性能）
        workers (bool): 是否设置 DataLoader 的 worker_init_fn 和 worker seed
    """

    # Python 随机模块
    random.seed(seed)
    # NumPy 随机模块
    np.random.seed(seed)
    # 环境变量（影响 Python hash、NVIDIA 行为等）
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 用于部分 cuBLAS 算法的确定性（PyTorch >= 1.8）

    # PyTorch 随机种子（CPU / GPU / 多卡）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 后端行为（是否启用 benchmark、是否使用确定性算法）
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # PyTorch 全局确定性算法控制（PyTorch >= 1.8）
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    # DataLoader 多线程 worker 的种子（如果配合 num_workers 使用）
    if workers:
        seed_base = seed

    print(f"[Seed Fixed] seed={seed}, deterministic={deterministic}, workers={workers}")


def dataloader_worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cosine_ramp_up(x, steps, max_val=0.25, min_val=0.0):
    """
    平滑从 min_val 上升到 max_val 的余弦函数。

    参数：
    - x: 当前步数（可为 float）
    - steps: 上升的总步数
    - max_val: 最终值（默认 0.25）
    - min_val: 起始值（默认 0.0）

    返回：
    - 当前步数对应的平滑值
    """
    if x <= 0:
        return min_val
    elif x >= steps:
        return max_val
    else:
        ratio = (1 - np.cos(np.pi * x / steps)) / 2
        return min_val + (max_val - min_val) * ratio
