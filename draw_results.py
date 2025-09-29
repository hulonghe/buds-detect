# -*- coding: utf-8 -*-
import os
import re
import argparse
from collections import defaultdict, namedtuple
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ========== 常量与映射 ==========
VAL_DEMO_PREFIX = "val-demo-"

ALL_MODELS_BASE = {
    "ours",
    "rtdetr-x.pt",
    "yolo11l.pt",
    "yolo11m.pt",
    "yolo12l.pt",
    "yolo12m.pt",
    "yolov5lu.pt",
    "yolov5mu.pt",
    "yolov8l.pt",
    "yolov8m.pt",
}

ALL_DATASETS = {
    "tea-buds-owns": "A",
    "teaRob.v9i.yolov11": "B",
    "tea-bud-3": "C",
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ========== 基础工具 ==========
def yolo_to_xyxy(xc, yc, w, h):
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]

def read_pred_file(txt_path: str) -> Tuple[List[List[float]], List[float]]:
    """
    预测标签：每行 `class xc yc w h conf`（均归一化）
    返回：boxes_norm [[x1,y1,x2,y2], ...]，scores [conf, ...]
    """
    boxes, scores = [], []
    if not os.path.isfile(txt_path):
        return boxes, scores
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            xc, yc, w, h, conf = map(float, parts[1:6])
            boxes.append(yolo_to_xyxy(xc, yc, w, h))
            scores.append(conf)
    return boxes, scores

def read_gt_file(txt_path: str) -> List[List[float]]:
    """
    GT 标签：每行 `class xc yc w h`（归一化）
    返回：[[x1,y1,x2,y2], ...]
    """
    boxes = []
    if not os.path.isfile(txt_path):
        return boxes
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append(yolo_to_xyxy(xc, yc, w, h))
    return boxes

def parse_dataset_and_model(folder_name: str) -> Tuple[str, str, str]:
    """
    输入：例如 'val-demo-tea-bud-3.yolo12m.pt2'
    输出：(dataset原名, 数据集别名[A/B/C], 模型名)
    """
    assert folder_name.startswith(VAL_DEMO_PREFIX), f"Invalid name: {folder_name}"
    tail = folder_name[len(VAL_DEMO_PREFIX):]
    for base_model in sorted(ALL_MODELS_BASE, key=len, reverse=True):
        pattern = re.escape(base_model) + r"(\d*)$"
        match = re.search(pattern, tail)
        if match:
            dataset = tail[: match.start()].rstrip(".")
            model = base_model + match.group(1)
            return dataset, ALL_DATASETS.get(dataset, dataset), model
    return tail, tail, "unknown"

def resolve_gt_dir(gt_root: str, dataset_alias: str, split: str) -> str:
    """
    优先使用：gt_root/{dataset_alias}/{split}/labels
    否则回退为 gt_root
    """
    candidate = os.path.join(gt_root, dataset_alias, split, "labels")
    return candidate if os.path.isdir(candidate) else gt_root

def collect_txt_ids(labels_dir: str) -> List[str]:
    if not os.path.isdir(labels_dir):
        return []
    ids = [os.path.splitext(fn)[0] for fn in os.listdir(labels_dir) if fn.lower().endswith('.txt')]
    ids.sort()
    return ids

def find_image(images_dir: str, stem: str) -> str:
    for ext in IMAGE_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    # 容错：大小写扩展名/任意匹配
    for fn in os.listdir(images_dir):
        name, ext = os.path.splitext(fn)
        if name == stem and ext.lower() in IMAGE_EXTS:
            return os.path.join(images_dir, fn)
    return ""

# ========== IoU 与匹配 ==========
def compute_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union

def match_tp_fp_fn(gt_boxes: List[List[float]],
                   pred_boxes: List[List[float]],
                   pred_scores: List[float],
                   iou_thresh: float = 0.5) -> Tuple[int, int, int, List[int]]:
    """
    在归一化坐标下进行匹配（贪心：按score降序对pred逐个匹配IoU>阈值且未被占用的GT）。
    返回：TP, FP, FN, match_idx（与pred等长，-1表示未匹配）
    """
    if len(pred_boxes) != len(pred_scores):
        # 兜底
        pred_scores = [1.0] * len(pred_boxes)
    order = np.argsort(-np.array(pred_scores))
    gt_used = [False] * len(gt_boxes)
    match_idx = [-1] * len(pred_boxes)
    TP = 0
    for oi in order:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = compute_iou_xyxy(pred_boxes[oi], g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            gt_used[best_j] = True
            match_idx[oi] = best_j
            TP += 1
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP
    return TP, FP, FN, match_idx

def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return (2 * tp) / denom if denom > 0 else 0.0

# ========== 可视化 ==========
def to_pixel_boxes(norm_boxes: List[List[float]], w: int, h: int) -> List[Tuple[int,int,int,int]]:
    pix = []
    for x1, y1, x2, y2 in norm_boxes:
        pix.append((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
    return pix

def draw_boxes(image: Image.Image,
               gt_boxes_norm: List[List[float]],
               pred_boxes_norm: List[List[float]],
               pred_scores: List[float],
               iou_thresh: float,
               footer_text: str = "") -> Image.Image:
    """
    在图像上绘制GT(绿)、Pred(红)。右下角写入统计信息或标题。
    """
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size

    gt_pix = to_pixel_boxes(gt_boxes_norm, w, h)
    pred_pix = to_pixel_boxes(pred_boxes_norm, w, h)
    # 绘制GT
    for (x1, y1, x2, y2) in gt_pix:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
    # 绘制Pred
    for k, (x1, y1, x2, y2) in enumerate(pred_pix):
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=3)
        if k < len(pred_scores):
            txt = f"{pred_scores[k]:.2f}"
            tx, ty = x1 + 3, max(0, y1 - 18)
            draw.rectangle([tx - 2, ty - 2, tx + 40, ty + 16], fill=(220, 30, 30))
            draw.text((tx, ty), txt, fill=(255, 255, 255))

    # 右下角信息
    if footer_text:
        pad = 6
        tw, th = draw.textlength(footer_text), 14
        x1 = max(0, w - int(tw) - 2 * pad)
        y1 = max(0, h - th - 2 * pad)
        draw.rectangle([x1, y1, w, h], fill=(0, 0, 0, 180))
        draw.text((x1 + pad, y1 + pad), footer_text, fill=(255, 255, 255))
    return im

def add_header(im: Image.Image, title: str) -> Image.Image:
    """
    在顶部加一行白底标题条（模型名/图名）
    """
    w, _ = im.size
    bar_h = 36
    canvas = Image.new("RGB", (w, im.height + bar_h), (255, 255, 255))
    canvas.paste(im, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    # 底部分隔线
    draw.line([(0, bar_h-1), (w, bar_h-1)], fill=(200, 200, 200), width=1)
    return canvas

def tile_images(rows: List[List[Image.Image]], col_titles: List[str], row_titles: List[str]) -> Image.Image:
    """
    将二维图像数组拼接为网格，并为每列增加列标题，为每行左侧加行标题条。
    """
    # 对齐每列宽度、每行高度
    cols = len(col_titles)
    rows_n = len(rows)
    # 统一每个cell的尺寸（按第一行第一列的大小缩放）
    cell_w, cell_h = rows[0][0].size

    # 列标题条高度
    top_h = 42
    left_w = 120  # 行标题栏宽度

    canvas_w = left_w + cols * cell_w
    canvas_h = top_h + rows_n * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    # 画列标题
    for c, title in enumerate(col_titles):
        x = left_w + c * cell_w
        draw.rectangle([x, 0, x + cell_w, top_h], fill=(255, 255, 255))
        draw.text((x + 10, 10), title, fill=(0, 0, 0))
        draw.line([(x, top_h-1), (x + cell_w, top_h-1)], fill=(200, 200, 200), width=1)

    # 画行标题与粘贴图像
    for r, row in enumerate(rows):
        # 行标题
        y = top_h + r * cell_h
        draw.rectangle([0, y, left_w, y + cell_h], fill=(255, 255, 255))
        draw.text((10, y + 10), row_titles[r], fill=(0, 0, 0))
        draw.line([(left_w-1, y), (left_w-1, y + cell_h)], fill=(200, 200, 200), width=1)
        # 单元格
        for c, im in enumerate(row):
            canvas.paste(im, (left_w + c * cell_w, y))
    return canvas

# ========== 主流程 ==========
def evaluate_per_image_f1_ours(pred_root: str,
                               dataset_name: str,
                               ours_subdir: str,
                               gt_dir: str,
                               iou_thresh: float) -> List[Tuple[str, float, Tuple[int,int,int]]]:
    """
    返回：(img_id, F1, (TP,FP,FN)) 列表（仅基于 Ours）
    """
    labels_dir = os.path.join(ours_subdir, "labels")
    ids = collect_txt_ids(labels_dir)
    results = []
    for img_id in ids:
        pred_txt = os.path.join(labels_dir, img_id + ".txt")
        gt_txt = os.path.join(gt_dir, img_id + ".txt")
        pred_boxes, pred_scores = read_pred_file(pred_txt)
        gt_boxes = read_gt_file(gt_txt)
        tp, fp, fn, _ = match_tp_fp_fn(gt_boxes, pred_boxes, pred_scores, iou_thresh)
        f1 = f1_from_counts(tp, fp, fn)
        results.append((img_id, f1, (tp, fp, fn)))
    # 按 F1 降序、TP 次序排序
    results.sort(key=lambda x: (x[1], x[2][0]), reverse=True)
    return results

def load_image_for_model(model_dir: str, img_id: str) -> Image.Image:
    images_dir = os.path.join(model_dir, "images")
    img_path = find_image(images_dir, img_id)
    if img_path == "":
        raise FileNotFoundError(f"找不到图片：{images_dir}/{img_id}.*")
    im = Image.open(img_path)
    return im

def visualize_cell(model_dir: str,
                   img_id: str,
                   gt_dir: str,
                   iou_thresh: float,
                   cell_size: Tuple[int,int]) -> Image.Image:
    """
    读取单元格需要的图像与标签，绘制并缩放到统一大小。
    """
    labels_dir = os.path.join(model_dir, "labels")
    pred_txt = os.path.join(labels_dir, img_id + ".txt")
    gt_txt = os.path.join(gt_dir, img_id + ".txt")

    pred_boxes, pred_scores = read_pred_file(pred_txt)
    gt_boxes = read_gt_file(gt_txt)

    im = load_image_for_model(model_dir, img_id)
    # 统计（用于角标）
    tp, fp, fn, _ = match_tp_fp_fn(gt_boxes, pred_boxes, pred_scores, iou_thresh)
    f1 = f1_from_counts(tp, fp, fn)
    footer = f"TP:{tp} FP:{fp} FN:{fn} | F1:{f1:.3f}"

    im_vis = draw_boxes(im, gt_boxes, pred_boxes, pred_scores, iou_thresh, footer_text=footer)
    # 统一 cell 尺寸
    if cell_size is not None:
        im_vis = im_vis.resize(cell_size, Image.BILINEAR)
    return im_vis

def main():
    parser = argparse.ArgumentParser(description="生成SCI论文风格的检测效果对比图（每数据集一张大图）")
    parser.add_argument("--pred-root", default=r"C:\Users\Administrator\Desktop\pred-labels",
                        required=False, type=str, help="包含若干 val-demo-* 子目录的根目录")
    parser.add_argument("--gt-root", default=r"E:\resources\datasets\tea-buds-database", required=False, type=str,
                        help="GT 根目录（可含 dataset 同名子目录）")
    parser.add_argument("--out-dir", default="./runs/qual-results", type=str,
                        help="输出目录")
    parser.add_argument("--iou-thresh", default=0.5, type=float,
                        help="单图F1与可视化统计的IoU阈值")
    parser.add_argument("--topk", default=4, type=int,
                        help="从 Ours 中挑选的最佳样本数")
    parser.add_argument("--cell-width", default=640, type=int,
                        help="拼接图中每个可视化单元的宽度（像素）")
    parser.add_argument("--cell-height", default=640, type=int,
                        help="拼接图中每个可视化单元的高度（像素）")
    parser.add_argument("--dataset-filter", default="", type=str,
                        help="仅处理包含该子串的数据集名（可空）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cell_size = (args.cell_width, args.cell_height)

    # 收集：dataset -> {model_name: model_dir}
    datasets_models: Dict[str, Dict[str, str]] = defaultdict(dict)
    dataset_alias_map: Dict[str, str] = {}

    for name in os.listdir(args.pred_root):
        subdir = os.path.join(args.pred_root, name)
        if not (os.path.isdir(subdir) and name.startswith(VAL_DEMO_PREFIX)):
            continue
        dataset, alias_dataset, model = parse_dataset_and_model(name)
        if args.dataset_filter and args.dataset_filter not in dataset:
            continue
        labels_dir = os.path.join(subdir, "labels")
        images_dir = os.path.join(subdir, "images")
        if not (os.path.isdir(labels_dir) and os.path.isdir(images_dir)):
            print(f"[WARN] 跳过：{subdir} 缺少 labels/ 或 images/")
            continue
        datasets_models[dataset][model] = subdir
        dataset_alias_map[dataset] = alias_dataset

    if not datasets_models:
        print("[ERROR] 未发现有效的 val-demo-* 结构。")
        return

    # 针对每个数据集生成大图
    for dataset, model_dirs in datasets_models.items():
        alias_ds = dataset_alias_map.get(dataset, dataset)
        gt_dir = resolve_gt_dir(args.gt_root, alias_ds, "val")

        if "ours" not in model_dirs:
            print(f"[WARN] 数据集 {dataset} 缺少 ours，跳过。")
            continue

        ours_dir = model_dirs["ours"]
        # 用 ours 计算单图F1并挑选 topk
        per_img = evaluate_per_image_f1_ours(args.pred_root, dataset, ours_dir, gt_dir, args.iou_thresh)
        if len(per_img) == 0:
            print(f"[WARN] 数据集 {dataset} 在 ours 上无可评图片，跳过。")
            continue
        topk = per_img[:max(1, min(args.topk, len(per_img)))]
        chosen_ids = [x[0] for x in topk]

        # 列 = 模型（按名称排序，ours放最前）
        model_names = list(model_dirs.keys())
        model_names.sort()
        if "ours" in model_names:
            model_names.remove("ours")
            model_names = ["ours"] + model_names

        # 行 = 选出的图片
        rows_images: List[List[Image.Image]] = []
        row_titles: List[str] = []
        for img_id in chosen_ids:
            row_cells = []
            for m in model_names:
                try:
                    cell = visualize_cell(model_dirs[m], img_id, gt_dir, args.iou_thresh, cell_size)
                except Exception as e:
                    # 出错时给一个占位空白
                    cell = Image.new("RGB", cell_size, (230, 230, 230))
                    d = ImageDraw.Draw(cell)
                    d.text((10, 10), f"{m}\n{img_id}\n{str(e)}", fill=(0, 0, 0))
                row_cells.append(cell)
            rows_images.append(row_cells)
            row_titles.append(img_id)

        # 拼成大图
        big = tile_images(rows_images, col_titles=model_names, row_titles=row_titles)
        # 标题与导出
        title = f"Dataset: {alias_ds} ({dataset})  |  Rows: Top-{len(chosen_ids)} from Ours  |  IoU≥{args.iou_thresh}"
        big = add_header(big, title)
        out_path = os.path.join(args.out_dir, f"{alias_ds}_{dataset}_qualitative_comparison.png")
        big.save(out_path)
        print(f"[OK] 已保存：{out_path}")

if __name__ == "__main__":
    main()
