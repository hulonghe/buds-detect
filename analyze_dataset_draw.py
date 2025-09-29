import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm

# 全局字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42  # 保证矢量字体嵌入（重要）
plt.rcParams['ps.fonttype'] = 42


# ----------------------
# 复用之前的工具函数
# ----------------------
def read_yolo_labels(label_path, img_w, img_h):
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) < 5:
                continue
            _, cx, cy, w, h = map(float, elems[:5])
            x_min = (cx - w / 2) * img_w
            y_min = (cy - h / 2) * img_h
            x_max = (cx + w / 2) * img_w
            y_max = (cy + h / 2) * img_h
            bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes


def process_image(img_path, dataset_root):
    img_name = os.path.basename(img_path)
    split = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    label_dir = os.path.join(dataset_root, split, "labels")
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    # ---- 1️⃣ 加速读图 ----
    img_np = cv2.imread(img_path)
    if img_np is None:
        return None
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_np.shape[:2]

    # ---- 2️⃣ 快速计算亮度 & Laplacian ----
    brightness = float(np.mean(img_np))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ---- 3️⃣ bbox 统计 ----
    widths, heights, ratios, cx, cy = [], [], [], [], []
    if os.path.exists(label_path):
        bboxes = read_yolo_labels(label_path, img_w, img_h)
        for x_min, y_min, x_max, y_max in bboxes:
            w, h = x_max - x_min, y_max - y_min
            widths.append(w)
            heights.append(h)
            ratios.append(w / h if h > 0 else 0)
            cx.append((x_min + x_max) / 2 / img_w)
            cy.append((y_min + y_max) / 2 / img_h)

    return widths, heights, ratios, cx, cy, brightness, lap_var


def analyze_dataset(dataset_root, max_workers=16):
    # 收集所有图片
    all_img_files = []
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(dataset_root, split, "images")
        if os.path.exists(img_dir):
            img_files = glob.glob(os.path.join(img_dir, "*.*"))
            all_img_files.extend([f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    total = len(all_img_files)
    if total == 0:
        print("[WARN] 未找到任何图片。")
        return None
    print(f"[INFO] 共 {total} 张图片，开始并行分析...")

    widths, heights, ratios, center_x, center_y = [], [], [], [], []
    brightness_list, laplacian_var_list = [], []

    # ✅ 多线程并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, img_path, dataset_root): img_path for img_path in all_img_files}
        for future in tqdm(as_completed(futures), total=total, desc="Analyzing", unit="img"):
            result = future.result()
            if result is None:
                continue
            w, h, r, cx, cy, b, l = result
            widths.extend(w)
            heights.extend(h)
            ratios.extend(r)
            center_x.extend(cx)
            center_y.extend(cy)
            brightness_list.append(b)
            laplacian_var_list.append(l)

    print("[OK] 数据集分析完成 ✅")

    return {
        'widths': np.array(widths),
        'heights': np.array(heights),
        'ratios': np.array(ratios),
        'center_x': np.array(center_x),
        'center_y': np.array(center_y),
        'brightness': np.array(brightness_list),
        'laplacian_var': np.array(laplacian_var_list)
    }


# ----------------------
# 多数据集绘图
# ----------------------
def plot_multi_dataset(datasets_stats, dataset_names):
    """
    datasets_stats: list of stats dict
    dataset_names: list of dataset name strings
    """
    n = len(datasets_stats)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # 保证 axes 变成 (1, 4)

    for i, stats in enumerate(datasets_stats):
        # 宽度直方图
        axes[i, 0].hist(stats['widths'], bins=30, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f"{dataset_names[i]} Width", fontsize=18)
        axes[i, 0].set_xlabel("Width", fontsize=16)
        axes[i, 0].set_ylabel("Frequency", fontsize=16)
        axes[i, 0].tick_params(axis='both', labelsize=15)

        # 高度直方图
        axes[i, 1].hist(stats['heights'], bins=30, color='salmon', edgecolor='black')
        axes[i, 1].set_title(f"{dataset_names[i]} Height", fontsize=18)
        axes[i, 1].set_xlabel("Height", fontsize=16)
        axes[i, 1].set_ylabel("Frequency", fontsize=16)
        axes[i, 1].tick_params(axis='both', labelsize=15)

        # 长宽比分布
        sns.histplot(stats['ratios'], bins=30, kde=True, ax=axes[i, 2], color='green')
        axes[i, 2].set_title(f"{dataset_names[i]} Aspect Ratio", fontsize=18)
        axes[i, 2].set_xlabel("W/H", fontsize=16)
        axes[i, 2].set_ylabel("Frequency", fontsize=16)
        axes[i, 2].tick_params(axis='both', labelsize=15)

        # 中心点热力图
        heatmap, xedges, yedges = np.histogram2d(
            stats['center_x'], stats['center_y'],
            bins=(50, 50),
            range=[[0, 1], [0, 1]]
        )
        # heatmap 绘制
        hm = sns.heatmap(
            heatmap.T,
            cmap="Reds",
            ax=axes[i, 3],
            cbar=True,
            xticklabels=8,
            yticklabels=8
        )
        axes[i, 3].set_title(f"{dataset_names[i]} Center Heatmap", fontsize=18)
        axes[i, 3].tick_params(axis='x', labelsize=15)
        axes[i, 3].tick_params(axis='y', labelsize=15)
        # ✅ 控制右侧 colorbar 的刻度字体大小
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    # 保存为矢量图
    fig_path = os.path.join("./runs/reg", "analyze_datasets.pdf")
    plt.savefig(fig_path, format='pdf', bbox_inches="tight")
    plt.close()
    print(f"[OK] 数据集可视化分析图已保存: {fig_path}")


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    dataset_list = ["A", "B", "C"]
    dataset_root_base = "E:/resources/datasets/tea-buds-database/"
    datasets_stats = []
    for d in dataset_list:
        stats = analyze_dataset(os.path.join(dataset_root_base, d))
        datasets_stats.append(stats)

    plot_multi_dataset(datasets_stats, dataset_list)
