import os
import csv
from utils.YoloDataset import YoloDataset
from utils.helper import get_train_transform

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def analyze_yolo_dataset(dataset, num_samples=2000, input_size=640, cls_names=None):
    """
    分析 YoloDataset 数据分布，并输出诊断结论（中文）
    Args:
        dataset: YoloDataset 实例
        num_samples: 最大抽样数量（避免大数据集过慢）
        input_size: 模型输入尺寸（例如 640），用于判定小/中/大目标
        cls_names: 类别名字列表（可选），用于输出更友好
    """

    cls_counter = Counter()
    num_boxes_per_img = []
    box_areas = []
    aspect_ratios = []
    centers_x, centers_y = [], []
    empty_count = 0

    total = min(len(dataset), num_samples)

    for i in tqdm(range(total), desc="分析数据集"):
        img, boxes, labels, _ = dataset[i]
        if len(boxes) == 0:
            empty_count += 1
            continue

        num_boxes_per_img.append(len(boxes))
        cls_counter.update(labels.tolist())

        for box in boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            w = max(0, x_max - x_min)
            h = max(0, y_max - y_min)
            area = w * h
            if area <= 0:
                continue
            box_areas.append(area)
            aspect_ratios.append(w / h if h > 0 else 0)
            centers_x.append((x_min + x_max) / 2)
            centers_y.append((y_min + y_max) / 2)

    # ================== 统计超级小/小/中/大目标比例 ==================
    pixel_areas = [a * (input_size ** 2) for a in box_areas]
    super_tiny, tiny, small1, small2, medium, large = 0, 0, 0, 0, 0, 0
    for pa in pixel_areas:
        if pa < 8 ** 2:
            super_tiny += 1
        elif pa < 16 ** 2:
            tiny += 1
        elif pa < 32 ** 2:
            small1 += 1
        elif pa < 48 ** 2:
            small2 += 1
        elif pa < 96 ** 2:
            medium += 1
        else:
            large += 1

    total_boxes = len(pixel_areas)
    super_tiny_ratio = super_tiny / total_boxes if total_boxes else 0
    tiny_ratio = tiny / total_boxes if total_boxes else 0
    small1_ratio = small1 / total_boxes if total_boxes else 0
    small2_ratio = small2 / total_boxes if total_boxes else 0
    medium_ratio = medium / total_boxes if total_boxes else 0
    large_ratio = large / total_boxes if total_boxes else 0

    # ================== 绘制可视化 ==================
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].hist(num_boxes_per_img, bins=20)
    axs[0, 0].set_title("每张图的目标数分布")

    axs[0, 1].bar(cls_counter.keys(), cls_counter.values())
    axs[0, 1].set_title("类别分布")

    axs[0, 2].hist(box_areas, bins=50, log=True)
    axs[0, 2].set_title("目标归一化面积分布")

    axs[1, 0].hist(aspect_ratios, bins=50, log=True)
    axs[1, 0].set_title("目标长宽比分布")

    axs[1, 1].hexbin(centers_x, centers_y, gridsize=40, cmap="viridis")
    axs[1, 1].set_title("目标中心点分布")

    axs[1, 2].bar(["空标注比例"], [empty_count / total])
    axs[1, 2].set_ylim(0, 1)
    axs[1, 2].set_title("无目标图像比例")

    plt.tight_layout()
    plt.show()

    # ================== 自动诊断 ==================
    report = []
    empty_ratio = empty_count / total
    mean_boxes = np.mean(num_boxes_per_img) if num_boxes_per_img else 0
    area_mean = np.mean(box_areas) if box_areas else 0
    area_median = np.median(box_areas) if box_areas else 0

    # 类别分布
    if len(cls_counter) > 1:
        cls_values = np.array(list(cls_counter.values()))
        imbalance_ratio = cls_values.max() / (cls_values.min() + 1e-6)
        if imbalance_ratio > 10:
            report.append("⚠️ 类别分布极度不均衡，可能导致训练严重偏差。建议：类别重采样、损失函数加权或数据扩充。")
        elif imbalance_ratio > 3:
            report.append("⚠️ 类别分布不均衡。建议：采用类别均衡采样或损失函数调权。")
        else:
            report.append("✅ 类别分布相对均衡。")
    else:
        report.append("ℹ️ 数据集中只有一个类别。")

    # 目标数量
    if mean_boxes < 0.5:
        report.append("⚠️ 平均每张图不到 0.5 个目标，存在大量空图，可能影响训练。建议：检查标注质量或剔除空图。")
    elif mean_boxes > 50:
        report.append("⚠️ 平均每张图超过 50 个目标，可能导致显存压力和训练困难。建议：裁剪图像或限制目标数。")
    else:
        report.append(f"✅ 平均每张图目标数合理（{mean_boxes:.2f} 个）。")

    # 目标面积
    if area_median < 0.001:
        report.append("⚠️ 数据集中小目标占多数，检测难度大。建议：提高输入分辨率或调整 anchor 尺寸。")
    elif area_median > 0.2:
        report.append("⚠️ 数据集中大目标占多数，可能影响小目标检测性能。建议：考虑缩小输入尺寸。")
    else:
        report.append("✅ 目标面积分布合理。")

    # 长宽比
    if np.max(aspect_ratios) > 10 or np.min(aspect_ratios) < 0.1:
        report.append("⚠️ 存在极端长宽比的目标，需确保 anchor 能覆盖。")
    else:
        report.append("✅ 长宽比分布正常。")

    # 空标注
    if empty_ratio > 0.5:
        report.append(f"⚠️ 超过 {empty_ratio:.1%} 的图像没有目标，浪费计算资源。建议：剔除或单独处理。")
    elif empty_ratio > 0.2:
        report.append(f"⚠️ 有 {empty_ratio:.1%} 的图像没有目标，需要确认是否合理。")
    else:
        report.append("✅ 无目标图像比例正常。")

    # 小中大目标比例 + 小目标细分
    report.append(
        f"📊 超级小目标: {super_tiny_ratio:.1%}, 极小目标: {tiny_ratio:.1%}, "
        f"小目标: {small1_ratio:.1%}, 偏小目标: {small2_ratio:.1%}, "
        f"中目标: {medium_ratio:.1%}, 大目标: {large_ratio:.1%}"
    )

    if super_tiny_ratio > 0.2:
        report.append("⚠️ 超级小目标比例过高（面积 < 8²），检测极难，建议提升输入分辨率或使用超分辨率方法。")
    if tiny_ratio + small1_ratio + small2_ratio > 0.6:
        report.append("⚠️ 小目标整体占比超过 60%，需要小目标优化策略。")
    elif large_ratio > 0.6:
        report.append("⚠️ 大目标过多，模型可能对小目标不敏感。")
    else:
        report.append("✅ 小/中/大目标比例分布较为均衡。")

    # ================== 输出报告 ==================
    print("\n====== 数据集统计信息 ======")
    print(f"总样本数: {total}")
    print(f"空标注图像比例: {empty_ratio:.2%}")
    print(f"类别分布: {cls_counter}")
    print(f"平均每张图目标数: {mean_boxes:.2f}")
    print(f"目标面积均值: {area_mean:.4f}, 中位数: {area_median:.4f}")

    print("\n====== 自动诊断结论 ======")
    for line in report:
        print(line)

    return {
        "cls_counter": cls_counter,
        "num_boxes_per_img": num_boxes_per_img,
        "box_areas": box_areas,
        "aspect_ratios": aspect_ratios,
        "empty_ratio": empty_ratio,
        "super_tiny_ratio": super_tiny_ratio,
        "tiny_ratio": tiny_ratio,
        "small1_ratio": small1_ratio,
        "small2_ratio": small2_ratio,
        "medium_ratio": medium_ratio,
        "large_ratio": large_ratio,
        "diagnosis": report
    }


def analyze_dataset_pair(data_root, size, mean, std, cls_names, num_samples=5000):
    """同时分析 train + val 并返回汇总结果"""
    subsets = {}
    for split in ["train", "val"]:
        dataset = YoloDataset(
            img_dir=os.path.join(data_root, split, "images"),
            label_dir=os.path.join(data_root, split, "labels"),
            img_size=size,
            normalize_label=True,
            cls_num=1,
            mean=mean,
            std=std,
            mode="train",
            max_memory_usage=0.0,
            mosaic_prob=0.3,
            transforms=get_train_transform(size, mean=mean, std=std, val=True),
            easy_fraction=1.0
        )

        results = analyze_yolo_dataset(
            dataset,
            input_size=size,
            num_samples=num_samples,
            cls_names=cls_names
        )
        subsets[split] = results

    return subsets


if __name__ == '__main__':
    size = 320
    data_names = [
        ['A', (0.4206, 0.502, 0.3179), (0.2162, 0.2199, 0.1967)],
    ]
    base_root = r"E:/resources/datasets/tea-buds-database/"

    all_results = []

    for data_name, mean, std in data_names:
        print("\n" + "=" * 50)
        print(f"开始分析数据集: {data_name}")
        print("=" * 50)

        data_root = os.path.join(base_root, data_name)
        subsets = analyze_dataset_pair(
            data_root, size, mean, std, cls_names=['tea-bud'], num_samples=10000
        )

        for split, res in subsets.items():
            row = {
                "dataset": data_name,
                "split": split,
                "samples": len(res["num_boxes_per_img"]),
                "empty_ratio": res["empty_ratio"],
                "mean_boxes": (sum(res["num_boxes_per_img"]) / max(1, len(res["num_boxes_per_img"]))),
                "super_tiny_ratio": res["super_tiny_ratio"],
                "tiny_ratio": res["tiny_ratio"],
                "small1_ratio": res["small1_ratio"],
                "small2_ratio": res["small2_ratio"],
                "medium_ratio": res["medium_ratio"],
                "large_ratio": res["large_ratio"],
                "diagnosis": " | ".join(res["diagnosis"])  # 多条诊断合并为一条
            }
            all_results.append(row)

    # ================== 导出 CSV ==================
    output_csv = "./runs/dataset_analysis.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ 所有结果已导出到 {output_csv}")
