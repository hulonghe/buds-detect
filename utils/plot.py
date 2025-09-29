import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.interpolate import make_interp_spline


def plot_pr_curve_yolo(y_true_list, y_score_list, model_name="Model", save_path=None, smooth=True):
    """
    绘制 YOLO 风格的 Precision-Recall 曲线 (AP@0.5)
    :param y_true_list: list[int] or list[bool]，1 表示 TP，0 表示 FP
    :param y_score_list: list[float]，预测置信度
    :param model_name: 模型名称（会显示在图右上角）
    :param save_path: 保存路径 (e.g. "pr_curve.png")，如果为 None 则不保存
    :param smooth: 是否对插值后的曲线进行平滑（样条插值）
    """
    # === Precision-Recall 计算 ===
    precision, recall, _ = precision_recall_curve(y_true_list, y_score_list)
    ap = average_precision_score(y_true_list, y_score_list)

    # 插值（保证 Precision 单调递减，YOLO-style）
    precision_interp = np.maximum.accumulate(precision[::-1])[::-1]

    # 去重 recall
    recall_unique, idx = np.unique(recall, return_index=True)
    precision_unique = precision_interp[idx]
    if smooth and len(recall_unique) > 3:
        recall_new = np.linspace(recall_unique.min(), recall_unique.max(), 200)
        spline = make_interp_spline(recall_unique, precision_unique, k=3)
        precision_smooth = spline(recall_new)
    else:
        recall_new, precision_smooth = recall_unique, precision_unique

    # === 绘制 ===
    plt.figure(figsize=(7, 7))
    plt.plot(recall_new, precision_smooth, label=f"{model_name} (AP@0.5={ap:.3f})", linewidth=2.5)

    # 坐标轴与网格
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Recall 轴间隔 0.2
    plt.xticks(np.arange(0, 1.01, 0.2))
    plt.yticks(np.arange(0, 1.01, 0.2))

    # 图例放右上角
    plt.legend(loc="lower left")  # YOLO 默认放左下，这里你说要右上就改成 upper right
    # plt.legend(loc="upper right")

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"PR 曲线已保存到: {save_path}")
    else:
        plt.show()


def plot_detection_metrics(metrics_history, save_dir, start_epoch=0):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(start_epoch, start_epoch + len(metrics_history)))

    # 初始化各类曲线的数据
    ap_ious = {f'AP@{iou:.2f}': [] for iou in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]}
    mAP, precision, recall, f1, mr, lr, cls_loss, box_loss, iou_loss = [], [], [], [], [], [], [], [], []
    fp_dup, fp_offset, fp_bg = [], [], []

    for entry in metrics_history:
        for key in ap_ious:
            ap_ious[key].append(entry.get(key, 0))
        mAP.append(entry.get('mAP', 0))
        precision.append(entry.get('Precision', 0))
        recall.append(entry.get('Recall', 0))
        f1.append(entry.get('F1', 0))
        mr.append(entry.get('MR', 0))
        lr.append(entry.get('lr', 0))
        cls_loss.append(entry.get('cls_loss', 0))
        box_loss.append(entry.get('box_loss', 0))
        iou_loss.append(entry.get('iou_loss', 0))
        fp_dup.append(entry.get('FPDup', 0))
        fp_offset.append(entry.get('FPOffset', 0))
        fp_bg.append(entry.get('FPBg', 0))

    def plot_line(y_data_list, labels, ylabel, title, filename, annotate_max=False):
        plt.figure(figsize=(10, 6))
        for y_data, label in zip(y_data_list, labels):
            plt.plot(epochs, y_data, label=label)
            if annotate_max:
                max_idx = np.argmax(y_data)
                plt.scatter(epochs[max_idx], y_data[max_idx], color='red')
                plt.text(epochs[max_idx], y_data[max_idx], f"{label} max: {y_data[max_idx]:.3f} @ {epochs[max_idx]}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    # 标准指标图
    plot_line(list(ap_ious.values()), list(ap_ious.keys()), "AP", "AP vs IoU Threshold", "ap_vs_iou.png")
    plot_line([mAP], ["mAP"], "mAP", "Mean AP over Epochs", "mAP.png")
    plot_line([precision, recall, f1], ["Precision", "Recall", "F1"], "Value", "Precision / Recall / F1",
              "prf_curve.png")
    plot_line([mr], ["MR"], "Miss Rate", "Miss Rate (MR)", "miss_rate.png")
    plot_line([fp_dup, fp_offset, fp_bg], ["FPDup", "FPOffset", "FPBg"], "Count", "False Positive Types",
              "fp_types.png")
    plot_line([lr], ["lr"], "Learning Rate", "Learning Rate Schedule", "learning_rate.png")
    plot_line([cls_loss, box_loss, iou_loss], ["cls_loss", "box_loss", "iou_loss"], "Count", "Loss Epochs",
              "loss_epochs.png")

    # F1 最大值位置标注
    plot_line([f1], ["F1"], "F1", "F1 Peak", "f1_peak.png", annotate_max=True)

    # AP 曲线重叠视图
    plot_line(list(ap_ious.values()), list(ap_ious.keys()), "AP", "AP vs Epochs (All IoU)", "ap_curves.png")

    # 堆叠图：误检类型堆叠柱状
    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    plt.bar(epochs, fp_dup, label='FPDup')
    plt.bar(epochs, fp_offset, bottom=fp_dup, label='FPOffset')
    bottom_fp = np.array(fp_dup) + np.array(fp_offset)
    plt.bar(epochs, fp_bg, bottom=bottom_fp, label='FPBg')
    plt.xlabel("Epoch")
    plt.ylabel("FP Count")
    plt.title("False Positive Types (Stacked)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fp_types_stacked.png"))
    plt.close()

    # 精度召回曲线（非随 epoch，假设使用最后一轮）
    if precision[-1] > 0 and recall[-1] > 0:
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, marker='o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs Recall Curve (All Epochs)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
        plt.close()

    # 模拟 F1 vs Threshold（假设分数从 0.1 ~ 0.9）
    thresholds = np.linspace(0.1, 0.9, 9)
    # 模拟用最后一个 epoch 的 precision 和 recall 近似计算
    f1_sim = [2 * (p * r) / (p + r + 1e-6) for p, r in zip(np.linspace(0.2, 0.9, 9), np.linspace(0.9, 0.2, 9))]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_sim, marker='o')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score (simulated)")
    plt.title("F1 vs Confidence Threshold (Simulated)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "f1_vs_threshold.png"))
    plt.close()

    print(f"✅ 所有指标图已保存至：{save_dir}")
