import json
import shutil

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from nn.bbox.EnhancedDynamicBoxDetector_ablation import DynamicBoxDetector
from utils.YoloDataset import YoloDataset
from utils.dynamic_postprocess import dynamic_postprocess
from utils.helper import custom_collate_fn, write_val_log, get_train_transform, set_seed
from utils.metrics import compute_ap_metrics
from torch.amp import autocast
import csv
import time
import torch
from tqdm import tqdm
import os


def save_predictions_to_txt(batch_pred_boxes, batch_pred_scores, batch_pred_labels, img_paths, save_dir):
    """
    将预测结果保存为 YOLO 格式 txt 文件 (class_id x_center y_center width height confidence)

    参数:
        batch_pred_boxes: list[ndarray(N, 4)] 归一化边框 (x1, y1, x2, y2)
        batch_pred_scores: list[ndarray(N,)] 置信度
        batch_pred_labels: list[ndarray(N,)] 分类 ID
        img_paths: list[str] 图片路径或文件名
        save_dir: 保存 txt 文件的目录
    """
    os.makedirs(save_dir, exist_ok=True)

    B = len(img_paths)

    for b in range(B):
        boxes = batch_pred_boxes[b]  # (N, 4)
        scores = batch_pred_scores[b]  # (N,)
        labels = batch_pred_labels[b]  # (N,)

        lines = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(scores[i])
            cls_id = int(labels[i])

            if conf <= 0:
                continue

            # 转换成 (xc, yc, w, h)
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}"
            lines.append(line)

        # 文件名
        img_name = img_paths[b]
        txt_path = os.path.join(save_dir, img_name + ".txt")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))


@torch.no_grad()
def validate_loader(model, device, data_loader, dataset,
                    t0=None, epoch=0, epochs=0,
                    score_thresh=0.25, iou_thresh=0.25,
                    file_base='train', use_softnms=False,
                    base_path='runs/reg', criterion=None,
                    warmup_epochs=5, weights_=None, nms_method="nms", is_finish=False):
    """
        数据真实框：
            boxes [B,N,4] 已经归一化数据
            labels [B,N] 框分类ID
        预测框：
            score_maps: [B, N, C] 未归一化的预测分类分数
            box_maps: [B, N, 4] 已归一化的预测框
    """

    model.eval()
    all_gts = []  # 供 mAP 用
    all_predictions = []  # 供 mAP 用
    all_pred_details = []  # 用于后续绘图或错误分析

    if t0 is None:
        t0 = time.time()
        log_file = f"{base_path}/{file_base}_{t0}.txt"
    else:
        log_file = f"{base_path}/{file_base}_{t0}.txt"

    epoch_loss = 0.0
    epoch_iou_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_box_loss = 0.0

    pbar = tqdm(data_loader, desc=f"Val | score={score_thresh} iou={iou_thresh}")
    for imgs, boxes, labels, log_targets, img_paths in pbar:
        if imgs is None:
            continue
        imgs = imgs.to(device)
        boxes = [t.to(device) for t in boxes]
        labels = [t.to(device) for t in labels]
        batch_size = imgs.size(0)

        with autocast(device_type=device, enabled=True):
            score_maps, box_maps, aux_logits = model(imgs)
            if criterion is not None:
                total_loss, loss_cls, loss_box, loss_iou = criterion(score_maps, box_maps, aux_logits,
                                                                     boxes, labels, log_targets,
                                                                     epoch, epochs, warmup_epochs, weights_)
                epoch_loss += total_loss.cpu().item()
                epoch_iou_loss += loss_iou.cpu().item()
                epoch_cls_loss += loss_cls.cpu().item()
                epoch_box_loss += loss_box.cpu().item()
            score_maps = torch.sigmoid(score_maps)

        score_maps = score_maps.detach().cpu()
        box_maps = box_maps.detach().cpu()
        batch_pred_boxes, batch_pred_scores, batch_pred_labels = dynamic_postprocess(
            score_maps, box_maps,
            img_size=(dataset.img_size, dataset.img_size),
            score_thresh=score_thresh, iou_thresh=iou_thresh,
            upscale=False, method="soft-nms" if use_softnms == True else nms_method
        )
        save_predictions_to_txt(batch_pred_boxes, batch_pred_scores, batch_pred_labels, img_paths,
                                os.path.join(base_path, "labels"))

        for i in range(batch_size):
            gt_boxes = boxes[i].cpu().numpy()
            pred_boxes = batch_pred_boxes[i].cpu().numpy() if isinstance(batch_pred_boxes[i], torch.Tensor) else \
                batch_pred_boxes[i]
            pred_scores = batch_pred_scores[i].cpu().numpy() if isinstance(batch_pred_scores[i], torch.Tensor) else \
                batch_pred_scores[i]
            pred_labels = batch_pred_labels[i].cpu().numpy() if isinstance(batch_pred_labels[i], torch.Tensor) else \
                batch_pred_labels[i]

            all_predictions.append((pred_boxes, pred_scores))
            all_gts.append(gt_boxes)

            all_pred_details.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            })

    epoch_loss /= len(data_loader)
    epoch_iou_loss /= len(data_loader)
    epoch_cls_loss /= len(data_loader)
    epoch_box_loss /= len(data_loader)

    # === AP / Precision 等指标 ===
    # === AP / Precision 等指标 ===
    metrics = compute_ap_metrics(
        all_predictions,
        all_gts,
        iou_thresholds=None,
        plot_iou_dist=False,
        plot_pr=is_finish,
        save_path=os.path.join(base_path, "PR.png"),
        model_name="Ours",
    )
    save_pr_to_json(metrics, os.path.join(base_path, "pr_result_source.json"))
    print(f"PR 数据已保存到 {os.path.join(base_path, 'pr_result_source.json')}")

    metrics.pop("PR", None)  # 如果存在就移除，不存在也不会报错
    # 打印日志
    print("\t".join([k for k in metrics.keys() if k != "PR"]))
    print("\t".join([
        f"{v:.4f}" if isinstance(v, float) else str(v)
        for k, v in metrics.items() if k != "PR"
    ]))

    # 写入文件
    val_log = write_val_log(epoch, epochs, metrics, log_file=log_file)

    return {
        "metrics": metrics,
        "predictions": all_pred_details,
        "gt_boxes": all_gts,
        "score_thresh": score_thresh,
        "iou_thresh": iou_thresh,
        "val_loss": epoch_loss,
        "val_iou_loss": epoch_iou_loss,
        "val_cls_loss": epoch_cls_loss,
        "val_box_loss": epoch_box_loss,
    }


def save_pr_to_json(pr_data, path):
    """
    pr_data: 可能包含 ndarray 的字典或嵌套结构
    """

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # ndarray -> list
        else:
            return obj

    pr_data = convert(pr_data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(pr_data, f, ensure_ascii=False, indent=4)


def compute_score(entry, weights=None):
    # 默认权重设置
    if weights is None:
        weights = {
            "mAP": 0.2,
            "AP@0.50": 0.6,
            "Precision": 0.2,
        }

    score = 0.0
    for k, w in weights.items():
        score += entry.get(k, 0) * w
    return score


def run_validation_grid(model_, DEVICE, train_loader_, train_dataset, val_loader_, val_dataset, use_softnms=False,
                        output_csv="validation_results.csv", score_thresh_list=[0.4], iou_thresh_list=[0.1],
                        nms_method="nms", base_path="runs/reg"):
    t0 = time.time()
    results = []

    best_score = -1
    best_config = {}

    for score_thresh in score_thresh_list:
        for iou_thresh in iou_thresh_list:
            # for dataset_name, loader_, dataset_ in [("train", train_loader_, train_dataset),
            #                                         ("val", val_loader_, val_dataset)]:
            for dataset_name, loader_, dataset_ in [("val", val_loader_, val_dataset)]:
                print(f"Validating {dataset_name} -> score_thresh: {score_thresh}, iou_thresh: {iou_thresh}")
                result = validate_loader(
                    model_, DEVICE, loader_, dataset_,
                    score_thresh=score_thresh,
                    iou_thresh=iou_thresh,
                    file_base=dataset_name,
                    t0=t0,
                    use_softnms=use_softnms,
                    nms_method=nms_method,
                    is_finish=True,
                    base_path=base_path
                )

                metrics = result["metrics"]
                entry = {
                    "dataset": dataset_name,
                    "score_thresh": score_thresh,
                    "iou_thresh": iou_thresh,
                    "mAP": metrics.get("mAP", 0),
                    "AP@0.50": metrics.get("AP@0.50", 0),
                    "Precision": metrics.get("Precision", 0),
                    "Recall": metrics.get("Recall", 0),
                    "F1": metrics.get("F1", 0),
                    "MR": metrics.get("MR", 0)
                }
                score = compute_score(entry)
                entry["score"] = score

                results.append(entry)

                # 使用加权得分选最佳组合（只考虑 val）
                if dataset_name == "val" and score > best_score:
                    best_score = score
                    best_config = entry

    print(f"\n✅ 所有验证完成，结果保存在: {output_csv}")
    print(f"🔥 最佳阈值配置（基于 val 集 mAP）:")
    for k, v in best_config.items():
        print(f"{k}: {v}")

    # 保存CSV
    if results:
        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    return best_score, best_config


def parse_folder_flags(folder: str):
    # 默认值
    is_fpn = False
    use_transformer = False
    use_refine = False

    # 解析 fpX
    if "-fpTrue" in folder:
        is_fpn = True
    elif "-fpFalse" in folder:
        is_fpn = False

    # 解析 trX
    if "-trTrue" in folder:
        use_transformer = True
    elif "-trFalse" in folder:
        use_transformer = False

    # 解析 reX
    if "-reTrue" in folder:
        use_refine = True
    elif "-reFalse" in folder:
        use_refine = False

    return is_fpn, use_transformer, use_refine


if __name__ == '__main__':
    set_seed(2025, False)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = (0.4868, 0.5291, 0.3377)
    std = (0.2017, 0.2022, 0.1851)
    # mean = (0.3908, 0.4763, 0.3021)
    # std = (0.179, 0.1821, 0.1636)
    # mean, std = (0.3908, 0.4763, 0.3021), (0.179, 0.1821, 0.1636)

    backbone_type = "resnet50"
    use_softnms = False
    nms_method = "nms"
    img_size = 320
    dropout = 0.0

    # data_name = "A"
    data_name = "B"
    # data_name = "C"
    # data_name = "D"
    data_root = r"E:/resources/datasets/tea-buds-database/" + data_name

    # 初始化数据
    train_dataset = YoloDataset(
        img_dir=os.path.join(data_root, "train", 'images'),
        label_dir=os.path.join(data_root, "train", 'labels'),
        img_size=img_size,
        cls_num=1,
        mean=mean,
        std=std,
        normalize_label=True,
        mode='val',
    )
    train_loader_ = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn,
                               num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1, timeout=60)

    val_dataset = YoloDataset(
        img_dir=os.path.join(data_root, "val", 'images'),
        label_dir=os.path.join(data_root, "val", 'labels'),
        img_size=img_size,
        cls_num=1,
        mean=mean,
        std=std,
        mode='val',
        normalize_label=True
    )
    val_loader_ = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1, timeout=60)

    # 遍历 runs/ 目录
    runs_dir = "./runs/ablation"
    best_score = -1
    best_config = {}
    best_path = ""
    all_results = []  # 存储所有 config_

    reg_path = os.path.join(runs_dir, "reg")
    if os.path.exists(reg_path):
        for item in os.listdir(reg_path):
            item_path = os.path.join(reg_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(reg_path)
    print(f"{reg_path} 已清空")

    for folder in os.listdir(runs_dir):
        # score_thresh_list = [0.4, 0.5, 0.6, 0.7, 0.75]
        # iou_thresh_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        score_thresh_list = [0.75]
        iou_thresh_list = [0.2]

        is_fpn, use_transformer, use_refine = parse_folder_flags(folder)

        folder_path = os.path.join(runs_dir, folder)
        if not os.path.isdir(folder_path) or folder == "reg":
            continue

        ckpt_path = os.path.join(folder_path, "model_epoch_best.pth")
        if not os.path.isfile(ckpt_path):
            print(f"跳过 {folder}：未找到 model_epoch_best.pth")
            continue

        print(f"验证模型：{folder}")

        model_ = DynamicBoxDetector(
            in_dim=img_size, hidden_dim=128, nhead=4, num_layers=2, cls_num=1,
            once_embed=True, is_split_trans=False,
            dropout=dropout, backbone_type=backbone_type,
            is_fpn=is_fpn,
            use_transformer=use_transformer,
            use_refine=use_refine
        ).to(DEVICE)
        model_.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        output_csv = os.path.join(reg_path, f"validation_{folder}.csv")
        score, config_ = run_validation_grid(
            model_,
            DEVICE,
            train_loader_,
            train_dataset,
            val_loader_,
            val_dataset,
            use_softnms=use_softnms,
            output_csv=output_csv,
            score_thresh_list=score_thresh_list, iou_thresh_list=iou_thresh_list,
            nms_method=nms_method,
            base_path=runs_dir,
        )

        # === 新增：存储到列表 ===
        config_row = {"folder": folder}
        config_row.update(config_)
        all_results.append(config_row)

        if score > best_score:
            best_score = score
            best_config = config_
            best_path = folder_path

    # === 循环结束后，写入 result.csv ===
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(reg_path, "result.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n📂 所有结果已保存到 {csv_path}")

    print(f"\n✅ 所有验证完成，最好的: {best_path}")
    print(f"🔥 最佳阈值配置:")
    for k, v in best_config.items():
        print(f"{k}: {v}")
