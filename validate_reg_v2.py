"""
validate_reg_v2.py - 验证脚本
使用cx,cy,w,h格式
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torchvision.ops import nms
from tqdm import tqdm

from nn.bbox.DynamicBoxDetectorV2 import DynamicBoxDetectorV2
from utils.YoloDataset import YoloDataset
from utils.helper import custom_collate_fn, write_val_log, print_metrics_table_row_style
from utils.metrics import compute_ap_metrics


def xywh2xyxy(boxes):
    """cx,cy,w,h -> x1,y1,x2,y2"""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def multiclass_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=0.25):
    if len(boxes) == 0:
        return boxes, scores, labels
    
    final_boxes = []
    final_scores = []
    final_labels = []
    
    for cls_id in torch.unique(labels):
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        keep_mask = cls_scores > score_thresh
        if keep_mask.sum() == 0:
            continue
        
        cls_boxes = cls_boxes[keep_mask]
        cls_scores = cls_scores[keep_mask]
        
        keep = nms(cls_boxes, cls_scores, iou_thresh)
        
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_labels.append(torch.full((len(keep),), cls_id, dtype=torch.long, device=boxes.device))
    
    if len(final_boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,))
    
    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_labels = torch.cat(final_labels, dim=0)
    
    return final_boxes, final_scores, final_labels


def save_predictions_to_txt(batch_pred_boxes, batch_pred_scores, batch_pred_labels, img_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for b, img_name in enumerate(img_paths):
        boxes = batch_pred_boxes[b]
        scores = batch_pred_scores[b]
        labels = batch_pred_labels[b]

        lines = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(scores[i])
            cls_id = int(labels[i])

            if conf <= 0:
                continue

            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}"
            lines.append(line)

        txt_path = os.path.join(save_dir, img_name + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))


@torch.no_grad()
def validate_loader_v2(model, device, data_loader, dataset,
                        t0=None, epoch=0, epochs=0,
                        score_thresh=0.25, iou_thresh=0.25,
                        file_base='val', base_path='runs/reg',
                        num_classes=1, use_tta=False,
                        warmup_epochs=5, weights_=None, is_finish=False):
    model.eval()
    all_gts = []
    all_predictions = []
    all_pred_details = []

    if t0 is None:
        t0 = time.time()
        log_file = f"{base_path}/{file_base}_{t0}.txt"
    else:
        log_file = f"{base_path}/{file_base}_{t0}.txt"

    pbar = tqdm(data_loader, desc=f"Val | score={score_thresh} iou={iou_thresh}")
    for batch_idx, (imgs, boxes, labels, aux_targets, img_paths) in enumerate(pbar):
        if imgs is None:
            continue

        imgs = imgs.to(device)
        boxes = [t.to(device) for t in boxes]
        labels = [t.to(device) for t in labels]
        batch_size = imgs.size(0)

        with autocast(device_type=device, enabled=True):
            cls_scores, reg_preds, iou_preds, _ = model(imgs)
        
        # 解码预测框
        decoded_boxes = model.decode_boxes(reg_preds, model.img_size)
        
        # 转换为xyxy用于NMS
        boxes_xyxy = xywh2xyxy(decoded_boxes)
        
        # 质量分数
        cls_scores_sigmoid = cls_scores.sigmoid()
        iou_scores_sigmoid = iou_preds.sigmoid()
        quality_scores = cls_scores_sigmoid * iou_scores_sigmoid
        
        # Debug
        if batch_idx == 0:
            max_score = quality_scores.max().item()
            mean_score = quality_scores.mean().item()
            print(f"[Val DEBUG] max_score={max_score:.4f}, mean_score={mean_score:.4f}")

        batch_pred_boxes = []
        batch_pred_scores = []
        batch_pred_labels = []

        for b in range(batch_size):
            boxes_per_img = boxes_xyxy[b]
            scores_per_img = quality_scores[b]
            
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for cls_id in range(num_classes):
                cls_scores_single = scores_per_img[:, cls_id]
                keep_mask = cls_scores_single > score_thresh
                
                if keep_mask.sum() == 0:
                    continue
                
                cls_boxes = boxes_per_img[keep_mask]
                cls_scores_filtered = cls_scores_single[keep_mask]
                
                if len(cls_boxes) == 0:
                    continue
                
                keep = nms(cls_boxes, cls_scores_filtered, iou_thresh)
                
                all_boxes.append(cls_boxes[keep])
                all_scores.append(cls_scores_filtered[keep])
                all_labels.append(torch.full((len(keep),), cls_id, dtype=torch.long, device=device))
            
            if len(all_boxes) > 0:
                final_boxes = torch.cat(all_boxes, dim=0)
                final_scores = torch.cat(all_scores, dim=0)
                final_labels = torch.cat(all_labels, dim=0)
            else:
                final_boxes = torch.zeros((0, 4), device=device)
                final_scores = torch.zeros((0,), device=device)
                final_labels = torch.zeros((0,), dtype=torch.long, device=device)
            
            batch_pred_boxes.append(final_boxes.cpu())
            batch_pred_scores.append(final_scores.cpu())
            batch_pred_labels.append(final_labels.cpu())

        save_predictions_to_txt(batch_pred_boxes, batch_pred_scores, batch_pred_labels, img_paths,
                                os.path.join(base_path, "labels"))

        for i in range(batch_size):
            gt_boxes = boxes[i].cpu().numpy()
            pred_boxes = batch_pred_boxes[i].numpy()
            pred_scores = batch_pred_scores[i].numpy()
            pred_labels = batch_pred_labels[i].numpy()

            all_predictions.append((pred_boxes, pred_scores, pred_labels))
            all_gts.append(gt_boxes)

            all_pred_details.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            })

    metrics = compute_ap_metrics_v2(
        all_predictions,
        all_gts,
        num_classes=num_classes,
        iou_thresholds=None,
        plot_pr=is_finish,
        save_path=os.path.join(base_path, "PR.png"),
    )
    
    print_metrics_table_row_style("Val:", metrics)

    val_log = write_val_log(epoch, epochs, metrics, log_file=log_file)

    return {
        "metrics": metrics,
        "predictions": all_pred_details,
        "gt_boxes": all_gts,
        "score_thresh": score_thresh,
        "iou_thresh": iou_thresh,
    }


def compute_ap_metrics_v2(predictions, gts, num_classes=1, iou_thresholds=None, plot_pr=False, save_path="pr.png"):
    if iou_thresholds is None:
        iou_thresholds = (0.5, 0.95, 0.05)
    
    all_metrics = {}
    
    if num_classes == 1:
        pred_boxes_list = [p[0] for p in predictions]
        pred_scores_list = [p[1] for p in predictions]
        metrics = compute_ap_metrics(
            list(zip(pred_boxes_list, pred_scores_list)),
            gts,
            iou_thresholds=iou_thresholds,
            plot_pr=plot_pr,
            save_path=save_path,
            model_name="Ours"
        )
        all_metrics.update(metrics)
    else:
        per_class_ap = []
        for cls_id in range(num_classes):
            cls_preds = []
            cls_gts = []
            for pred, gt in zip(predictions, gts):
                pred_boxes, pred_scores, pred_labels = pred
                
                cls_mask = pred_labels == cls_id
                cls_pred_boxes = pred_boxes[cls_mask]
                cls_pred_scores = pred_scores[cls_mask]
                
                cls_preds.append((cls_pred_boxes, cls_pred_scores))
                cls_gts.append(gt)
            
            if len(cls_gts) > 0:
                metrics = compute_ap_metrics(
                    cls_preds, cls_gts,
                    iou_thresholds=iou_thresholds,
                    plot_pr=False,
                    save_path=None
                )
                cls_ap = metrics.get('mAP', 0.0)
                per_class_ap.append(cls_ap)
                all_metrics[f'AP_{cls_id}'] = cls_ap
        
        if len(per_class_ap) > 0:
            all_metrics['mAP'] = np.mean(per_class_ap)
            all_metrics['mAP50'] = np.mean([ap for ap in per_class_ap if not np.isnan(ap)])
    
    return all_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r"E:/resources/datasets/tea-buds-database/A")
    parser.add_argument('--img_size', type=int, default=320)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--score_thresh', type=float, default=0.001)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    val_dataset = YoloDataset(
        img_dir=os.path.join(args.data_root, 'val', 'images'),
        label_dir=os.path.join(args.data_root, 'val', 'labels'),
        img_size=args.img_size,
        normalize_label=False,
        cls_num=args.num_classes,
        mode='val',
        max_memory_usage=0.0,
        mosaic_prob=0.0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    model = DynamicBoxDetectorV2(
        img_size=args.img_size,
        num_classes=args.num_classes,
        backbone_type='cspdarknet',
        fpn_type='pafpn',
        use_dfl=False
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Val dataset: {len(val_dataset)} images")

    results = validate_loader_v2(
        model, args.device, val_loader, val_dataset,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        num_classes=args.num_classes,
        base_path='runs/val_v2'
    )

    print(f"Results: {results['metrics']}")
