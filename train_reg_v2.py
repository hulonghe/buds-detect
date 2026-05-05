"""
train_reg_v2.py - 工业级茶叶芽检测训练脚本

特性:
1. 多尺度训练 (Multi-Scale Training)
2. P2-P5 FPN (小目标友好)
3. OTA匹配 + 质量焦距损失
4. EMA指数移动平均
5. TTA推理增强
6. 多类别支持
7. 修复scheduler bug
"""

import os
import csv
import random
from datetime import datetime
import uuid
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from loss.DetectionLossV4 import DetectionLossV4
from nn.bbox.DynamicBoxDetectorV2 import DynamicBoxDetectorV2
from utils.YoloDataset import YoloDataset
from utils.helper import write_train_log, print_gpu_usage, custom_collate_fn, \
    get_train_transform, set_seed, print_metrics_table_row_style
from utils.ema import EMAWrapper
from utils.tta import TTA
from optim.OneCyclePlateauJumpLR import OneCyclePlateauJumpLR
from utils.plot import plot_detection_metrics
from torch.utils.tensorboard import SummaryWriter

seed = 2025


def schedule_easy_fraction(epoch, total_epochs,
                           start=0.3, end=1.0,
                           mode="linear",
                           warmup_epochs=0, start_epoch=0):
    import math
    if epoch < warmup_epochs:
        return end
    elif epoch < start_epoch:
        return start

    adj_epoch = epoch - warmup_epochs
    adj_total = max(1, total_epochs - warmup_epochs)

    if mode == "linear":
        frac = start + (end - start) * (adj_epoch / adj_total)
    elif mode == "cosine":
        frac = end - (end - start) * (math.cos(math.pi * adj_epoch / adj_total) * 0.5 + 0.5)
    elif mode == "step":
        step_size = adj_total // 3
        frac = start + (end - start) * (adj_epoch // step_size) / 3
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return max(min(frac, 1.0), 0.0)


def log_dict_as_table(writer, tag: str, data: dict, step=0):
    lines = ["| 参数 | 值 |", "|------|-----|"]
    for k, v in data.items():
        v = str(v).replace('\n', ' ').replace('|', '/')
        lines.append(f"| {k} | {v} |")
    markdown = "\n".join(lines)
    writer.add_text(tag, markdown, step)


def generate_filename_from_args(base_path="runs/reg", is_uid=False, **kwargs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]

    def short_key(k):
        return k[:2].lower()

    def safe_str(k, v):
        if v == '' or v is None:
            v = 'none'
        if isinstance(v, (list, tuple)):
            v = '_'.join(map(str, v))
        return str(v)

    parts = []
    for k, v in kwargs.items():
        if k not in ['base_path', 'is_uid', 'file_base']:
            sk = short_key(k)
            sv = safe_str(k, v)
            if len(sv) > 8:
                sv = sv[:8]
            parts.append(f"{sk}{sv}")

    if is_uid:
        return f"{base_path}/{uid}"
    return f"{base_path}/{timestamp}_{uid}"


def train_one_epoch_v2(model, criterion, ema, train_loader, optimizer, scaler, device,
                       epoch, epochs, warmup_epochs, multi_scale=False,
                       img_base_size=640, ms_range=(0.5, 1.5), scheduler=None,
                       gradient_accumulation=1):
    """
    训练一个epoch，支持梯度累积
    """
    model.train()
    epoch_loss = 0.0
    epoch_iou_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_box_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Train E{epoch + 1}/{epochs}")
    for batch_idx, (imgs, boxes, labels, aux_targets, img_paths) in enumerate(pbar):
        if imgs is None:
            continue

        imgs = imgs.to(device, non_blocking=True)
        boxes = [t.to(device, non_blocking=True) for t in boxes]
        labels = [t.to(device, non_blocking=True) for t in labels]

        with autocast(device_type=device, enabled=True):
            cls_scores, reg_preds, iou_preds, features = model(imgs)
            
            if batch_idx == 0:
                progress = min(epoch / max(epochs * 0.5, 1), 1.0)
                dynamic_iou = 0.05 + 0.25 * progress
                print(f"[Epoch {epoch+1}] loss=... dynamic_iou={dynamic_iou:.3f}")
            
            total_loss, loss_cls, loss_box, loss_iou = criterion(
                cls_scores, reg_preds, iou_preds, features,
                boxes, labels, aux_targets,
                epoch, epochs, warmup_epochs
            )
            
            if not torch.isfinite(total_loss):
                print(f"[WARN] Loss is not finite: {total_loss}, skipping batch")
                optimizer.zero_grad()
                continue
                
            total_loss = total_loss / gradient_accumulation
            
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        num_batches += 1
        epoch_loss += total_loss.detach().cpu().item() * gradient_accumulation
        epoch_iou_loss += loss_iou.cpu().item() * gradient_accumulation
        epoch_cls_loss += loss_cls.cpu().item() * gradient_accumulation
        epoch_box_loss += loss_box.cpu().item() * gradient_accumulation

        lr_cur = optimizer.param_groups[0]['lr']
        pbar.set_postfix(
            loss=f"{total_loss.item() * gradient_accumulation:.4f}",
            lr=f"{lr_cur:.5f}",
        )

    # FIXED: Scheduler step moved here (after epoch)
    if scheduler is not None and epoch >= warmup_epochs:
        scheduler.step()

    if num_batches > 0:
        epoch_loss /= num_batches
        epoch_iou_loss /= num_batches
        epoch_cls_loss /= num_batches
        epoch_box_loss /= num_batches

    return epoch_loss, epoch_iou_loss, epoch_cls_loss, epoch_box_loss


def train_v2(epochs=1000,
             train_loader=None,
             img_size=640,
             num_classes=1,
             device="cuda",
             lr=1e-4,
             warmup_epochs=1,
             val_dataset=None,
             val_loader=None,
             backbone_type="cspdarknet",
             iou_loss_type='ciou',
             cls_type='quality_focal',
             gamma=2.0,
             alpha=0.25,
             iou_thresh=0.3,
             score_thresh=0.001,
             weight_decay=0.0005,
             base_path="./runs",
             data_name=None,
              ckpt_path=None,
              m_name="default",
             dropout=0.1,

             # 新增参数
             multi_scale=True,
             ms_range=(0.5, 1.5),
             use_ema=True,
             ema_decay=0.9999,
             use_tta=False,
             fpn_out_channels=256,
             extra_p2=True,

             loss_version='v4',
             simota_topk=30,
             center_radius=2.5):
    t0 = time.time()

    # 模型参数 - 简化版
    model_kwargs = dict(
        img_size=img_size,
        num_classes=num_classes,
        backbone_type='cspdarknet',
        fpn_type='pafpn',
        use_dfl=False,
        hidden_dim=128,
        dropout=0.0,
        device=device,
        fpn_out_channels=128,
        extra_p2=False
    )

    train_params = {
        'epochs': epochs,
        'img_size': img_size,
        'num_classes': num_classes,
        'lr': lr,
        'warmup_epochs': warmup_epochs,
        'backbone_type': backbone_type,
        'iou_loss_type': iou_loss_type,
        'cls_type': cls_type,
        'iou_thresh': iou_thresh,
        'score_thresh': score_thresh,
        'gamma': gamma,
        'alpha': alpha,
        'device': device,
        'multi_scale': multi_scale,
        'ms_range': ms_range,
        'use_ema': use_ema,
        'ema_decay': ema_decay,
        'use_tta': use_tta,
        'loss_version': loss_version,
    }

    base_path = generate_filename_from_args(
        base_path=base_path,
        data_name=data_name,
        m_name=m_name,
        epochs=epochs,
        size=img_size,
        lr=lr,
        warmup=warmup_epochs,
        backbone=backbone_type,
        iou_loss=iou_loss_type,
        cls_type=cls_type,
        iou=iou_thresh,
        score=score_thresh,
        dropout=dropout,
        multi_scale=multi_scale,
        ema=use_ema,
        loss=loss_version
    )
    print(f"save path: {base_path}")
    os.makedirs(base_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{base_path}/tf-logs')
    log_dict_as_table(writer, 'Config/Train_Params', train_params)

    print(f"{'img_size':<24}: {img_size}")
    print(f"{'num_classes':<24}: {num_classes}")
    print(f"{'multi_scale':<24}: {multi_scale}")
    print(f"{'ms_range':<24}: {ms_range}")
    print(f"{'use_ema':<24}: {use_ema}")
    print(f"{'backbone_type':<24}: {backbone_type}")

    scaler = GradScaler()

    print(model_kwargs)
    log_dict_as_table(writer, 'Config/Model_Params', model_kwargs)
    model = DynamicBoxDetectorV2(**model_kwargs).to(device)

    # EMA
    ema = None
    if use_ema:
        ema = EMAWrapper(model, decay=ema_decay, device=device)
        print(f"EMA enabled with decay={ema_decay}")

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {ckpt_path}")

    # Loss
    criterion = DetectionLossV4(
        num_classes=num_classes,
        cls_weight=1.0,
        box_weight=2.5,
        # iou_weight=0.0,
        fg_iou_threshold=0.5,
        # bg_iou_threshold=0.5,
        device=device
    ).to(device)
    print(f"Loss Version: V4 (Simplified BCE + L1)")
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = OneCyclePlateauJumpLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
        warmup_epochs=warmup_epochs
    )

    # TTA (for inference)
    tta = None
    if use_tta:
        tta = TTA(scales=[0.75, 1.0, 1.25], flip=True)
        print("TTA enabled")

    best_fitness = 0.0
    best_epoch = 0

    for epoch in range(epochs):

        # 更新easy_fraction
        if hasattr(train_loader.dataset, 'update_selected_indices'):
            easy_frac = schedule_easy_fraction(epoch, epochs, start=1.0, end=1.0)
            train_loader.dataset.update_selected_indices(easy_frac)

        # 训练
        epoch_loss, epoch_iou_loss, epoch_cls_loss, epoch_box_loss = train_one_epoch_v2(
            model, criterion, ema, train_loader, optimizer, scaler, device,
            epoch, epochs, warmup_epochs,
            multi_scale=False, img_base_size=img_size, ms_range=ms_range,
            scheduler=scheduler, gradient_accumulation=1
        )

        writer.add_scalar('EpochTrain/LossAll', epoch_loss, epoch)
        writer.add_scalar('EpochTrain/LossCls', epoch_cls_loss, epoch)
        writer.add_scalar('EpochTrain/LossBox', epoch_box_loss, epoch)
        writer.add_scalar('EpochTrain/LossIou', epoch_iou_loss, epoch)
        writer.add_scalar('EpochTrain/Lr', optimizer.param_groups[0]['lr'], epoch)

        log = write_train_log(
            epoch, epochs,
            epoch_loss, epoch_iou_loss, epoch_cls_loss, epoch_box_loss,
            optimizer.param_groups[0]['lr'], log_file=f"{base_path}/train_{t0}.txt"
        )
        print(f"\n{log}")

        # 验证
        if val_loader is not None and (epoch + 1) % 5 == 0:
            from validate_reg_v2 import validate_loader_v2
            # 使用EMA模型验证
            val_model = model
            if ema is not None:
                ema.apply_shadow()
                val_model = ema

            val_results = validate_loader_v2(
                val_model, device, val_loader, val_dataset,
                t0=t0, epoch=epoch, epochs=epochs,
                score_thresh=score_thresh, iou_thresh=iou_thresh,
                file_base='val', base_path=base_path,
                warmup_epochs=warmup_epochs,
                num_classes=num_classes
            )
            val_map = val_results['metrics'].get('mAP', 0.0)
            val_loss = 0.0

            # 恢复原始模型
            if ema is not None:
                ema.restore()

            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/mAP', val_map, epoch)

            # 保存最佳模型
            fitness = val_map
            if fitness > best_fitness:
                best_fitness = fitness
                best_epoch = epoch
                save_path = f"{base_path}/best.pt"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'fitness': fitness,
                }, save_path)
                print(f"Saved best model: {save_path} (mAP: {fitness:.4f})")

        # 定期保存
        if (epoch + 1) % 50 == 0:
            save_path = f"{base_path}/epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)

    # 保存最终模型
    final_path = f"{base_path}/last.pt"
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    print(f"Training completed! Best mAP: {best_fitness:.4f} at epoch {best_epoch + 1}")
    print(f"Final model saved to: {final_path}")

    writer.close()
    return model


def build_dataloader_v2(img_dir, label_dir, img_size=640, batch_size=16,
                        num_workers=8, mode='train', cls_num=1,
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                        mosaic_prob=0.0, easy_fraction=1.0):
    """构建数据加载器"""
    dataset = YoloDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        normalize_label=True,
        transforms=get_train_transform(img_size, mean=mean, std=std, val=False) if mode == 'train' else None,
        cls_num=cls_num,
        mean=mean,
        std=std,
        mode=mode,
        max_memory_usage=0.7,
        mosaic_prob=mosaic_prob,
        easy_fraction=easy_fraction
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=(mode == 'train'),
        persistent_workers=True if num_workers > 0 else False
    )

    return dataset, loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r"E:/resources/datasets/tea-buds-database/C")
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='cspdarknet')
    parser.add_argument('--multi_scale', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_ema', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--use_tta', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    set_seed(seed)

    # 构建数据加载器
    train_dataset, train_loader = build_dataloader_v2(
        img_dir=os.path.join(args.data_root, 'train', 'images'),
        label_dir=os.path.join(args.data_root, 'train', 'labels'),
        img_size=args.img_size,
        batch_size=args.batch_size,
        mode='train',
        cls_num=args.num_classes,
        mosaic_prob=0.0
    )

    val_dataset, val_loader = build_dataloader_v2(
        img_dir=os.path.join(args.data_root, 'val', 'images'),
        label_dir=os.path.join(args.data_root, 'val', 'labels'),
        img_size=args.img_size,
        batch_size=args.batch_size * 2,
        num_workers=8,
        mode='val',
        cls_num=args.num_classes
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    # 训练
    model = train_v2(
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        img_size=args.img_size,
        num_classes=args.num_classes,
        device=args.device,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        backbone_type=args.backbone,
        multi_scale=args.multi_scale,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        use_tta=args.use_tta,
        data_name=os.path.basename(args.data_root)
    )
