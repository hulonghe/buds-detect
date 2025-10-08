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
from loss.DetectionBoxLoss import DetectionLoss
from nn.bbox.EnhancedDynamicBoxDetector_ablation import DynamicBoxDetector
from utils.YoloDataset import YoloDataset
from utils.helper import write_train_log, print_gpu_usage, custom_collate_fn, \
    clear_gpu_cache, get_train_transform, set_seed
from validate_reg import validate_loader
from torch.amp import autocast, GradScaler
from optim.OneCyclePlateauJumpLR import OneCyclePlateauJumpLR
from utils.plot import plot_detection_metrics
from torch.utils.tensorboard import SummaryWriter
import itertools

seed = 2025


# torch.set_num_threads(4)  # 设物理核数的一半
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def schedule_easy_fraction(epoch, total_epochs,
                           start=0.3, end=1.0,
                           mode="linear",
                           warmup_epochs=0, start_epoch=0):
    """
    课程学习式样本比例调度函数，支持 warmup 阶段

    Args:
        epoch (int): 当前 epoch (从 0 开始计数)
        total_epochs (int): 总训练轮数
        start (float): 初始样本比例 (0~1)
        end (float): 最终样本比例 (0~1)
        mode (str): 调度方式 ["linear", "step", "cosine"]
        warmup_epochs (int): 前多少个 epoch 固定用 end (不做调度)
        start_epoch (int): 前多少个 epoch 固定用 start (不做调度)

    Returns:
        float: 当前 epoch 应该使用的样本比例 (easy_fraction)
    """
    import math

    # warmup 阶段：直接使用 end（全部样本）
    if epoch < warmup_epochs:
        return end
    elif epoch < start_epoch:
        return start

    # 调整 epoch 和总轮数，剔除 warmup 部分
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
    """将字典以Markdown表格形式写入TensorBoard"""
    lines = ["| 参数 | 值 |", "|------|-----|"]
    for k, v in data.items():
        v = str(v).replace('\n', ' ').replace('|', '/')  # 防止表格混乱
        lines.append(f"| {k} | {v} |")
    markdown = "\n".join(lines)
    writer.add_text(tag, markdown, step)


def generate_filename_from_args(base_path="runs/reg", is_uid=False, **kwargs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]

    # 函数：将键名缩短为前两个字母
    def short_key(k):
        return k[:2].lower()  # 取前两个字母并小写

    def safe_str(k, v):
        v = 'none' if v == '' else v
        return f"{short_key(k)}{str(v).replace('.', '_')}"  # 避免小数点冲突文件名

    # 将所有键值对处理成字符串
    parts = [safe_str(k, v) for k, v in kwargs.items()]

    # 根据是否需要 UID 生成文件名
    if is_uid:
        name = "-".join(parts) + f"-{timestamp}_{uid}"
    else:
        name = "-".join(parts)

    return f"{base_path}/{name}"


def compute_score(entry, weights=None):
    # 默认权重设置
    if weights is None:
        weights = {
            "mAP": 0.3,
            "AP@0.50": 0.6,
            "Precision": 0.1,
        }

    score = 0.0
    for k, w in weights.items():
        score += entry.get(k, 0) * w
    return score


def is_better_metrics(metrics: dict, best_metrics: dict, primary_key='AP@0.50', secondary_key='F1', delta=1e-4) -> bool:
    """
    判断当前 metrics 是否比 best_metrics 更优，是否保存模型。

    参数：
        metrics       当前轮次的指标（dict）
        best_metrics  历史最佳指标（dict）
        primary_key   主评估指标（默认用 AP@0.50）
        secondary_key 次指标（辅助判别）
        delta         允许的最小提升阈值（防止浮动误差）

    返回：
        True  当前更优，建议保存
        False 不更优，不保存
    """
    if best_metrics is None:
        return True  # 没有历史记录，肯定保存

    current_primary = metrics.get(primary_key, 0)
    best_primary = best_metrics.get(primary_key, 0)

    if current_primary > best_primary + delta:
        return True  # 明显更好
    elif abs(current_primary - best_primary) < delta:
        # 主指标相近，再比较次指标
        current_secondary = metrics.get(secondary_key, 0)
        best_secondary = best_metrics.get(secondary_key, 0)
        if current_secondary > best_secondary + delta:
            return True
    return False


def save_metrics_to_csv(metrics_list, filename="metrics.csv"):
    if not metrics_list:
        print("⚠️ 空的 metrics_list，无法保存。")
        return

    # 自动收集所有字段名，防止不同 row 字段不一致导致写入报错
    all_fieldnames = sorted(set().union(*[m.keys() for m in metrics_list]))

    # 自动创建文件夹（若路径中带文件夹）
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    try:
        with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            for row in metrics_list:
                # 自动补全缺失字段为空串或 0（根据需要修改）
                full_row = {key: row.get(key, "") for key in all_fieldnames}
                writer.writerow(full_row)

        print(f"✅ 已保存 {len(metrics_list)} 条指标记录到 {filename}")

    except Exception as e:
        print(f"❌ 写入 CSV 出错：{e}")


def detect_plateau(metrics_history, key='mAP', window=5, delta=0.005):
    """
    检测指标震荡高原：最近 window 个周期内无明显提升
    """
    if len(metrics_history) < window + 1:
        return False
    values = [m[key] for m in metrics_history[-(window + 1):]]
    baseline = values[0]
    max_later = max(values[1:])
    return (max_later - baseline) < delta


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_model_components(model, epoch, epochs, warmup_epochs):
    """
    针对无预训练 transformer 的冻结策略。
    """
    if epoch > warmup_epochs:
        freeze_backbone = True
    else:
        freeze_backbone = False

    for name, module in model.backbone.items():
        if name in ['conv1', 'layer1']:
            for p in module.parameters():
                p.requires_grad = freeze_backbone


def train_one(model, device, epoch, epochs, warmup_epochs,
              train_loader, criterion, optimizer,
              scheduler, scheduler2, t0, scaler=None,
              base_path='runs/reg', writer=None, weights_=None, freeze_model=False):
    model.train()

    # 冻结策略
    if freeze_model:
        freeze_model_components(model, epoch, epochs, warmup_epochs)

    epoch_loss = 0.0
    epoch_iou_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_box_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
    for batch_idx, (imgs, boxes, labels, aux_targets, _) in enumerate(pbar):
        if imgs is None:
            continue
        imgs = imgs.to(device)
        boxes = [t.to(device) for t in boxes]
        labels = [t.to(device) for t in labels]
        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type=device, enabled=True):
            pred_scores, pred_boxes, aux_logits = model(imgs)
            total_loss, loss_cls, loss_box, loss_iou = criterion(pred_scores, pred_boxes, aux_logits,
                                                                 boxes, labels, aux_targets,
                                                                 epoch, epochs, warmup_epochs, weights_)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()

        epoch_loss += total_loss.detach().cpu().item()
        epoch_iou_loss += loss_iou.cpu().item()
        epoch_cls_loss += loss_cls.cpu().item()
        epoch_box_loss += loss_box.cpu().item()

        lr_cur = optimizer.param_groups[0]['lr']
        gpu_info = print_gpu_usage()
        pbar.set_postfix(
            loss=f"{total_loss.item():.6f}",
            # iou=f"{loss_iou.item():.6f}",
            # cls=f"{loss_cls.item():.6f}",
            # box=f"{loss_box.item():.6f}",
            lr=f"{lr_cur:.6f}",
            gpu=gpu_info,
        )

        writer.add_scalar('BatchTrain/LossAll', total_loss.item())
        writer.add_scalar('BatchTrain/LossIou', loss_iou.item())
        writer.add_scalar('BatchTrain/LossCls', loss_cls.item())
        writer.add_scalar('BatchTrain/LossBox', loss_box.item())
        writer.add_scalar('BatchTrain/Lr', lr_cur)

    if scheduler is None:
        scheduler2.step()

    epoch_loss /= len(train_loader)
    epoch_iou_loss /= len(train_loader)
    epoch_cls_loss /= len(train_loader)
    epoch_box_loss /= len(train_loader)

    last_iou_loss = epoch_iou_loss
    last_box_loss = epoch_box_loss
    last_cls_loss = epoch_cls_loss

    writer.add_scalar('EpochTrain/LossAll', epoch_loss, epoch)
    writer.add_scalar('EpochTrain/LossCls', epoch_cls_loss, epoch)
    writer.add_scalar('EpochTrain/LossBox', epoch_box_loss, epoch)
    writer.add_scalar('EpochTrain/LossIou', epoch_box_loss, epoch)

    log = write_train_log(
        epoch, epochs,
        epoch_loss, epoch_iou_loss, epoch_cls_loss, epoch_box_loss,
        optimizer.param_groups[0]['lr'], log_file=f"{base_path}/train_{t0}.txt"
    )
    print(f"\n{log}")

    return last_cls_loss, last_box_loss, last_iou_loss


def trains(epochs=1000, train_loader=None, img_size=640, device="cuda",
           lr=1e-4, warmup_epochs=5, val_dataset=None, val_loader=None, backbone_type="resnet34",
           iou_loss_type='ciou',  # ciou,diou,eiou,giou
           box_loss_type='',  # balanced,gwd,'l1'
           cls_type="focal",  # focal,vari,bce
           gamma=1.0, alpha=0.25, iou_thresh=0.5, score_thresh=0.25,
           freeze_model=False, weight_decay=0.0005, base_path="/root/autodl-tmp/runs", data_name=None,
           ckpt_path=None, weights_=None, m_name="default", dropout=0.1,
           use_transformer=True, use_refine=True, is_fpn=True):
    t0 = time.time()
    use_softnms = False
    sum_weighted = True

    train_params = {
        'epochs': epochs,
        'img_size': img_size,
        'lr': lr,
        'warmup_epochs': warmup_epochs,
        'backbone_type': backbone_type,
        'iou_loss_type': iou_loss_type,
        'box_loss_type': box_loss_type or 'none',
        'cls_type': cls_type,
        'iou_thresh': iou_thresh,
        'score_thresh': score_thresh,
        'gamma': gamma,
        'alpha': alpha,
        'sum_weighted': sum_weighted,
        'device': device,
        'use_softnms': use_softnms
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
        box_loss=box_loss_type,
        cls_type=cls_type,
        iou=iou_thresh,
        score=score_thresh,
        gamma=gamma,
        alpha=alpha,
        dropout=dropout,
        fpn=is_fpn,
        transformer=use_transformer,
        refine=use_refine,
    )
    print(f"save path: {base_path}")
    os.makedirs(base_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'./runs/tf-logs')
    log_dict_as_table(writer, 'Config/Train_Params', train_params)

    print(f"{'iou_loss_type':<24}: {iou_loss_type}")
    print(f"{'box_loss_type':<24}: {box_loss_type}")
    print(f"{'iou_thresh':<24}: {iou_thresh}")
    print(f"{'score_thresh':<24}: {score_thresh}")
    print(f"{'sum_weighted':<24}: {sum_weighted}")
    print(f"{'weighted':<24}: {weights_}")
    print(f"{'cls_type':<24}: {cls_type}")
    print(f"{'gamma':<24}: {gamma}")
    print(f"{'alpha':<24}: {alpha}")

    scaler = GradScaler()  # 创建一个梯度缩放器
    model_kwargs = dict(
        in_dim=img_size, hidden_dim=128, nhead=4, num_layers=2, cls_num=1,
        once_embed=True, is_split_trans=False, is_fpn=is_fpn,
        dropout=dropout, backbone_type=backbone_type, device=device,
        use_transformer=use_transformer, use_refine=use_refine
    )
    print(model_kwargs)
    log_dict_as_table(writer, 'Config/Model_Params', model_kwargs)
    model = DynamicBoxDetector(**model_kwargs).to(device)
    if ckpt_path is not None:
        # 加载模型
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = DetectionLoss(iou_thresh=iou_thresh, score_thresh=score_thresh,
                              iou_type=iou_loss_type, box_type=box_loss_type, cls_type=cls_type,
                              device_type=device, sum_weighted=sum_weighted,
                              gamma=gamma, alpha=alpha).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_dataset) // batch_size_

    scheduler2 = None
    scheduler_kwargs = dict(
        max_lr=lr, total_steps=epochs * steps_per_epoch,
        pct_start=0.1, div_factor=10, final_div_factor=100, plateau_up_count=0,
        plateau_up_steps=steps_per_epoch * 2, plateau_up_strategy='arithmetic',
        plateau_down_count=0, plateau_down_steps=steps_per_epoch * 2, plateau_down_strategy='exponential',
        num_jumps=0, jump_magnitude=1.5, jump_once_steps=steps_per_epoch,
        anneal_strategy='cos', num_cycles=1, cycle_decay=1,
    )
    print(scheduler_kwargs)
    scheduler = OneCyclePlateauJumpLR(optimizer, **scheduler_kwargs)
    log_dict_as_table(writer, 'Config/Scheduler_Params', scheduler_kwargs)

    list_cls_loss = []
    list_box_loss = []
    list_iou_loss = []

    metrics_history = []
    best_metrics = None
    best_score = -1
    best_ap_score = -1
    best_ap_metrics = None
    easy_fraction_start = train_dataset.easy_fraction
    for epoch in range(epochs):
        if train_dataset.easy_fraction < 1.0:
            n_easy = schedule_easy_fraction(epoch, epochs, start=easy_fraction_start, mode="cosine",
                                            warmup_epochs=warmup_epochs, start_epoch=25)
            train_dataset.update_selected_indices(n_easy)

        now_lr = lr
        for param_group in optimizer.param_groups:
            now_lr = param_group['lr']
            break

        last_cls_loss, last_box_loss, last_iou_loss = train_one(model, device,
                                                                epoch, epochs, warmup_epochs, train_loader,
                                                                criterion, optimizer, scheduler, scheduler2, t0,
                                                                scaler, base_path=base_path, writer=writer,
                                                                weights_=weights_, freeze_model=freeze_model)
        list_cls_loss.append(last_cls_loss)
        list_box_loss.append(last_box_loss)
        list_iou_loss.append(last_iou_loss)

        # 前期不做验证
        if epoch >= warmup_epochs:
            results = validate_loader(model, device, val_loader, val_dataset, t0, epoch, epochs,
                                      score_thresh=0.4, iou_thresh=0.15,
                                      file_base='train', use_softnms=use_softnms, base_path=base_path,
                                      criterion=criterion, warmup_epochs=warmup_epochs, weights_=weights_)
            metrics = results["metrics"]
            metrics['lr'] = now_lr
            metrics['epoch'] = epoch
            metrics['cls_loss'] = results["val_cls_loss"]
            metrics['box_loss'] = results["val_box_loss"]
            metrics['iou_loss'] = results["val_iou_loss"]
            metrics['loss'] = results["val_loss"]
            metrics_history.append(metrics)

            for name, value in metrics.items():
                writer.add_scalar(f'Val/{name}', value, epoch)

            log = write_train_log(
                epoch, epochs,
                results["val_loss"], results["val_iou_loss"],
                results["val_cls_loss"], results["val_box_loss"],
                optimizer.param_groups[0]['lr'],
                log_file=f"{base_path}/train_{t0}.txt",
                mode="Val"
            )
            print(f"\n{log}")

            score = results["val_loss"]
            ap_score = compute_score(metrics)
            if best_metrics is None:
                best_metrics = metrics
                best_score = score
            elif best_metrics is not None and score < best_score:
                best_score = score
                best_metrics = metrics
                if epoch > 10:
                    torch.save(model.state_dict(), f'{base_path}/model_epoch_loss_best.pth')

            if best_ap_metrics is None:
                best_ap_metrics = metrics
                best_ap_score = ap_score
            elif best_ap_metrics is not None and ap_score > best_ap_score:
                best_ap_score = ap_score
                best_ap_metrics = metrics
                if epoch > 10:
                    torch.save(model.state_dict(), f'{base_path}/model_epoch_best.pth')

            print(
                f"best ap: {best_ap_metrics['AP@0.50']} epoch: {best_ap_metrics['epoch']} mAP: {best_ap_metrics['mAP']}")
            print(f"best low loss: {best_metrics['AP@0.50']} epoch: {best_metrics['epoch']} mAP: {best_metrics['mAP']}")

        clear_gpu_cache()

    # 打印日志
    print('best loss:')
    print("\t".join(best_metrics.keys()))
    print("\t".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in best_metrics.values()]))
    print('best ap loss:')
    print("\t".join(best_ap_metrics.keys()))
    print("\t".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in best_ap_metrics.values()]))

    torch.save(model.state_dict(), f'{base_path}/model_epoch_last.pth')
    save_metrics_to_csv(metrics_history, f"{base_path}/val_metrics_log.csv")
    plot_detection_metrics(metrics_history, save_dir=f"{base_path}", start_epoch=warmup_epochs + 1)

    return best_metrics


def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    set_seed(seed, False, True)
    base_path = "./runs"
    data_root_base = r"E:/resources/datasets/tea-buds-database/"
    
    iou_loss_types = ['ciou']
    box_loss_types = ['sosa']
    cls_loss_types = ['focal']

    freeze_model = False
    size = 320
    epochs_ = 100
    warmup_epochs_ = 1
    batch_size_ = 32
    weight_decay = 0.0001
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = None
    weights_ = [10.0, 1.2, 5.0]  #- cls,box,iou
    m_name = "default"
    dropout = 0.01
    lrs_ = [0.001]
    use_transformers = [True]
    use_refines = [True]
    is_fpns = [True]


    backbone_types = ["resnet101"]
    data_names = [
        # ['A', (0.4206, 0.502, 0.3179), (0.2162, 0.2199, 0.1967)],
        ['A-old', (0.4192, 0.5019, 0.3103), (0.2146, 0.2186, 0.1904)],
        # ['B', (0.4868, 0.5291, 0.3377), (0.2017, 0.2022, 0.1851)],
        # ['C', (0.3908, 0.4763, 0.3021), (0.179, 0.1821, 0.1636)],
        # ['D', (0.4553, 0.5044, 0.3957), (0.2096, 0.2159, 0.1845)]
    ]
    gammas = [1.0]
    alphas = [0.5]
    scores_thresh = [0.01]
    ious_thresh = [0.15]

    # 遍历所有组合
    best_result = None
    best_config = None
    for (backbone_type, score_thresh, iou_thresh, lr_, iou_loss_type, box_loss_type, cls_type,
         gamma, alpha, (data_name, mean, std),
         use_transformer, use_refine, is_fpn) in itertools.product(
        backbone_types, scores_thresh, ious_thresh, lrs_,
        iou_loss_types, box_loss_types, cls_loss_types, gammas, alphas, data_names,
        use_transformers, use_refines, is_fpns):
        print(
            f"\n训练组合: backbone={backbone_type}, "
            f"iou={iou_loss_type}, box={box_loss_type}, cls={cls_type},"
            f" gamma={gamma}, alpha={alpha}, lr={lr_}, data_name={data_name}")
        data_root = os.path.join(data_root_base, data_name)

        train_dataset = YoloDataset(
            img_dir=os.path.join(data_root + r"/train", 'images'),
            label_dir=os.path.join(data_root + r"/train", 'labels'),
            img_size=size, normalize_label=True, cls_num=1,
            mean=mean, std=std, mode="train",
            mosaic_prob=0.3,
            # transforms=get_train_transform(size, mean=mean, std=std, val=False),
            easy_fraction=1.0
        )
        train_loader_ = DataLoader(train_dataset, batch_size=batch_size_,
                                   shuffle=True, num_workers=4,
                                   collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True,
                                   prefetch_factor=1, worker_init_fn=worker_init_fn)

        val_dataset_ = YoloDataset(
            img_dir=os.path.join(data_root + r"/val", 'images'),
            label_dir=os.path.join(data_root + r"/val", 'labels'),
            img_size=size, cls_num=1,
            mean=mean, std=std, normalize_label=True, mode="val",
            mosaic_prob=0.0,
            # transforms=get_train_transform(size, mean=mean, std=std, val=True),
        )
        val_loader_ = DataLoader(val_dataset_, batch_size=batch_size_, shuffle=False,
                                 collate_fn=custom_collate_fn, num_workers=4,
                                 pin_memory=True, persistent_workers=True,
                                 prefetch_factor=1)

        # 调用训练函数，返回验证指标（如 mAP）
        result = trains(
            epochs_, train_loader_, size, device_,
            lr=lr_,
            warmup_epochs=warmup_epochs_,
            backbone_type=backbone_type,
            val_dataset=val_dataset_, val_loader=val_loader_,
            iou_thresh=iou_thresh, score_thresh=score_thresh,
            # 循环参数
            iou_loss_type=iou_loss_type, box_loss_type=box_loss_type, cls_type=cls_type,
            gamma=gamma, alpha=alpha,
            freeze_model=freeze_model, weight_decay=weight_decay,
            base_path=base_path,
            data_name=data_name,
            ckpt_path=ckpt_path,
            weights_=weights_,
            m_name=m_name,
            dropout=dropout,
            use_transformer=use_transformer, use_refine=use_refine, is_fpn=is_fpn
        )

        val_metric = result.get("AP@0.50", 0)
        if best_result is None or val_metric > best_result:
            best_result = val_metric
            best_config = {
                "backbone_type": backbone_type,
                "iou_loss_type": iou_loss_type,
                "box_loss_type": box_loss_type,
                "cls_type": cls_type,
                "gamma": gamma,
                "alpha": alpha
            }

    print("\n✅ 最佳组合:")
    print(best_config)
    print("验证指标(AP@0.50):", best_result)
