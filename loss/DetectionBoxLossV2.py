import os
from torch.amp import autocast, GradScaler
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from loss.SigmoidFocalLoss import SigmoidFocalLoss
from nn.bbox.EnhancedDynamicBoxDetector import DynamicBoxDetector
from utils.YoloDataset import YoloDataset
from utils.bbox import dynamic_postprocess
from utils.ciou import complete_ciou
from utils.draw import draw_boxes
from utils.gwd import gwd_loss
from utils.helper import clear_gpu_cache, set_seed
import cv2


class DetectionLoss(nn.Module):

    def __init__(self, score_thresh=0.25, iou_thresh=0.25,
                 iou_type="ciou", box_type="gwd",
                 device_type="cpu"):
        super().__init__()
        self.cls_loss_fn = SigmoidFocalLoss(gamma=1.0, alpha=0.8, reduction='mean', device_type=device_type)
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.iou_type = iou_type
        self.box_type = box_type
        self.device_type = device_type

    def forward(self, pred_scores, pred_boxes, target_boxes, target_labels, epoch, epochs, warmup_epoch,
            weights=(1.0, 1.0, 1.0)):
        device = pred_scores.device
        B, N, C = pred_scores.shape

        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_iou = torch.tensor(0.0, device=device)

        for b in range(B):
            pred_logits = pred_scores[b]      # [N, C]，每个框的分类概率（sigmoid 或 softmax 后）
            pred_box = pred_boxes[b]          # [N, 4]，预测框坐标
            gt_boxes = target_boxes[b].to(device)      # [M, 4]
            gt_labels = target_labels[b].to(device)    # [M]

            num_gt = gt_boxes.size(0)
            if num_gt == 0:
                # 没有GT，只计算背景分类损失
                loss_cls += self.cls_loss_fn(pred_logits, None)
                continue

            # ======================== 正负样本匹配 =========================
            pos_mask, matched_gt_idx = self.dynamic_assign(
                pred_box, pred_logits, gt_boxes, gt_labels,
                epoch=epoch, warmup_epoch=warmup_epoch
            )

            if pos_mask.sum() == 0:
                loss_cls += self.cls_loss_fn(pred_logits, targets=None, no_positive=True)
                continue

            # ======================== 类别分布动态权重 =====================
            with torch.no_grad():
                class_counts = torch.bincount(gt_labels, minlength=C).float() + 1e-6
                class_weights = (1.0 / class_counts)
                class_weights /= class_weights.max()  # 归一化

            # ======================== soft label 分类目标 ===================
            target_cls = torch.zeros_like(pred_logits)  # [N, C]
            target_label = gt_labels[matched_gt_idx]    # [num_pos]
            target_cls[pos_mask, target_label] = 0.95   # 默认 soft 置信度

            # === IoU-aware 分类增强（高IoU预测框给予更高标签值）===
            pred_pos_boxes = pred_box[pos_mask]
            matched_gt_boxes = gt_boxes[matched_gt_idx]
            with torch.no_grad():
                ious = complete_ciou(pred_pos_boxes, matched_gt_boxes).clamp(0, 1)
                target_cls[pos_mask, target_label] = 0.7 + 0.25 * ious  # soft标签随IoU波动

            # === 分类损失 ===
            loss_cls += self.cls_loss_fn(pred_logits, target_cls) * class_weights[target_label].mean()

            # ======================== 回归分支：位置误差 ===================
            box_sizes = matched_gt_boxes[:, 2:] - matched_gt_boxes[:, :2]  # [num_pos, 2]
            area = box_sizes[:, 0] * box_sizes[:, 1] + 1e-6
            size_weights = (1.0 / area).clamp(0.1, 10.0)  # 小目标给予更大权重

            # === GWD / smooth_l1 ===
            if self.box_type == "gwd":
                loss_box += (gwd_loss(pred_pos_boxes, matched_gt_boxes) * size_weights).mean()
            else:
                loss_box += (F.smooth_l1_loss(pred_pos_boxes, matched_gt_boxes, reduction='none').sum(dim=1) * size_weights).mean()

            # ======================== 回归分支：IoU loss ====================
            if self.iou_type in {"ciou", "all"}:
                ciou = complete_ciou(pred_pos_boxes, matched_gt_boxes)
                iou_weights = (ciou.detach() ** 2).clamp(min=0.01)  # 准确预测给予更大loss权重
                loss_iou += ((1.0 - ciou) * iou_weights).mean()

            elif self.iou_type == "giou":
                giou = generalized_box_iou(pred_pos_boxes, matched_gt_boxes)
                loss_iou += (1.0 - giou).mean()

        # ======================== Batch 平均 =========================
        loss_cls /= B
        loss_box /= B
        loss_iou /= B

        # ======================== 自动权重融合 or 固定权重 =========================
        cls_w, box_w, iou_w = weights
        if cls_w == 0.0 and box_w == 0.0 and iou_w == 0.0:
            ws = torch.softmax(torch.stack([-loss_cls, -loss_box, -loss_iou]), dim=0)
            total_loss = ws[0] * loss_cls + ws[1] * loss_box + ws[2] * loss_iou
        else:
            total_loss = cls_w * loss_cls + box_w * loss_box + iou_w * loss_iou

        return total_loss, loss_cls.detach(), loss_box.detach(), loss_iou.detach()

    
    def dynamic_assign(self, pred_boxes, pred_cls_scores, gt_boxes, gt_labels,
                   topk=50, epoch=0, warmup_epoch=10):
        N, M = pred_boxes.size(0), gt_boxes.size(0)
        device = pred_boxes.device

        topk = max(1, min(int(N * 0.2), M * 10))
        iou_thresh = self.iou_thresh
        if epoch < warmup_epoch:
            iou_thresh /= 10000
            topk *= 5

        if M == 0 or N == 0:
            return torch.zeros(N, dtype=torch.bool, device=device), torch.zeros(0, dtype=torch.long, device=device)

        ious = box_iou(pred_boxes, gt_boxes)  # [N, M]
        valid_mask = ious > iou_thresh

        pred_probs = pred_cls_scores.softmax(dim=-1)  # [N, C]
        tgt_labels_exp = gt_labels.unsqueeze(0).expand(N, M)
        cls_cost = -torch.log(torch.gather(pred_probs, 1, tgt_labels_exp))  # [N, M]
        cls_cost[~valid_mask] = 1e5

        cost = cls_cost - 3.0 * ious  # 综合 cost

        topk = min(topk, N)
        topk_cost, topk_idx = torch.topk(cost.T, k=topk, largest=False)
        matching_matrix = torch.zeros((N, M), device=device)
        matching_matrix[topk_idx.reshape(-1), torch.arange(M, device=device).repeat_interleave(topk)] = 1

        pos_mask = matching_matrix.sum(dim=1) > 0
        matched_gt_idx = matching_matrix[pos_mask].argmax(dim=1)

        # === warmup 阶段伪样本增强 ===
        if epoch < warmup_epoch:
            pos_ratio = pos_mask.float().mean().item()
            if pos_ratio < 0.05:
                extra_ratio = 0.3 * (1 - epoch / warmup_epoch)
            elif pos_ratio < 0.3:
                extra_ratio = 0.05 * (1 - epoch / warmup_epoch)
            else:
                extra_ratio = 0.0

            if extra_ratio > 0:
                num_extra = int(extra_ratio * N)

                score_max = pred_probs.max(dim=1).values
                iou_max = ious.max(dim=1).values
                score_norm = (score_max - score_max.min()) / (score_max.max() - score_max.min() + 1e-6)
                iou_norm = (iou_max - iou_max.min()) / (iou_max.max() - iou_max.min() + 1e-6)
                hybrid_score = 0.5 * score_norm + 0.5 * iou_norm

                hybrid_score[pos_mask] = -1.0
                extra_indices = torch.topk(hybrid_score, num_extra)[1]

                box_centers = (pred_boxes[extra_indices, :2] + pred_boxes[extra_indices, 2:]) / 2
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
                box_sizes = pred_boxes[extra_indices, 2:] - pred_boxes[extra_indices, :2]
                gt_sizes = gt_boxes[:, 2:] - gt_boxes[:, :2]
                dist = ((box_centers.unsqueeze(1) - gt_centers.unsqueeze(0)) ** 2).sum(-1)
                size_diff = (box_sizes.unsqueeze(1) - gt_sizes.unsqueeze(0)).abs().sum(-1)
                total_cost = dist + size_diff
                best_gt = total_cost.argmin(dim=1)

                best_gt_labels = gt_labels[best_gt]
                prob_extra = pred_probs[extra_indices, best_gt_labels]

                soft_target = 0.5 + 0.4 * hybrid_score[extra_indices].clamp(0, 1)  # [0.5, 0.9]
                with torch.amp.autocast(enabled=False, device_type=self.device_type):
                    cls_extra = F.binary_cross_entropy(prob_extra, soft_target, reduction='none')

                cls_cost[extra_indices, best_gt] = 0.5 * cls_extra
                pos_mask[extra_indices] = True
                matched_gt_idx = torch.cat([matched_gt_idx, best_gt], dim=0)

        return pos_mask, matched_gt_idx


def generate_synthetic_detection_data(batch_size=4, image_size=(224, 224), start_idx=0):
    """
    生成简化但有逻辑结构的合成目标检测数据。
    每张图像为白底，绘制不同颜色的矩形作为目标。
    """
    data_root = r"E:\resources\datasets\tea-buds-database\teaRob.v9i.yolov11"
    mean = (0.4868, 0.5291, 0.3377)
    std = (0.2017, 0.2022, 0.1851)
    train_dataset = YoloDataset(
        img_dir=os.path.join(data_root + r"\train", 'images'),
        label_dir=os.path.join(data_root + r"\train", 'labels'),
        img_size=image_size[0],
        normalize_label=True,
        cls_num=1,
        mean=mean,
        std=std,
        transforms=None
    )

    imgs = []
    boxes_batch = []
    labels_batch = []
    class_names = ['bud']

    for idx in range(batch_size):
        img_tensor, boxes, labels, gt_heatmaps, gt_bboxes, gt_masks, gt_coords, gt_sigma_map = train_dataset[
            idx + start_idx]
        imgs.append(img_tensor)
        boxes_batch.append(boxes)
        labels_batch.append(labels)

    return torch.stack(imgs), boxes_batch, labels_batch, class_names


def profile_memory_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # MB
        return mem_allocated, mem_reserved
    return 0.0, 0.0


def profile_cpu_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # MB


def check_requires_grad(loss, pred_scores, pred_boxes):
    print("=== requires_grad 检查 ===")

    print(f"[pred_scores] requires_grad: {pred_scores.requires_grad}")
    print(f"[pred_boxes] requires_grad: {pred_boxes.requires_grad}")

    if isinstance(loss, torch.Tensor):
        print(f"[loss] requires_grad: {loss.requires_grad}")

    # 针对 loss 相关的反向追踪：
    loss_grad = torch.autograd.grad(loss, pred_scores, retain_graph=True, allow_unused=True)[0]
    print(f"[loss -> pred_scores] grad None? {loss_grad is None}")


def main():
    epochs = 500
    warmup_epoch = 5
    img_size = 640
    num_classes = 5
    batch_size = 16
    include_backward = True  # <- 可选：是否执行 loss.backward()
    iou_loss_type = 'giou'
    box_loss_type = ''
    lr_ = 0.0001

    scaler = GradScaler()  # 创建一个梯度缩放器
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = DynamicBoxDetector(in_dim=img_size,
                               hidden_dim=256,
                               nhead=4,
                               num_layers=2, cls_num=num_classes,
                               once_embed=True,
                               is_split_trans=False,
                               is_fpn=False,
                               head_group=1,
                               cls_head_kernel_size=1,
                               backbone_type="resnet34", device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr_)
    loss_fn = DetectionLoss(iou_type=iou_loss_type, box_type=box_loss_type, device_type=device_str).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_,
        total_steps=epochs,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=10,  # 初始 lr = max_lr / 10
        final_div_factor=100  # 最终 lr = max_lr / 100
    )

    cls_losses, box_losses, giou_losses, total_losses = [], [], [], []
    forward_times, backward_times = [], []
    mem_cuda_allocs_fwd, mem_cuda_allocs_bwd = [], []

    print("===> Running benchmark...")

    last_cls_loss = 0.0
    last_box_loss = 0.0
    last_iou_loss = 0.0

    imgs, boxes, labels, class_names = generate_synthetic_detection_data(batch_size, (img_size, img_size))

    for epoch in tqdm(range(epochs), desc="Benchmarking"):
        model.train()

        if epoch < warmup_epoch:
            weights_loss = (0, 0, 0)
        else:
            # 对较难任务给予更小的权重，防止主导优化。
            eps = 1e-8
            inv_cls = 1.0 / (last_cls_loss + eps)
            inv_box = 1.0 / (last_box_loss + eps)
            inv_iou = 1.0 / (last_iou_loss + eps)
            sum_inv = inv_cls + inv_box + inv_iou
            cls_w = inv_cls / sum_inv
            box_w = inv_box / sum_inv
            giou_w = inv_iou / sum_inv
            weights_loss = (cls_w, box_w, giou_w)

        boxes = [t.to(device) for t in boxes]
        labels = [t.to(device) for t in labels]
        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type=device_str):
            pred_scores, pred_boxes = model(imgs.to(device))
            # Forward pass
            torch.cuda.synchronize()
            start_fwd = time.time()
            loss, loss_cls, loss_box, loss_iou = loss_fn(pred_scores, pred_boxes, boxes, labels, epoch, warmup_epoch,
                                                         weights_loss)

        torch.cuda.synchronize()
        end_fwd = time.time()
        forward_times.append((end_fwd - start_fwd) * 1000)

        if include_backward:
            torch.cuda.synchronize()
            start_bwd = time.time()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            torch.cuda.synchronize()
            end_bwd = time.time()
            backward_times.append((end_bwd - start_bwd) * 1000)

            alloc_bwd, _ = profile_memory_cuda()
            mem_cuda_allocs_bwd.append(alloc_bwd)

        alloc_fwd, _ = profile_memory_cuda()
        mem_cuda_allocs_fwd.append(alloc_fwd)

        cls_losses.append(loss_cls.item())
        box_losses.append(loss_box.item())
        giou_losses.append(loss_iou.item())
        total_losses.append(loss.item())

        last_iou_loss = loss_iou.item()
        last_box_loss = loss_box.item()
        last_cls_loss = loss_cls.item()

        clear_gpu_cache()

    def summary(name, values):
        arr = np.array(values)
        print(
            f"{name:<25}: mean = {arr.mean():.4f}, std = {arr.std():.4f}, min = {arr.min():.4f}, max = {arr.max():.4f}")

    print("\n=== Benchmark Summary ===")
    summary("Forward Time (ms)", forward_times)
    if include_backward:
        summary("Backward Time (ms)", backward_times)

    summary("Total Loss", total_losses)
    summary("Cls Loss", cls_losses)
    summary(f"{box_loss_type} Box Reg Loss", box_losses)
    summary(f"{iou_loss_type} Loss", giou_losses)

    summary("CUDA Allocated (MB, FWD)", mem_cuda_allocs_fwd)
    if include_backward:
        summary("CUDA Allocated (MB, BWD)", mem_cuda_allocs_bwd)

    print(f"CPU Memory (approx): {profile_cpu_memory():.2f} MB")

    # 重新提取
    # imgs, boxes, labels, class_names = generate_synthetic_detection_data(batch_size, (img_size, img_size), batch_size)
    model.eval()
    score_map, box_map = model(imgs.to(device))
    pred_boxes, pred_scores, pred_labels = dynamic_postprocess(score_map.detach(), box_map.detach(),
                                                               img_size=(img_size, img_size),
                                                               score_thresh=0.75, iou_thresh=0.75,
                                                               nms=True, upscale=False)
    imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255  # 转为 [B, H, W, 3]，像素范围 [0,255]
    imgs_np = imgs_np.astype(np.uint8)

    for b in range(len(imgs_np)):
        img = imgs_np[b]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 真实框
        boxes_gt = boxes[b].cpu()
        labels_gt = labels[b].cpu()
        pred_boxe, pred_score, pred_label = pred_boxes[b], pred_scores[b], pred_labels[b]

        img = draw_boxes(img, boxes_gt, labels_gt, color=(0, 255, 0), label_text=class_names)  # 绿色 GT
        img = draw_boxes(img, pred_boxe, pred_label, pred_score, color=(0, 0, 255), label_text=class_names)  # 红色预测

        cv2.imshow(f"Image {b}", img)
        cv2.waitKey(0)  # 按任意键继续
        cv2.destroyAllWindows()


if __name__ == "__main__":
    set_seed(2025, False)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
