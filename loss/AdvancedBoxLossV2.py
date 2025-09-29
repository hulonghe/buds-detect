import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss


class AdvancedBoxLossV2(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_reg=1.0, lambda_iou=1.0,
                 lambda_wasserstein=1.0, lambda_size_penalty=1.0,
                 topk=3, region=5):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_iou = lambda_iou
        self.lambda_wasserstein = lambda_wasserstein
        self.lambda_size_penalty = lambda_size_penalty
        self.topk = topk
        self.region = region

    def forward(self, pred_score_map, pred_box_map, gt_boxes_, img_size=(640, 640), imgs=None):
        """
        pred_score_map: [B, 1, Hf, Wf] raw logits (no sigmoid)
        pred_box_map: [B, 4, Hf, Wf], in normalized (cx, cy, w, h)
        gt_boxes: list of [num_obj_i, 4] in (x1,y1,x2,y2), normalized
        """
        B, _, Hf, Wf = pred_score_map.shape
        w, h = img_size
        device = pred_score_map.device
        # 0. 缩放至特征图坐标
        gt_boxes = self.scaled_bboxes(gt_boxes_, img_size, (Hf, Wf))
        # 1. 自动生成 soft label mask（中心区域 / Top-k 匹配）
        gt_masks, soft_scores = self.generate_soft_foreground_mask(
            gt_boxes, (Hf, Wf), img_size, device
        )

        # 2. 分类损失（直接用 logits + soft mask）
        if self.lambda_cls > 0:
            prob = torch.sigmoid(pred_score_map)
            alpha = 0.25
            gamma = 2.0
            cls_loss_list = []
            for b in range(B):
                # 1. 获取当前样本预测
                prob_b = prob[b, 0]
                logit_b = pred_score_map[b, 0]
                target_b = soft_scores[b, 0]  # soft label
                mask_b = gt_masks[b, 0]  # binary mask：正样本区域

                # 2. focal 计算
                pt = target_b * prob_b + (1 - target_b) * (1 - prob_b)
                focal_weight = (alpha * (1 - pt).pow(gamma)).detach()

                # 3. 正样本 loss
                pos_mask = mask_b > 0
                pos_loss = F.binary_cross_entropy_with_logits(
                    logit_b[pos_mask],
                    target_b[pos_mask],
                    weight=focal_weight[pos_mask],
                    reduction='sum'
                )

                # 4. 负样本挖掘
                neg_mask = mask_b == 0
                neg_loss_all = F.binary_cross_entropy_with_logits(
                    logit_b[neg_mask],
                    target_b[neg_mask],
                    weight=focal_weight[neg_mask],
                    reduction='none'
                )

                # 选 top-K hard negative
                num_pos = pos_mask.sum().item()
                some_min_value = 50
                num_neg = min(self.topk * num_pos + 50, some_min_value)
                if num_neg > 0:
                    neg_loss_topk, _ = torch.topk(neg_loss_all, num_neg)
                    neg_loss = neg_loss_topk.sum()
                else:
                    neg_loss = torch.tensor(0.0, device=neg_loss_all.device)

                # 5. 总 loss 归一化
                eps = 1.0
                normalizer = num_pos + num_neg + eps  # eps 比如 1.0
                total_loss = (pos_loss + neg_loss) / normalizer
                cls_loss_list.append(total_loss)

            cls_loss = torch.stack([torch.tensor(l, device=pred_score_map.device, dtype=torch.float32)
                                    if not isinstance(l, torch.Tensor) else l
                                    for l in cls_loss_list]).mean()
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_reg_loss, total_iou_loss = 0., 0.
        total_wass_loss, total_sz_loss = 0., 0.

        for b in range(B):
            if gt_boxes[b].numel() == 0:
                continue

            # 3. 匹配预测框与GT中心区域
            pos_mask = gt_masks[b, 0] > 0
            if pos_mask.sum() == 0:
                continue

            pred_boxes = pred_box_map[b].permute(1, 2, 0)[pos_mask]  # [N_pos, 4] in (cx,cy,w,h)
            pred_boxes = self.cxcywh_to_xyxy(pred_boxes)
            # 映射到特征图坐标（Hf, Wf）
            pred_boxes[:, [0, 2]] *= Wf
            pred_boxes[:, [1, 3]] *= Hf

            gt_boxes_rescaled = gt_boxes[b]  # [num_obj, 4] normalized
            matched_gt = self.select_best_gt(pos_mask, gt_boxes_rescaled, Hf, Wf)

            # 4. IoU loss
            if self.lambda_iou > 0:
                ciou_loss_matrix = complete_box_iou_loss(pred_boxes, matched_gt)  # [N_pos, N_gt]
                iou_loss = ciou_loss_matrix.diag().mean()  # 取对角线上每个预测框与其匹配 GT 的 CIoU loss
            else:
                iou_loss = torch.tensor(0.0, device=device)

            # 5. Box regression（Smooth L1）,按 box 尺寸归一化再计算 L1，以避免大框主导损失。
            if self.lambda_reg > 0:
                scale = (matched_gt[:, 2:] - matched_gt[:, :2]).clamp(min=math.sqrt(Hf * Wf))
                normed_diff = (pred_boxes - matched_gt) / torch.cat([scale, scale], dim=1)
                reg_loss = F.smooth_l1_loss(normed_diff, torch.zeros_like(normed_diff))
            else:
                reg_loss = torch.tensor(0.0, device=device)

            # 6. Wasserstein loss (中心 + 尺寸分布)
            pred_ctr = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
            gt_ctr = (matched_gt[:, :2] + matched_gt[:, 2:]) / 2
            pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2])
            gt_wh = (matched_gt[:, 2:] - matched_gt[:, :2])
            if self.lambda_wasserstein > 0:
                wass_loss = F.smooth_l1_loss(pred_ctr, gt_ctr) + F.smooth_l1_loss(pred_wh, gt_wh)
            else:
                wass_loss = torch.tensor(0.0, device=device)

            # 7. 尺寸惩罚（防止极大/极小框）
            if self.lambda_size_penalty > 0:
                denom = gt_wh.clamp(min=1.0)
                size_penalty = ((pred_wh - gt_wh).abs() / denom).mean()
            else:
                size_penalty = torch.tensor(0.0, device=device)

            # 聚合
            total_reg_loss += reg_loss
            total_iou_loss += iou_loss
            total_wass_loss += wass_loss
            total_sz_loss += size_penalty

        num_batches = max(1, B)
        cls_loss *= self.lambda_cls
        total_reg_loss /= num_batches
        total_reg_loss *= self.lambda_reg
        total_iou_loss /= num_batches
        total_iou_loss *= self.lambda_iou
        total_wass_loss /= num_batches
        total_wass_loss *= self.lambda_wasserstein
        total_sz_loss /= num_batches
        total_sz_loss *= self.lambda_size_penalty

        total_loss = (cls_loss + total_reg_loss + total_iou_loss + total_wass_loss + total_sz_loss)
        return total_loss, cls_loss, total_reg_loss, total_iou_loss, total_wass_loss, total_sz_loss

    @staticmethod
    def generate_soft_foreground_mask(gt_boxes, feature_size, img_size, device="cpu"):
        """
        gt_boxes: List[Tensor[num_obj_i, 4]] in absolute pixel (x1,y1,x2,y2)
        feature_size: (Hf, Wf)
        img_size: (H, W)
        """
        B = len(gt_boxes)
        Hf, Wf = feature_size
        H, W = img_size

        mask_all = torch.zeros((B, 1, Hf, Wf), dtype=torch.float32, device=device)
        soft_all = torch.zeros_like(mask_all)

        for b in range(B):
            boxes = gt_boxes[b]  # absolute coords: x1,y1,x2,y2
            if boxes.numel() == 0:
                continue

            # 将 boxes 缩放到特征图尺度
            boxes_feat = boxes.clone()
            scale_x = Wf / W
            scale_y = Hf / H
            boxes_feat[:, [0, 2]] *= scale_x
            boxes_feat[:, [1, 3]] *= scale_y

            N_gt = boxes_feat.shape[0]
            widths = boxes_feat[:, 2] - boxes_feat[:, 0]
            heights = boxes_feat[:, 3] - boxes_feat[:, 1]
            avg_radius = (widths + heights) / 4  # 每个 box 的平均半径

            x_grid = torch.arange(Wf, device=device).float()
            y_grid = torch.arange(Hf, device=device).float()
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')  # [Hf, Wf]

            for i_gt in range(N_gt):
                x1, y1, x2, y2 = boxes_feat[i_gt]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                radius = avg_radius[i_gt].clamp(min=1.0)  # 防止过小

                dist = torch.sqrt(((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2)  # 归一化后的欧式距离

                soft_mask = torch.clamp(1 - dist / (1 + 1e-6), min=0.0)  # 线性衰减
                mask_all[b, 0] = torch.max(mask_all[b, 0], (soft_mask > 0).float())
                soft_all[b, 0] = torch.max(soft_all[b, 0], soft_mask)

        return mask_all, soft_all

    @staticmethod
    def select_best_gt(pos_mask, gt_boxes, Hf, Wf):
        # 将每个 pos 匹配一个最近中心点的 GT box（用于回归目标）
        yy, xx = torch.where(pos_mask)
        cx = (xx.float() + 0.5) / Wf
        cy = (yy.float() + 0.5) / Hf
        pred_centers = torch.stack([cx, cy], dim=1)  # [N_pos, 2]

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # [N_gt, 2]
        dists = torch.cdist(pred_centers, gt_centers)  # [N_pos, N_gt]
        indices = torch.argmin(dists, dim=1)
        matched_gt = gt_boxes[indices]

        return matched_gt

    @staticmethod
    def cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def scaled_bboxes(gt_boxes, source_img_size, target_img_size):
        """
        gt_boxes: [num_obj, 4] in absolute pixel (x1,y1,x2,y2)
        """
        scaled_gt_boxes = []
        img_w, img_h = source_img_size
        Wf, Hf = target_img_size

        if img_w == Wf and img_h == Hf:
            return gt_boxes

        for boxes in gt_boxes:
            if boxes.numel() == 0:
                scaled_gt_boxes.append(boxes)
                continue

            boxes_scaled = boxes.clone()
            boxes_scaled[:, [0, 2]] = boxes_scaled[:, [0, 2]] / img_w * Wf  # x1, x2 -> scale to Wf
            boxes_scaled[:, [1, 3]] = boxes_scaled[:, [1, 3]] / img_h * Hf  # y1, y2 -> scale to Hf
            scaled_gt_boxes.append(boxes_scaled)

        return scaled_gt_boxes
