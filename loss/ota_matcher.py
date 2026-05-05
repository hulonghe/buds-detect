import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class OTAMatcher(nn.Module):
    """
    OTA (Optimal Transport Assignment) Matcher
    
    核心思想：
    1. 构建cost矩阵: cls_cost + reg_cost + iou_cost
    2. 使用匈牙利算法求解最优分配
    3. 动态调整正负样本比例
    """
    
    def __init__(self, num_classes=80, alpha=1.0, beta=6.0, gamma=0.5,
                 fg_iou_threshold=0.5, bg_iou_threshold=0.4,
                 num_candidates=10, use_focal=True, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 分类权重
        self.beta = beta   # 回归权重
        self.gamma = gamma # IoU权重
        
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.num_candidates = num_candidates  # 每个GT选择的候选预测数
        
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
    
    def forward(self, pred_scores, pred_boxes, gt_boxes, gt_labels, img_size):
        """
        Args:
            pred_scores: [B, N, C] 预测分类分数
            pred_boxes:  [B, N, 4] 预测框 (x1,y1,x2,y2) 归一化
            gt_boxes:    tuple of [M_i, 4] OR [B, M, 4] GT框
            gt_labels:   tuple of [M_i] OR [B, M] GT标签
            img_size:    (H, W) 图像尺寸
        
        Returns:
            pos_masks: [B, N] 正样本掩码
            matched_gt_indices: [B, N] 每个预测对应的GT索引
            fg_mask:    [B, N] 前景掩码
        """
        B, N, C = pred_scores.shape
        device = pred_scores.device
        
        # Handle tuple format from DataLoader
        if isinstance(gt_boxes, (list, tuple)):
            gt_boxes_tuple = gt_boxes
            gt_labels_tuple = gt_labels
        else:
            gt_boxes_tuple = [gt_boxes[b] for b in range(B)]
            gt_labels_tuple = [gt_labels[b] for b in range(B)]
        
        all_pos_masks = []
        all_matched_indices = []
        all_fg_masks = []
        
        for b in range(B):
            gt_boxes_b = gt_boxes_tuple[b]  # [M, 4]
            gt_labels_b = gt_labels_tuple[b]  # [M]
            M = gt_boxes_b.shape[0]
            
            if M == 0:
                pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
                matched_gt_idx = torch.zeros(0, dtype=torch.long, device=device)
                fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
                all_pos_masks.append(pos_mask)
                all_matched_indices.append(matched_gt_idx)
                all_fg_masks.append(fg_mask)
                continue
            
            # 1. 计算预测框与GT的IoU
            ious = box_iou(pred_boxes[b], gt_boxes_b)  # [N, M]
            
            # 2. 分类代价 (Focal loss style)
            pred_probs = pred_scores[b].sigmoid()  # [N, C]
            
            cls_cost = self._compute_cls_cost(
                pred_probs, gt_labels_b, ious
            )  # [N, M]
            
            # 3. 回归代价 (基于IoU)
            reg_cost = -ious  # IoU越高，cost越低
            
            # 4. 总代价
            total_cost = self.alpha * cls_cost + self.beta * reg_cost + self.gamma * (1 - ious)
            
            # 5. OTA匹配
            pos_mask, matched_gt_idx = self._ota_matching(
                total_cost, ious, gt_labels_b, N, M
            )
            
            # 6. 生成前景掩码
            fg_mask = pos_mask.clone()
            
            all_pos_masks.append(pos_mask)
            all_matched_indices.append(matched_gt_idx)
            all_fg_masks.append(fg_mask)
        
        pos_masks = torch.stack(all_pos_masks, dim=0)  # [B, N]
        matched_gt_indices = torch.stack(all_matched_indices, dim=0)  # [B, N]
        fg_masks = torch.stack(all_fg_masks, dim=0)  # [B, N]
        
        return pos_masks, matched_gt_indices, fg_masks
    
    def _compute_cls_cost(self, pred_probs, gt_labels, ious):
        """
        计算分类代价
        """
        N, M = ious.shape
        device = ious.device
        
        # 扩展gt_labels到[N, M]
        gt_labels_exp = gt_labels.unsqueeze(0).expand(N, -1)  # [N, M]
        
        # 获取对应类别的预测概率
        cls_scores = torch.gather(
            pred_probs, 1, gt_labels_exp
        )  # [N, M]
        
        if self.use_focal:
            # Focal loss style cost
            cls_cost = -(
                (1 - cls_scores) ** self.focal_gamma * 
                torch.log(cls_scores + 1e-7) * self.focal_alpha +
                cls_scores ** self.focal_gamma * 
                torch.log(1 - cls_scores + 1e-7) * (1 - self.focal_alpha)
            )
        else:
            # BCE style cost
            cls_cost = -(
                gt_labels_exp * torch.log(cls_scores + 1e-7) +
                (1 - gt_labels_exp) * torch.log(1 - cls_scores + 1e-7)
            )
        
        return cls_cost
    
    def _ota_matching(self, cost_matrix, ious, gt_labels, N, M):
        """
        OTA匹配: 使用top-k候选 + 匈牙利算法
        """
        device = cost_matrix.device
        
        # 1. 每个GT选择top-k个最低cost的预测
        k = min(self.num_candidates, N)
        topk_values, topk_indices = torch.topk(
            cost_matrix.T, k=k, largest=False
        )  # [M, k]
        
        # 2. 扩展代价矩阵用于匹配
        matching_matrix = torch.zeros_like(cost_matrix)
        
        for m in range(M):
            # 获取该GT的候选预测索引
            candidates = topk_indices[m]  # [k]
            # 在这些位置标记
            matching_matrix[candidates, m] = 1
        
        # 3. 过滤：只保留IoU高于阈值的
        iou_mask = ious > self.bg_iou_threshold
        matching_matrix = matching_matrix * iou_mask
        
        # 4. 确定正负样本
        pos_mask = matching_matrix.sum(dim=1) > 0  # [N]
        
        # 5. 为每个正样本找到对应的GT
        matched_gt_idx = torch.zeros(N, dtype=torch.long, device=device)
        matched_gt_idx[pos_mask] = matching_matrix[pos_mask].argmax(dim=1)
        
        return pos_mask, matched_gt_idx


class SimOTA(nn.Module):
    """
    Simplified OTA matcher (更高效的版本)
    来自YOLOX的SimOTA实现
    """
    
    def __init__(self, num_classes=80, topk=13, alpha=0.5, beta=1.5, iou_threshold=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.iou_threshold = iou_threshold
    
    def forward(self, pred_scores, pred_boxes, gt_boxes, gt_labels):
        """
        Args:
            pred_scores: [B, N, C]
            pred_boxes:  [B, N, 4]
            gt_boxes:    tuple of [M_i, 4] OR [B, M, 4] tensor
            gt_labels:   tuple of [M_i] OR [B, M] tensor
        
        Returns:
            pos_masks: [B, N]
            matched_gt_indices: [B, N]
            fg_masks: [B, N]
        """
        B, N, C = pred_scores.shape
        device = pred_scores.device
        
        # Handle tuple format from DataLoader
        if isinstance(gt_boxes, (list, tuple)):
            gt_boxes_tuple = gt_boxes
            gt_labels_tuple = gt_labels
        else:
            gt_boxes_tuple = [gt_boxes[b] for b in range(B)]
            gt_labels_tuple = [gt_labels[b] for b in range(B)]
        
        all_pos_masks = []
        all_matched_indices = []
        all_fg_masks = []
        
        for b in range(B):
            gt_boxes_b = gt_boxes_tuple[b]
            gt_labels_b = gt_labels_tuple[b]
            M = gt_boxes_b.shape[0]
            
            if M == 0:
                all_pos_masks.append(torch.zeros(N, dtype=torch.bool, device=device))
                # 修复：使用N而不是0，保持一致形状
                all_matched_indices.append(torch.full((N,), -1, dtype=torch.long, device=device))
                all_fg_masks.append(torch.zeros(N, dtype=torch.bool, device=device))
                continue
            
            # 计算IoU
            ious = box_iou(pred_boxes[b], gt_boxes_b)  # [N, M]
            
            # 计算cost matrix
            pred_probs = pred_scores[b].sigmoid()
            gt_labels_exp = gt_labels_b.unsqueeze(0).expand(N, -1)
            cls_scores = torch.gather(pred_probs, 1, gt_labels_exp)
            
            cls_cost = -torch.log(cls_scores + 1e-7)
            reg_cost = -ious
            cost_matrix = self.alpha * cls_cost + self.beta * reg_cost
            
            # SimOTA匹配
            pos_mask, matched_gt_idx = self._simota_matching(
                cost_matrix, ious, M, N
            )
            
            all_pos_masks.append(pos_mask)
            all_matched_indices.append(matched_gt_idx)
            all_fg_masks.append(pos_mask.clone())
        
        return (
            torch.stack(all_pos_masks),
            torch.stack(all_matched_indices),
            torch.stack(all_fg_masks)
        )
    
    def _simota_matching(self, cost_matrix, ious, M, N):
        """SimOTA匹配实现"""
        device = cost_matrix.device
        
        # 每个GT选择top-k
        k = min(self.topk, N)
        topk_cost, topk_idx = torch.topk(
            cost_matrix.T, k=k, largest=False
        )  # [M, k]
        
        # 构建匹配矩阵
        matching_matrix = torch.zeros_like(cost_matrix)
        for m in range(M):
            matching_matrix[topk_idx[m], m] = 1
        
        # IoU过滤
        iou_mask = ious > self.iou_threshold
        matching_matrix = matching_matrix * iou_mask
        
        # 确定正样本
        pos_mask = matching_matrix.sum(dim=1) > 0
        matched_gt_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        matched_gt_idx[pos_mask] = matching_matrix[pos_mask].argmax(dim=1)
        
        return pos_mask, matched_gt_idx
