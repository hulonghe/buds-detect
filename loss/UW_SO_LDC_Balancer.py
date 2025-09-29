import torch
import torch.nn as nn
import torch.nn.functional as F

class UW_LDC_GCR_Balancer(nn.Module):
    def __init__(self, num_losses, T=1.0, ldc_thresh=2.0, ldc_scale=0.5, eps=1e-6):
        super().__init__()
        self.K = num_losses
        self.T = T
        self.ldc_thresh = ldc_thresh
        self.ldc_scale = ldc_scale
        self.eps = eps

    def forward(self, losses, grad_list=None):
        """
        losses: List[Tensor], each scalar
        grad_list: List[Tensor], gradient vectors of each loss w.r.t. model params or shared features
        """
        L = torch.stack(losses)
        s = 1.0 / (L + self.eps)
        w = torch.softmax(s / self.T, dim=0)

        # --- LDC 控制 ---
        L_mean = L.mean()
        mask = (L / (L_mean + self.eps)) > self.ldc_thresh
        if mask.any():
            w[mask] *= self.ldc_scale
            w = w / (w.sum() + self.eps)

        # --- GCR 梯度一致性调整 ---
        if grad_list is not None:
            # 构建梯度向量矩阵 G
            G = torch.stack([g.view(-1) for g in grad_list], dim=0)  # [K, D]
            G = F.normalize(G, p=2, dim=1)  # cosine normalize
            sim_matrix = G @ G.T  # [K, K]
            mean_sim = sim_matrix.mean(dim=1)  # 每个 loss 与其他的平均余弦相似度
            w = w * mean_sim  # 权重乘以一致性评分
            w = w / (w.sum() + self.eps)

        final_loss = (w * L).sum()
        return final_loss, w.detach()
