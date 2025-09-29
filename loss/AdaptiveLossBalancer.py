import torch
import torch.nn as nn

class AdaptiveLossBalancer(nn.Module):
    def __init__(self, num_losses, alpha=0.9, T=2.0, eps=1e-6):
        super().__init__()
        self.num_losses = num_losses
        self.alpha = alpha    # EMA 衰减率
        self.T = T            # DWA 温度参数
        self.eps = eps
        # 初始化 EMA 和前两步损失历史（可注册为 buffer，不参与梯度）
        self.register_buffer('ema_loss', torch.zeros(num_losses))
        self.register_buffer('prev_loss', torch.ones(num_losses))
        self.register_buffer('prev2_loss', torch.ones(num_losses))

    def forward(self, losses):
        """
        losses: list 或张量列表，长度 = num_losses。
        返回加权后的总损失（标量）。
        """
        # 将 losses 转为同形张量
        L = torch.stack(losses).detach()  # detach() 防止梯度流入历史
        # 更新 EMA：EMA = alpha * EMA + (1-alpha) * current_loss
        self.ema_loss = self.alpha * self.ema_loss + (1 - self.alpha) * L
        
        # 计算 DWA 权重因子：仅当有前两步历史时使用
        if (self.prev_loss is not None) and (self.prev2_loss is not None):
            # 避免 division by zero
            ratio = (self.prev_loss + self.eps) / (self.prev2_loss + self.eps)
            dva_weights = torch.exp(ratio / self.T)
        else:
            dva_weights = torch.ones_like(L)
        
        # 更新历史（为下次计算做准备）
        self.prev2_loss = self.prev_loss.clone()
        self.prev_loss = L.clone()
        
        # 计算组合权重：weight_i = (1/EMA_i) * DWA_factor_i
        inv_ema = 1.0 / (self.ema_loss + self.eps)
        raw_w = inv_ema * dva_weights
        # 归一化权重（可用 softmax 或除以 sum）
        w = raw_w / (raw_w.sum() + self.eps)
        
        # 计算加权损失
        weighted_losses = w * L
        return weighted_losses.sum()
