import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', from_logits=True, device_type="cpu"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device_type = device_type
        self.from_logits = from_logits  # 是否为原始 logits

    def forward(self, inputs, targets=None, no_positive=False):
        if inputs is not None and not torch.isfinite(inputs).all():
            print("inputs 非法数值：", inputs)
        if targets is not None and not torch.isfinite(targets).all():
            print("targets 非法数值：", targets)

        inputs = inputs.view(-1)
        if no_positive or targets is None:
            targets = torch.zeros_like(inputs)
        targets = targets.view(-1)

        if self.from_logits:
            prob = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            prob = inputs.clamp(min=1e-6, max=1 - 1e-6)  # 避免 log(0)
            ce_loss = None
            with torch.amp.autocast(enabled=False, device_type=self.device_type):
                ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
