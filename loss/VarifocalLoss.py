import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', from_logits=True, device_type='cpu'):
        """
        alpha: 控制负样本调权（仅用于 target=0）
        gamma: 焦点调制因子
        from_logits: 输入是否为原始 logits（需做 sigmoid）
        device_type: 用于自动混合精度时的设备类型
        """
        super(VarifocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits
        self.device_type = device_type

        
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
            prob = inputs.clamp(min=1e-6, max=1 - 1e-6)
            with torch.amp.autocast(enabled=False, device_type=self.device_type):
                ce_loss = F.binary_cross_entropy(prob, targets, reduction='none')

        # focal 权重构建逻辑（Varifocal 不同点）
        # 正样本（targets > 0）用 iou 分数做权重
        # 负样本（targets == 0）用 alpha * prob^gamma 做调制
        focal_weight = torch.where(
            targets > 0.0,
            targets,  # 正样本：用 iou 分数作为权重
            self.alpha * prob.pow(self.gamma)  # 负样本：和 Focal 类似
        )

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
