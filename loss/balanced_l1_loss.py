import torch
import torch.nn as nn

class BalancedL1Loss(nn.Module):
    def __init__(self, beta=1.0, alpha=0.5, gamma=1.5, reduction='mean'):
        super(BalancedL1Loss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.b = torch.exp(torch.tensor(gamma / alpha)) - 1  # Tensor，不转float
        beta_t = torch.tensor(beta, dtype=self.b.dtype)

        self.C = (gamma * beta_t
                  - alpha * torch.log(self.b * beta_t + 1)
                  + self.alpha * beta_t)
        
        print("self.C =", self.C.item())

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            self.alpha / self.b * torch.log(self.b * diff + 1) - self.alpha * diff,
            self.gamma * diff - self.C
        )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
