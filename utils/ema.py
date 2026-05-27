import copy
import torch
import torch.nn as nn


class ModelEMA:
    """指数移动平均：平滑模型参数，提升泛化"""

    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


class SWA:
    """随机权重平均：保存最后 N 个 epoch 的模型权重并平均"""

    def __init__(self, swa_start_epoch=80, swa_freq=1):
        self.swa_start_epoch = swa_start_epoch
        self.swa_freq = swa_freq
        self.swa_state = None
        self.swa_count = 0

    def maybe_update(self, model, epoch):
        if epoch < self.swa_start_epoch or epoch % self.swa_freq != 0:
            return False

        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if self.swa_state is None:
            self.swa_state = state
            self.swa_count = 1
        else:
            for k in self.swa_state:
                self.swa_state[k] = (self.swa_state[k] * self.swa_count + state[k]) / (self.swa_count + 1)
            self.swa_count += 1
        return True

    def apply_to(self, model):
        if self.swa_state is not None:
            model.load_state_dict(self.swa_state)
