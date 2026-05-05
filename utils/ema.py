"""
EMA: Exponential Moving Average
用于稳定训练和提升推理性能
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA(nn.Module):
    """
    Exponential Moving Average
    
    使用方法:
    ema_model = EMA(model, decay=0.9999)
    
    训练时:
    ema_model.update_params()
    
    推理时:
    ema_model.apply_shadow()
    # 执行推理
    ema_model.restore()
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # 保存原始参数
        self.shadow = {}
        self.backup = {}
        
        # 初始化shadow参数
        self.register_buffer('update_count', torch.tensor(0))
        self._register()
    
    def _register(self):
        """注册原始参数的hook"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    @torch.no_grad()
    def update_params(self):
        """更新EMA参数"""
        self.update_count += 1
        decay = self.decay
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """用EMA参数替换模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


class EMAWrapper:
    """
    更易用的EMA包装器
    
    用法:
    ema = EMAWrapper(model, decay=0.9999, device='cuda')
    
    # 训练循环
    for epoch in range(epochs):
        train(...)
        ema.update()
    
    # 推理
    ema.apply_shadow()
    validate(...)
    ema.restore()
    """
    
    def __init__(self, model, decay=0.9999, device='cuda', warmup_epochs=0):
        self.model = model
        self.decay = decay
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # 初始化shadow
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)
        
        self.backup = {}
        self.epoch = 0
    
    def update(self):
        """每epoch更新一次"""
        self.epoch += 1
        
        # Warmup阶段不更新
        if self.epoch <= self.warmup_epochs:
            return
        
        decay = min(self.decay, (1 + self.epoch) / (10 + self.epoch))
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param.data
    
    def apply_shadow(self):
        """应用EMA参数"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self):
        """保存EMA状态"""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'epoch': self.epoch
        }
    
    def load_state_dict(self, state_dict):
        """加载EMA状态"""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']
        self.epoch = state_dict['epoch']
    
    def eval(self):
        """切换到评估模式"""
        self.model.eval()
    
    def train(self, mode=True):
        """切换到训练模式"""
        self.model.train(mode)
        return self
    
    def __call__(self, *args, **kwargs):
        """代理调用"""
        return self.model(*args, **kwargs)
    
    @property
    def training(self):
        return self.model.training
    
    def parameters(self, recurse=True):
        return self.model.parameters(recurse)
    
    def named_parameters(self, recurse=True):
        return self.model.named_parameters()


class MultiScaleEMA(nn.Module):
    """
    多尺度EMA (用于不同epoch阶段的EMA)
    """
    
    def __init__(self, model, decay_schedule=None):
        super().__init__()
        
        if decay_schedule is None:
            decay_schedule = {
                0: 0.999,
                50: 0.9995,
                100: 0.9997,
                200: 0.9999,
            }
        
        self.decay_schedule = decay_schedule
        self.model = model
        
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, epoch):
        # 获取当前decay
        decay = 0.999
        for threshold, d in self.decay_schedule.items():
            if epoch >= threshold:
                decay = d
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()


# 简化版EMA
def build_ema(model, decay=0.9999, device='cpu', warmup_epochs=0):
    """构建EMA"""
    return EMAWrapper(model, decay, device, warmup_epochs)


if __name__ == '__main__':
    # 测试
    model = nn.Linear(10, 2)
    ema = build_ema(model)
    
    # 模拟训练
    for i in range(100):
        # 模拟参数更新
        with torch.no_grad():
            for param in model.parameters():
                param += torch.randn_like(param) * 0.01
        ema.update()
    
    # 测试应用
    print("原始参数:", list(model.parameters())[0][0, :3])
    ema.apply_shadow()
    print("EMA参数:", list(model.parameters())[0][0, :3])
    ema.restore()
    print("恢复参数:", list(model.parameters())[0][0, :3])
