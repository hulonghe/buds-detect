import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Optional, Dict, Tuple


# -------------------------
# 基础模块
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, g=1, act=True, norm_type='bn', gn_groups=8):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=min(gn_groups, out_c), num_channels=out_c)
        else:
            self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(4, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class LayerNorm2d(nn.Module):
    """按通道做 LayerNorm，适合卷积特征"""
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
        self.c = c

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, (self.c,), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# -------------------------
# 更稳的轻量残差块
# ConvNeXt-ish / 大核 DWConv + PWConv
# -------------------------
class LiteConvBlock(nn.Module):
    """
    适合从零训练的轻量块：
    DWConv -> Norm -> PWConv -> Act -> PWConv -> residual
    - 用 layer scale 稳定训练
    - 比 attention 更省显存，也更容易收敛
    """
    def __init__(self, c, kernel_size=7, expansion=2.0, norm_type='bn', gn_groups=8, use_se=False):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size, 1, kernel_size // 2, groups=c, bias=False)

        if norm_type == 'ln':
            self.norm = LayerNorm2d(c)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=min(gn_groups, c), num_channels=c)
        else:
            self.norm = nn.BatchNorm2d(c)

        hidden = max(8, int(c * expansion))
        self.pw1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden, c, 1, bias=False)

        # layer scale，防止从零训练时残差过强
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-6)
        self.se = SEBlock(c) if use_se else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = y * self.gamma
        y = self.se(y)
        return x + y


# -------------------------
# 下采样模块
# -------------------------
class Downsample(nn.Module):
    def __init__(self, in_c, out_c, norm_type='bn', gn_groups=8):
        super().__init__()
        self.block = ConvBNAct(in_c, out_c, k=3, s=2, p=1, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x):
        return self.block(x)


# -------------------------
# Aux Head
# -------------------------
class AuxHead(nn.Module):
    def __init__(self, in_c, mid=128, num_classes=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_c, mid, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(mid, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.act(self.conv(x))
        x = self.flatten(x)
        return self.fc(x)


# -------------------------
# 更轻更稳的 EHT Backbone V2
# -------------------------
class EHTBackboneV2(nn.Module):
    """
    输出: ([c3, c4, c5], aux_logits)
    目标:
    - 从零训练更稳
    - 轻量化
    - 只在 c3 侧重上下文，后面尽量纯卷积
    """
    def __init__(self,
                 in_ch: int = 3,
                 base_c: int = 24,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0,
                 norm_type: str = 'gn',   # 无预训练时，batch 小建议 gn
                 gn_groups: int = 8,
                 use_checkpoint: bool = False,
                 aux_supervision: bool = True,
                 num_classes: int = 1):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.aux_supervision = aux_supervision
        self.norm_type = norm_type

        c1 = int(base_c * width_mult)
        c2 = int(base_c * 2 * width_mult)
        c3 = int(base_c * 4 * width_mult)
        c4 = int(base_c * 8 * width_mult)
        c5 = int(base_c * 12 * width_mult)   # 原来是 16 倍，这里收一点，更轻

        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, c1, k=3, s=2, p=1, norm_type=norm_type, gn_groups=gn_groups),
            ConvBNAct(c1, c1, k=3, s=1, p=1, norm_type=norm_type, gn_groups=gn_groups),
        )

        # stride=4
        self.down1 = Downsample(c1, c2, norm_type=norm_type, gn_groups=gn_groups)

        # stage2: 偏局部，小目标/边缘信息
        n2 = max(1, int(2 * depth_mult))
        self.stage2 = nn.Sequential(
            *[
                LiteConvBlock(c2, kernel_size=5, expansion=2.0, norm_type=norm_type, gn_groups=gn_groups, use_se=(i == n2 - 1))
                for i in range(n2)
            ]
        )

        # stride=8
        self.down2 = Downsample(c2, c3, norm_type=norm_type, gn_groups=gn_groups)

        # stage3: 只在这里稍微加强上下文
        n3 = max(2, int(3 * depth_mult))
        blocks3 = []
        for i in range(n3):
            blocks3.append(
                LiteConvBlock(
                    c3,
                    kernel_size=9 if i == 0 else 7,
                    expansion=2.0,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                    use_se=(i == 0 or i == n3 - 1)
                )
            )
        self.stage3 = nn.Sequential(*blocks3)

        # stride=16
        self.down3 = Downsample(c3, c4, norm_type=norm_type, gn_groups=gn_groups)

        # stage4: 语义增强，但仍然保持轻
        n4 = max(1, int(2 * depth_mult))
        self.stage4 = nn.Sequential(
            *[
                LiteConvBlock(
                    c4,
                    kernel_size=7,
                    expansion=1.5,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                    use_se=(i == 0)
                )
                for i in range(n4)
            ]
        )

        # stride=32
        self.down4 = Downsample(c4, c5, norm_type=norm_type, gn_groups=gn_groups)

        # stage5: 只保留极少量块，别继续堆重
        n5 = max(1, int(1 * depth_mult))
        self.stage5 = nn.Sequential(
            *[
                LiteConvBlock(
                    c5,
                    kernel_size=5,
                    expansion=1.5,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                    use_se=True
                )
                for _ in range(n5)
            ]
        )

        if aux_supervision:
            self.aux2 = AuxHead(c2, mid=96, num_classes=num_classes)
            self.aux3 = AuxHead(c3, mid=128, num_classes=num_classes)

        self._init_weights()

    def _init_weights(self):
        # 比你原来的 xavier_uniform_(gain=0.1) 更适合从零训练
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _maybe_checkpoint(self, module, x):
        if self.use_checkpoint and self.training:
            return cp.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x) -> Tuple[list, Optional[Dict[str, torch.Tensor]]]:
        x = self.stem(x)           # stride=2

        c2 = self.down1(x)         # stride=4
        c2 = self._maybe_checkpoint(self.stage2, c2)

        c3 = self.down2(c2)        # stride=8
        c3 = self._maybe_checkpoint(self.stage3, c3)

        c4 = self.down3(c3)        # stride=16
        c4 = self._maybe_checkpoint(self.stage4, c4)

        c5 = self.down4(c4)        # stride=32
        c5 = self._maybe_checkpoint(self.stage5, c5)

        aux_logits = None
        if self.aux_supervision:
            aux_logits = {
                'small': self.aux2(c2),
                'normal': self.aux3(c3),
            }

        return [c3, c4, c5], aux_logits


# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
    model = EHTBackboneV2(
        in_ch=3,
        base_c=24,
        width_mult=1.0,
        depth_mult=1.0,
        norm_type='gn',          # 小 batch 更稳
        use_checkpoint=False,
        aux_supervision=True,
        num_classes=1
    )
    model.eval()
    img = torch.randn(2, 3, 320, 320)
    feats, aux = model(img)
    for i, f in enumerate(feats, start=3):
        print(f"c{i}.shape = {f.shape}")
    print({k: v.shape for k, v in aux.items()})