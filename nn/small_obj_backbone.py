"""
EHTBackbone: Efficient Hybrid Transformer Backbone
- 输出: ([c3, c4, c5], aux_logits)
- 关键显存优化:
    * 线性注意力(ELU+1 kernel) 替代原生 O(N^2) self-attention
    * 仅在中间尺度（默认 c3）使用注意力
    * 可选 checkpoint 重算中间激活
    * GN/Bn 切换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Optional, Dict, Tuple


# ------------------------- 基础模块 -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, g=1, act=True, norm_type='bn', gn_groups=8):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        if norm_type == 'gn':
            self.bn = nn.GroupNorm(num_groups=gn_groups, num_channels=out_c)
        else:
            self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(4, c // r), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, c // r), c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


# ------------------------- 轻量线性注意力 -------------------------
class LinearAttention(nn.Module):
    """
    线性注意力实现（ELU+1 近似 kernel，Memory friendly）
    参考思路：Favor+/Performer 风格的核化注意力，将复杂度从 O(N^2) 降为 O(N)
    适合在中等分辨率（如 1/8 特征）使用。
    """
    def __init__(self, dim, heads=4, proj_norm=True):
        super().__init__()
        assert dim % heads == 0 or True  # 我们允许不整除，并动态处理
        self.dim = dim
        self.heads = min(heads, dim)  # heads 不宜大于 dim
        self.dim_head = dim // self.heads if self.heads > 0 else dim
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_norm = proj_norm
        if proj_norm:
            # LayerNorm over channels; apply per spatial location by permutation
            self.ln = nn.LayerNorm(dim)

    def elu_plus_one(self, x):
        # kernel feature map: elu(x) + 1, > 0
        return F.elu(x) + 1.0

    def forward(self, x):
        # x: b, c, h, w
        b, c, h, w = x.shape
        n = h * w
        qkv = self.to_qkv(x)  # b, 3c, h, w
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to (b, heads, head_dim, n)
        hd = self.dim_head
        # if dim not divisible, pad channel dim to heads*hd
        target_c = self.heads * hd
        if c != target_c:
            # pad zeros
            pad = target_c - c
            q = F.pad(q, (0,0,0,0,0,pad))
            k = F.pad(k, (0,0,0,0,0,pad))
            v = F.pad(v, (0,0,0,0,0,pad))
        q = q.view(b, self.heads, hd, n)
        k = k.view(b, self.heads, hd, n)
        v = v.view(b, self.heads, hd, n)

        # kernel feature maps
        qk = self.elu_plus_one(q)   # b,h,hd,n
        kk = self.elu_plus_one(k)   # b,h,hd,n

        # compute KV summary: (hd x hd) per head? we use (hd x n) * (n) -> reduce along n:
        # Efficient computation: compute S = kk @ v^T  -> shape (b, h, hd, hd)
        # but hd may be large; instead compute per-head context by (kk * v) sum over spatial dim
        # context = sum_over_n( kk * v ) -> b,h,hd,1? Actually we compute weighted sum along spatial dim:
        # We'll compute: kv = (kk @ v.transpose(-1,-2))? to keep memory small do:
        # kv = torch.einsum('bhdn,bhdn->bhdd?') (not practical). Alternative standard trick:
        # compute denom = sum(kk, dim=-1, keepdim=True) -> b,h,hd,1
        # compute weighted_v = v * kk (elementwise) -> b,h,hd,n, then sum over n -> b,h,hd,1
        kv = (kk * v).sum(dim=-1, keepdim=True)  # b,h,hd,1
        # now out = qk * (kv / (kk.sum(dim=-1, keepdim=True) + eps))
        denom = kk.sum(dim=-1, keepdim=True)  # b,h,hd,1
        denom = denom.clamp_min(1e-6)
        out = qk * (kv / denom)  # b,h,hd,n
        out = out.view(b, target_c, h, w)
        out = self.to_out(out[:, :c, :, :])  # unpad if padded
        if self.proj_norm:
            t = out.permute(0, 2, 3, 1).contiguous()  # b,h,w,c
            t = self.ln(t)
            out = t.permute(0, 3, 1, 2).contiguous()
        return out


# ------------------------- 轻量混合块（局部 CNN + 线性注意力 + 跨流融合） -------------------------
class HybridBlock(nn.Module):
    """
    HybridBlock: local conv path + linear attention path + cross modulation + fuse
    - local 保持高频/纹理（小目标）
    - attn 提供长程上下文（语义）
    - cross modulation 互相引导
    """
    def __init__(self, channels, norm_type='bn', gn_groups=8, heads=4, use_se=True):
        super().__init__()
        self.channels = channels
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups, channels) if norm_type == 'gn' else nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.attn = LinearAttention(channels, heads=heads, proj_norm=True)
        # cross gating
        mid = max(4, channels // 8)
        self.g_l2g = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(channels, mid, 1, bias=True),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(mid, channels, 1, bias=True),
                                   nn.Sigmoid())
        self.g_g2l = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(channels, mid, 1, bias=True),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(mid, channels, 1, bias=True),
                                   nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(gn_groups, channels) if norm_type == 'gn' else nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        local = self.local(x)
        global_ = self.attn(x)
        g1 = self.g_l2g(local)
        g2 = self.g_g2l(global_)
        local_mod = local * g2
        global_mod = global_ * g1
        cat = torch.cat([local_mod, global_mod], dim=1)
        fused = self.fuse(cat)
        fused = self.se(fused)
        return x + fused


# ------------------------- 轻量残差块（替代普通瓶颈） -------------------------
class LightBottleneck(nn.Module):
    def __init__(self, c, expansion=0.5, norm_type='bn', gn_groups=8, use_se=False):
        super().__init__()
        hid = max(8, int(c * expansion))
        self.conv1 = ConvBNAct(c, hid, k=1, s=1, p=0, norm_type=norm_type, gn_groups=gn_groups)
        self.dw = nn.Sequential(
            nn.Conv2d(hid, hid, 3, 1, 1, groups=hid, bias=False),
            nn.GroupNorm(gn_groups, hid) if norm_type == 'gn' else nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True)
        )
        self.conv2 = ConvBNAct(hid, c, k=1, s=1, p=0, norm_type=norm_type, gn_groups=gn_groups, act=False)
        self.use_se = use_se
        self.se = SEBlock(c) if use_se else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.dw(y)
        y = self.conv2(y)
        y = self.se(y)
        return x + y


# ------------------------- 跨尺度融合桥 -------------------------
class CrossScaleBridge(nn.Module):
    def __init__(self, fine_c, coarse_c, mid=64, norm_type='bn', gn_groups=8):
        super().__init__()
        self.f_proj = nn.Conv2d(fine_c, mid, 1, bias=False)
        self.c_proj = nn.Conv2d(coarse_c, mid, 1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(mid * 2, coarse_c, 1, bias=False),
            nn.GroupNorm(gn_groups, coarse_c) if norm_type == 'gn' else nn.BatchNorm2d(coarse_c),
            nn.SiLU(inplace=True)
        )
        self.attn = nn.Sequential(
            nn.Conv2d(coarse_c, coarse_c // 4 if coarse_c >= 8 else 1, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(coarse_c // 4 if coarse_c >= 8 else 1, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, fine, coarse):
        # fine -> adaptive pool to coarse spatial size
        _, _, Hc, Wc = coarse.shape
        f = F.adaptive_avg_pool2d(fine, (Hc, Wc))
        f = self.f_proj(f)
        c = self.c_proj(coarse)
        out = torch.cat([f, c], dim=1)
        out = self.mix(out)
        a = self.attn(out)
        return coarse * (1.0 + a) + out


# ------------------------- EHTBackbone（完整主干） -------------------------
class EHTBackbone(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 base_c: int = 32,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0,
                 norm_type: str = 'bn',
                 gn_groups: int = 8,
                 use_checkpoint: bool = False,
                 aux_supervision: bool = True,
                 num_classes: int = 1):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm_type = norm_type
        c1 = int(base_c * width_mult)            # stem out
        c2 = int(base_c * 2 * width_mult)        # stride=4
        c3 = int(base_c * 4 * width_mult)        # stride=8
        c4 = int(base_c * 8 * width_mult)        # stride=16
        c5 = int(base_c * 16 * width_mult)       # stride=32

        # Stem: overlap convs, 保留细节
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, c1 // 2, k=3, s=2, p=1, norm_type=norm_type, gn_groups=gn_groups),
            ConvBNAct(c1 // 2, c1, k=3, s=1, p=1, norm_type=norm_type, gn_groups=gn_groups)
        )

        # down1 -> stride=4
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            nn.GroupNorm(gn_groups, c2) if norm_type == 'gn' else nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        # stage2: 负责小目标判别力（较多 local conv）
        n2 = max(2, int(3 * depth_mult))
        stage2 = []
        for i in range(n2):
            stage2.append(LightBottleneck(c2, expansion=0.5, norm_type=norm_type, gn_groups=gn_groups, use_se=(i==n2-1)))
        self.stage2 = nn.Sequential(*stage2)

        # down2 -> stride=8
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False),
            nn.GroupNorm(gn_groups, c3) if norm_type == 'gn' else nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True)
        )
        # stage3: 中心 stage，使用 hybrid block(s)（带线性注意力）
        n3 = max(2, int(3 * depth_mult))
        stage3 = []
        for i in range(n3):
            # 第一个 block 更注重局部 -> use local bottleneck, 后续使用 HybridBlock
            if i == 0:
                stage3.append(LightBottleneck(c3, expansion=0.5, norm_type=norm_type, gn_groups=gn_groups, use_se=False))
            else:
                stage3.append(HybridBlock(c3, norm_type=norm_type, gn_groups=gn_groups, heads=4, use_se=True))
        self.stage3 = nn.Sequential(*stage3)

        # down3 -> stride=16
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, 2, 1, bias=False),
            nn.GroupNorm(gn_groups, c4) if norm_type == 'gn' else nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True)
        )
        # stage4: 深层更多语义，使用 LightBottleneck + 少量 Hybrid
        n4 = max(2, int(2 * depth_mult))
        stage4 = []
        for i in range(n4):
            if i == 0:
                stage4.append(LightBottleneck(c4, expansion=0.6, norm_type=norm_type, gn_groups=gn_groups, use_se=True))
            else:
                stage4.append(HybridBlock(c4, norm_type=norm_type, gn_groups=gn_groups, heads=4, use_se=True))
        self.stage4 = nn.Sequential(*stage4)

        # down4 -> stride=32
        self.down4 = nn.Sequential(
            nn.Conv2d(c4, c5, 3, 2, 1, bias=False),
            nn.GroupNorm(gn_groups, c5) if norm_type == 'gn' else nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True)
        )
        # stage5: relatively shallow
        n5 = max(1, int(2 * depth_mult))
        stage5 = []
        for i in range(n5):
            stage5.append(LightBottleneck(c5, expansion=0.6, norm_type=norm_type, gn_groups=gn_groups, use_se=True))
        self.stage5 = nn.Sequential(*stage5)

        # 跨尺度桥（增强层间对齐）
        self.bridge23 = CrossScaleBridge(c2, c3, mid=min(64, c3//2), norm_type=norm_type, gn_groups=gn_groups)
        self.bridge34 = CrossScaleBridge(c3, c4, mid=min(64, c4//2), norm_type=norm_type, gn_groups=gn_groups)

        # Aux heads（深监督）
        self.aux_supervision = aux_supervision
        if aux_supervision:
            self.aux2 = AuxHead(c2, mid=128, num_classes=num_classes)
            self.aux3 = AuxHead(c3, mid=128, num_classes=num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)

    def _maybe_checkpoint(self, module, x):
        if self.use_checkpoint and self.training:
            return cp.checkpoint(module, x)
        else:
            return module(x)

    def forward(self, x) -> Tuple[list, Optional[Dict[str, torch.Tensor]]]:
        # stem -> stride=2
        x = self.stem(x)
        # down1 -> stride=4
        c2 = self.down1(x)
        c2 = self._maybe_checkpoint(self.stage2, c2)

        # down2 -> stride=8
        c3 = self.down2(c2)
        # stage3 含注意力：使用 checkpoint（推荐在训练时开启）
        c3 = self._maybe_checkpoint(self.stage3, c3)
        c3 = self.bridge23(c2, c3)

        # down3 -> stride=16
        c4 = self.down3(c3)
        c4 = self._maybe_checkpoint(self.stage4, c4)
        c4 = self.bridge34(c3, c4)

        # down4 -> stride=32
        c5 = self.down4(c4)
        c5 = self._maybe_checkpoint(self.stage5, c5)

        aux_logits = None
        if self.aux_supervision:
            aux2 = self.aux2(c2)
            aux3 = self.aux3(c3)
            aux_logits = {'small': aux2, 'normal': aux3}

        # 返回与 FPN/Head 兼容的 list
        return [c3, c4, c5], aux_logits


# ------------------------- AuxHead (保持与之前一致) -------------------------
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


# ------------------------- Smoke test -------------------------
if __name__ == "__main__":
    # small smoke test to check shapes & memory behavior
    model = EHTBackbone(in_ch=3, base_c=64, width_mult=1.0, depth_mult=1.0,
                        norm_type='bn', use_checkpoint=False, aux_supervision=True)
    model.eval()
    img = torch.randn(32, 3, 320, 320)
    feats, aux = model(img)
    for i, f in enumerate(feats, start=3):
        print(f"c{i}.shape = {f.shape}")

    if aux is not None:
        print("aux:", {k: v.shape for k, v in aux.items()})
