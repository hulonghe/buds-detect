"""
tea_bud_fpn_replace.py
替代 FPN 的多尺度融合模块（专为小目标/茶芽设计）
接口：
    model = TeaBudFPNReplace(in_channels_list, out_channels, repeats=2, use_cbam=False, small_branch=False)
    fused_feats = model(feats)  # fused_feats: list of tensors, same order/resolutions as feats
    # 若 small_branch=True，则返回 (fused_feats, small_feat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选 CBAM：若没有 nn.cbam，会自动降级为 Identity
try:
    from nn.cbam import CBAM

    HAS_CBAM = True
except Exception:
    CBAM = None
    HAS_CBAM = False


# --------------------------
# 基础轻量卷积（Depthwise Separable + GN + SiLU）
# --------------------------
class SeparableConvGN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, use_cbam=False, dropout=0.0):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=padding, dilation=dilation, groups=in_ch,
                            bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.attn = CBAM(out_ch) if (use_cbam and HAS_CBAM) else nn.Identity()
        self.gn = nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.attn(x)
        x = self.gn(x)
        x = self.drop(x)
        return self.act(x)


# --------------------------
# BiFPN 风格的可学习权重归一化（支持任意输入数量）
# --------------------------
class FastNormalizedWeights(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs):
        # inputs: list of tensors（shape must match）
        assert len(inputs) == self.w.shape[0]
        w = F.relu(self.w)
        denom = torch.sum(w) + 1e-6
        w = w / denom
        out = 0
        for i, inp in enumerate(inputs):
            out = out + w[i] * inp
        return out


# --------------------------
# Cross-scale 全局上下文补偿（轻量）
# --------------------------
class CrossScaleContext(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)  # [B,C,1,1]
        g = self.act(self.fc1(g))
        g = torch.sigmoid(self.fc2(g))
        return x * g


# --------------------------
# Gated Fusion：在原始尺度与融合尺度之间做门控
# --------------------------
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = max(1, channels // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GroupNorm(max(1, mid // 8), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1)
        )

    def forward(self, orig, fused):
        # 简单统计融合输入以获得门控因子
        x = 0.5 * (orig + fused)
        g = torch.sigmoid(self.gate(x))
        return orig * (1 - g) + fused * g


# --------------------------
# 单个 BiFPNBlock：top-down + bottom-up 加权融合（支持任意 num_levels）
# 输入：list of tensors [P0 (high res), P1, ..., Pn-1 (low res)]
# 返回：list of fused tensors（同 order）
# --------------------------
class BiFPNBlock(nn.Module):
    def __init__(self, num_levels, channels, use_cbam=False, dropout=0.0):
        super().__init__()
        self.num_levels = num_levels
        self.channels = channels

        # 层间 conv（作用于每个节点的输出）
        self.node_convs = nn.ModuleList([SeparableConvGN(channels, channels, use_cbam=False, dropout=dropout)
                                         for _ in range(num_levels)])

        # top-down 融合权重与 conv（每个非最高尺度一个节点）
        self.td_weights = nn.ModuleList([FastNormalizedWeights(2) for _ in range(num_levels - 1)])
        self.td_conv = nn.ModuleList([SeparableConvGN(channels, channels, use_cbam=False, dropout=0.0)
                                      for _ in range(num_levels - 1)])

        # bottom-up 融合权重与 conv（每个非最低尺度一个节点）
        self.bu_weights = nn.ModuleList([FastNormalizedWeights(2) for _ in range(num_levels - 1)])
        self.bu_conv = nn.ModuleList([SeparableConvGN(channels, channels, use_cbam=False, dropout=0.0)
                                      for _ in range(num_levels - 1)])

    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        # inputs assumed already channel-aligned to `channels`
        lateral = [inputs[i] for i in range(self.num_levels)]

        # top-down pathway (from low-res -> high-res)
        td = lateral.copy()
        for i in range(self.num_levels - 1, 0, -1):
            up = F.interpolate(td[i], size=td[i - 1].shape[-2:], mode='bilinear', align_corners=False)
            fused = self.td_weights[i - 1]([lateral[i - 1], up])
            td[i - 1] = self.td_conv[i - 1](fused)  # conv on fused

        # bottom-up pathway (from high-res -> low-res)
        bu = td.copy()
        for i in range(0, self.num_levels - 1):
            down = F.max_pool2d(bu[i], kernel_size=2)  # downsample by 2
            fused = self.bu_weights[i]([bu[i + 1], down])
            bu[i + 1] = self.bu_conv[i](fused)

        # per-node final conv + residual
        out = []
        for i in range(self.num_levels):
            node = self.node_convs[i](bu[i]) + lateral[i]  # residual connection to preserve high-res detail
            out.append(node)
        return out


# --------------------------
# 可选 small-target 特征合成：把若干低层上采样并拼到 P0（仅当 small_branch=True 时启用）
# 返回一个 high-res 特征（与 P0 同分辨率）
# --------------------------
class SmallTargetComposer(nn.Module):
    def __init__(self, channels, num_up=2, use_cbam=False, dropout=0.0):
        """
        num_up: 使用低于 P0 的最多 num_up 层（P1..Pnum_up）上采样并拼接到 P0
        """
        super().__init__()
        self.num_up = num_up
        # reduce each upsampled feature to channels before concat
        self.reduce = SeparableConvGN(channels, channels, use_cbam=use_cbam, dropout=dropout)
        self.fuse = SeparableConvGN(channels * (num_up + 1), channels, use_cbam=use_cbam, dropout=dropout)
        self.context = CrossScaleContext(channels)

    def forward(self, feats):
        # feats: [P0, P1, P2, ...]
        p0 = feats[0]
        to_cat = [p0]
        for i in range(1, min(len(feats), self.num_up + 1)):
            feat = feats[i]
            feat = F.interpolate(feat, size=p0.shape[-2:], mode='bilinear', align_corners=False)
            feat = self.reduce(feat)
            to_cat.append(feat)
        x = torch.cat(to_cat, dim=1)
        x = self.fuse(x)
        x = self.context(x)
        return x


# --------------------------
# 最终替换模块：TeaBudFPN
# - 只输出融合后的多尺度特征（与输入层数和分辨率一致）
# - optional small_branch 可返回 single high-res small_target_feature
# --------------------------
class TeaBudFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, repeats=2, use_cbam=False,
                 small_branch=False, small_num_up=2, dropout=0.05):
        """
        in_channels_list: backbone 多尺度输出通道列表，顺序要求： [P0(highest res), P1, ..., Pn-1(lowest)]
        out_channels: 融合后每层通道数
        repeats: 堆叠多少个 BiFPNBlock
        use_cbam: 是否在 channel-align 时使用 CBAM（需 nn.cbam 可用）
        small_branch: 是否额外输出 small_target_feature（高分辨率，便于小目标检测）
        small_num_up: small branch 上采样层数（P1..Psmall_num_up -> P0）
        """
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.repeats = repeats
        self.small_branch = small_branch

        # 输入通道对齐
        self.input_align = nn.ModuleList([
            SeparableConvGN(in_ch, out_channels, use_cbam=use_cbam, dropout=dropout) for in_ch in in_channels_list
        ])

        # BiFPN blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(self.num_levels, out_channels, use_cbam=use_cbam, dropout=dropout) for _ in range(repeats)
        ])

        # 最后的 cross-scale context + gated fusion（保留细节）
        self.contexts = nn.ModuleList([CrossScaleContext(out_channels) for _ in range(self.num_levels)])
        self.gates = nn.ModuleList([GatedFusion(out_channels) for _ in range(self.num_levels)])

        # small branch composer (optional)
        if small_branch:
            self.small_composer = SmallTargetComposer(out_channels, num_up=small_num_up, use_cbam=use_cbam,
                                                      dropout=dropout)

    def forward(self, feats):
        """
        feats: list of tensors [P0, P1, ..., Pn-1]
        returns:
            if small_branch is False: fused_feats (list)
            else: (fused_feats, small_feat)
        """
        assert len(feats) == self.num_levels, "输入特征数量与初始化时 in_channels_list 不一致"

        # 1) 通道对齐
        aligned = [self.input_align[i](feats[i]) for i in range(self.num_levels)]

        # 2) 堆叠 BiFPNBlocks
        fused = aligned
        for block in self.bifpn_blocks:
            fused = block(fused)

        # 3) cross-scale context + gated fusion (与 aligned 保持细节)
        out = []
        for i in range(self.num_levels):
            ctx = self.contexts[i](fused[i])
            gated = self.gates[i](aligned[i], ctx)  # 使用 aligned 保留原始高频细节
            out.append(gated)

        if self.small_branch:
            small = self.small_composer(fused)
            return out, small
        return out


# --------------------------
# minimal sanity check (optional)
# --------------------------
if __name__ == "__main__":
    B = 2
    feats = [
        torch.randn(B, 128, 80, 80),  # P1
        torch.randn(B, 256, 40, 40),  # P2
        torch.randn(B, 512, 20, 20),  # P3
    ]
    model = TeaBudFPN(in_channels_list=[128, 256, 512], out_channels=128, repeats=2, use_cbam=False,
                             small_branch=True)
    fused, small = model(feats)
    print([f.shape for f in fused])  # 每层均为 out_channels 且空间尺寸与输入对应
    print("small:", small.shape)  # 若开启 small_branch，则返回高分辨率特征（同 P0 尺寸）
