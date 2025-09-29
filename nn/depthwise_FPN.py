import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.cbam import CBAM  # 你已有的 CBAM 模块


# ---------------- DepthwiseSeparableConv + 可选 CBAM ----------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 use_cbam=True, dropout=0.0, norm="GN", dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.attn = CBAM(out_channels) if use_cbam else nn.Identity()
        self.norm = nn.GroupNorm(max(1, out_channels // 8), out_channels) if norm == "GN" else nn.BatchNorm2d(
            out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.attn(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.act(x)


class LocalRefineC0(nn.Module):
    """
    面向小目标的局部增强块：
    - DWConv + (可选)CBAM 抑制背景
    - 膨胀 DWConv(3x3, dilation=2) 扩展有效感受野但不过度糊边
    - Pointwise 1x1 融合
    - 可学习门控 gamma（初始0）做残差融合，稳定训练
    """

    def __init__(self, dim, dropout=0.1, norm="GN", use_cbam=True):
        super().__init__()
        norm2d = (lambda c: nn.GroupNorm(max(1, c // 8), c)) if norm == "GN" else nn.BatchNorm2d
        self.block = nn.Sequential(
            DepthwiseSeparableConv(dim, dim, use_cbam=use_cbam, dropout=dropout, norm=norm),
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False),
            norm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        # 学习到“该不该强化、强化多少”
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x + self.gamma * self.block(x)


class DynamicDepthwiseFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout=0.1,
                 is_fpn=True, use_cbam=True, norm="GN",
                 init_weights=(0.6, 0.25, 0.15)):
        """
        feats 输入顺序 = [c0, c1, c2]，对应原来的 [C3, C4, C5]
        外层的 refine_head3/4/5 仍由你的检测 head 挂接使用；此处不改。
        """
        super().__init__()
        assert len(in_channels_list) == 3, "当前实现按三层金字塔设计（c0/c1/c2）"
        self.is_fpn = is_fpn
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.norm = norm

        # 1) lateral 对齐
        self.lateral_convs = nn.ModuleList([
            DepthwiseSeparableConv(c, out_channels, use_cbam=use_cbam, dropout=dropout, norm=norm)
            for c in in_channels_list
        ])

        # 2) FPN 融合卷积
        self.fpn_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, use_cbam=False, dropout=0.0, norm=norm)
            for _ in range(self.num_levels)
        ])

        # 3) 仅 c0 的局部增强（内置 refine）
        self.c0_refine = LocalRefineC0(out_channels, dropout=dropout, norm=norm, use_cbam=True)

        # 4) 输出正则
        self.out_norm = nn.GroupNorm(max(1, out_channels // 8), out_channels) if norm == "GN" else nn.BatchNorm2d(
            out_channels)
        self.out_dropout = nn.Dropout2d(p=dropout)

        # 5) 层级权重（可学习），初始化偏向 c0
        w = torch.tensor(init_weights, dtype=torch.float32)
        assert w.numel() == self.num_levels, "init_weights 长度需与金字塔层数一致"
        self.level_weights = nn.Parameter(w)

    def forward(self, feats):
        """
        feats: [c0, c1, c2]  (B, C_i, H/8, W/8), (H/16), (H/32)
        return: [p0_refined, p1, p2] 供后续 cls/reg/外层 refine_head* 使用
        """
        assert len(feats) == self.num_levels

        # lateral 对齐
        laterals = [self.lateral_convs[i](f) for i, f in enumerate(feats)]

        # top-down 融合
        if self.is_fpn:
            for i in range(self.num_levels - 1, 0, -1):
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False
                )
            outs = [self.fpn_convs[i](laterals[i]) + laterals[i] for i in range(self.num_levels)]
        else:
            outs = laterals

        # 内部仅增强 c0（小目标层）
        outs[0] = self.c0_refine(outs[0])

        # 归一化权重（softmax），放大 c0 贡献但保持总能量可控
        lw = torch.softmax(self.level_weights, dim=0)

        # 输出正则 + 按层缩放（让下游 head 感知到“层重要性”）
        outs = [self.out_dropout(self.out_norm(outs[i])) * lw[i] for i in range(self.num_levels)]

        return outs  # 供外层 cls/reg + 你的 refine_head3/4/5 使用


# ---------------- 测试 ----------------
if __name__ == "__main__":
    hidden_dim = 128
    feats = [
        torch.randn(1, 128, 80, 80),  # P1
        torch.randn(1, 256, 40, 40),  # P2
        torch.randn(1, 512, 20, 20)  # P3
    ]
    fpn = DynamicDepthwiseFPN(
        in_channels_list=[128, 256, 512],
        out_channels=hidden_dim,
        dropout=0.1,
        is_fpn=True,
        use_cbam=True,
        norm="GN"
    )
    out_feats = fpn(feats)
    for i, f in enumerate(out_feats):
        print(f"P{i} shape:", f.shape)
