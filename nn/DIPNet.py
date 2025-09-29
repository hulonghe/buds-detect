import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Dynamic Interaction Pyramid Network（DIPNet）
一个融合了图神经网络思想、动态路由、稀疏连接和注意力机制的结构，作为 FPN 的替代。
"""


class EdgeAttention(nn.Module):
    def __init__(self, num_feats, channels):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_feats)])

    def forward(self, feats):
        # feats 是 list of [B, C, H, W]，已经统一到相同 channels
        max_h = max([f.shape[2] for f in feats])
        max_w = max([f.shape[3] for f in feats])

        proj_feats = []
        for i, f in enumerate(feats):
            f_proj = self.proj[i](f)
            if f_proj.shape[2] != max_h or f_proj.shape[3] != max_w:
                f_proj = F.interpolate(f_proj, size=(max_h, max_w), mode='nearest')
            proj_feats.append(f_proj)

        B, _, H, W = proj_feats[0].shape
        N = len(proj_feats)
        attn_matrix = torch.zeros(B, N, N, device=feats[0].device)

        for i in range(N):
            for j in range(N):
                f_i = proj_feats[i].flatten(2)  # [B, C, HW]
                f_j = proj_feats[j].flatten(2)
                attn = F.cosine_similarity(f_i, f_j, dim=1)  # [B, HW]
                attn_matrix[:, i, j] = attn.mean(-1)  # [B]

        return attn_matrix.softmax(dim=-1)  # [B, N, N]


class DIPBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.num_feats = len(in_channels_list)
        self.out_channels = out_channels

        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.update = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in range(self.num_feats)
        ])
        self.edge_attn = EdgeAttention(num_feats=self.num_feats, channels=out_channels)

    def forward(self, feats):
        feats = [proj(f) for proj, f in zip(self.proj, feats)]  # -> [B, out_channels, H, W]

        B, C, H, W = feats[0].shape
        n = len(feats)
        attn = self.edge_attn(feats)  # [B, n, n]

        updated = []
        for i in range(n):
            target = feats[i]
            agg = 0
            for j in range(n):
                src = feats[j]
                if src.shape[-2:] != target.shape[-2:]:
                    src = F.interpolate(src, size=target.shape[-2:], mode='bilinear', align_corners=False)
                weight = attn[:, i, j].view(B, 1, 1, 1)
                agg += weight * src
            updated.append(self.update[i](agg))
        return updated


class DIPNet(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(DIPBlock(in_channels_list, out_channels))  # 第一个 block 使用原始输入通道
        for _ in range(1, num_blocks):
            self.blocks.append(DIPBlock([out_channels] * len(in_channels_list), out_channels))  # 后续统一为 out_channels

    def forward(self, feats):
        for block in self.blocks:
            feats = block(feats)
        return feats


# ================== 示例运行 ==================
if __name__ == '__main__':
    P3 = torch.randn(2, 128, 80, 80)
    P4 = torch.randn(2, 256, 40, 40)
    P5 = torch.randn(2, 512, 20, 20)

    model = DIPNet(in_channels_list=[128, 256, 512], out_channels=160, num_blocks=2)
    outputs = model([P3, P4, P5])

    for i, out in enumerate(outputs):
        print(f"P{i + 3} shape: {out.shape}")
