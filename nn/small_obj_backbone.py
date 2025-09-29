import gc
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        p = k // 2
        self.dw = ConvBNAct(in_c, in_c, k=k, s=s, p=p, g=in_c)
        self.pw = ConvBNAct(in_c, out_c, k=1, s=1, p=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(4, c // r), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, c // r), c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class CBAM(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(c, max(4, c // r), 1, bias=False), nn.SiLU(True),
            nn.Conv2d(max(4, c // r), c, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ch
        xs = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        sp = torch.sigmoid(self.spatial(xs))
        return x * sp


class BlurPool(nn.Module):
    """抗混叠下采样，减少小目标细节损失"""

    def __init__(self, channels, stride=2):
        super().__init__()
        assert stride in [2]
        filt = torch.tensor([1., 2., 1.], dtype=torch.float32)
        kernel = (filt[:, None] * filt[None, :])[None, None, ...]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel.repeat(channels, 1, 1, 1))
        self.stride = stride
        self.groups = channels
        self.pad = (1, 1, 1, 1)

    def forward(self, x):
        return F.conv2d(F.pad(x, self.pad, mode='reflect'),
                        self.kernel, stride=self.stride, groups=self.groups)


# --------- 轻量 ASPP（不降采样，扩感受野） ---------
class LiteASPP(nn.Module):
    def __init__(self, c, out_c):
        super().__init__()
        self.branch1 = ConvBNAct(c, out_c // 4, k=1, s=1, p=0)
        self.branch2 = ConvBNAct(c, out_c // 4, k=3, s=1, p=1, g=1)
        self.branch3 = nn.Conv2d(c, out_c // 4, 3, 1, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c // 4)
        self.branch4 = nn.Conv2d(c, out_c // 4, 3, 1, padding=3, dilation=3, bias=False)
        self.bn4 = nn.BatchNorm2d(out_c // 4)
        self.act = nn.SiLU(inplace=True)
        self.merge = ConvBNAct(out_c, out_c, k=1, s=1, p=0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.act(self.bn3(self.branch3(x)))
        b4 = self.act(self.bn4(self.branch4(x)))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.merge(out)


# --------- CSP 残差块 ---------
class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True, e=0.5, use_dw=False):
        super().__init__()
        hid = int(c * e)
        Conv = DepthwiseSeparableConv if use_dw else ConvBNAct
        self.cv1 = ConvBNAct(c, hid, k=1, s=1, p=0)
        self.cv2 = Conv(hid, c, k=3, s=1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CSPBlock(nn.Module):
    def __init__(self, c, n=3, e=0.5, use_dw=False, attn=None):
        super().__init__()
        hid = int(c * e)
        self.cv1 = ConvBNAct(c, hid, 1, 1, 0)
        self.cv2 = ConvBNAct(c, hid, 1, 1, 0)
        blocks = []
        for _ in range(n):
            blocks.append(Bottleneck(hid, shortcut=True, e=1.0, use_dw=use_dw))
        self.blocks = nn.Sequential(*blocks)
        self.cv3 = ConvBNAct(hid * 2, c, 1, 1, 0)
        self.attn = attn(c) if attn is not None else nn.Identity()

    def forward(self, x):
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        y = torch.cat([y1, y2], dim=1)
        y = self.cv3(y)
        return self.attn(y)


# --------- SPPF ---------
class SPPF(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        hid = c // 2
        self.cv1 = ConvBNAct(c, hid, 1, 1, 0)
        self.cv2 = ConvBNAct(hid * 4, c, 1, 1, 0)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.k // 2)
        y2 = F.max_pool2d(y1, kernel_size=self.k, stride=1, padding=self.k // 2)
        y3 = F.max_pool2d(y2, kernel_size=self.k, stride=1, padding=self.k // 2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------- 修改：使 ConvBNAct 支持 GN/BN ----------
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


# （保持 DepthwiseSeparableConv、SE、CBAM、BlurPool、LiteASPP、Bottleneck、CSPBlock、SPPF 等原实现）
# 仅确保 ConvBNAct 的构造调用传入 norm_type 参数（下面 backbone 使用）

# ---------- 新增：中间辅助头（用于深监督） ----------
class AuxHead(nn.Module):
    """轻量的辅助头：GlobalPool -> FC -> logit（可用于分类性/回归性辅助损失）"""

    def __init__(self, in_c, mid=128, num_classes=1):  # 对检测主干，aux用于增强特征学习，不作为最终检测head
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_c, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(mid, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


# ---------- 主干修改版 ----------
class SmallObjBackbone(nn.Module):
    def __init__(self, in_ch=3, base_c=64, width_mult=1.0, depth_mult=1.0,
                 use_dw=False, attn_type='cbam', norm_type='bn', gn_groups=8,
                 use_pretrained_stem=False, freeze_stem=False, aux_supervision=True,
                 num_classes=1):
        super().__init__()
        c1 = int(base_c * width_mult)  # ~64
        c2 = int(base_c * 2 * width_mult)  # ~128
        c3 = int(base_c * 4 * width_mult)  # ~256
        c4 = int(base_c * 8 * width_mult)  # ~512

        BlockAttention = CBAM if attn_type == 'cbam' else (lambda c: SE(c))

        # ---- Stem（依旧轻量，但支持替换为预训练stem） ----
        if use_pretrained_stem:
            # 这只是接口示例：你可以在训练脚本里加载 torchvision 的 pretrained MobileNet/EfficientNet 并把前 N 层替换到 self.stem
            # 这里保留原始小型三层 stem 以便直接使用
            pass

        self.stem = nn.Sequential(
            ConvBNAct(in_ch, c1 // 2, 3, 2, norm_type=norm_type, gn_groups=gn_groups),  # stride 2
            ConvBNAct(c1 // 2, c1 // 2, 3, 1, norm_type=norm_type, gn_groups=gn_groups),
            ConvBNAct(c1 // 2, c1, 3, 1, norm_type=norm_type, gn_groups=gn_groups),
        )
        self.down1 = nn.Sequential(BlurPool(c1), ConvBNAct(c1, c1, 1, 1, 0, norm_type=norm_type, gn_groups=gn_groups))

        # C2
        n2 = max(2, int(3 * depth_mult))
        self.c2 = nn.Sequential(
            CSPBlock(c1, n=n2, e=0.5, use_dw=use_dw, attn=BlockAttention),
            LiteASPP(c1, c1),
        )

        # C3
        n3 = max(2, int(3 * depth_mult))
        self.down2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2, norm_type=norm_type, gn_groups=gn_groups), )
        self.c3 = nn.Sequential(
            CSPBlock(c2, n=n3, e=0.5, use_dw=use_dw, attn=BlockAttention),
            LiteASPP(c2, c2),
        )

        # C4
        n4 = max(2, int(3 * depth_mult))
        self.down3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2, norm_type=norm_type, gn_groups=gn_groups), )
        self.c4 = nn.Sequential(
            CSPBlock(c3, n=n4, e=0.5, use_dw=use_dw, attn=BlockAttention),
            SPPF(c3, k=5),
        )

        # C5
        n5 = max(1, int(2 * depth_mult))
        self.down4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2, norm_type=norm_type, gn_groups=gn_groups), )
        self.c5 = nn.Sequential(
            CSPBlock(c4, n=n5, e=0.5, use_dw=use_dw, attn=BlockAttention),
        )

        # 辅助头（深监督）
        self.aux_supervision = aux_supervision
        if aux_supervision:
            # 输出维度可按需设定（这里假设二分类aux，若用于别的任务请改）
            self.aux2 = AuxHead(c1, mid=128, num_classes=num_classes)  # 对应 c2 层（小目标关键）
            self.aux3 = AuxHead(c2, mid=128, num_classes=num_classes)  # 对应 c3
            # 不强制对 c4/c5 加 aux（避免过多参数），按需可加

        # 可选冻结 stem
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False

        # 初始化
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN/GN 已默认初始化
                pass
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)  # stride 2
        x = self.down1(x)  # stride 4
        c2 = self.c2(x)  # stride 4 (小目标层)

        x = self.down2(c2)  # stride 8
        c3 = self.c3(x)

        x = self.down3(c3)  # stride 16
        c4 = self.c4(x)

        x = self.down4(c4)  # stride 32
        c5 = self.c5(x)

        aux_logits = None
        if self.aux_supervision:
            aux2 = self.aux2(c2)  # scalar/logit per image
            aux3 = self.aux3(c3)
            aux_logits = {'small': aux2, 'normal': aux3}

        # 关键修正：返回包含 c2（便于检测头获取高分辨率信息）
        return [c2, c3, c4], aux_logits


def get_memory_usage():
    """获取显存和CPU内存占用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
    else:
        allocated, reserved = 0, 0

    ram = psutil.Process().memory_info().rss / 1024 ** 2
    return allocated, reserved, ram


@torch.no_grad()
def test_backbone_memory_repeated(batch_size=8, img_size=320, device="cuda", repeat=10):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = SmallObjBackbone(in_ch=3).to(device)

    for i in range(1, repeat + 1):
        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
        outputs, _ = model(dummy_input)

        print(f"\n=== Run {i}/{repeat} ===")
        for j, feat in enumerate(outputs, start=1):
            print(f"C{j}: {feat.shape}")

        allocated, reserved, ram = get_memory_usage()
        print(f"显存 allocated: {allocated:.2f} MB")
        print(f"显存 reserved : {reserved:.2f} MB")
        print(f"CPU RAM used  : {ram:.2f} MB")


if __name__ == "__main__":
    test_backbone_memory_repeated(batch_size=32, img_size=320, device="cuda", repeat=100)
