import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------
# SE注意力模块
# ---------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


# ---------------------
# 基础卷积块：Conv -> GN -> SiLU
# ---------------------
def ConvBNAct(in_channels, out_channels, k=3, s=1, p=1, g=1, d=1, norm="GN"):
    layers = [nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p, dilation=d, groups=g, bias=False)]
    if norm == "GN":
        layers.append(nn.GroupNorm(max(1, out_channels // 8), out_channels))
    elif norm == "BN":
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------
# 分类头
# ---------------------
class ClsHead(nn.Module):
    def __init__(self, hidden_dim, cls_num):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 3, 1, p=1, d=1, g=1, norm="GN"),
            ConvBNAct(hidden_dim, hidden_dim, 3, 1, p=2, d=2, g=1, norm="GN"),  # 膨胀卷积
            SEBlock(hidden_dim),
            nn.Conv2d(hidden_dim, cls_num, 1)
        )
        self.last_layer = self.block[-1]  # 增加方便访问属性

    def forward(self, x):
        return self.block(x)


# ---------------------
# 回归头
# ---------------------
class RegHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 3, 1, p=1, d=1, g=1, norm="GN"),
            ConvBNAct(hidden_dim, hidden_dim, 3, 1, p=2, d=2, g=1, norm="GN"),  # 膨胀卷积
            SEBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),  # 通道融合
            nn.Conv2d(hidden_dim, 4, 1)  # 输出回归框
        )
        self.last_layer = self.block[-1]  # 增加方便访问属性

    def forward(self, x):
        return self.block(x)


# ---------------------
# 精修头（高分辨率 P0 层）
# ---------------------
class RefineHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(hidden_dim + 4, hidden_dim, 3, 1, p=1, d=1, g=1, norm="GN"),
            ConvBNAct(hidden_dim, hidden_dim, 3, 1, p=2, d=2, g=1, norm="GN"),  # dilated conv
            SEBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 4, 1)
        )
        self.last_layer = self.block[-1]  # 增加方便访问属性

    def forward(self, x):
        return self.block(x)


# ---------------------
# 测试示例
# ---------------------
if __name__ == "__main__":
    hidden_dim = 128
    cls_num = 5  # 茶芽分类数量

    cls_head = ClsHead(hidden_dim, cls_num)
    reg_head = RegHead(hidden_dim)
    refine_head = RefineHead(hidden_dim)

    x = torch.randn(1, hidden_dim, 80, 80)  # 主特征图输入
    reg_feat = torch.randn(1, 4, 80, 80)  # 上一层回归输出，拼接给 refine_head

    cls_out = cls_head(x)  # [1, cls_num, 80, 80]
    reg_out = reg_head(x)  # [1, 4, 80, 80]
    refine_in = torch.cat([x, reg_feat], dim=1)  # 拼接高分辨率回归特征
    refine_out = refine_head(refine_in)  # [1, 4, 80, 80]

    print("分类输出：", cls_out.shape)
    print("回归输出：", reg_out.shape)
    print("精修输出：", refine_out.shape)
