import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights,
    ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

from nn.FPN import DynamicFPN
from nn.depthwise_FPN import DynamicDepthwiseFPN
from nn.head_model import ClsHead, RegHead, RefineHead
from nn.small_obj_backbone import SmallObjBackbone
from nn.tea_bud_FPN import TeaBudFPN


def init_all_heads(model, cls_num, prior_prob=0.01):
    """
    初始化检测器的分类 head、回归 head、refine head。
    结构为 Conv + GroupNorm + SiLU。
    特别地，对分类 head 最后一层 bias 做先验初始化，以便 focal loss 更稳定。
    """

    def init_head(head, is_cls=False, is_reg=False):
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                # 中间层卷积
                if (is_cls and m.out_channels != cls_num) or (is_reg and m.out_channels != 4):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                # 分类 head 最后一层卷积
                elif is_cls and m.out_channels == cls_num:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        # focal loss 先验偏置: bias = -log((1-p)/p)
                        nn.init.constant_(m.bias, float(-torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))))

                # 回归 head 最后一层卷积
                elif is_reg and m.out_channels == 4:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    # 遍历并初始化所有 head
    for name, module in model.named_modules():
        if "score_head" in name:
            init_head(module, is_cls=True)
        elif "reg_head" in name or "refine_head" in name:
            init_head(module, is_reg=True)


class OptimizedDynamicBoxDetector(nn.Module):
    def __init__(self,
                 in_channels=3, in_dim=320,
                 hidden_dim=128,
                 cls_num=1,
                 num_layers=1,
                 nhead=4,
                 once_embed=False,
                 is_split_trans=False,  # 是否分开 transformer，默认合并更适合小目标
                 is_fpn=True,
                 topk=300,
                 dropout=0.1,
                 backbone_type='smallobjnet',
                 device='cuda',
                 freeze_backbone_layers=True,
                 max_span_cells=4  # 解码时宽高最多跨的格子数，增强小目标表现
                 ):
        super().__init__()
        self.is_split_trans = is_split_trans
        self.once_embed = once_embed
        self.topk = topk
        self.device = device
        self.backbone_type = backbone_type
        self.max_span_cells = float(max_span_cells)

        # ---------------- Backbone ---------------- #
        if backbone_type == 'smallobjnet':
            backbone = SmallObjBackbone(in_ch=in_channels)
            # 统一成 ModuleDict 接口
            self.backbone = nn.ModuleDict({
                'conv1': nn.Identity(),  # 已在 forward 内处理，不需要
                'layer1': nn.Identity(),
                'layer2': nn.Identity(),
                'layer3': nn.Identity(),
                'layer4': nn.Identity(),
                '_impl': backbone
            })
        else:
            if backbone_type == 'resnet18':
                backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif backbone_type == 'resnet34':
                backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            elif backbone_type == 'resnet50':
                backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            elif backbone_type == 'resnet101':
                backbone = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            elif backbone_type == 'resnet152':
                backbone = models.resnet152(weights=ResNet152_Weights.DEFAULT)
            else:
                raise ValueError("不支持的 backbone 类型")
            self.backbone = nn.ModuleDict({
                'conv1': nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
                'layer1': backbone.layer1,  # C2
                'layer2': backbone.layer2,  # C3
                'layer3': backbone.layer3,  # C4
                'layer4': backbone.layer4  # C5
            })

        # 是否冻结浅层
        if freeze_backbone_layers:
            for name, module in self.backbone.items():
                if name in ['conv1', 'layer1']:
                    for p in module.parameters():
                        p.requires_grad = False

        # ---------------- 动态推断特征层 ---------------- #
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, in_dim, in_dim)
            feats = self._extract_features(dummy)  # 自动获取特征层
            in_channels_list = [f.shape[1] for f in feats]
            print(f"{'in_channels_list':<24}: {in_channels_list}")
            for feat in feats:
                print(f"{'in_channels_feat':<24}: {feat.shape}")

        # ---------------- FPN ---------------- #
        self.fpn = DynamicFPN(in_channels_list, hidden_dim)
        # self.fpn = TeaBudFPN(in_channels_list, hidden_dim, dropout=dropout,use_cbam=True)

        # 为不同尺度建立 scale embedding
        self.num_levels = len(in_channels_list)
        self.scale_embed = nn.Embedding(self.num_levels, hidden_dim)

        # ---------------- 位置编码 ---------------- #
        H2, W2 = feats[0].shape[-2:]
        self.row_embed = nn.Embedding(H2, hidden_dim // 2)
        self.col_embed = nn.Embedding(W2, hidden_dim // 2)

        # ---------------- Transformer ---------------- #
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 动态创建 head
        for i in range(self.num_levels):
            setattr(self, f"score_head{i}", ClsHead(hidden_dim, cls_num))
            setattr(self, f"reg_head{i}", RegHead(hidden_dim))

        # refine head：放在最高分辨率的第0层
        self.refine_head0 = RefineHead(hidden_dim)

        # 初始化
        init_all_heads(self, cls_num)

    def _extract_features(self, x):
        if self.backbone_type == 'smallobjnet':
            return self.backbone['_impl'](x)  # 直接返回 [C2,C3,C4,C5]
        else:
            x = self.backbone['conv1'](x)
            c2 = self.backbone['layer1'](x)
            c3 = self.backbone['layer2'](c2)
            c4 = self.backbone['layer3'](c3)
            c5 = self.backbone['layer4'](c4)
            return [c3, c4, c5]

    def forward(self, x):
        B = x.size(0)
        device = x.device
        features = self._extract_features(x)
        feats = self.fpn(features)  # 输出 N 层特征

        spatial_shapes = [f.shape[-2:] for f in feats]
        f_flat_list, pos_list = [], []

        # 位置编码
        for idx, (feat, (H, W)) in enumerate(zip(feats, spatial_shapes)):
            row_pos = self.row_embed(torch.arange(H, device=device))
            col_pos = self.col_embed(torch.arange(W, device=device))
            pos = torch.cat([
                row_pos.unsqueeze(1).repeat(1, W, 1),
                col_pos.unsqueeze(0).repeat(H, 1, 1)
            ], dim=-1)  # H x W x hidden
            pos = pos.flatten(0, 1).unsqueeze(0).repeat(B, 1, 1)

            scale_code = self.scale_embed(torch.full((B, H * W), idx, dtype=torch.long, device=device))
            pos = pos + scale_code

            f_flat = feat.flatten(2).permute(0, 2, 1)
            f_flat_list.append(f_flat)
            pos_list.append(pos)

        # transformer 编码
        if self.is_split_trans:
            feats_trans = [self.transformer(f_flat + pos) for f_flat, pos in zip(f_flat_list, pos_list)]
        else:
            f_all = torch.cat(f_flat_list, dim=1)
            pos_all = torch.cat(pos_list, dim=1)
            f_trans = self.transformer(f_all + pos_all)
            split_sizes = [H * W for H, W in spatial_shapes]
            feats_trans = torch.split(f_trans, split_sizes, dim=1)

        # 还原回 feature map
        feat_ts = [
            f.permute(0, 2, 1).reshape(B, -1, H, W)
            for f, (H, W) in zip(feats_trans, spatial_shapes)
        ]

        # ---------------- Head 输出 ---------------- #
        scores, boxes = [], []
        for i, feat_t in enumerate(feat_ts):
            score = getattr(self, f"score_head{i}")(feat_t)
            reg = getattr(self, f"reg_head{i}")(feat_t)

            # refine 只在第0层
            if i == 0:
                delta = self.refine_head0(torch.cat([feat_t, reg], dim=1))
                reg = reg + delta

            box = self.decode_boxes(reg, feat_t, max_span_cells=self.max_span_cells)

            s_flat = score.flatten(2).permute(0, 2, 1)
            b_flat = box.flatten(2).permute(0, 2, 1)

            scores.append(s_flat)
            boxes.append(b_flat)

        score = torch.cat(scores, dim=1)
        box = torch.cat(boxes, dim=1)
        return score, box

    def decode_boxes(self, reg, feat, max_span_cells=4.0):
        """
        解码回归框：
          - tx,ty: tanh 限制在 cell 内偏移
          - w,h: sigmoid 缩放到 [0, max_span_cells * cell_size]
        输出归一化坐标 (0~1)
        """
        B, _, H, W = reg.shape
        device = reg.device

        tx = torch.tanh(reg[:, 0])
        ty = torch.tanh(reg[:, 1])

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        cell_w = 1.0 / float(W)
        cell_h = 1.0 / float(H)

        tw = reg[:, 2]
        th = reg[:, 3]
        w = torch.sigmoid(tw) * (max_span_cells * cell_w)
        h = torch.sigmoid(th) * (max_span_cells * cell_h)

        cx = (tx + grid[:, 0]).clamp(0., 1.)
        cy = (ty + grid[:, 1]).clamp(0., 1.)

        x_min = (cx - w / 2).clamp(0., 1.)
        y_min = (cy - h / 2).clamp(0., 1.)
        x_max = (cx + w / 2).clamp(0., 1.)
        y_max = (cy + h / 2).clamp(0., 1.)

        return torch.stack([x_min, y_min, x_max, y_max], dim=1)
