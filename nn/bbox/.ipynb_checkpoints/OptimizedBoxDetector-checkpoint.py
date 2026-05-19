import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from nn.FPN import DynamicFPN
from nn.depthwise_FPN import DynamicDepthwiseFPN
from nn.small_obj_backbone import EHTBackboneV2


def init_all_heads(model, cls_num):
    """优化初始化：分类头 gain=1.0，回归/Refine gain=0.5"""
    def init_head(head, is_cls=False, is_reg=False):
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                gain = 1.0 if is_cls else 0.5
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    for h in [model.score_head3, model.score_head4, model.score_head5]:
        init_head(h, is_cls=True)
    for h in [model.reg_head3, model.reg_head4, model.reg_head5]:
        init_head(h, is_reg=True)
    if model.use_refine:
        for h in [model.refine_head3, model.refine_head4, model.refine_head5]:
            init_head(h, is_reg=True)


class OptimizedBoxDetector(nn.Module):
    def __init__(self,
                 in_channels=3, in_dim=256, hidden_dim=256, cls_num=1,
                 num_layers=2, nhead=8,
                 once_embed=False, is_split_trans=False, is_fpn=True,
                 head_group=1, cls_head_kernel_size=3,
                 topk=300, dropout=0.2, backbone_type='resnet50', device='cpu',
                 use_transformer=True, use_refine=True, fpn_weights=(0.5, 0.3, 0.2)):
        super().__init__()
        self.once_embed = once_embed
        self.is_split_trans = is_split_trans
        self.use_transformer = use_transformer
        self.use_refine = use_refine
        self.backbone_type = backbone_type
        self.device = device

        # ---------------- Backbone ---------------- #
        if backbone_type == 'smallobjnet':
            backbone = EHTBackboneV2(
                in_ch=in_channels, base_c=24, width_mult=1.0,
                depth_mult=1.0, norm_type='gn', use_checkpoint=False,
                aux_supervision=True, num_classes=cls_num
            )
            self.backbone = nn.ModuleDict({
                '_impl': backbone,
                'conv1': nn.Identity(), 'layer1': nn.Identity(),
                'layer2': nn.Identity(), 'layer3': nn.Identity(),
                'layer4': nn.Identity()
            })
        else:
            if backbone_type == 'resnet18':
                res = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif backbone_type == 'resnet34':
                res = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            elif backbone_type == 'resnet50':
                res = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            elif backbone_type == 'resnet101':
                res = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            else:
                raise ValueError("Unsupported backbone")
            self.backbone = nn.ModuleDict({
                'conv1': nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool),
                'layer1': res.layer1, 'layer2': res.layer2,
                'layer3': res.layer3, 'layer4': res.layer4
            })

        # 动态获取 FPN 输入通道
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, in_dim, in_dim)
            feats, _ = self._extract_features(dummy)
            in_channels_list = [f.shape[1] for f in feats]

        self.fpn = DynamicDepthwiseFPN(
            in_channels_list, hidden_dim, dropout=dropout, is_fpn=is_fpn,
            use_cbam=True, init_weights=fpn_weights
        )

        # 获取 P3 尺寸用于位置编码
        with torch.no_grad():
            fpn_p = self.fpn(feats)
            h3, w3 = fpn_p[0].shape[-2:]
            h4, w4 = fpn_p[1].shape[-2:]
            h5, w5 = fpn_p[2].shape[-2:]
            print(f"{'FPN P_h_w':<24}: P3={h3}x{w3}, P4={h4}x{w4}, P5={h5}x{w5}")

        # Embedding
        self.scale_embed = nn.Embedding(in_channels, hidden_dim)
        self.row_embed = nn.Embedding(h3, hidden_dim // 2, device=device)
        self.col_embed = nn.Embedding(w3, hidden_dim // 2, device=device)

        # ---------------- Transformer (仅 P3) ---------------- #
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---------------- Heads ---------------- #
        def make_cls_head(ks=3):
            pad = ks // 2
            return nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, ks, padding=pad, groups=head_group),
                nn.GroupNorm(hidden_dim // 4, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, cls_num, 1)
            )

        def make_reg_head():
            C = hidden_dim
            return nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, groups=4),
                nn.GroupNorm(32, C),
                nn.SiLU(),
                nn.Conv2d(C, C, 3, padding=1),
                nn.GroupNorm(32, C),
                nn.SiLU(),
                nn.Conv2d(C, 4, 1)
            )

        self.score_head3 = make_cls_head(cls_head_kernel_size)
        self.score_head4 = make_cls_head(cls_head_kernel_size)
        self.score_head5 = make_cls_head(cls_head_kernel_size)
        self.reg_head3 = make_reg_head()
        self.reg_head4 = make_reg_head()
        self.reg_head5 = make_reg_head()

        if self.use_refine:
            def make_refine():
                return nn.Sequential(
                    nn.Conv2d(hidden_dim + 4, hidden_dim, 3, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, 4, 1)
                )
            self.refine_head3 = make_refine()
            self.refine_head4 = make_refine()
            self.refine_head5 = make_refine()

        init_all_heads(self, cls_num)

    def _extract_features(self, x):
        if self.backbone_type == 'smallobjnet':
            return self.backbone['_impl'](x)
        else:
            x = self.backbone['conv1'](x)
            c2 = self.backbone['layer1'](x)
            c3 = self.backbone['layer2'](c2)
            c4 = self.backbone['layer3'](c3)
            c5 = self.backbone['layer4'](c4)
            return [c3, c4, c5], None

    def decode_boxes(self, reg, grid, stride, scale=1.0):
        """稳定的 decode: sigmoid offset + log-space size"""
        tx = (torch.sigmoid(reg[:, 0]) - 0.5) * 2 * scale
        ty = (torch.sigmoid(reg[:, 1]) - 0.5) * 2 * scale
        cx = (tx + grid[:, 0]).clamp(0., 1.)
        cy = (ty + grid[:, 1]).clamp(0., 1.)
        w = torch.exp(reg[:, 2]) * stride
        h = torch.exp(reg[:, 3]) * stride
        w = w.clamp(max=1.0)
        h = h.clamp(max=1.0)
        x_min = (cx - w / 2).clamp(0., 1.)
        y_min = (cy - h / 2).clamp(0., 1.)
        x_max = (cx + w / 2).clamp(0., 1.)
        y_max = (cy + h / 2).clamp(0., 1.)
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def forward(self, x):
        B, _, _, _ = x.shape
        device = x.device

        # Backbone + FPN
        features, _ = self._extract_features(x)
        p3, p4, p5 = self.fpn(features)

        _, C3, H3, W3 = p3.shape
        _, C4, H4, W4 = p4.shape
        _, C5, H5, W5 = p5.shape

        # stride
        stride3, stride4, stride5 = 1.0/H3, 1.0/H4, 1.0/H5

        # ===== Position Embedding =====
        pos_list, f_flat_list = [], []
        for idx, (feat, H, W) in enumerate(zip([p3, p4, p5], [H3, H4, H5], [W3, W4, W5])):
            if self.once_embed:
                row_pos = self.row_embed(torch.arange(H, device=device))
                col_pos = self.col_embed(torch.arange(W, device=device))
                pos = torch.cat([row_pos.unsqueeze(1).repeat(1,W,1),
                                 col_pos.unsqueeze(0).repeat(H,1,1)], dim=-1)
            else:
                row_pos = self.row_embed(torch.arange(H, device=device))
                col_pos = self.col_embed(torch.arange(W, device=device))
                pos = torch.cat([row_pos.unsqueeze(1).repeat(1,W,1),
                                 col_pos.unsqueeze(0).repeat(H,1,1)], dim=-1)
            pos = pos.flatten(0,1).unsqueeze(0).repeat(B,1,1)
            scale_code = self.scale_embed(torch.full((B,H*W), idx, dtype=torch.long, device=device))
            pos = pos + scale_code
            f_flat_list.append(feat.flatten(2).permute(0,2,1))
            pos_list.append(pos)

        # ===== Transformer only on P3 =====
        if self.use_transformer:
            f_trans3 = self.transformer(f_flat_list[0]+pos_list[0])
            f_trans_list = [f_trans3, f_flat_list[1], f_flat_list[2]]
        else:
            f_trans_list = f_flat_list

        # reshape
        feat3_t = f_trans_list[0].permute(0,2,1).reshape(B,C3,H3,W3)
        feat4_t = f_trans_list[1].permute(0,2,1).reshape(B,C4,H4,W4)
        feat5_t = f_trans_list[2].permute(0,2,1).reshape(B,C5,H5,W5)

        # ===== Heads =====
        score3 = self.score_head3(feat3_t)
        reg3 = self.reg_head3(feat3_t)
        score4 = self.score_head4(feat4_t)
        reg4 = self.reg_head4(feat4_t)
        score5 = self.score_head5(feat5_t)
        reg5 = self.reg_head5(feat5_t)

        # ===== Refine =====
        if self.use_refine:
            reg3 = reg3 + self.refine_head3(torch.cat([feat3_t, reg3], dim=1))
            reg4 = reg4 + self.refine_head4(torch.cat([feat4_t, reg4], dim=1))
            reg5 = reg5 + self.refine_head5(torch.cat([feat5_t, reg5], dim=1))

        # ===== Grid & decode =====
        def make_grid(H,W):
            gy,gx = torch.meshgrid(torch.linspace(0,1,H,device=device),
                                   torch.linspace(0,1,W,device=device), indexing='ij')
            return torch.stack([gx,gy], dim=0).unsqueeze(0).repeat(B,1,1,1)

        grid3, grid4, grid5 = make_grid(H3,W3), make_grid(H4,W4), make_grid(H5,W5)
        box3 = self.decode_boxes(reg3, grid3, stride3, 0.5)
        box4 = self.decode_boxes(reg4, grid4, stride4, 1.0)
        box5 = self.decode_boxes(reg5, grid5, stride5, 2.0)

        # flatten
        score = torch.cat([s.flatten(2).permute(0,2,1) for s in [score3, score4, score5]], dim=1)
        box = torch.cat([b.flatten(2).permute(0,2,1) for b in [box3, box4, box5]], dim=1)

        return score, box, None