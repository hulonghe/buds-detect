import time

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights
from torchinfo import summary
import torch.nn.functional as F

from nn.FPN import DynamicFPN
from nn.depthwise_FPN import DynamicDepthwiseFPN
from nn.small_obj_backbone import SmallObjBackbone


def init_all_heads(model, cls_num):
    """
    初始化检测器的分类 head、回归 head、refine head.
    默认适配 Conv + GroupNorm + SiLU 结构。
    """

    def init_head(head, is_cls=False, is_reg=False):
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                # 中间层 Conv
                if (is_cls and m.out_channels != cls_num) or (is_reg and m.out_channels != 4):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                # 分类 head 最后一层 Conv
                elif is_cls and m.out_channels == cls_num:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                # 回归 head / refine head 最后一层 Conv
                elif is_reg and m.out_channels == 4:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    # 分类 heads
    for h in [model.score_head3, model.score_head4, model.score_head5]:
        init_head(h, is_cls=True)

    # 回归 heads
    for h in [model.reg_head3, model.reg_head4, model.reg_head5]:
        init_head(h, is_reg=True)
    for h in [model.refine_head3, model.refine_head4, model.refine_head5]:
        init_head(h, is_reg=True)


class DynamicBoxDetector(nn.Module):
    def __init__(self,
                 in_channels=3, in_dim=256, hidden_dim=256, cls_num=1,
                 num_layers=2, nhead=8,
                 once_embed=False, is_split_trans=False, is_fpn=True,
                 head_group=1, cls_head_kernel_size=3,
                 topk=300, dropout=0.2, backbone_type='resnet50', device='cpu'):
        super().__init__()
        self.is_split_trans = is_split_trans
        self.once_embed = once_embed
        self.topk = topk
        self.backbone_type = backbone_type

        # ---------------- Backbone ---------------- #
        if backbone_type == 'smallobjnet':
            backbone = SmallObjBackbone(in_ch=in_channels, num_classes=cls_num)
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

        # 动态推断通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, in_dim, in_dim)
            feats, _ = self._extract_features(dummy)
            in_channels_list = [f.shape[1] for f in feats]  # [C3, C4, C5]
            print(f"{'in_channels_list':<24}: {in_channels_list}")

        # self.fpn = DynamicFPN(in_channels_list, hidden_dim)
        self.fpn = DynamicDepthwiseFPN(in_channels_list, hidden_dim, dropout=dropout, is_fpn=is_fpn)

        with torch.no_grad():
            fpn_p = self.fpn(feats)
            h3, w3 = fpn_p[0].shape[-2:]
            h4, w4 = fpn_p[1].shape[-2:]
            h5, w5 = fpn_p[2].shape[-2:]
            print(f"{'PANet P_h_w':<24}: P3={h3}x{w3}, P4={h4}x{w4}, P5={h5}x{w5}")

        # Scale embedding
        self.scale_embed = nn.Embedding(in_channels, hidden_dim)
        # Positional embedding (基于 P3 尺度大小)，后续会 crop 到对应尺度
        self.row_embed = nn.Embedding(h3, hidden_dim // 2, device=device)
        self.col_embed = nn.Embedding(w3, hidden_dim // 2, device=device)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类head
        def make_cls_head(ks=3):
            padding = ks // 2
            return nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, ks, padding=padding, groups=head_group),
                nn.GroupNorm(hidden_dim // 4, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, cls_num, 1)
            )

        # Box head
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
        self.reg_head3 = make_reg_head()
        self.score_head4 = make_cls_head(cls_head_kernel_size)
        self.reg_head4 = make_reg_head()
        self.score_head5 = make_cls_head(cls_head_kernel_size)
        self.reg_head5 = make_reg_head()

        # Refine head
        def make_refine_head():
            return nn.Sequential(
                nn.Conv2d(hidden_dim + 4, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, 4, 1)
            )

        self.refine_head3 = make_refine_head()
        self.refine_head4 = make_refine_head()
        self.refine_head5 = make_refine_head()

        # 初始化 reg_head
        init_all_heads(self, cls_num)

    def _extract_features(self, x):
        if self.backbone_type == 'smallobjnet':
            return self.backbone['_impl'](x)  # 直接返回 [C2,C3,C4,C5]
        else:
            x = self.backbone['conv1'](x)
            c2 = self.backbone['layer1'](x)   # /4
            c3 = self.backbone['layer2'](c2)
            c4 = self.backbone['layer3'](c3)
            c5 = self.backbone['layer4'](c4)
            return [c3, c4, c5], None

    def inverse_sigmoid(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # 先 clamp 防止 x=0 或 x=1 导致 log(0)
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))

    def decode_boxes(self, reg, grid, scale=0.2):
        """
            scal：可调节范围，如0.2表示[-0.2,0.2]
        """
        # 对 tx, ty 使用 tanh() 限制范围，再乘以 scale
        tx = torch.tanh(reg[:, 0]) * scale
        ty = torch.tanh(reg[:, 1]) * scale

        # 计算预测中心点，结合 grid 坐标
        cx = (tx + grid[:, 0]).clamp(0., 1.)
        cy = (ty + grid[:, 1]).clamp(0., 1.)

        # 对 tw, th 直接使用 clamp 限制下界防止为0
        w = reg[:, 2].clamp(min=1e-3, max=1.)
        h = reg[:, 3].clamp(min=1e-3, max=1.)

        # 计算边界框的左上、右下点
        x_min = (cx - w / 2).clamp(0., 1.)
        y_min = (cy - h / 2).clamp(0., 1.)
        x_max = (cx + w / 2).clamp(0., 1.)
        y_max = (cy + h / 2).clamp(0., 1.)

        return torch.stack([x_min, y_min, x_max, y_max], dim=1)  # [B,4,H,W]

    def forward(self, x):
        B = x.size(0)
        device = x.device
        # Backbone + FPN
        features, aux_logits = self._extract_features(x)
        p3, p4, p5 = self.fpn(features)  # PANet 输出

        # 取出每层尺寸
        _, C3, H3, W3 = p3.shape
        _, C4, H4, W4 = p4.shape
        _, C5, H5, W5 = p5.shape
        # Flatten + position + scale embedding
        f_flat_list, pos_list, spatial_shapes = [], [], [(H3, W3), (H4, W4), (H5, W5)]

        pos_full = None
        if self.once_embed:
            # === 计算最大尺寸，构造共享位置编码 ===
            H_max, W_max = max(H3, H4, H5), max(W3, W4, W5)
            # 构造 row/col embedding （只做一次）
            row_pos_all = self.row_embed(torch.arange(H_max, device=device))  # [H_max, hidden//2]
            col_pos_all = self.col_embed(torch.arange(W_max, device=device))  # [W_max, hidden//2]

            # 拼接生成共享二维位置编码 [H_max, W_max, hidden]
            pos_full = torch.cat([
                row_pos_all.unsqueeze(1).repeat(1, W_max, 1),  # [H_max, W_max, hidden//2]
                col_pos_all.unsqueeze(0).repeat(H_max, 1, 1)  # [H_max, W_max, hidden//2]
            ], dim=-1)  # [H_max, W_max, hidden]

        for idx, (feat, (H, W)) in enumerate(zip([p3, p4, p5], spatial_shapes)):
            if self.once_embed:
                # 裁剪共享位置编码
                pos = pos_full[:H, :W]  # [H, W, hidden]
            else:
                # 构造行/列 embedding（基于 P3 大小）
                # 如果是 P4/P5，需要从 P3 embedding crop 大小
                row_pos = self.row_embed(torch.arange(H3 if idx == 0 else H, device=device)[:H])  # [H,hidden//2]
                col_pos = self.col_embed(torch.arange(W3 if idx == 0 else W, device=device)[:W])  # [W,hidden//2]
                pos = torch.cat([
                    row_pos.unsqueeze(1).repeat(1, W, 1),
                    col_pos.unsqueeze(0).repeat(H, 1, 1)
                ], dim=-1)  # [H, W, hidden]
            pos = pos.flatten(0, 1).unsqueeze(0).repeat(B, 1, 1)  # [B, H*W, hidden]
            # Scale embedding
            scale_code = self.scale_embed(torch.full((B, H * W), idx, dtype=torch.long, device=device))
            pos = pos + scale_code
            f_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden]
            f_flat_list.append(f_flat)
            pos_list.append(pos)

        if self.is_split_trans:
            feats_trans = []
            for f_flat, pos in zip(f_flat_list, pos_list):
                f_trans_scale = self.transformer(f_flat + pos)  # 序列短：1600/400/100
                feats_trans.append(f_trans_scale)
        else:
            f_all = torch.cat(f_flat_list, dim=1)  # [B, S_total, hidden]
            pos_all = torch.cat(pos_list, dim=1)
            f_trans = self.transformer(f_all + pos_all)  # [B, S_total, hidden]
            # 拆分回多尺度
            split_sizes = [H3 * W3, H4 * W4, H5 * W5]
            feats_trans = torch.split(f_trans, split_sizes, dim=1)

        feat3_t = feats_trans[0].permute(0, 2, 1).reshape(B, -1, H3, W3)
        feat4_t = feats_trans[1].permute(0, 2, 1).reshape(B, -1, H4, W4)
        feat5_t = feats_trans[2].permute(0, 2, 1).reshape(B, -1, H5, W5)
        # 分类 & 回归 head
        score3 = self.score_head3(feat3_t)  # [B,cls_num,H3,W3]
        reg3 = self.reg_head3(feat3_t)  # [B,4,H3,W3]
        score4 = self.score_head4(feat4_t)
        reg4 = self.reg_head4(feat4_t)
        score5 = self.score_head5(feat5_t)
        reg5 = self.reg_head5(feat5_t)

        # 生成网格
        grid_y3, grid_x3 = torch.meshgrid(
            torch.linspace(0, 1, H3, device=device),
            torch.linspace(0, 1, W3, device=device), indexing='ij'
        )
        grid3 = torch.stack([grid_x3, grid_y3], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H3, W3]

        grid_y4, grid_x4 = torch.meshgrid(
            torch.linspace(0, 1, H4, device=device),
            torch.linspace(0, 1, W4, device=device), indexing='ij'
        )
        grid4 = torch.stack([grid_x4, grid_y4], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        grid_y5, grid_x5 = torch.meshgrid(
            torch.linspace(0, 1, H5, device=device),
            torch.linspace(0, 1, W5, device=device), indexing='ij'
        )
        grid5 = torch.stack([grid_x5, grid_y5], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # # reg head 输出 4通道，改为 [tx, ty, th, tw]
        # # tx, ty: 中心点偏移，tw, th: 宽高，均为相对比例，范围[0,1]
        # # 转换为xyxy格式
        # box3 = self.decode_boxes(reg3, grid3)
        # box4 = self.decode_boxes(reg4, grid4)
        # box5 = self.decode_boxes(reg5, grid5)

        # —— 迭代微调 —— 
        # 1) 将第一版 reg3 和 feat3_t 在 channel 维度拼接  
        #    注意 reg3 形状 [B,4,H3,W3] 与 feat3_t [B,C,H3,W3]
        feat3_cat = torch.cat([feat3_t, reg3], dim=1)  # [B, C+4, H3, W3]
        # 2) 通过 refine head 预测增量 delta3  
        delta3 = self.refine_head3(feat3_cat)  # [B, 4, H3, W3]
        # 3) 更新 reg3，再次 decode  
        reg3_refined = reg3 + delta3  # [B,4,H3,W3]
        # 4) 最终框  
        box3 = self.decode_boxes(reg3_refined, grid3, 1.0)  # [B, H3*W3, 4]
        # P4 / P5 同理：
        feat4_cat = torch.cat([feat4_t, reg4], dim=1)
        delta4 = self.refine_head4(feat4_cat)
        reg4_refined = reg4 + delta4
        box4 = self.decode_boxes(reg4_refined, grid4, 1.0)

        feat5_cat = torch.cat([feat5_t, reg5], dim=1)
        delta5 = self.refine_head5(feat5_cat)
        reg5_refined = reg5 + delta5
        box5 = self.decode_boxes(reg5_refined, grid5, 1.0)

        # flatten 拼接
        score3_flat = score3.flatten(2).permute(0, 2, 1)
        score4_flat = score4.flatten(2).permute(0, 2, 1)
        score5_flat = score5.flatten(2).permute(0, 2, 1)
        score = torch.cat([score3_flat, score4_flat, score5_flat], dim=1)  # [B, N_all, num_cls] 范围 [0, 1]
        box3_flat = box3.flatten(2).permute(0, 2, 1)
        box4_flat = box4.flatten(2).permute(0, 2, 1)
        box5_flat = box5.flatten(2).permute(0, 2, 1)
        box = torch.cat([box3_flat, box4_flat, box5_flat], dim=1)  # [B, N_all, 4] 范围 [0, 1]

        return score, box, aux_logits


def measure_latency(model, device, input_size=(1, 3, 320, 320), warmup=10, repeat=50):
    model.eval()
    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model(x)

        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        for _ in range(repeat):
            _ = model(x)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.time()

    avg_time = (end - start) / repeat
    fps = 1 / avg_time
    return avg_time * 1000, fps  # ms, FPS


if __name__ == '__main__':
    image_size = 320
    device_ = "cuda" if torch.cuda.is_available() else "cpu"
    model = DynamicBoxDetector(in_dim=image_size,
                               hidden_dim=256,
                               nhead=4,
                               num_layers=2,
                               cls_num=1,
                               once_embed=True,
                               is_split_trans=True,
                               is_fpn=True,
                               head_group=1,
                               cls_head_kernel_size=1,
                               backbone_type="resnet50", device=device_).to(device_)
    # 输入维度根据实际修改
    summary(model, input_size=(1, 3, image_size, image_size))

    latency, fps = measure_latency(model, device_, input_size=(1, 3, image_size, image_size))
    print(f"Inference time: {latency:.2f} ms, FPS: {fps:.2f}")
