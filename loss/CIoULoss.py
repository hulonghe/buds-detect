import torch
import math


class CIoULoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super(CIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds: (N, 4) (cx, cy, w, h)
        targets: (N, 4) (cx, cy, w, h)
        """
        # 计算 CIoU
        ciou = self.box_ciou(preds, targets)

        # CIoU 损失 = 1 - CIoU
        loss = 1 - ciou

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def box_ciou(pred_boxes, target_boxes, eps=1e-7):
        """
        计算两组边界框之间的 CIoU（Complete IoU）损失。

        参数:
            pred_boxes: [N, 4]，预测框 (cx, cy, w, h)，归一化或绝对坐标
            target_boxes: [N, 4]，真实框 (cx, cy, w, h)，坐标系统与 pred 相同
            eps: 防止除零的小常数

        返回:
            ciou: [N]，每个预测框的 CIoU 值 ∈ [-1, 1]，越大越好
        """

        # 分离中心与宽高
        px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        gx, gy, gw, gh = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        # 转为左上、右下角坐标
        p_xmin = px - pw / 2
        p_ymin = py - ph / 2
        p_xmax = px + pw / 2
        p_ymax = py + ph / 2

        g_xmin = gx - gw / 2
        g_ymin = gy - gh / 2
        g_xmax = gx + gw / 2
        g_ymax = gy + gh / 2

        # —— 1. IoU 部分 ——
        inter_xmin = torch.max(p_xmin, g_xmin)
        inter_ymin = torch.max(p_ymin, g_ymin)
        inter_xmax = torch.min(p_xmax, g_xmax)
        inter_ymax = torch.min(p_ymax, g_ymax)

        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * \
                     torch.clamp(inter_ymax - inter_ymin, min=0)
        area_p = pw * ph
        area_g = gw * gh
        union_area = area_p + area_g - inter_area + eps
        iou = inter_area / union_area

        # —— 2. 中心距离项 ——
        center_dist = (px - gx) ** 2 + (py - gy) ** 2

        # —— 3. 包围盒对角线距离 ——
        enclose_xmin = torch.min(p_xmin, g_xmin)
        enclose_ymin = torch.min(p_ymin, g_ymin)
        enclose_xmax = torch.max(p_xmax, g_xmax)
        enclose_ymax = torch.max(p_ymax, g_ymax)

        enclose_diag = (enclose_xmax - enclose_xmin) ** 2 + \
                       (enclose_ymax - enclose_ymin) ** 2 + eps

        # —— 4. Aspect Ratio 一致性项 ——
        v = (4 / (3.14159265 ** 2)) * torch.pow(
            torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps)), 2)

        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        # —— 5. CIoU 公式 ——
        ciou = iou - (center_dist / enclose_diag + alpha * v)

        return ciou
