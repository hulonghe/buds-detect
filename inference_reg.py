import os
import random
import shutil
import torch
import cv2
from torch.amp import autocast
from nn.bbox.EnhancedDynamicBoxDetector import DynamicBoxDetector
from utils.YoloDataset import YoloDataset, denormalize
from utils.dynamic_postprocess import dynamic_postprocess
from utils.helper import set_seed


# 绘制预测框（橙色）和真实框（绿色）
def draw_boxes(image, pred_boxes, pred_scores=None, gt_boxes=None):
    image = image.copy()

    # 绘制预测框
    if pred_boxes is not None:
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box.astype(int)
            if x1 > x2 or y1 > y2:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
            if pred_scores is not None:
                label = f"{pred_scores[i]:.2f}"
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

    # 绘制真实框
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(image, "GT", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


def infer_with_boxes(model, img_tensor, img_size=320, boxes=None,
                     score_thresh=0.25, iou_thresh=0.25,
                     use_softnms=False, save_path="pred_result.jpg",
                     nms_method="nms"):
    """
    只做检测 + 绘制预测框，不使用 Grad-CAM
    """
    img_resized = denormalize(img_tensor.permute(1, 2, 0).cpu().numpy(), mean=mean, std=std)
    img_tensor = img_tensor.unsqueeze(0)

    with autocast(device_type=DEVICE, enabled=True):
        score_maps, box_maps, _ = model(img_tensor.to(DEVICE))
    score_maps = torch.sigmoid(score_maps)

    batch_pred_boxes, batch_pred_scores, batch_pred_labels = dynamic_postprocess(
        score_maps, box_maps,
        img_size=(img_size, img_size),
        score_thresh=score_thresh, iou_thresh=iou_thresh,
        upscale=True, method=nms_method
    )

    pred_boxes = batch_pred_boxes[0]
    pred_scores = batch_pred_scores[0]

    # 还原 GT 框（如果提供的话）
    if boxes is not None:
        boxes[:, [0, 2]] *= img_resized.shape[0]
        boxes[:, [1, 3]] *= img_resized.shape[1]

    # 绘制预测框（如果有）
    overlay = draw_boxes(
        img_resized,
        pred_boxes.detach().cpu().numpy(),
        pred_scores.detach().cpu().numpy(),
        boxes
    )
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if len(pred_scores) == 0:
        print(f"❌ 没有检测到目标，已保存到 {save_path}")
    else:
        print(f"✅ 预测结果保存到 {save_path}")

    return overlay


if __name__ == "__main__":
    set_seed(2025, False)
    img_size = 320
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = (0.4868, 0.5291, 0.3377)
    std = (0.2017, 0.2022, 0.1851)

    # mean = (0.3908, 0.4763, 0.3021)
    # std = (0.179, 0.1821, 0.1636)

    # mean, std = (0.3908, 0.4763, 0.3021), (0.179, 0.1821, 0.1636)

    backbone_type = "resnet50"
    use_softnms = False
    nms_method = "nms"

    # === 加载模型 ===
    model = DynamicBoxDetector(
        in_dim=img_size, hidden_dim=128, nhead=4, num_layers=2, cls_num=1,
        once_embed=True, is_split_trans=False, is_fpn=True,
        dropout=0.0, backbone_type=backbone_type, device=DEVICE
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(
            "./runs/"
            "dteaRob_v9i_yolov11-mdefault-e100-i320-l0_0001-w1-bresnet50-iciou-bsosa-cfocal-i0_25-s0_4-g1_0-a0_5"
            "/model_epoch_best.pth",
            map_location=DEVICE)
    )
    model.eval()

    data_name = "teaRob.v9i.yolov11"
    # data_name = "tea-buds-owns"
    # data_name = "tea-bud-3"
    # data_name = "tea-bud-4"
    data_root = r"E:/resources/datasets/tea-buds-database/" + data_name
    test_dataset = YoloDataset(
        img_dir=os.path.join(data_root + r"/val", 'images'),
        label_dir=os.path.join(data_root + r"/val", 'labels'),
        img_size=img_size,
        cls_num=1,
        mean=mean,
        std=std,
        normalize_label=True,
        mode="val",
        max_memory_usage=0.7, mosaic_prob=0.0,
    )

    save_root = "runs/a-test-infer"
    # 如果目录已存在，先删掉再创建
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)

    target_layers = [
        model.fpn.fpn_convs[0],
        model.fpn.fpn_convs[1],
        model.fpn.fpn_convs[2],
    ]

    # === 循环图片，依次生成预测结果（仅框） ===
    sample_indices = random.sample(range(len(test_dataset)), 300)
    for img_idx in sample_indices:
        img_tensor, boxes, labels = test_dataset[img_idx]
        real_idx = test_dataset.selected_indices[img_idx]
        img_name = test_dataset.img_files[real_idx]
        img_id = os.path.splitext(img_name)[0]

        save_path = os.path.join(save_root, f"pred_{img_idx}_{img_id}.jpg")
        result = infer_with_boxes(
            model, img_tensor,
            img_size=img_size,
            boxes=boxes.cpu().numpy().copy(),
            score_thresh=0.7, iou_thresh=0.25,
            save_path=save_path,
            nms_method=nms_method
        )
        print(f"[OK] {img_id} -> {save_path}")
