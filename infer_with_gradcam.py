import os
import random
import shutil

import torch
import cv2
from torch.amp import autocast

from inference_reg import draw_boxes
from nn.bbox.EnhancedDynamicBoxDetector import DynamicBoxDetector
from utils.YoloDataset import YoloDataset, denormalize
from utils.dynamic_postprocess import dynamic_postprocess
from utils.grad_CAM import GradCAM
from utils.helper import set_seed, get_train_transform

# -------------------- 检测任务推理 + Grad-CAM --------------------
def infer_with_gradcam(model, gradcam, img_tensor, img_size=320, boxes=None,
                       score_thresh=0.25, iou_thresh=0.25,
                       use_softnms=False, save_path="gradcam_result.jpg",
                       mode="fusion", fusion_weights=None, nms_method="nms"):
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
    # 画预测框
    if boxes is not None:
        boxes[:, [0, 2]] *= img_resized.shape[0]
        boxes[:, [1, 3]] *= img_resized.shape[1]
    if len(pred_scores) == 0:
        overlay = draw_boxes(img_resized,
                             pred_boxes.detach().cpu().numpy(),
                             pred_scores.detach().cpu().numpy(), boxes)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"没有检测到目标 保存到 {save_path}")
        return None

    # 选择最高分预测框作为目标
    top_idx = pred_scores.argmax()
    target_score = pred_scores[top_idx]  # 保留梯度用于 Grad-CAM

    # 生成 Grad-CAM
    cam_img = gradcam.generate_cam(target_score, mode=mode, weights=fusion_weights, output_size=(img_size, img_size))
    overlay = gradcam.overlay_gradcam_on_image(cam_img, img_resized, alpha=0.6, mode=mode)

    overlay = draw_boxes(overlay,
                         pred_boxes.detach().cpu().numpy(),
                         pred_scores.detach().cpu().numpy(), boxes)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM 保存到 {save_path}")

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
            # dteaRob_v9i_yolov11-mdefault-e100-i320-l0_0001-w1-bresnet50-iciou-bsosa-cfocal-i0_25-s0_25-g1_0-a0_5-d0_1
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

    save_root = "runs/a-test"
    # 如果目录已存在，先删掉再创建
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)

    target_layers = [
        model.fpn.fpn_convs[0],
        model.fpn.fpn_convs[1],
        model.fpn.fpn_convs[2],
    ]

    # === 循环图片 × 层，依次生成 Grad-CAM ===
    sample_indices = random.sample(range(len(test_dataset)), 300)
    for img_idx in sample_indices:
        img_tensor, boxes, labels = test_dataset[img_idx]
        real_idx = test_dataset.selected_indices[img_idx]  # ✅ 映射回真实索引
        img_name = test_dataset.img_files[real_idx]
        img_id = os.path.splitext(img_name)[0]

        for idx, mode_name in enumerate(['side_by_side', 'fusion']):
            gradcam = GradCAM(model, target_layers=target_layers, use_plus=True, topk=10)

            save_path = os.path.join(
                save_root, f"cam_{mode_name}_{img_idx}_{img_id}.jpg"
            )
            result = infer_with_gradcam(
                model, gradcam, img_tensor,
                img_size=img_size,
                boxes=boxes.cpu().numpy().copy(),
                score_thresh=0.7, iou_thresh=0.25,
                save_path=save_path,
                mode=mode_name,
                nms_method=nms_method
            )
            print(f"[OK] {img_id} -> {save_path}")
