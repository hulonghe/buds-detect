import os
import cv2
import shutil
import numpy as np


def smart_crop_yolo(img_dir, label_dir, output_dir, img_size=(640, 640)):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in img_filenames:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

        # 如果没有标注文件，直接跳过
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 读取 YOLO 标签：class cx cy bw bh
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue  # 没有目标，跳过

        bboxes = []
        for line in lines:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            bboxes.append([x1, y1, x2, y2, cls])

        bboxes = np.array(bboxes)

        # 所有目标的边界框范围
        x_min = np.min(bboxes[:, 0])
        y_min = np.min(bboxes[:, 1])
        x_max = np.max(bboxes[:, 2])
        y_max = np.max(bboxes[:, 3])

        # 计算中心区域并扩展
        crop_cx = (x_min + x_max) / 2
        crop_cy = (y_min + y_max) / 2
        crop_w = max(x_max - x_min, img_size[0])
        crop_h = max(y_max - y_min, img_size[1])

        # 扩张到最小尺寸
        crop_w = max(crop_w, img_size[0])
        crop_h = max(crop_h, img_size[1])

        # 计算左上角坐标
        crop_x1 = int(crop_cx - crop_w / 2)
        crop_y1 = int(crop_cy - crop_h / 2)
        crop_x2 = int(crop_x1 + crop_w)
        crop_y2 = int(crop_y1 + crop_h)

        # 限制在图像范围内
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w, crop_x2)
        crop_y2 = min(h, crop_y2)

        # 实际裁剪
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # 更新标注框
        new_bboxes = []
        for box in bboxes:
            x1, y1, x2, y2, cls = box
            new_x1 = x1 - crop_x1
            new_y1 = y1 - crop_y1
            new_x2 = x2 - crop_x1
            new_y2 = y2 - crop_y1

            new_cx = (new_x1 + new_x2) / 2 / (crop_x2 - crop_x1)
            new_cy = (new_y1 + new_y2) / 2 / (crop_y2 - crop_y1)
            new_bw = (new_x2 - new_x1) / (crop_x2 - crop_x1)
            new_bh = (new_y2 - new_y1) / (crop_y2 - crop_y1)

            new_bboxes.append(f"{int(cls)} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

        # 保存图片和标注
        out_img_path = os.path.join(output_dir, 'images', img_name)
        out_label_path = os.path.join(output_dir, 'labels', os.path.splitext(img_name)[0] + '.txt')

        cv2.imwrite(out_img_path, cropped_img)
        with open(out_label_path, 'w') as f:
            f.write('\n'.join(new_bboxes))

    print(f"处理完成，共处理 {len(img_filenames)} 张图像。")


if __name__ == '__main__':
    smart_crop_yolo(
        img_dir=r'/root/autodl-tmp/tea-buds-owns/dataset/train/images',
        label_dir=r'/root/autodl-tmp/tea-buds-owns/dataset/train/labels',
        output_dir=r'/root/autodl-tmp/tea-buds-owns/dataset/train',
        img_size=(640, 640),
    )
