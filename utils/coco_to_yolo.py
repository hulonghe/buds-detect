import os
import json
import argparse


def coco_to_yolo(json_file, image_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 构建类别ID映射（COCO类别id不一定是连续的）
    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
    print("类别映射:", categories)

    # 建立image_id到图像文件名的映射
    images = {img['id']: img['file_name'] for img in data['images']}

    # 遍历所有标注
    annotations = data['annotations']
    count = 0

    for ann in annotations:
        image_id = ann['image_id']
        category_id = categories[ann['category_id']]
        bbox = ann['bbox']  # [x_min, y_min, width, height]

        # 获取图像尺寸
        image_path = os.path.join(image_dir, images[image_id])
        if not os.path.exists(image_path):
            print(f"警告：找不到图像文件 {image_path}")
            continue

        from PIL import Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        # 转换为YOLO格式
        x, y, w, h = bbox
        xc = (x + w / 2) / img_w
        yc = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h

        # 准备写入内容
        line = f"{category_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"

        # 写入txt文件
        txt_file_name = os.path.splitext(images[image_id])[0] + ".txt"
        txt_file_path = os.path.join(output_dir, txt_file_name)

        with open(txt_file_path, 'a') as f:
            f.write(line)

        count += 1

    print(f"共处理了 {count} 条标注，生成标签文件在: {output_dir}")


if __name__ == '__main__':
    json_file = f"/root/autodl-tmp/coco2017/annotations/instances_train2017.json"
    image_dir = f"/root/autodl-tmp/coco2017/train2017"
    output_dir = f"/root/autodl-tmp/coco2017/train2017_labels"

    coco_to_yolo(json_file, image_dir, output_dir)
