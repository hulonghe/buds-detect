import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# 类别映射：VOC object name -> class id (0~19)
def get_voc_class_name_to_id():
    return {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }

# 解析 target 中的标注信息
def parse_target(target):
    boxes = []
    labels = []

    class_name_to_id = get_voc_class_name_to_id()

    objects = target['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]  # 如果只有一个 object，包装成列表统一处理

    for obj in objects:
        name = obj['name']
        if name not in class_name_to_id:
            continue  # 跳过未知类别

        bbox = obj['bndbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_name_to_id[name])

    return boxes, labels


# 将边界框转为 YOLO 格式
def convert_to_yolo_format(boxes, img_w, img_h):
    yolo_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        yolo_boxes.append([xc, yc, w, h])
    return yolo_boxes


# 主函数：遍历数据集并生成 YOLO 标签
def voc_to_yolo(voc_dataset, output_dir, image_output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)

    for i, (image, target) in enumerate(voc_dataset):
        image: Image.Image = transforms.ToPILImage()(image).convert("RGB")
        img_w, img_h = image.size
        image_id = os.path.splitext(target['annotation']['filename'])[0]

        # 获取标注信息
        boxes, labels = parse_target(target)
        if not boxes:
            print(f"跳过无标注图片: {image_id}")
            continue

        # 转换为 YOLO 格式
        yolo_boxes = convert_to_yolo_format(boxes, img_w, img_h)

        # 保存标签
        txt_file = os.path.join(output_dir, f"{image_id}.txt")
        with open(txt_file, 'w') as f:
            for cls_id, box in zip(labels, yolo_boxes):
                line = f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                f.write(line)

        # 可选：保存图像到指定路径
        if image_output_dir:
            image.save(os.path.join(image_output_dir, f"{image_id}.jpg"))

        if i % 100 == 0:
            print(f"已处理 {i} 张图片")

    print(f"转换完成，标签文件保存至: {output_dir}")


if __name__ == '__main__':
    # 数据预处理（仅转为 Tensor）
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载 VOC2012 数据集
    voc_dataset = datasets.VOCDetection(
        root='/root/autodl-tmp/VOC2012',     # 指向 VOCdevkit 目录
        year='2012',
        image_set='val',                   # 可改为 val 或 trainval
        download=False,
        transform=transform
    )

    # 设置输出目录
    output_dir = '/root/autodl-tmp/VOC2012/val_labels'           # 标签文件输出路径
    image_output_dir = "/root/autodl-tmp/VOC2012/val_images"     # 图像文件输出路径（可选）

    # 开始转换
    voc_to_yolo(voc_dataset, output_dir, image_output_dir=image_output_dir)