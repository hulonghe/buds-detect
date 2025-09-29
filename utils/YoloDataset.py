import os
import random
import time

import numpy as np
import psutil
import torch
import torch.utils.data as data
import cv2
from utils.helper import get_train_transform
from utils.bbox import cxywh_to_xyxy


class YoloDataset(data.Dataset):
    def __init__(self,
                 img_dir=None, label_dir=None,
                 img_size=640,
                 normalize_label=False,
                 transforms=None,
                 feats_size=None, strides=None,
                 cls_num=1,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 mode="train",
                 max_memory_usage=0.7, mosaic_prob=0.0,
                 easy_fraction=1.0):
        """
        img_dir: 存放图片的目录
        label_dir: 存放 txt 标注的目录
        img_size: 训练时输入的统一尺寸（长、宽相同）
        normalize_label: 边框标记是否归一化[0,1]
        transforms: 可选的数据增强/预处理
        feats_size: list of [H,W] 特征图组参数
        mosaic_prob: 多图拼接增强的概率
        max_memory_usage: 最大内存使用量，0 到 1 之间的浮动
        easy_fraction: 使用多少比例简单样本
        """
        super().__init__()
        self.mode = mode
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transforms = transforms
        self.normalize_label = normalize_label
        self.mean = mean
        self.std = std
        self.feats_size = feats_size
        self.strides_list = strides
        self.cls_num = cls_num
        self.history_hm_loss_status = None
        self.history_bbox_loss_status = None
        self.mosaic_prob = mosaic_prob  # 多图拼接的概率
        self.max_memory_usage = max_memory_usage  # 最大内存使用比例
        self.easy_fraction = easy_fraction
        self.selected_indices = []

        # 计算允许的最大缓存大小
        self.max_cache_size = self._get_max_cache_size()

        # 列出所有图片文件名（不含后缀）
        self.img_files, self.label_counts = self.get_valid_image_files(img_dir, label_dir)
        print(f"{mode}{' datasets':<24}: {len(self.img_files)}")
        print(f"{mode}{' label_counts':<24}: {self.label_counts}")

        # 缓存加载的图像和标签
        self.image_cache = {}
        self.label_cache = {}

        # ✅ 预计算样本容易度
        self.sample_difficulty = self.evaluate_sample_difficulty()
        self.update_selected_indices(easy_fraction)

    def update_selected_indices(self, easy_fraction):
        self.easy_fraction = easy_fraction  # ✅ 保存
        # ✅ 根据 easy_fraction 截断
        n_keep = int(len(self.sample_difficulty) * self.easy_fraction)
        self.selected_indices = [idx for idx, _ in self.sample_difficulty[:n_keep]]
        print(f"Using {self.easy_fraction * 100:.1f}% easy samples: {len(self)}")

    def evaluate_sample_difficulty(self):
        """
        给每个样本计算“容易度分数”并排序
        简单定义：平均目标面积越大、目标数量越少 => 越容易
        """
        difficulties = []
        for idx, img_name in enumerate(self.img_files):
            img_id = os.path.splitext(img_name)[0]
            label_path = os.path.join(self.label_dir, img_id + '.txt')

            # 读取所有标注
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f if len(line.strip().split()) == 5]

            if len(labels) == 0:
                score = -1e6  # 无效
            else:
                wh = []
                for cls, cx, cy, w, h in labels:
                    wh.append(float(w) * float(h))  # YOLO 格式相对坐标
                avg_area = np.mean(wh)
                num_objs = len(labels)
                # 简单打分公式：面积大 → 容易，数量少 → 容易
                score = avg_area - 0.01 * num_objs

            difficulties.append((idx, score))

        # ✅ 按 score 从大到小排序（越大越容易）
        difficulties.sort(key=lambda x: x[1], reverse=True)
        return difficulties

    def _get_max_cache_size(self):
        """
        计算主内存的最大缓存占用内存 (GB)
        """
        # 获取系统总内存
        total_memory = psutil.virtual_memory().total  # 返回的是字节数
        max_cache_size = total_memory * self.max_memory_usage  # 计算最大缓存内存
        return max_cache_size  # 返回字节数

    def count_valid_labels(self, label_path):
        """
        统计YOLO格式标注文件中的有效标记数：
        - 每行必须有5个数值；
        - 宽高（w, h）必须 > 0；
        """
        if not os.path.isfile(label_path):
            return 0

        valid_count = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls, cx, cy, w, h = map(float, parts)
                    if w > 0 and h > 0:
                        valid_count += 1
                except ValueError:
                    continue  # 有非数字内容
        return valid_count

    def get_valid_image_files(self, img_dir, label_dir):
        valid_img_files = []
        label_all = 0

        for f in os.listdir(img_dir):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            name, _ = os.path.splitext(f)
            label_path = os.path.join(label_dir, f"{name}.txt")

            label_num = self.count_valid_labels(label_path)
            if label_num > 0:
                label_all += label_num
                valid_img_files.append(f)

        valid_img_files.sort()
        return valid_img_files, label_all

    def load_image(self, img_path, resize=True):
        """
        先对图像进行等比压缩，保持宽高比，然后再缓存。默认resize为True，即等比缩放图像到接近 img_size 的尺寸。
        """

        orig_w, orig_h, new_w, new_h = None, None, None, None
        if img_path not in self.image_cache:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 获取原始图像的宽高
            orig_h, orig_w, _ = img.shape
            if resize:
                # 计算等比缩放的比例
                scale = min(self.img_size * 2.0 / orig_w, self.img_size * 2.0 / orig_h)  # 保持宽高比进行缩放
                # 按照计算的比例缩放图像
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img = cv2.resize(img, (new_w, new_h))

            # 存储到缓存
            if self.max_cache_size > 0:
                self.image_cache[img_path] = (img, orig_w, orig_h, new_w, new_h)
            else:
                return img, orig_w, orig_h, new_w, new_h

        return self.image_cache[img_path]

    def load_label(self, label_path, orig_w, orig_h, new_w, new_h):
        """
        读取并调整标签坐标，以适应图像的缩放。
        """
        if label_path not in self.label_cache:
            boxes = []
            with open(label_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_c, y_c, w, h = map(float, parts)
                    # 计算缩放比例
                    scale_x = new_w / orig_w
                    scale_y = new_h / orig_h
                    x_min, y_min, x_max, y_max = cxywh_to_xyxy(x_c, y_c, w, h, (orig_w, orig_h))
                    # 调整坐标：按比例调整 x_c, y_c, w, h
                    x_min = x_min * scale_x
                    y_min = y_min * scale_y
                    x_max = x_max * scale_x
                    y_max = y_max * scale_y
                    # 检查并修正x坐标
                    if x_min >= x_max:
                        x_max = x_min + 0.01  # 微小偏移量修正
                    # 检查并修正y坐标
                    if y_min >= y_max:
                        y_max = y_min + 0.01
                    # 将标签添加到列表中
                    boxes.append([int(cls), x_min, y_min, x_max, y_max])
            self.label_cache[label_path] = np.array(boxes) if len(boxes) > 0 else np.zeros((0, 5))

        return self.label_cache[label_path]

    def mosaic_augmentation(self, img_list, label_list):
        import random, cv2
        target_size = self.img_size
        # 随机中心点
        xc = int(random.uniform(0.25 * target_size, 1.75 * target_size))
        yc = int(random.uniform(0.25 * target_size, 1.75 * target_size))

        # 初始化画布
        new_img = np.full((2 * target_size, 2 * target_size, 3), 114, dtype=np.uint8)
        new_labels = []

        for i in range(4):
            img = img_list[i]
            labels = label_list[i]

            h0, w0 = img.shape[:2]
            # 随机缩放
            scale = random.uniform(0.5, 1.5)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # 计算拼接区域
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), max(yc - new_h, 0), xc, yc
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - new_h, 0), min(xc + new_w, 2 * target_size), yc
                x1b, y1b, x2b, y2b = 0, new_h - (y2a - y1a), min(new_w, x2a - x1a), new_h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), yc, xc, min(2 * target_size, yc + new_h)
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), 0, new_w, min(y2a - y1a, new_h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + new_w, 2 * target_size), min(2 * target_size, yc + new_h)
                x1b, y1b, x2b, y2b = 0, 0, min(new_w, x2a - x1a), min(new_h, y2a - y1a)

            # 粘贴到画布
            new_img[y1a:y2a, x1a:x2a] = img_resized[y1b:y2b, x1b:x2b]

            if labels is not None and len(labels) > 0:
                labels = labels.copy()
                labels[:, 1] = labels[:, 1] * (new_w / w0) + x1a - x1b
                labels[:, 2] = labels[:, 2] * (new_h / h0) + y1a - y1b
                labels[:, 3] = labels[:, 3] * (new_w / w0) + x1a - x1b
                labels[:, 4] = labels[:, 4] * (new_h / h0) + y1a - y1b
                # 限制范围
                labels[:, 1:] = labels[:, 1:].clip(0, 2 * target_size)
                # ⚠️ 过滤掉非法框（宽或高 <= 1 像素的框）
                w = labels[:, 3] - labels[:, 1]
                h = labels[:, 4] - labels[:, 2]
                valid = (w > 1) & (h > 1)
                labels = labels[valid]
                if len(labels) > 0:
                    new_labels.append(labels)

        # 合并所有标签
        if len(new_labels):
            new_labels = np.concatenate(new_labels, 0)
        else:
            new_labels = np.zeros((0, 5))
        return new_img, new_labels

    def __len__(self):
        # ✅ 用筛选后的索引
        return len(self.selected_indices)

    def __getitem__(self, index):
        if index >= len(self.selected_indices):
            raise IndexError(
                f"Index out of range after update_easy_fraction，index={index}, selected_indices={len(self.selected_indices)},self len={len(self)}")

        real_idx = self.selected_indices[index]  # ✅ 映射回真实索引
        img_name = self.img_files[real_idx]
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_id + '.txt')

        # 1. 读取图像和标签
        img, orig_w, orig_h, new_w, new_h = self.load_image(img_path, resize=True)  # 默认进行压缩后加载
        boxes = self.load_label(label_path, orig_w, orig_h, new_w, new_h)  # 标签根据压缩后的尺寸进行处理

        # 2. 进行多图拼接增强（根据概率）
        if 0 < random.random() < self.mosaic_prob and self.mode == 'train':
            # 随机选择3张其他图片进行拼接
            selected_idx = random.sample(range(len(self.img_files)), 3)
            img_list = [img]  # 包括当前图像
            label_list = [boxes]  # 当前标签
            for idx in selected_idx:
                img_name_ = self.img_files[idx]
                img_path_ = os.path.join(self.img_dir, img_name_)
                label_path_ = os.path.join(self.label_dir, os.path.splitext(img_name_)[0] + '.txt')
                pj_img, orig_w, orig_h, new_w, new_h = self.load_image(img_path_, resize=True)
                img_list.append(pj_img)
                label_list.append(self.load_label(label_path_, orig_w, orig_h, new_w, new_h))

            img, boxes = self.mosaic_augmentation(img_list, label_list)

        # 3. 应用数据增强 / 预处理（可选）
        if self.transforms:
            try:
                transformed = self.transforms(image=img, bboxes=boxes[:, 1:], labels=boxes[:, 0])
                img_tensor = transformed['image']
                transformed_boxes = np.hstack((np.array(transformed['labels']).reshape(-1, 1), transformed['bboxes']))
            except Exception as e:
                print(e)
                print(boxes)
        else:
            # -------------------------
            # Step 1: 计算缩放比例，保持长宽比
            # -------------------------
            h, w = img.shape[:2]
            scale = self.img_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            # -------------------------
            # Step 2: 缩放图像
            # -------------------------
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # -------------------------
            # Step 3: 计算上下左右的填充量，并填充
            # -------------------------
            pad_h, pad_w = self.img_size - new_h, self.img_size - new_w
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2

            img_padded = cv2.copyMakeBorder(
                img_resized, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )

            # -------------------------
            # Step 4: 归一化
            # -------------------------
            img_norm = img_padded.astype(np.float32) / 255.0
            mean = np.array(self.mean)[None, None, :]  # (1,1,3)
            std = np.array(self.std)[None, None, :]  # (1,1,3)
            img_norm = (img_norm - mean) / std

            # -------------------------
            # Step 5: 转为 PyTorch Tensor (C, H, W)
            # -------------------------
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()

            # Step 6: 处理 bounding boxes (格式: [label, x1, y1, x2, y2])
            # -------------------------
            boxes = boxes.copy().astype(np.float32)

            # 取出标签列 (不参与缩放/平移)
            labels = boxes[:, 0].astype(np.int32)

            # 取出坐标列
            coords = boxes[:, 1:5]

            # 缩放并加上偏移
            coords[:, [0, 2]] = coords[:, [0, 2]] * scale + left  # x1, x2
            coords[:, [1, 3]] = coords[:, [1, 3]] * scale + top  # y1, y2

            # 裁剪到图像范围
            coords[:, 0::2] = np.clip(coords[:, 0::2], 0, self.img_size)  # x1, x2
            coords[:, 1::2] = np.clip(coords[:, 1::2], 0, self.img_size)  # y1, y2

            # 拼回 [label, x1, y1, x2, y2]
            transformed_boxes = np.concatenate([labels[:, None], coords], axis=1)

        if self.normalize_label:
            transformed_boxes[:, [1, 3]] /= self.img_size  # x1, x2
            transformed_boxes[:, [2, 4]] /= self.img_size  # y1, y2

        boxes = torch.tensor(transformed_boxes[:, 1:], dtype=torch.float32)
        labels = torch.tensor(transformed_boxes[:, 0], dtype=torch.int64)

        return img_tensor, boxes, labels, img_id


def denormalize(image, mean, std):
    # 反归一化，确保 std 和 mean 在这里是正确的
    img = image * std + mean  # 使用与训练时相同的 mean 和 std
    img = np.clip(img, 0, 1)  # 保证数据值在 [0, 1] 之间
    return (img * 255).astype(np.uint8)  # 转换为 [0, 255] 范围，并转换为 uint8 类型


def show_images_with_boxes(train_dataset, mean, std):
    """
    显示训练数据集中的每一张图像，绘制标记框，并在每次按下回车键后显示下一张图像。

    :param train_dataset: 一个数据集迭代器，包含 (img_tensor, boxes, labels) 的元组。
                          img_tensor 是图像的 Tensor 或 NumPy 数组，boxes 和 labels 是标记框及其对应信息。
    """
    for img_tensor, boxes, labels, _ in train_dataset:
        # 如果 img_tensor 是一个 Tensor，转为 NumPy 数组
        if isinstance(img_tensor, np.ndarray):
            img = img_tensor
        else:
            img = img_tensor.numpy()  # 假设 img_tensor 是 PyTorch Tensor，转换为 NumPy 数组
            boxes = boxes.numpy()
            # 检查图像的形状，如果是 (3, H, W)，则转换为 (H, W, 3)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # 转换为 (H, W, 3)
        img = denormalize(img, mean, std)
        # 图像格式转换，确保它是正确的格式 (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 绘制标记框
        for box, label in zip(boxes, labels):
            # 假设 boxes 格式是 [class_id, x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            # 将框的位置转换为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 绘制矩形框
            color = (0, 255, 0)  # 绿色框（你可以根据 class_id 更改颜色）
            thickness = 1  # 线宽
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            # 可以在框上添加类标签，显示 class_id
            label = f"c{label}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 显示图像
        cv2.imshow('Image with Boxes', img)
        # 等待回车键按下后继续显示下一张图像
        key = cv2.waitKey(0)  # 按下回车键才继续
        if key == 27:  # 如果按下 ESC 键则退出
            break

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_size = 320
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = (0.4206, 0.502, 0.3179)
    std = (0.2162, 0.2199, 0.1967)
    backbone_type = "resnet50"
    use_softnms = False

    # data_name = "teaRob.v9i.yolov11"
    data_name = "A"
    # data_name = "tea-bud-3"
    # data_name = "tea-bud-4"
    data_root = r"E:/resources/datasets/tea-buds-database/" + data_name

    # 初始化数据
    train_dataset = YoloDataset(
        img_dir=os.path.join(data_root + r"/val", 'images'),
        label_dir=os.path.join(data_root + r"/val", 'labels'),
        img_size=img_size,
        normalize_label=False,
        cls_num=1,
        mean=mean,
        std=std,
        mode="train",
        max_memory_usage=0.0, mosaic_prob=0.3,
        transforms=get_train_transform(img_size, mean=mean, std=std, val=False),
        easy_fraction=1.0
    )

    start_time = time.time()
    show_images_with_boxes(train_dataset, mean, std)
    print("--- %s seconds ---" % (time.time() - start_time))
