import random
import numpy as np
import cv2


class YOLOAugmentor:
    def __init__(self, img_size=320, close_mosaic=False, mixup_prob=0.2, copy_paste_prob=0.4):
        self.img_size = img_size
        self.close_mosaic = close_mosaic
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob

    # ---------------- Mosaic ----------------
    def mosaic_augmentation(self, img_list, label_list):
        target_size = self.img_size
        if self.close_mosaic:
            return img_list[0], label_list[0]

        # ---------------- Mosaic 中心点，轻微偏移 ----------------
        margin = target_size // 16  # 偏移范围 ±1/8 target_size
        xc = target_size // 2 + random.randint(-margin, margin)
        yc = target_size // 2 + random.randint(-margin, margin)

        # mosaic 临时画布扩大为 2*target_size，以容纳拼接和偏移
        canvas_size = target_size * 2
        new_img = np.full((canvas_size, canvas_size, 3), 114, dtype=np.uint8)
        new_labels = []

        for i in range(4):
            img = img_list[i]
            labels = label_list[i].copy()
            if len(labels) == 0:
                continue

            h0, w0 = img.shape[:2]

            # ---------------- 随机裁剪包含目标区域 ----------------
            x_min, y_min = labels[:, 1:3].min(0)
            x_max, y_max = labels[:, 3:5].max(0)
            expand = random.uniform(0.1, 0.3)
            x_min = max(0, int(x_min - expand * (x_max - x_min)))
            y_min = max(0, int(y_min - expand * (y_max - y_min)))
            x_max = min(w0, int(x_max + expand * (x_max - x_min)))
            y_max = min(h0, int(y_max + expand * (y_max - y_min)))

            img_crop = img[y_min:y_max, x_min:x_max]
            labels[:, [1, 3]] -= x_min
            labels[:, [2, 4]] -= y_min

            # ---------------- 缩放 ----------------
            scale = random.uniform(1.0, 1.1)
            new_w = int(img_crop.shape[1] * scale)
            new_h = int(img_crop.shape[0] * scale)
            img_resized = cv2.resize(img_crop, (new_w, new_h))

            labels[:, [1, 3]] *= new_w / img_crop.shape[1]
            labels[:, [2, 4]] *= new_h / img_crop.shape[0]

            # ---------------- 四分区位置 ----------------
            if i == 0:  # 左上
                x1a, y1a = max(xc - new_w, 0), max(yc - new_h, 0)
            elif i == 1:  # 右上
                x1a, y1a = xc, max(yc - new_h, 0)
            elif i == 2:  # 左下
                x1a, y1a = max(xc - new_w, 0), yc
            else:  # 右下
                x1a, y1a = xc, yc

            x2a = x1a + min(new_w, canvas_size - x1a)
            y2a = y1a + min(new_h, canvas_size - y1a)

            quadrant_w, quadrant_h = x2a - x1a, y2a - y1a

            # ---------------- 裁剪 img 和 bbox 到 quadrant ----------------
            img_resized_cropped = img_resized[:quadrant_h, :quadrant_w]
            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, quadrant_w)
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, quadrant_h)

            # ---------------- 剔除过小目标 ----------------
            w = labels[:, 3] - labels[:, 1]
            h = labels[:, 4] - labels[:, 2]
            border = 2
            valid = (w > 2) & (h > 2) & (labels[:, 1] > border) & (labels[:, 2] > border) & \
                    (labels[:, 3] < quadrant_w - border) & (labels[:, 4] < quadrant_h - border)
            labels = labels[valid]
            if len(labels) == 0:
                continue

            # ---------------- 粘贴到 mosaic ----------------
            new_img[y1a:y2a, x1a:x2a] = img_resized_cropped
            labels[:, [1, 3]] += x1a
            labels[:, [2, 4]] += y1a
            new_labels.append(labels)

        new_labels = np.concatenate(new_labels, 0) if len(new_labels) else np.zeros((0, 5))

        # ---------------- Copy-Paste ----------------
        if random.random() < self.copy_paste_prob:
            new_img, new_labels = self.copy_paste(new_img, new_labels)

        # ---------------- MixUp ----------------
        if random.random() < self.mixup_prob:
            new_img, new_labels = self.mixup(new_img, new_labels, img_list[0], label_list[0])

        # ---------------- 最终裁剪为 target_size ----------------
        y_start = (canvas_size - target_size) // 2
        x_start = (canvas_size - target_size) // 2
        new_img = new_img[y_start:y_start + target_size, x_start:x_start + target_size]
        if len(new_labels):
            new_labels[:, [1, 3]] -= x_start
            new_labels[:, [2, 4]] -= y_start
            new_labels[:, [1, 3]] = new_labels[:, [1, 3]].clip(0, target_size)
            new_labels[:, [2, 4]] = new_labels[:, [2, 4]].clip(0, target_size)

        return new_img, self.clean_boxes(new_labels, target_size, target_size)

    # ---------------- MixUp ----------------
    def mixup(self, img1, labels1, img2, labels2, alpha=0.5):
        lam = np.random.beta(alpha, alpha)
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        mixed_img = (lam * img1 + (1 - lam) * img2_resized).astype(np.uint8)
        mixed_labels = np.concatenate((labels1, labels2), axis=0) if len(labels2) else labels1
        return mixed_img, mixed_labels

    # ---------------- Copy-Paste ----------------
    def copy_paste(self, img, labels, max_paste=50, iou_thresh=0.1):
        if len(labels) == 0:
            return img, labels

        new_img = img.copy()
        new_labels = labels.copy()
        h, w = img.shape[:2]
        num_paste = min(len(labels), max_paste)
        chosen = np.random.choice(len(labels), num_paste, replace=False)

        for idx in chosen:
            cls, x1, y1, x2, y2 = labels[idx]
            bw, bh = int(x2 - x1), int(y2 - y1)
            if bw < 5 or bh < 5:
                continue
            obj_crop = img[int(y1):int(y2), int(x1):int(x2)]
            if obj_crop.size == 0:
                continue

            # 尝试随机放置
            for _ in range(10):
                dx = np.random.randint(0, max(1, w - bw))
                dy = np.random.randint(0, max(1, h - bh))
                if self.is_valid_position(dx, dy, bw, bh, new_labels, iou_thresh):
                    break
            else:
                continue

            # 随机翻转
            if random.random() < 0.5:
                obj_crop = cv2.flip(obj_crop, 1)

            ph, pw = obj_crop.shape[:2]
            if dy + ph > h: ph = h - dy
            if dx + pw > w: pw = w - dx
            obj_crop = obj_crop[:ph, :pw]

            # 粘贴
            new_img[dy:dy + ph, dx:dx + pw] = obj_crop
            new_labels = np.vstack((new_labels, np.array([[cls, dx, dy, dx + pw, dy + ph]])))

        return new_img, new_labels

    # ---------------- IoU 检查 ----------------
    @staticmethod
    def is_valid_position(dx, dy, pw, ph, existing_boxes, iou_thresh=0.1):
        for b in existing_boxes:
            x1, y1, x2, y2 = b[1:5]
            xx1 = max(x1, dx)
            yy1 = max(y1, dy)
            xx2 = min(x2, dx + pw)
            yy2 = min(y2, dy + ph)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area1 = (x2 - x1) * (y2 - y1)
            area2 = pw * ph
            iou = inter / (area1 + area2 - inter)
            if iou > iou_thresh:
                return False
        return True

    # ---------------- 边界清理 ----------------
    @staticmethod
    def clean_boxes(boxes, img_w, img_h, min_size=10):
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 5))
        boxes = boxes.copy()
        boxes[:, 1] = boxes[:, 1].clip(0, img_w - 1)
        boxes[:, 3] = boxes[:, 3].clip(0, img_w - 1)
        boxes[:, 2] = boxes[:, 2].clip(0, img_h - 1)
        boxes[:, 4] = boxes[:, 4].clip(0, img_h - 1)
        w = boxes[:, 3] - boxes[:, 1]
        h = boxes[:, 4] - boxes[:, 2]
        valid = (w > min_size) & (h > min_size)
        return boxes[valid]
