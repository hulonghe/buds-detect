import numpy as np
from torchvision import datasets
import torch


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, transforms, mean, std):
        self.mean = mean
        self.std = std
        self._dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=False)
        self.transforms = transforms
        self.class_name_to_id = {
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20
        }

    def __getitem__(self, idx):
        image, target = self._dataset[idx]
        image = image.convert("RGB")

        bboxes = []
        labels = []

        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['name'])

        if self.transforms is not None:
            # 转换为 numpy array
            image_np = np.array(image)
            transformed = self.transforms(
                image=image_np,
                bboxes=bboxes,
                labels=labels
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # 构造目标字典
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor([self.class_to_idx[label] - 1 for label in labels], dtype=torch.int64),
        }

        return image, target

    def __len__(self):
        return len(self._dataset)

    def denormalize_albumentations(self, img: torch.Tensor, to_numpy=True):
        """
        反标准化 albumentations 处理过的图像张量，返回 [0, 255] 的 uint8 图像（RGB）
        输入 img: Tensor[3, H, W]，标准化后的图像
        """
        if self.transforms:
            mean = torch.tensor(self.mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            std = torch.tensor(self.std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            img = img * std + mean  # 反标准化
            img = img.clamp(0, 1)

        if to_numpy:
            img = (img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, C]
        return img

    @property
    def class_to_idx(self):
        return self.class_name_to_id
