import numpy as np
import torch
from torchvision import datasets
from pycocotools.coco import COCO


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None, mean=[0.4697, 0.4465, 0.4073], std=[0.2388, 0.2338, 0.2371]):
        """
        COCO Custom Dataset 类，支持 albumentations 数据增强和目标格式构建
        :param root: 图像文件夹路径 (如 'autodl-tmp/coco2017/train2017')
        :param annFile: COCO 标注文件路径 (如 'autodl-tmp/coco2017/annotations/instances_train2017.json')
        :param transforms: albumentations 的转换操作
        :param mean: 图像标准化均值
        :param std: 图像标准化标准差
        """
        self.root = root
        self.annFile = annFile
        self.coco = COCO(annFile)
        self.transforms = transforms
        self.mean = mean
        self.std = std

        # 常用的80个类别的名称列表
        self.common_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
            'toothbrush'
        ]
        
        # 创建类别名到ID的映射
        self.category_name_to_id = {name: idx for idx, name in enumerate(self.common_categories)}
        self.category_id_to_name = {idx: name for idx, name in enumerate(self.common_categories)}

        # 获取所有图片ID
        self.img_ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        targets = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = self._load_image(path)  # 假设_load_image方法返回PIL.Image对象

        bboxes = []
        labels = []

        for target in targets:
            if target['category_id'] not in self.category_name_to_id.values():
                continue  # 忽略非目标类别

            bbox = target['bbox']  # [x, y, width, height]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h

            # 检查边框合法性
            if x_min >= x_max or y_min >= y_max:
                continue

            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.category_id_to_name[target['category_id']])

        if self.transforms is not None:
            image_np = np.array(image)
            transformed = self.transforms(
                image=image_np,
                bboxes=bboxes,
                labels=[self.category_name_to_id[label] for label in labels]  # 注意这里直接转换为类别索引
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # 构建目标字典
        target_dict = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),  # 直接使用transformed后的label
        }

        return image, target_dict

    def _load_image(self, path):
        return datasets.folder.pil_loader(f"{self.root}/{path}")

    def __len__(self):
        return len(self.img_ids)

    def denormalize_albumentations(self, img: torch.Tensor, to_numpy=True):
        """
        反标准化图像张量，返回 [0, 255] 的 uint8 图像（RGB）
        输入 img: Tensor[3, H, W]，标准化后的图像
        """
        mean = torch.tensor(self.mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)

        if to_numpy:
            img = (img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, C]
        return img

    @property
    def class_to_idx(self):
        return self.category_name_to_id

    @property
    def idx_to_class(self):
        return self.category_id_to_name