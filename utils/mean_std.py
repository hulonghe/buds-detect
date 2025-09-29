import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_image(path, image_size):
    try:
        img = cv2.imread(path)  # BGR
        if img is None:
            return None
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype(np.float32) / 255.0  # Normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        return mean, std
    except Exception as e:
        return None


def get_mean_std(image_dir, image_size=640, sample_size=None, num_workers=8):
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))

    if sample_size is not None and sample_size < len(image_paths):
        image_paths = random.sample(image_paths, sample_size)

    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    valid_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_image, path, image_size): path for path in image_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing mean/std"):
            result = future.result()
            if result is None:
                continue
            mean, std = result
            mean_sum += mean
            std_sum += std
            valid_count += 1

    if valid_count == 0:
        raise ValueError("没有有效图像处理成功")

    mean_tensor = torch.tensor(mean_sum / valid_count, dtype=torch.float32)
    std_tensor = torch.tensor(std_sum / valid_count, dtype=torch.float32)
    # 保留四位小数并转为 tuple
    mean_rounded = tuple(round(v, 4) for v in mean_tensor.tolist())
    std_rounded = tuple(round(v, 4) for v in std_tensor.tolist())

    return mean_rounded, std_rounded


if __name__ == '__main__':
    # data_name = "teaRob.v9i.yolov11"
    # data_name = "tea-buds-owns"
    # data_name = "tea-bud-3"
    data_name = "C"
    data_root = r"E:/resources/datasets/tea-buds-database/" + data_name

    mean, std = get_mean_std(
        image_dir=data_root,
        image_size=320,
        num_workers=4
    )

    print("mean:", mean)
    print("std:", std)
