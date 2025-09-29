import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture


class ScaleClusterNet(nn.Module):
    def __init__(self, in_dim=5, hidden=16, num_scales=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_scales)
        )

    def forward(self, x):
        return self.net(x)


def generate_boxes(img_size=(320, 320), boxes=None):
    if boxes is None:
        boxes = []
    img_w, img_h = img_size
    all_features = []
    for B in range(len(boxes)):
        n_boxes = len(boxes[B])
        for N in range(n_boxes):
            box = boxes[B][N]
            x1, y1, x2, y2 = box
            w, h = box[2] - box[0], box[3] - box[1]
            cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
            sqrt_area = np.sqrt(w * h)
            all_features.append([sqrt_area, cx, cy, w / img_w, h / img_h])

    return torch.tensor(all_features, dtype=torch.float32)


def generate_random_boxes(img_size=(320, 320), num_images=1000, min_boxes=1, max_boxes=5, box_range=(2, 320)):
    img_w, img_h = img_size
    boxes = []
    for _ in range(num_images):
        box_list = []
        n_boxes = np.random.randint(min_boxes, max_boxes + 1)
        for _ in range(n_boxes):
            w, h = np.random.uniform(box_range[0], box_range[1], 2)
            x1 = np.random.uniform(0, img_w - w)
            y1 = np.random.uniform(0, img_h - h)
            x2 = x1 + w
            y2 = y1 + h
            box = [x1, y1, x2, y2]
            box_list.append(box)

        boxes.append(box_list)

    all_features = generate_boxes(img_size=img_size, boxes=boxes)
    return torch.tensor(all_features, dtype=torch.float32)


def init_pseudo_labels_gmm(features: torch.Tensor, strides: list[int], img_size=(320, 320)):
    img_area = img_size[0] * img_size[1]
    abs_area = features[:, 0] ** 2
    area_ratio = (abs_area / img_area).unsqueeze(1).numpy()

    gmm = GaussianMixture(n_components=len(strides), random_state=0)
    raw_labels = gmm.fit_predict(area_ratio)

    # 面积小的标签映射为大感受野
    mean_area = [area_ratio[raw_labels == i].mean() for i in range(len(strides))]
    sorted_label_map = np.argsort(mean_area)[::-1]  # 小目标 → 大感受野
    labels = torch.tensor([sorted_label_map[i] for i in raw_labels], dtype=torch.long)

    return labels


def init_pseudo_labels_kde(features: torch.Tensor, strides: list[int], img_size=(320, 320)) -> torch.Tensor:
    """
    使用 KDE 密度估计 + 极小值作为分段点的方式，替代原始 quantile 方法。
    自动获取感受野归属的伪标签，满足小目标给大感受野的原则。
    """
    img_area = img_size[0] * img_size[1]
    abs_area = features[:, 0] ** 2
    area_ratio = abs_area / img_area
    area_ratio_np = area_ratio.cpu().numpy()

    num_scales = len(strides)

    # KDE 密度估计
    kde = gaussian_kde(area_ratio_np)
    x_vals = np.linspace(area_ratio_np.min(), area_ratio_np.max(), 500)
    y_vals = kde(x_vals)

    # 寻找局部极小值作为分段边界（至多 num_scales-1 个）
    minima_idx = argrelextrema(y_vals, np.less)[0]
    split_candidates = x_vals[minima_idx]

    # 如果极小值太少，不足以分 num_scales，就 fallback 使用 quantile
    if len(split_candidates) < (num_scales - 1):
        print("⚠️ KDE 切分点不足，回退为 quantile 分段")
        quantile_points = torch.linspace(0, 1, steps=num_scales + 1)[1:-1]
        bins = torch.quantile(area_ratio, quantile_points).cpu().numpy()
    else:
        # 选出均匀间隔的 (num_scales - 1) 个分段点
        bins = np.linspace(0, len(split_candidates) - 1, num_scales - 1).astype(int)
        bins = split_candidates[bins]

    # 根据 bin 分段 → 标签（越大越大目标）
    labels = np.zeros_like(area_ratio_np, dtype=np.int64)
    for i, b in enumerate(bins):
        labels[area_ratio_np > b] += 1

    # 小目标给大感受野：反转标签（label 0 是最大感受野，对应小目标）
    labels = (num_scales - 1) - labels

    return torch.tensor(labels, dtype=torch.long)


def init_pseudo_labels(features: torch.Tensor, strides: list[int], img_size=(320, 320)):
    """
    动态划分伪标签区域（小中大目标），根据 box 占图像面积比例
    并按“小目标给大尺度”原则进行反向分配。
    """
    img_area = img_size[0] * img_size[1]
    abs_area = features[:, 0] ** 2
    area_ratio = abs_area / img_area

    num_scales = len(strides)
    # 自动生成区间点，例如3尺度 → [0.33, 0.66]，4尺度 → [0.25, 0.5, 0.75]
    quantile_points = torch.linspace(0, 1, steps=num_scales + 1)[1:-1]
    bins = torch.quantile(area_ratio, quantile_points)

    # 分配标签（label越大，目标越大）
    labels = torch.zeros_like(area_ratio, dtype=torch.long)
    for i, b in enumerate(bins):
        labels[area_ratio > b] += 1
    # 小目标给大尺度（label反向映射到stride）
    labels = (num_scales - 1) - labels

    return labels


def train_unsupervised_model(features: torch.Tensor,
                             pseudo_labels: torch.Tensor,
                             num_scales: int = 3,
                             hidden: int = 16,
                             lr: float = 1e-2,
                             epochs: int = 30,
                             batch_size: int = 256):
    """
    使用小批量训练 ScaleClusterNet，执行无监督伪标签学习
    """
    device = features.device
    dataset = TensorDataset(features, pseudo_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = ScaleClusterNet(in_dim=features.shape[1], hidden=hidden, num_scales=num_scales).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    tqdm_bar = tqdm(range(epochs), desc="训练模型")
    for epoch in tqdm_bar:
        net.train()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = net(x_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataset)
        tqdm_bar.set_postfix(loss=avg_loss)

    return net


def get_sqrt_area_range_by_scale(net, features, feats_size, num_scales=3):
    net.eval()
    with torch.no_grad():
        logits = net(features)
        preds = torch.argmax(logits, dim=1)
        sqrt_areas = features[:, 0]

    # 按类别统计每类面积均值
    class_avg = []
    for s in range(num_scales):
        areas = sqrt_areas[preds == s]
        avg = areas.mean().item() if len(areas) > 0 else float('inf')
        class_avg.append((s, avg))

    # 面积小的类别 → 分配给大感受野（stride小的）
    sorted_classes = sorted(class_avg, key=lambda x: x[1])  # 从小到大
    class_to_feat_idx = {
        class_id: feat_idx for feat_idx, (class_id, _) in enumerate(sorted_classes)
    }

    # 重新统计并输出
    final_ranges = [None] * num_scales
    for class_id, feat_idx in class_to_feat_idx.items():
        areas = sqrt_areas[preds == class_id]
        if len(areas) > 0:
            min_a, max_a = areas.min().item(), areas.max().item()
            final_ranges[feat_idx] = (min_a, max_a)
        else:
            final_ranges[feat_idx] = (None, None)

    return final_ranges


def train_DEC_(img_size, strides, feats_size, features, epochs=100):
    # print("📌 初始化伪标签（按感受野归属）...")
    pseudo_labels = init_pseudo_labels_gmm(features, strides, img_size)
    # print("🧠 训练极简自监督聚类网络...")
    net = train_unsupervised_model(features, pseudo_labels, num_scales=len(strides), epochs=epochs)
    # print("📈 每个尺度对应 box sqrt(area) 范围：")
    area_ranges = get_sqrt_area_range_by_scale(net, features, feats_size, num_scales=len(strides))

    return area_ranges


# ========================= MAIN =========================
if __name__ == '__main__':
    strides = [8, 16, 32]
    feats_size = [(40, 40), (20, 20), (10, 10)]
    img_size = (320, 320)

    print("⏳ 正在生成随机框并构造特征...")
    features = generate_random_boxes(img_size=img_size, num_images=2000)

    area_ranges = train_DEC_(img_size, strides, feats_size, features, epochs=200)
    for i, (min_val, max_val) in enumerate(area_ranges):
        if min_val is not None:
            print(f"  尺度 {i + 1} {feats_size[i]} (stride={strides[i]}): sqrt(area) ∈ [{min_val:.2f}, {max_val:.2f}]")
        else:
            print(f"  尺度 {i + 1} {feats_size[i]} (stride={strides[i]}): 无数据")
