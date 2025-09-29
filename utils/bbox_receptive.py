import numpy as np
from sklearn.cluster import KMeans


def scale_box_to_level(box, img_size, stride):
    H, W = img_size
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 缩放至当前层级下的坐标
    scaled_w = w / stride
    scaled_h = h / stride
    scaled_cx = cx / stride
    scaled_cy = cy / stride

    return {
        'scaled_w': scaled_w,
        'scaled_h': scaled_h,
        'scaled_cx': scaled_cx,
        'scaled_cy': scaled_cy,
        'scaled_s': np.sqrt(scaled_w * scaled_h),
        'scaled_ratio': scaled_w / scaled_h,
    }


def extract_features_for_all_boxes(labels_boxes, img_size, strides):
    all_features = []

    for b in range(len(labels_boxes)):
        N = len(labels_boxes[b])
        for n in range(N):
            box = labels_boxes[b][n]
            features_per_level = []
            for i, stride in enumerate(strides):
                feat = scale_box_to_level(box, img_size, stride)
                feature_vec = [
                    feat['scaled_s'],  # 尺度
                    feat['scaled_ratio'],  # 宽高比
                    feat['scaled_cx'],  # 中心x
                    feat['scaled_cy'],  # 中心y
                ]
                features_per_level.append(feature_vec)
            all_features.append(features_per_level)

    return np.array(all_features)  # shape: [total_boxes, num_levels, 4]


def assign_boxes_by_clustering(features, num_levels):
    total_boxes, _, feat_dim = features.shape
    cluster_centers = np.zeros((num_levels, feat_dim))

    # 先初始化聚类中心（可以随机或根据统计）
    for level in range(num_levels):
        cluster_centers[level] = np.mean(features[:, level], axis=0)

    kmeans = KMeans(n_clusters=num_levels, init=cluster_centers, n_init=1)
    kmeans.fit(features.reshape(-1, feat_dim))

    # 每个框对应哪个层级
    assignments = kmeans.predict(features.reshape(-1, feat_dim))
    assignments = assignments.reshape(total_boxes, -1)

    # 最终每个框只属于一个层级（最近的那个）
    final_assignments = []
    for i in range(total_boxes):
        counts = np.bincount(assignments[i])
        best_level = np.argmax(counts)
        final_assignments.append(best_level)

    return final_assignments


def calculate_receptive_fields_with_clustering(img_size, labels_boxes, feats_size, strides):
    num_levels = len(strides)
    level_w = [[] for _ in range(num_levels)]
    level_h = [[] for _ in range(num_levels)]
    # 提取特征
    features = extract_features_for_all_boxes(labels_boxes, img_size, strides)
    # 使用聚类分配层级
    assignments = assign_boxes_by_clustering(features, num_levels)

    # 收集每个层级对应的宽高
    box_idx = 0
    for b in range(len(labels_boxes)):
        N = len(labels_boxes[b])
        for n in range(N):
            box = labels_boxes[b][n]
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            level = assignments[box_idx]
            level_w[level].append(w)
            level_h[level].append(h)
            box_idx += 1

    # 计算感受野大小
    rf_sizes = []
    for i in range(num_levels):
        if level_w[i]:
            max_w = max(level_w[i])
            max_h = max(level_h[i])
            rf_sizes.append((max_h, max_w))
        else:
            rf_sizes.append((0, 0))

    for i in range(num_levels):
        print(f"Level {i} 收到 {len(level_w[i])} 个目标")

    return rf_sizes
