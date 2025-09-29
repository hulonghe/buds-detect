import os
import re
import pandas as pd

def find_best_models_by_prefix(root_folder='runs'):
    # 权重配置
    weights = {
        'mAP': 0.3,
        'AP@0.50': 0.6,
        'Precision': 0.1
    }

    # 核心指标
    core_metrics = ["epoch", "mAP", "AP@0.50", "Precision", "Recall", "F1", "score"]

    # 存储每个子文件夹的最佳行和路径
    group_best_models = {}

    # 遍历所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'val_metrics_log.csv' in filenames:
            print(dirpath)
            csv_path = os.path.join(dirpath, 'val_metrics_log.csv')
            try:
                df = pd.read_csv(csv_path)

                # 确保需要的列都存在
                required_cols = ['mAP', 'AP@0.50', 'Precision']
                if not all(col in df.columns for col in required_cols):
                    print(f"跳过文件 {csv_path}：缺少必要列。")
                    continue

                # 计算加权得分
                df['score'] = sum(df[col] * weight for col, weight in weights.items())

                # 找到得分最高的那一行
                best_row = df.loc[df['score'].idxmax()]

                # 提取组前缀：匹配 *-e100-
                folder_name = os.path.basename(dirpath)
                match = re.search(r'.*-e100-', folder_name)
                if match:
                    group_key = match.group(0)  # 分组 key
                else:
                    group_key = "其他"

                # 如果该组还没记录，或者当前模型分数更高 → 更新
                if (group_key not in group_best_models) or \
                   (best_row['score'] > group_best_models[group_key]['data']['score']):
                    group_best_models[group_key] = {
                        'folder': dirpath,
                        'data': best_row.to_dict()
                    }

            except Exception as e:
                print(f"读取文件 {csv_path} 出错: {e}")

    # 转换成表格
    rows = []
    for group, entry in group_best_models.items():
        row = {k: entry['data'].get(k, None) for k in core_metrics}
        row['group'] = group
        row['folder'] = entry['folder']
        rows.append(row)

    df_result = pd.DataFrame(rows)

    # 调整列顺序
    df_result = df_result[["group"] + core_metrics + ["folder"]]

    print("\n各组最佳模型核心指标：")
    print(df_result.to_string(index=False))

    return df_result


# 使用
find_best_models_by_prefix("./runs/ablation")
