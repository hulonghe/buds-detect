import os
import pandas as pd

def find_best_models(root_folder='runs'):
    # 权重配置
    weights = {
        'mAP': 0.3,
        'AP@0.50': 0.6,
        'Precision': 0.1
    }

    # 核心指标
    core_metrics = ["epoch", "mAP", "AP@0.50", "Precision", "Recall", "F1", "score"]

    # 存储每个子文件夹的最佳行
    best_models = []

    # 遍历所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'val_metrics_log.csv' in filenames:
            # if "reFalse" not in dirpath:
            #     continue
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

                # 保存结果
                entry = {k: best_row.get(k, None) for k in core_metrics}
                entry['folder'] = dirpath
                best_models.append(entry)

            except Exception as e:
                print(f"读取文件 {csv_path} 出错: {e}")

    # 转换成 DataFrame
    df_result = pd.DataFrame(best_models)

    # 按 score 从高到低排序
    df_result = df_result.sort_values(by="score", ascending=False).reset_index(drop=True)

    print("\n所有文件夹最佳模型（按 score 排序）：")
    print(df_result.to_string(index=False))

    return df_result


# 使用
find_best_models("./runs/ablation")
