import os
import shutil

def reorganize_dataset(source_root, target_root):
    """
    将 source_root 下的所有文件（包括子目录）复制到 target_root 中，
    保留原来的相对路径结构。
    """
    if not os.path.exists(source_root):
        raise ValueError(f"Source path {source_root} 不存在")
    os.makedirs(target_root, exist_ok=True)

    for root, dirs, files in os.walk(source_root):
        # 计算相对路径
        rel_path = os.path.relpath(root, source_root)
        target_dir = os.path.join(target_root, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            # 拷贝文件，如果想移动则用 shutil.move()
            shutil.copy2(src_file, dst_file)
            # 如果需要移动而不是复制，可以使用：
            # shutil.move(src_file, dst_file)

    print(f"完成！已将 {source_root} 的所有文件复制到 {target_root}")

# -------------------
# 使用示例
# -------------------
if __name__ == "__main__":
    source = r"E:\resources\datasets\tea-buds-database\tea-buds-owns\labels\train"
    target = r"E:\resources\datasets\tea-buds-database\tea-buds-owns\train\labels"
    reorganize_dataset(source, target)
