import os


def compare_filenames(folder_a, folder_b, recursive=False, extensions=None):
    """
    比较两个文件夹的文件名是否一致，返回差异。

    参数:
        folder_a: 第一个文件夹路径
        folder_b: 第二个文件夹路径
        recursive: 是否递归子目录，默认 False
        extensions: 可选，只比对特定文件类型，如 ['.jpg','.png']

    返回:
        dict:
            'only_in_a': 只在 A 中的文件名
            'only_in_b': 只在 B 中的文件名
            'in_both': 两个文件夹都有的文件名
    """

    def list_files(folder):
        file_list = []
        if recursive:
            for root, _, files in os.walk(folder):
                for f in files:
                    if extensions is None or os.path.splitext(f)[1].lower() in extensions:
                        file_list.append(f)
        else:
            for f in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, f)):
                    if extensions is None or os.path.splitext(f)[1].lower() in extensions:
                        file_list.append(f)
        return set(file_list)

    set_a = list_files(folder_a)
    set_b = list_files(folder_b)

    only_in_a = sorted(list(set_a - set_b))
    only_in_b = sorted(list(set_b - set_a))
    in_both = sorted(list(set_a & set_b))

    return {
        'only_in_a': only_in_a,
        'only_in_b': only_in_b,
        'in_both': in_both
    }


# -----------------------
# 使用示例
# -----------------------
if __name__ == "__main__":
    folder1 = r"C:\Users\Administrator\Desktop\pred-labels\val-demo-teaRob.v9i.yolov11.yolo12l.pt\labels"
    folder2 = r"C:\Users\Administrator\Desktop\pred-labels\val-demo-teaRob.v9i.yolov11.ours\labels"

    diff = compare_filenames(folder1, folder2, recursive=True, extensions=['.txt', '.png'])
    print("只在 A 中的文件:", diff['only_in_a'])
    print("只在 B 中的文件:", diff['only_in_b'])
    print("两个文件夹都有的文件:", diff['in_both'])
