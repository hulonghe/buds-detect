import os


def export_all_py_to_txt(output_file='all_py_files.txt'):
    base_dir = os.getcwd()
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir).replace('\\', '/')
                    out_f.write(f"# ===== {rel_path} =====\n")
                    try:
                        with open(full_path, 'r', encoding='utf-8') as py_file:
                            content = py_file.read()
                            out_f.write(content)
                    except Exception as e:
                        out_f.write(f"# [读取失败: {e}]\n")
                    out_f.write('\n\n')  # 添加分隔空行


if __name__ == '__main__':
    export_all_py_to_txt()
