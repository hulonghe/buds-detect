import os
import cv2
import shutil


def compress_images(input_dir, output_dir, max_width=1600, image_exts={'.jpg', '.jpeg', '.png'}):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in image_exts:
                continue

            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 读取图片
            img = cv2.imread(input_path)
            if img is None:
                print(f"⚠️ 跳过无法读取的图像: {input_path}")
                continue

            h, w = img.shape[:2]

            if w > max_width:
                scale = max_width / w
                new_w = max_width
                new_h = int(h * scale)
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized_img)
                print(f"✅ 压缩保存: {rel_path}  -> {new_w}x{new_h}")
            else:
                shutil.copy2(input_path, output_path)
                print(f"➡️ 跳过压缩: {rel_path} （宽度 {w}）")

            count += 1

    print(f"\n✅ 完成，共处理图像：{count} 张。")


# 示例用法（直接调用）
if __name__ == "__main__":
    input_dir = r"/root/autodl-tmp/teaRob.v9i.yolov11/train/images"
    output_dir = r"/root/autodl-tmp/teaRob.v9i.yolov11/train/resized_images"
    compress_images(input_dir, output_dir, max_width=1200)
