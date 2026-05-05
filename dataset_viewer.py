import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk, ImageDraw
import glob


class YOLODatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO数据集查看器和裁剪工具")
        self.root.geometry("1200x800")

        self.dataset_dir = ""
        self.images_dir = ""
        self.labels_dir = ""
        self.output_dir = ""
        self.image_files = []
        self.current_index = 0

        self.canvas = None
        self.photo = None
        self.original_image = None
        self.display_image = None

        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None

        self.bbox_coords = None

        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_select = tk.Button(top_frame, text="选择数据集文件夹", command=self.select_folder)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.lbl_info = tk.Label(top_frame, text="请选择数据集文件夹")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(top_frame, text="")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind("<Return>", self.on_enter_pressed)
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.btn_prev = tk.Button(bottom_frame, text="上一张 (←)", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(bottom_frame, text="下一张 (→)", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.lbl_image_info = tk.Label(bottom_frame, text="")
        self.lbl_image_info.pack(side=tk.LEFT, padx=20)

        help_text = "操作说明: 鼠标左键画框 → 按回车保存crop区域"
        self.lbl_help = tk.Label(bottom_frame, text=help_text, fg="blue")
        self.lbl_help.pack(side=tk.RIGHT, padx=5)

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择数据集文件夹")
        if not folder:
            return

        images_dir = os.path.join(folder, "images")
        labels_dir = os.path.join(folder, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            messagebox.showerror("错误", "所选文件夹必须包含 'images' 和 'labels' 子文件夹")
            return

        self.dataset_dir = folder
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = os.path.join(folder, "images_crops")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
        self.image_files = [f for f in self.image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not self.image_files:
            messagebox.showerror("错误", "images文件夹中没有找到图片文件")
            return

        self.current_index = 0
        self.lbl_info.config(text=f"数据集: {os.path.basename(folder)}")
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)

        self.load_current_image()

    def load_current_image(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.current_index]
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]

        self.original_image = Image.open(img_path).convert("RGB")
        img_width, img_height = self.original_image.size

        label_path = os.path.join(self.labels_dir, f"{name_without_ext}.txt")
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            x1 = int((x_center - width / 2) * img_width)
                            y1 = int((y_center - height / 2) * img_height)
                            x2 = int((x_center + width / 2) * img_width)
                            y2 = int((y_center + height / 2) * img_height)
                            bboxes.append((cls, x1, y1, x2, y2))

        self.display_image = self.original_image.copy()
        draw = ImageDraw.Draw(self.display_image)

        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
        for i, (cls, x1, y1, x2, y2) in enumerate(bboxes):
            color = colors[cls % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 15), f"cls:{cls}", fill=color)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 1000
            canvas_height = 700

        img_w, img_h = self.display_image.size
        scale = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        self.display_image = self.display_image.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.current_image_scale = scale
        self.current_image_offset_x = 0
        self.current_image_offset_y = 0

        self.current_bboxes = bboxes
        self.current_image_size = (img_width, img_height)

        self.lbl_status.config(text=f"标注数量: {len(bboxes)}")
        self.lbl_image_info.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
        self.bbox_coords = None

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        if self.drawing:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline="white", width=2, dash=(5, 3)
            )

    def on_mouse_up(self, event):
        self.drawing = False
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)

        self.bbox_coords = (x1, y1, x2, y2)

        self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=3, tags="crop_box")

    def on_enter_pressed(self, event):
        if self.bbox_coords is None:
            messagebox.showwarning("警告", "请先使用鼠标画一个框")
            return

        self.save_crop()

    def save_crop(self):
        if not self.image_files or self.bbox_coords is None:
            return

        x1_canvas, y1_canvas, x2_canvas, y2_canvas = self.bbox_coords
        scale = self.current_image_scale

        x1_img = int(x1_canvas / scale)
        y1_img = int(y1_canvas / scale)
        x2_img = int(x2_canvas / scale)
        y2_img = int(y2_canvas / scale)

        img_width, img_height = self.current_image_size
        x1_img = max(0, min(x1_img, img_width))
        y1_img = max(0, min(y1_img, img_height))
        x2_img = max(0, min(x2_img, img_width))
        y2_img = max(0, min(y2_img, img_height))

        if x2_img <= x1_img or y2_img <= y1_img:
            messagebox.showerror("错误", "裁剪区域无效")
            return

        crop_img = self.original_image.crop((x1_img, y1_img, x2_img, y2_img))
        crop_width, crop_height = crop_img.size

        valid_bboxes = []
        for cls, bx1, by1, bx2, by2 in self.current_bboxes:
            if bx2 > x1_img and bx1 < x2_img and by2 > y1_img and by1 < y2_img:
                new_x1 = max(0, bx1 - x1_img)
                new_y1 = max(0, by1 - y1_img)
                new_x2 = min(crop_width, bx2 - x1_img)
                new_y2 = min(crop_height, by2 - y1_img)

                if new_x2 > new_x1 and new_y2 > new_y1:
                    x_center = (new_x1 + new_x2) / 2 / crop_width
                    y_center = (new_y1 + new_y2) / 2 / crop_height
                    width = (new_x2 - new_x1) / crop_width
                    height = (new_y2 - new_y1) / crop_height
                    valid_bboxes.append((cls, x_center, y_center, width, height))

        img_path = self.image_files[self.current_index]
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]

        crop_img_name = f"{name_without_ext}_crop.jpg"
        crop_img_path = os.path.join(self.output_dir, crop_img_name)
        crop_img.save(crop_img_path, quality=95)

        label_path = os.path.join(self.output_dir, f"{name_without_ext}_crop.txt")
        with open(label_path, 'w') as f:
            for cls, xc, yc, w, h in valid_bboxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        messagebox.showinfo("保存成功", f"裁剪图片已保存到:\n{crop_img_path}\n标注数量: {len(valid_bboxes)}")

        self.canvas.delete("crop_box")
        self.bbox_coords = None

    def prev_image(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self, event=None):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODatasetViewer(root)
    root.mainloop()
