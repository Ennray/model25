import os

labels_dir = "E:/work/yolov8/ultralytics/datasets/VisDrone/val/labels"
images_dir = "E:/work/yolov8/ultralytics/datasets/VisDrone/val/images"
output_dir = "E:/work/yolov8/ultralytics/datasets/VisDrone/val/labels2"

os.makedirs(output_dir, exist_ok=True)

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    image_file = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))
    if not os.path.exists(image_file):
        continue

    # 这里需要知道图片尺寸
    from PIL import Image
    img = Image.open(image_file)
    img_w, img_h = img.size

    new_lines = []
    with open(os.path.join(labels_dir, label_file), "r") as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            x1 = float(parts[0])
            y1 = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])
            class_id = int(parts[5]) - 1  # VisDrone 类别从 1 开始，YOLO 从 0 开始

            # 计算中心点
            x_center = x1 + w / 2
            y_center = y1 + h / 2

            # 归一化
            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h

            # 构造 YOLO 格式行
            new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
            new_lines.append(new_line)

    with open(os.path.join(output_dir, label_file), "w") as out_f:
        out_f.writelines(new_lines)

print(" 标签已成功转换为 YOLO 格式！")
