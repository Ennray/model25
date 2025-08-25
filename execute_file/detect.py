from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./runs/detect/train5/weights/best.pt")
    results = model.predict(
        source="./datasets/VisDrone/test/images",
        save=True,
        data="./datasets/UAV_yaml/VisDrone.yaml",
        batch=4
    )

    # 遍历每张图片的结果
    for i, result in enumerate(results):
        num_boxes = len(result.boxes)
        print(f"Image {i+1}: Detected {num_boxes} objects.")