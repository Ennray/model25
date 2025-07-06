from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train5/weights/best.pt")
    model.predict(
        source="datasets/VisDrone/test/images",
        save=True,
        data="datasets/UAV_yaml/VisDrone.yaml",  # yaml 文件路径
        batch=4
    )
