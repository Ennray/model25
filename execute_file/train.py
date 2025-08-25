from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./yolov8x.pt")
    model.train(
        data='./datasets/UAV_yaml/VisDrone.yaml',
        epochs=300,
        batch=4,
        lr0=0.01
    )

