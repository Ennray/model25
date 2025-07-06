conda create -n yolov8_cuda python=3.10 -y

conda activate yolov8_cuda

pip install ultralytics


yolo detect train data=datasets/UAV_yaml/VisDrone.yaml model=yolov8x.yaml pretrained=yolov8x.pt epochs=300 batch=4 lr0=0.01 

yolo detect val data=datasets/UAV_yaml/VisDrone.yaml model=runs/train5/weights/best.pt batch=4