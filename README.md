``` 
1.1 虚拟环境搭建
#创建虚拟环境，python>3.8版本即可，这里我是用的是3.10
conda create -n yolov8_cuda python=3.10 -y

#激活虚拟环境
conda activate yolov8_cuda

1.2 关于设备配置
可以运行以下命令进行测试，其会分别输出Pytorch的版本号、输出编译Pytorch时使用的cuda版本、检查当前系统是否检测到cuda(也就是是否能用GPU)、并打印GPU名称，
例如我的Pytorch版本是2.5.1，cuda版本是12.1
python GPU_test.py

1.3 关于数据结构
我是按照VisDrone的数据集格式进行搭建的，如下（仅仅适用于VisDrone.yaml）：
datasets/
└── VisDrone/
    ├── test/
    │   ├── images/
    │   └── labels/ 
    ├── train/
    │   └── images/     
    └── val/
        ├── images/
        └── labels/   

1.4 关于数据集脚本
数据集脚本的路径为datasets\UAV_yaml\VisDrone.yaml，VisDrone.yaml文件里包含test、train、val的路径设置以及分类

1.5 如何运行
#如果需要使用yolo命令则安装ultralytics和yolo
pip install ultralytics
pip install yolo

#当你支持yolo命令时可以直接使用以下命令进行训练
yolo detect train data=datasets/UAV_yaml/VisDrone.yaml model=yolov8x.yaml pretrained=yolov8x.pt epochs=300 batch=4 lr0=0.01

#当你支持yolo命令时可以直接使用以下命令进行检测
yolo detect val data=datasets/UAV_yaml/VisDrone.yaml model=runs/train5/weights/best.pt batch=4

#如果不能使用yolo命令可以直接使用以下命令进行训练（可自行更改）
python test.py

#如果不能使用yolo命令可以直接使用以下命令进行检测（可自行更改）
python detect.py

``` 