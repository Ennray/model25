# save_pixel_boxes.py
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor

class MyPredictor(DetectionPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 准备输出文件
        self.out_file = Path('pixel_boxes.txt')
        self.out_file.write_text('')   # 清空旧内容

    def write_results(self, idx, results, batch):
        super().write_results(idx, results, batch)   # 让父类正常画框/写 txt

        # 取出当前 batch 中的第 idx 张图的信息
        r = results[idx]
        img_path = Path(batch['im_file'][idx]).name  # 原始图片文件名

        # 把每个框写进文件
        with self.out_file.open('a') as f:
            for *xyxy, conf, cls in reversed(r.boxes.data):
                x1, y1, x2, y2 = map(int, xyxy)
                f.write(f'{img_path} {int(cls)} {x1} {y1} {x2} {y2} {conf:.4f}\n')

# --------------- 主程序 ---------------
model = YOLO('yolov8n.pt')
model.predictor = MyPredictor(
    overrides={'model': model.model, 'source': 'images/'}  # 可以是单图或文件夹
)
model.predict()