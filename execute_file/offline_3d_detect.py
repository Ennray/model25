
"""
offline_3d.py
离线文件夹：每 30 张图 → 1 个目标的 3D 结果
"""

import os, glob, cv2, numpy as np
from ultralytics import YOLO
from collections import deque

'''
模型2.2.2.1单机视场内物体尺寸识别算法
输出为（编号、以检测框中心点位圆点的x轴坐标、y轴坐标、z轴坐标、识别物的宽度、识别物的高度、识别物的深度）
'''

# ---------- 固定相机参数 ----------
IMG_W   = 1920
IMG_H   = 1080
FOV_H_D = 60.0
RANGE_M = 1200.0

FX = IMG_W / (2 * np.tan(np.radians(FOV_H_D) / 2))
FY = FX * IMG_H / IMG_W
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# ---------- 工具 ----------
def bbox2cam3d(x1, y1, x2, y2):
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    x = (u - CX) / FX * RANGE_M
    y = (v - CY) / FY * RANGE_M
    z = RANGE_M
    w = (x2 - x1) / FX * RANGE_M
    h = (y2 - y1) / FY * RANGE_M
    return np.array([x, y, z]), np.array([w, h, 0.0])

class Target3D:
    def __init__(self, max_len=30):
        self.centers = deque(maxlen=max_len)
        self.whds    = deque(maxlen=max_len)

    def add(self, center, whd):
        self.centers.append(center)
        self.whds.append(whd)

    def final(self):
        if not self.centers:
            return None, None
        c_mean = np.array(self.centers).mean(axis=0)
        w_max  = np.array(self.whds)[:,0].max()
        h_max  = np.array(self.whds)[:,1].max()
        d_max  = np.array(self.whds)[:,2].max()
        return c_mean, np.array([w_max, h_max, d_max])

# ---------- 主流程 ----------
def main(config: dict | None = None) -> dict:
    img_dir   = './datasets/airplane_detect_images'  # 图片文件夹
    out_file  = './result_3d.txt'
    win_size  = 30                # 每 30 张算 1 个目标
    model     = YOLO('./runs/detect/train8/weights/best.pt')   # 权重路径
    out_info = []

    # 读取所有图片并按文件名排序
    img_list = sorted(glob.glob(os.path.join(img_dir, '*.*')))

    if len(img_list) < win_size:
        print(f'图片不足 {win_size} 张，无法计算！')
        return

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('target_id,cx,cy,cz,w,h,d\n')


        for start in range(0, len(img_list), win_size):
            end   = start + win_size
            imgs  = img_list[start:end]
            if len(imgs) < win_size:
                break  # 不足 30 张直接跳过

            target = Target3D(max_len=win_size)
            target_id = start // win_size + 1

            for img_path in imgs:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                results = model.predict(frame, conf=0.25, verbose=False)
                for r in results:
                    # 取置信度最高的框（若多框可加 NMS）
                    if len(r.boxes) == 0:
                        continue
                    best = r.boxes.xyxy[0].cpu().numpy()
                    center, whd = bbox2cam3d(*best)
                    target.add(center, whd)

            c_final, whd_final = target.final()
            if c_final is not None:
                line = f"{target_id},{c_final[0]:.3f},{c_final[1]:.3f},{c_final[2]:.3f}," \
                       f"{whd_final[0]:.3f},{whd_final[1]:.3f},{whd_final[2]:.3f}\n"
                f.write(line)
                #==============输出为（编号、以检测框中心点位圆点的x轴坐标、y轴坐标、z轴坐标、识别物的宽度、识别物的高度、识别物的深度）===================
                out_info.append([target_id, c_final[0], c_final[1], c_final[2], whd_final[0], whd_final[1], whd_final[2]])
                print(f"目标 {target_id}: 3D中心 {c_final} 尺寸 {whd_final}")
    out = {
        "size_result": out_info,
    }

    print("全部完成，结果已写入", out_file)
    return out

if __name__ == '__main__':
    result = main()