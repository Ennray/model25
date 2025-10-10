# tracking.py
import subprocess
import sys
from pathlib import Path

'''
封装模型5跟踪重捕，模型路径、输入视频路径、本地输出视频路径默认如下（需修改）
'''

# === 统一维护默认参数（都写死在这里） ===
DEFAULT_MODEL  = r"E:\work\model25789\runs\detect\train2\best.pt"
DEFAULT_INPUT  = r"E:\work\model25789\datasets\UAV_tracking_video\tracking_2.mp4"
# 固定输出目录与文件名
DEFAULT_OUTPUT = r"E:\work\model25789\runs\out\output_result_tracking_2.2.mp4"
DEFAULT_CONF   = 0.8

def _ensure_parent_dir(path_str: str):
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

def run_tracking():
    """
    使用写死的路径运行 tracking；生成 mp4 后返回“输出文件的绝对路径”。
    """
    # 规范化路径
    model_path  = Path(DEFAULT_MODEL).expanduser()
    input_path  = Path(DEFAULT_INPUT).expanduser()
    output_path = Path(DEFAULT_OUTPUT).expanduser()
    conf        = DEFAULT_CONF

    # 确保输出目录存在
    _ensure_parent_dir(str(output_path))

    # drone_tracking.py 的绝对路径（相对本文件定位更稳妥）
    base_dir = Path(__file__).resolve().parent
    drone_script = base_dir /"drone_tracking.py"

    python_exec = sys.executable or "python"
    cmd = [
        python_exec, str(drone_script),
        "--model", str(model_path),
        "--input", str(input_path),
        "--output", str(output_path),
        "--conf", str(conf),
    ]
    # 等价命令：python .\execute_file\drone_tracking.py --model ... --input ... --output ... --conf ...
    subprocess.run(cmd, check=True)

    return str(output_path.resolve())

if __name__ == "__main__":
    out = run_tracking()
    print(f"[tracking] 输出视频：{out}")
