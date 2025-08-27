# tracking.py
import subprocess
import sys
from pathlib import Path

# —— 写死模型/输入/输出路径 & 置信度 ——
DEFAULT_MODEL  = r"E:\work\model25789\runs\detect\train1\best.pt"
DEFAULT_INPUT  = r"E:\work\model25789\datasets\UAV_tracking_video\input_video.mp4"
DEFAULT_OUTPUT = r"E:\work\model25789\runs\out\output_result_tracking.mp4"
DEFAULT_CONF   = 0.5

def _ensure_parent_dir(path_str: str):
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

def run_tracking():
    """
    使用写死的路径运行 tracking；
    生成 mp4 后返回“输出文件的绝对路径”（字符串）。
    """
    model_path  = Path(DEFAULT_MODEL).expanduser()
    input_path  = Path(DEFAULT_INPUT).expanduser()
    output_path = Path(DEFAULT_OUTPUT).expanduser()
    conf        = DEFAULT_CONF

    _ensure_parent_dir(str(output_path))

    # 用相对 tracking.py 的位置来定位 drone_tracking.py，更稳妥
    base_dir = Path(__file__).resolve().parent
    drone_script = base_dir / "execute_file" / "drone_tracking.py"

    python_exec = sys.executable or "python"
    cmd = [
        python_exec, str(drone_script),
        "--model", str(model_path),
        "--input", str(input_path),
        "--output", str(output_path),
        "--conf", str(conf),
    ]
    subprocess.run(cmd, check=True)

    return str(output_path.resolve())

if __name__ == "__main__":
    out = run_tracking()
    print(f"[tracking] 输出视频：{out}")
