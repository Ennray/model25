# server_track_api.py
import os
import mimetypes
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import BadRequest
import tracking as tracking  # 调用上面写死路径的 tracking

app = Flask(__name__)
app.json.sort_keys = False
app.json.ensure_ascii = False

# 固定的保存与运行路径
DATA_DIR = Path(r"E:\work\model25789\datasets\UAV_tracking_video")
FIXED_INPUT_NAME = "input_video.mp4"  # 所有上传视频都覆盖为这个名字

def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "video/mp4"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/upload_and_track", methods=["POST"])
def upload_and_track():

    try:
        # 1) 取文件
        f = request.files.get("media") or request.files.get("video")
        if not f or not f.filename:
            raise BadRequest("请使用 multipart/form-data 上传文件，字段名为 'media' 或 'video'。")

        # 2) 保存到固定路径（覆盖旧文件）
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fixed_input_path = DATA_DIR / FIXED_INPUT_NAME
        f.save(str(fixed_input_path))

        # 3) 触发 tracking（使用 tracking.py 中写死的路径）
        output_path = tracking.run_tracking()  # 返回输出 mp4 的绝对路径
        if not os.path.isfile(output_path):
            raise BadRequest(f"输出视频未生成：{output_path}")

        # 4) 返回视频；默认下载，如需在线播放把 as_attachment 改为 False
        return send_file(
            output_path,
            mimetype=_guess_mime(output_path),
            as_attachment=True,
            download_name=os.path.basename(output_path),
        )

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"内部错误: {e}"}), 500

if __name__ == "__main__":
    # 如需对外服务可改 host="0.0.0.0"
    app.run(port=5000, debug=True)
