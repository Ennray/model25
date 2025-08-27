# server_track_api.py
import os
import mimetypes
from flask import Flask, jsonify, send_file
from werkzeug.exceptions import BadRequest
from tracking import run_tracking  # 只调用写死路径的 tracking

'''
模型5跟踪重捕模型接口
'''

app = Flask(__name__)
app.json.sort_keys = False
app.json.ensure_ascii = False

def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "video/mp4"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/track", methods=["GET"])
def track_and_download():
    """
    零参数：调用 tracking.py 的写死配置，生成 mp4 并直接作为“下载文件”返回。
    """
    try:
        output_path = run_tracking()  # 只跑写死的那一套
        if not os.path.isfile(output_path):
            raise BadRequest(f"输出视频未生成：{output_path}")

        return send_file(
            output_path,
            mimetype=_guess_mime(output_path),
            as_attachment=True,  # 默认下载
            download_name=os.path.basename(output_path),
        )
    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"内部错误: {e}"}), 500

if __name__ == "__main__":
    # 如需对外访问可改 host="0.0.0.0"
    app.run(port=5000, debug=True)
