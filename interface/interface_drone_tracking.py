import os
import mimetypes
from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import BadRequest
import numpy as np
from execute_file import offline_3d_detect as SizeRecognition

app = Flask(__name__)

# JSON 输出设置
app.json.sort_keys = False
app.json.ensure_ascii = False

def to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj

def _extract_config_from_request():
    """
    统一从 POST JSON 或 GET Query 中取 config。
    - POST: { "config": {...} } 或直接 {...}
    - GET:  ?input=...&type=video&conf=0.5 等（自动拼成 config）
    """
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise BadRequest("JSON body 必须是对象（dict）。")
        config = payload.get("config") if "config" in payload else payload
        if config is not None and not isinstance(config, dict):
            raise BadRequest("config 必须是对象（dict）。")
        return config or {}
    else:  # GET
        cfg = {}
        # 常用参数（按你实际 SizeRecognition.main 需要的键名调整）
        if 'input' in request.args:
            cfg['input'] = request.args.get('input')
        if 'type' in request.args:
            cfg['type'] = request.args.get('type')  # image / video
        if 'conf' in request.args:
            try:
                cfg['conf'] = float(request.args.get('conf'))
            except Exception:
                raise BadRequest("conf 必须是浮点数。")
        return cfg

def _pick_output_path(result):
    """
    从 SizeRecognition.main 的返回值中提取视频输出路径。
    兼容：返回字符串 / dict 的多种键名。
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("output", "output_path", "video_path", "save_path"):
            if key in result and isinstance(result[key], str):
                return result[key]
    # 兜底：有时返回里有 'data' 再包一层
    if isinstance(result, dict) and isinstance(result.get('data'), dict):
        for key in ("output", "output_path", "video_path", "save_path"):
            v = result['data'].get(key)
            if isinstance(v, str):
                return v
    return None

@app.route('/drone_tracking_main', methods=['GET', 'POST'])
def size_recognition_main():
    """
    处理并直接返回视频文件。
    - POST JSON: {"config": {"input": "...", "type": "video", "conf": 0.5, ...}}
    - GET: /size_recognition_main?input=...&type=video&conf=0.5
    额外 query:
      - download=1  -> 以附件下载；默认 inline 预览
    """
    try:
        # 1) 拿到 config
        config = _extract_config_from_request()

        if not isinstance(config, dict):
            raise BadRequest("config 必须是对象（dict）。")

        # 2) 调用你的主流程
        result = SizeRecognition.main(config)

        # 3) 提取输出视频路径
        output_path = _pick_output_path(result)
        if not output_path:
            # 如果你的 main 直接把路径放在 result 之外，也可以改成固定路径
            raise BadRequest("未从结果中解析到输出视频路径，请确保 main 返回了 'output' 或 'output_path'。")

        # 转成绝对路径并校验
        output_path = os.path.abspath(output_path)
        if not os.path.isfile(output_path):
            raise BadRequest(f"输出视频不存在：{output_path}")

        # 4) 选择 mimetype（默认 mp4）
        mime, _ = mimetypes.guess_type(output_path)
        if not mime:
            mime = "video/mp4"

        # 5) 是否下载
        download_flag = request.args.get('download', '0').lower() in ('1', 'true', 'yes')

        # 6) 直接返回视频文件
        # 注：send_file 默认设置 Content-Length；inline 显示更利于浏览器直接播放
        return send_file(
            output_path,
            mimetype=mime,
            as_attachment=download_flag,
            download_name=os.path.basename(output_path)
        )

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        # 捕获其它异常，避免 500 返回无结构信息
        return jsonify({"status": "error", "message": f"内部错误: {e}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
