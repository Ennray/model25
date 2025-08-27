import json
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import numpy as np
import offline_3d_detect as SizeRecognition


'''
模型2.2.2.1单机视场内物体尺寸识别接口
'''

app = Flask(__name__)

# # 关闭键排序 & 允许中文直出
# app.config['JSON_SORT_KEYS'] = False      # 不按键名排序（保留插入顺序）
# app.config['JSON_AS_ASCII'] = False       # 中文不转义

# Flask>=2.3 可以这样写（任选其一，别同时写两套）:
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
        # Python 3.7+ 字典本身按插入顺序保序
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj

@app.route('/execute_main', methods=['GET'])
def execute_main():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise BadRequest("JSON body 必须是对象（dict）。")

        config = payload.get("config") if "config" in payload else payload
        if config is not None and not isinstance(config, dict):
            raise BadRequest("config 必须是对象（dict）。")

        result = SizeRecognition.main(config)
        result_jsonable = to_jsonable(result)
        return jsonify({"status": "ok", "data": result_jsonable}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
