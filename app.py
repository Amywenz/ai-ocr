import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
from werkzeug.utils import secure_filename

from plate_utils import find_license_plates

app = Flask(__name__)

# 所有相对路径都基于代码所在目录，避免从别的目录启动时读写到错误位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'ocr_result')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
# 改完 templates/index.html 后刷新浏览器即可生效，不用重启服务
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载OCR模型
# 优先使用原始 PaddleHub 模型；PaddleHub 不可用时回退到接口兼容的 RapidOCR 引擎
try:
    import paddlehub as hub
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    print("Model loaded successfully. (PaddleHub)")
except Exception as e:
    print(f"PaddleHub unavailable, falling back to RapidOCR: {e}")
    from ocr_engine import RapidOCREngine
    ocr = RapidOCREngine()
    print(f"Fallback model loaded successfully. ({ocr.name})")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path, is_license_plate=False):
    """处理图像并返回OCR结果"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image"

        results = ocr.recognize_text(
            images=[image],
            use_gpu=False,
            output_dir=RESULT_FOLDER,
            visualization=True,
            box_thresh=0.5,
            text_thresh=0.5
        )

        if not results:
            return None, "No text detected"

        result = results[0]
        data = result['data']
        save_path = result['save_path']

        if not data:
            return None, "No text found in image"

        if is_license_plate:
            # 过滤车牌号码（含分隔点归一化、被切分车牌框的合并）
            return find_license_plates(data), save_path
        else:
            # 返回所有文本
            texts = [{
                'text': info['text'],
                'confidence': info['confidence'],
                'text_box_position': info['text_box_position']
            } for info in data]
            return texts, save_path

    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    """通用OCR API接口"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        texts, save_path = process_image(filepath, is_license_plate=False)

        if texts is None:
            return jsonify({'error': save_path}), 400

        return jsonify({
            'texts': texts,
            'result_image': save_path
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/license_plate', methods=['POST'])
def license_plate_api():
    """车牌识别API接口"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        plates, save_path = process_image(filepath, is_license_plate=True)

        if plates is None:
            return jsonify({'error': save_path}), 400

        return jsonify({
            'license_plates': plates,
            'result_image': save_path
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/results/<filename>')
def serve_result(filename):
    """提供结果图像访问"""
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
