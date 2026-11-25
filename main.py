import os
import sys

import cv2

from plate_utils import find_license_plates

# 相对路径统一基于代码目录，避免从别的目录启动时找错文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 检查文件是否存在
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True


# text.txt 里面存放了待处理图片的路径信息
# 一个文件一行
with open(os.path.join(BASE_DIR, 'test.txt'), 'r', encoding='utf-8') as f:
    raw_paths = [line.strip() for line in f if line.strip()]

# 相对路径按代码目录解析，绝对路径保持不变
test_img_path = [
    p if os.path.isabs(p) else os.path.join(BASE_DIR, p) for p in raw_paths
]
test_img_path = [p for p in test_img_path if check_file_exists(p)]

print("Loaded image paths:", test_img_path)

# 加载OCR模型：优先 PaddleHub，不可用时回退到接口兼容的 RapidOCR
try:
    import paddlehub as hub
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    print("Model loaded successfully. (PaddleHub)")
except Exception as e:
    print(f"PaddleHub unavailable, falling back to RapidOCR: {e}")
    from ocr_engine import RapidOCREngine
    ocr = RapidOCREngine()
    print(f"Fallback model loaded successfully. ({ocr.name})")

# 读取测试文件夹test.txt中的照片路径
np_images = [cv2.imread(image_path) for image_path in test_img_path]

# 检查图像是否正确加载
if any(image is None for image in np_images):
    print("Some images failed to load. Please check the image paths.")
    sys.exit(1)

# 识别文字
results = ocr.recognize_text(
    images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
    use_gpu=False,    # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
    output_dir=os.path.join(BASE_DIR, 'ocr_result'),  # 图片的保存路径，默认设为 ocr_result；
    visualization=True,  # 是否将识别结果保存为图片文件；
    box_thresh=0.5,     # 检测文本框置信度的阈值；
    text_thresh=0.5     # 识别中文文本置信度的阈值；
)

# 检查识别结果
if not results:
    print("No results found. Please check the images and model.")
else:
    for result in results:
        data = result['data']
        save_path = result['save_path']
        if not data:
            print(f"No text found in image saved at {save_path}")
            continue

        print(f"\n===== {save_path} =====")
        print(f"共识别 {len(data)} 段文本：")
        for information in data:
            print(f"  [{information['confidence']:.4f}] {information['text']}")

        # 过滤车牌号码
        license_plates = find_license_plates(data)
        if license_plates:
            print("车牌识别结果：")
            for plate in license_plates:
                print('  text: ', plate['text'], '\n  confidence: ',
                      plate['confidence'], '\n  text_box_position: ',
                      plate['text_box_position'])
        else:
            print("未识别到车牌")
