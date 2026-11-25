"""
OCR 引擎适配层（PaddleHub 兼容）

背景：项目原始依赖 `paddlehub.Module(name="chinese_ocr_db_crnn_mobile")`。
PaddleHub 2.x 已停止维护，其模型托管服务与 PaddlePaddle 3.x 不兼容，
且官方模型仓库经常不可用，导致旧环境无法直接复原。

本模块用 RapidOCR（PP-OCRv4 模型导出为 ONNX，与 chinese_ocr_db_crnn_mobile
同属百度 PP-OCR 系列）实现一个接口完全对齐的替代引擎，app.py 无需改动业务逻辑。
"""

import os
import time

import cv2
import numpy as np


class RapidOCREngine:
    """对外暴露与 paddlehub.Module.recognize_text 相同的方法签名与返回结构。"""

    def __init__(self, **kwargs):
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self.name = "chinese_ocr_db_crnn_mobile (RapidOCR/PP-OCRv4-onnx)"

    def recognize_text(self,
                       images,
                       use_gpu=False,
                       output_dir='ocr_result',
                       visualization=True,
                       box_thresh=0.5,
                       text_thresh=0.5):
        """
        参数与 PaddleHub 保持一致：
          images: List[np.ndarray] BGR 图像
          output_dir: 可视化结果保存目录
          visualization: 是否保存可视化图片

        返回：
          [{'data': [{'text', 'confidence', 'text_box_position'}], 'save_path': 'ocr_result/xxx.jpg'}]
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for image in images:
            raw, _ = self._engine(image)

            data = []
            for item in (raw or []):
                box, text, score = item[0], item[1], float(item[2])
                if score < text_thresh:
                    continue
                data.append({
                    'text': text,
                    'confidence': score,
                    'text_box_position': [[int(x), int(y)] for x, y in box],
                })

            save_path = ''
            if visualization:
                save_path = self._visualize(image, data, output_dir)

            results.append({'data': data, 'save_path': save_path})

        return results

    @staticmethod
    def _visualize(image, data, output_dir):
        """绘制检测框与识别文字。

        OpenCV 内置字体不支持中文（会显示成 ???），这里用 PIL + 系统中文字体绘制。
        """
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)

        font = None
        for font_path in (r"C:\Windows\Fonts\msyh.ttc",
                          r"C:\Windows\Fonts\simhei.ttf",
                          r"C:\Windows\Fonts\simsun.ttc"):
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 18)
                break

        for info in data:
            pts = [tuple(p) for p in info['text_box_position']]
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

            label = f"{info['text']} {info['confidence']:.2f}"
            x, y = pts[0]
            if font is not None:
                draw.text((x, max(y - 22, 0)), label, font=font, fill=(255, 0, 0))

        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        filename = f"ndarray_{time.time()}.jpg"
        full_path = os.path.join(output_dir, filename)
        cv2.imwrite(full_path, canvas)

        # 返回 Web 可访问的相对路径（正斜杠），前端 split('/') 取文件名即可
        return f"ocr_result/{filename}"
