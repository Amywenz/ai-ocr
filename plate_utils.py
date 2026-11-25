"""车牌文本的归一化、校验与检测框合并（app.py 与 main.py 共用）"""

import re

# 省份简称 + 字母 + 4~6 位字母数字（普通车牌 7 位、新能源车牌 8 位）
PLATE_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵黔粤青藏川宁琼使领'
PLATE_PATTERN = re.compile(rf'^[{PLATE_PROVINCES}][A-Z][A-Z0-9]{{4,6}}$')


def normalize_plate_text(text):
    """去掉车牌中的分隔点、空格等噪声并转为大写，例如 '辽A·2524N' -> '辽A2524N'"""
    return re.sub(r'[·•.\-\s]', '', text).upper()


def is_plate_text(text):
    return bool(PLATE_PATTERN.match(normalize_plate_text(text)))


def join_overlap(head, tail):
    """拼接两段文本；若边界字符重复（相邻检测框有重叠），去掉重复的那个"""
    if head and tail and head[-1].upper() == tail[0].upper():
        tail = tail[1:]
    return head + tail


def merge_line_boxes(data, gap_ratio=1.2):
    """把同一水平行上被切成多段的相邻文本框拼接成一个候选文本

    部分图片（尤其是新能源绿牌）车牌的检测框会被切成 "京A-D" + "D06088" 两段，
    直接逐框匹配会漏检，这里按“垂直方向重叠 + 水平间距足够近”合并。
    """
    boxes = []
    for info in data:
        xs = [p[0] for p in info['text_box_position']]
        ys = [p[1] for p in info['text_box_position']]
        boxes.append({
            'text': info['text'],
            'confidence': info['confidence'],
            'x0': min(xs), 'x1': max(xs),
            'y0': min(ys), 'y1': max(ys),
        })
    boxes.sort(key=lambda b: b['x0'])

    merged = []
    for box in boxes:
        if merged:
            last = merged[-1]
            overlap = min(last['y1'], box['y1']) - max(last['y0'], box['y0'])
            min_height = min(last['y1'] - last['y0'], box['y1'] - box['y0'])
            gap = box['x0'] - last['x1']
            if min_height > 0 and overlap / min_height > 0.5 and gap < gap_ratio * min_height:
                merged[-1] = {
                    'text': join_overlap(last['text'], box['text']),
                    'confidence': min(last['confidence'], box['confidence']),
                    'x0': last['x0'], 'x1': box['x1'],
                    'y0': min(last['y0'], box['y0']), 'y1': max(last['y1'], box['y1']),
                }
                continue
        merged.append(dict(box))

    return merged


def find_license_plates(data):
    """从 OCR 结果中筛出车牌，返回 [{'text', 'confidence', 'text_box_position'}, ...]"""
    plates = []
    for info in data:
        if is_plate_text(info['text']):
            plates.append({
                'text': normalize_plate_text(info['text']),
                'confidence': info['confidence'],
                'text_box_position': info['text_box_position'],
            })

    # 逐框没匹配到，可能是车牌被切成多段，合并同行相邻框后重试
    if not plates:
        for group in merge_line_boxes(data):
            if is_plate_text(group['text']):
                plates.append({
                    'text': normalize_plate_text(group['text']),
                    'confidence': group['confidence'],
                    'text_box_position': [
                        [group['x0'], group['y0']],
                        [group['x1'], group['y0']],
                        [group['x1'], group['y1']],
                        [group['x0'], group['y1']],
                    ],
                })

    return plates
