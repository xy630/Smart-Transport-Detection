"""
工具函数模块
Utility Functions Module

提供通用的辅助函数
"""

import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


def ensure_dir(directory):
    """
    确保目录存在

    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=0.6, text_color=(255, 255, 255),
                               bg_color=(0, 0, 0), thickness=1, padding=5):
    """
    在图像上绘制带背景的文本

    Args:
        img: 输入图像
        text: 文本内容
        position: 位置 (x, y)
        font: 字体
        font_scale: 字体大小
        text_color: 文本颜色
        bg_color: 背景颜色
        thickness: 线条粗细
        padding: 内边距
    """
    x, y = position

    # 获取文本尺寸
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # 绘制背景
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )

    # 绘制文本
    cv2.putText(
        img, text, (x, y),
        font, font_scale, text_color, thickness
    )


def resize_frame(frame, scale=None, width=None, height=None):
    """
    调整帧大小

    Args:
        frame: 输入帧
        scale: 缩放比例
        width: 目标宽度
        height: 目标高度

    Returns:
        resized: 调整后的帧
    """
    h, w = frame.shape[:2]

    if scale is not None:
        new_w, new_h = int(w * scale), int(h * scale)
    elif width is not None and height is not None:
        new_w, new_h = width, height
    elif width is not None:
        new_w, new_h = width, int(h * (width / w))
    elif height is not None:
        new_w, new_h = int(w * (height / h)), height
    else:
        return frame

    return cv2.resize(frame, (new_w, new_h))


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IOU

    Args:
        box1: 边界框1 (x1, y1, x2, y2)
        box2: 边界框2 (x1, y1, x2, y2)

    Returns:
        iou: IOU 值
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 计算交集面积
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # 计算两个框的面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    iou = inter_area / union_area
    return iou


def apply_blur(frame, kernel_size=(5, 5)):
    """
    对图像应用模糊

    Args:
        frame: 输入帧
        kernel_size: 卷积核大小

    Returns:
        blurred: 模糊后的帧
    """
    return cv2.GaussianBlur(frame, kernel_size, 0)


def enhance_contrast(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    增强图像对比度 (CLAHE)

    Args:
        frame: 输入帧
        clip_limit: 对比度限制
        tile_grid_size: 网格大小

    Returns:
        enhanced: 增强后的帧
    """
    # 转换为 LAB 色彩空间
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)

    # 合并通道
    lab = cv2.merge([l, a, b])

    # 转换回 BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def save_frame(frame, output_dir="outputs", prefix="frame"):
    """
    保存帧到文件

    Args:
        frame: 输入帧
        output_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        filepath: 保存的文件路径
    """
    ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, frame)

    return filepath


def create_color_palette(n):
    """
    创建颜色调色板

    Args:
        n: 颜色数量

    Returns:
        colors: 颜色列表
    """
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]),
            cv2.COLOR_HSV2BGR
        )[0][0]
        colors.append(tuple(map(int, color)))
    return colors


def draw_fps(frame, fps, position=(10, 30)):
    """
    在帧上绘制 FPS

    Args:
        frame: 输入帧
        fps: FPS 值
        position: 位置

    Returns:
        frame: 绘制了 FPS 的帧
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    return frame


def draw_chinese_text(img, text, position, font_size=20, text_color=(255, 255, 255), bg_color=None):
    """
    在图像上绘制中文文本（使用 PIL）

    Args:
        img: 输入图像 (OpenCV格式，BGR)
        text: 文本内容
        position: 位置 (x, y)
        font_size: 字体大小
        text_color: 文本颜色 (BGR)
        bg_color: 背景颜色，如果为 None 则不绘制背景

    Returns:
        img: 绘制了文本的图像
    """
    # 将 OpenCV 图像转换为 PIL 图像 (BGR -> RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试加载系统中文字体
    try:
        # 尝试不同的字体路径（包括更多可能的字体）
        font_paths = [
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",      # 微软雅黑粗体
            "C:/Windows/Fonts/STSong.ttf",      # 宋体
            "C:/Windows/Fonts/STHeiti.ttf",     # 黑体
            "C:/Windows/Fonts/KaiTi.ttf",       # 楷体
            "C:/Windows/Fonts/STKaiti.ttf",     # 楷体
            "C:/Windows/Fonts/STFangsong.ttf",  # 仿宋
            "C:/Windows/Fonts/FangSong.ttf",    # 仿宋
            "C:/Windows/Fonts/YanKai.ttf",      # 颜楷
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # Linux
            "/usr/share/fonts/truetype/arphic/ukai.ttc",       # Linux
            "/usr/share/fonts/truetype/arphic/uming.ttc",       # Linux
            "/Library/Fonts/Songti.ttc",        # macOS
            "/Library/Fonts/Hiragino Sans GB.ttc",  # macOS
            "/Library/Fonts/Heiti.ttc",         # macOS
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    continue

        if font is None:
            # 使用默认字体，但中文可能显示为方块
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 获取文本尺寸
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x, y = position

    # 如果有背景颜色，绘制背景
    if bg_color is not None:
        # 转换颜色格式
        bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle(
            [x, y - text_height, x + text_width, y],
            fill=bg_color_rgb
        )

    # 绘制文本
    text_color_rgb = (text_color[2], text_color[1], text_color[0])
    draw.text((x, y - text_height), text, font=font, fill=text_color_rgb)

    # 转换回 OpenCV 格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img, text_width, text_height


if __name__ == "__main__":
    # 测试代码
    print("[INFO] 工具模块测试")

    # 测试创建颜色调色板
    colors = create_color_palette(10)
    print(f"[INFO] 创建了 {len(colors)} 种颜色")

    # 测试目录创建
    test_dir = "outputs/test_dir"
    ensure_dir(test_dir)
    print(f"[INFO] 目录已创建: {test_dir}")

    print("[INFO] 工具模块测试完成")
