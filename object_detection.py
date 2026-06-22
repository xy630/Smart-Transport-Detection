"""
目标检测模块 - 基于 YOLOv8
Object Detection Module - Based on YOLOv8

功能：
- 实时检测车辆、行人、非机动车等交通目标
- 提供边界框、类别、置信度信息
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import MODEL_CONFIG, DETECTION_CLASSES, TRAFFIC_TARGETS
from utils import draw_chinese_text


class ObjectDetector:
    """目标检测器类"""

    def __init__(self, model_name=None, conf_threshold=None, device='cpu'):
        """
        初始化目标检测器

        Args:
            model_name: 模型名称，默认使用配置中的设置
            conf_threshold: 置信度阈值
            device: 运行设备 (cpu/cuda)
        """
        self.model_name = model_name or MODEL_CONFIG["detection_model"]["model_name"]
        self.conf_threshold = conf_threshold or MODEL_CONFIG["detection_model"]["conf_threshold"]
        self.device = device

        # 加载模型
        self.model = None
        self._load_model()

        # 类别名称映射
        self.class_names = DETECTION_CLASSES

        # 颜色映射 (对应正确的YOLOv8类别编号)
        self.colors = {
            0: (255, 0, 0),        # person (0) - 蓝色
            1: (0, 255, 255),      # bicycle (1) - 黄色
            2: (0, 165, 255),      # car (2) - 橙色
            3: (0, 255, 0),        # motorcycle (3) - 绿色
            5: (128, 0, 128),      # bus (5) - 紫色
            7: (0, 0, 139),        # truck (7) - 深红色
        }

    def _load_model(self):
        """加载 YOLOv8 模型"""
        try:
            print(f"[INFO] 正在加载目标检测模型: {self.model_name}")
            self.model = YOLO(self.model_name)
            print(f"[INFO] 模型加载成功，设备: {self.device}")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            raise

    def detect(self, frame):
        """
        在单帧图像上进行目标检测

        Args:
            frame: 输入图像 (numpy array, BGR格式)

        Returns:
            results: 检测结果，包含边界框、类别、置信度
        """
        if self.model is None:
            raise ValueError("模型未加载")

        # 执行检测
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        return results

    def filter_traffic_targets(self, results):
        """
        过滤出交通相关目标

        Args:
            results: YOLOv8 检测结果

        Returns:
            filtered_boxes: 过滤后的检测框列表
        """
        filtered = []

        # 车辆类别（需要严格过滤）
        vehicle_classes = [2, 5, 7]  # car, bus, truck
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 车辆类别需要更高的置信度
                if cls_id in vehicle_classes and conf < 0.45:
                    continue
                
                # 只保留交通目标
                if cls_id in TRAFFIC_TARGETS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    should_filter = False
                    
                    # 获取图像尺寸
                    if result.orig_img is not None:
                        img_height, img_width = result.orig_img.shape[:2]
                        
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 对于车辆类，严格过滤路边物体
                        if cls_id in vehicle_classes:
                            # 排除太靠近左右边缘的检测（路边建筑、公交车站）
                            left_edge_margin = img_width * 0.15
                            right_edge_margin = img_width * 0.15
                            is_near_left_edge = center_x < left_edge_margin
                            is_near_right_edge = center_x > img_width - right_edge_margin
                            
                            # 排除靠近边缘且上边界在上半部分的检测（路边物体）
                            is_top_half = y1 < img_height * 0.7
                            
                            # 排除高度过高的检测（建筑）
                            is_too_tall = bbox_height > img_height * 0.40
                            
                            # 排除宽度过宽的检测（大型建筑）
                            is_too_wide = bbox_width > img_width * 0.30
                            
                            # 排除长宽比异常的检测（建筑物长宽比通常与车辆不同）
                            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1
                            is_abnormal_aspect = aspect_ratio < 0.4 or aspect_ratio > 5.0
                            
                            # 排除靠近边缘且置信度低的检测
                            low_confidence = conf < 0.40
                            
                            # 排除底部边缘检测（路边物体）
                            is_near_bottom_edge = y2 > img_height * 0.95
                            
                            if is_near_left_edge or is_near_right_edge:
                                if is_top_half or is_too_tall or is_too_wide or low_confidence or is_abnormal_aspect or is_near_bottom_edge:
                                    should_filter = True
                    
                    if not should_filter:
                        filtered.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_id': cls_id,
                            'class_name': self.class_names.get(cls_id, 'unknown'),
                            'confidence': conf
                        })

        return filtered

    def draw_detections(self, frame, detections):
        """
        在图像上绘制检测结果

        Args:
            frame: 输入图像
            detections: 检测结果列表

        Returns:
            annotated_frame: 绘制了检测结果的图像
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['confidence']

            # 获取颜色
            color = self.colors.get(cls_id, (255, 255, 255))

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签（使用中文支持）
            label = f"{cls_name} {conf:.2f}"
            annotated, _, _ = draw_chinese_text(
                annotated, label, (x1, y1 - 2), font_size=14,
                text_color=(255, 255, 255), bg_color=color
            )

        return annotated

    def get_detection_summary(self, detections):
        """
        获取检测统计信息

        Args:
            detections: 检测结果列表

        Returns:
            summary: 统计字典
        """
        summary = {
            'total': len(detections),
            'by_class': {}
        }

        for det in detections:
            cls_name = det['class_name']
            summary['by_class'][cls_name] = summary['by_class'].get(cls_name, 0) + 1

        return summary


# 测试代码
if __name__ == "__main__":
    # 创建检测器
    detector = ObjectDetector()

    # 读取测试图像
    test_image = cv2.imread("videos/test.jpg")

    if test_image is not None:
        # 执行检测
        results = detector.detect(test_image)
        detections = detector.filter_traffic_targets(results)

        # 绘制结果
        annotated = detector.draw_detections(test_image, detections)

        # 显示统计信息
        summary = detector.get_detection_summary(detections)
        print(f"[INFO] 检测统计: {summary}")

        # 保存结果
        cv2.imwrite("outputs/detection_result.jpg", annotated)
        print("[INFO] 结果已保存到 outputs/detection_result.jpg")
    else:
        print("[WARNING] 未找到测试图像，请将测试图像放入 videos 目录")
