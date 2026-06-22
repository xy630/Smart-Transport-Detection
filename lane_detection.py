"""
车道线检测模块 - 基于 YOLOv8-seg
Lane Detection Module - Based on YOLOv8-seg

功能：
- 车道线实例分割
- 车道区域可视化
- 车道偏离预警
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import MODEL_CONFIG, LANE_CONFIG


class LaneDetector:
    """车道线检测器类"""

    def __init__(self, model_name=None, conf_threshold=0.30, device='cpu'):
        """
        初始化车道线检测器

        Args:
            model_name: 模型名称
            conf_threshold: 置信度阈值
            device: 运行设备
        """
        self.model_name = model_name or MODEL_CONFIG["segmentation_model"]["model_name"]
        self.conf_threshold = conf_threshold
        self.device = device

        # 车道线配置
        self.lane_color = LANE_CONFIG["lane_color"]
        self.lane_alpha = LANE_CONFIG["lane_alpha"]
        self.fill_lane = LANE_CONFIG["fill_lane"]

        # 车道偏离预警配置
        self.lane_departure_warning = LANE_CONFIG["lane_departure_warning"]
        self.departure_threshold = LANE_CONFIG["departure_threshold"]

        # 加载模型
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 YOLOv8-seg 模型"""
        try:
            print(f"[INFO] 正在加载车道线分割模型: {self.model_name}")
            self.model = YOLO(self.model_name)
            print(f"[INFO] 车道线分割模型加载成功，设备: {self.device}")
        except Exception as e:
            print(f"[ERROR] 车道线分割模型加载失败: {e}")
            raise

    def detect(self, frame):
        """
        检测车道线

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            results: 检测/分割结果
        """
        if self.model is None:
            raise ValueError("模型未加载")

        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        return results

    def get_lane_masks(self, results):
        """
        提取车道线掩码（更严格的过滤逻辑）

        Args:
            results: YOLOv8-seg 检测结果

        Returns:
            masks: 车道线掩码列表
        """
        masks = []

        for result in results:
            if result.masks is not None:
                for i, mask in enumerate(result.masks):
                    # 获取置信度（更高的阈值）
                    if result.boxes is not None and i < len(result.boxes):
                        conf = float(result.boxes[i].conf[0])
                        if conf < 0.7:  # 更高的置信度阈值
                            continue
                    
                    mask_data = mask.data[0].cpu().numpy()
                    height, width = mask_data.shape
                    
                    # 计算掩码面积
                    area = np.sum(mask_data > 0.5)
                    
                    # 更严格的面积过滤
                    min_area = 200
                    max_area = (height * width) * 0.15  # 不超过图像的15%
                    
                    if area < min_area or area > max_area:
                        continue
                    
                    # 获取掩码的边界框
                    coords = np.argwhere(mask_data > 0.5)
                    if len(coords) == 0:
                        continue
                    
                    # 更严格的位置过滤：只保留图像下2/3部分
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    # 只保留完全在图像下半部分的区域
                    if y_max < height * 0.4:
                        continue
                    
                    masks.append(mask_data)

        return masks

    def draw_lane_regions(self, frame, masks):
        """
        在图像上绘制车道线区域

        Args:
            frame: 输入图像
            masks: 车道线掩码列表

        Returns:
            annotated: 绘制了车道线的图像
        """
        annotated = frame.copy()

        # 创建车道线图层
        lane_overlay = np.zeros_like(annotated, dtype=np.uint8)

        for mask in masks:
            # 将掩码转换为二值图像
            binary_mask = (mask * 255).astype(np.uint8)

            # 找到轮廓
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 绘制填充区域
            if self.fill_lane:
                cv2.fillPoly(lane_overlay, contours, self.lane_color)

            # 绘制边界线
            cv2.polylines(lane_overlay, contours, True, (0, 255, 0), 3)

        # 混合图层
        annotated = cv2.addWeighted(
            annotated, 1 - self.lane_alpha,
            lane_overlay, self.lane_alpha, 0
        )

        return annotated

    def draw_lane_lines(self, frame, masks):
        """
        绘制车道线边界（不填充）

        Args:
            frame: 输入图像
            masks: 车道线掩码列表

        Returns:
            annotated: 绘制了车道线的图像
        """
        annotated = frame.copy()

        for mask in masks:
            # 将掩码转换为二值图像
            binary_mask = (mask * 255).astype(np.uint8)

            # 找到轮廓
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 绘制边界线
            cv2.polylines(annotated, contours, True, self.lane_color, 3)

        return annotated

    def check_lane_departure(self, frame, vehicle_box):
        """
        检查车辆是否偏离车道

        Args:
            frame: 当前帧
            vehicle_box: 车辆边界框 (x1, y1, x2, y2)

        Returns:
            is_departing: 是否正在偏离
            deviation: 偏离程度
        """
        if not self.lane_departure_warning:
            return False, 0.0

        x1, y1, x2, y2 = vehicle_box
        frame_width = frame.shape[1]

        # 计算车辆中心点
        vehicle_center_x = (x1 + x2) / 2

        # 计算偏离程度（相对于图像宽度）
        deviation = abs(vehicle_center_x - frame_width / 2) / frame_width

        is_departing = deviation > self.departure_threshold

        return is_departing, deviation

    def get_lane_info(self, masks):
        """
        获取车道线信息

        Args:
            masks: 车道线掩码列表

        Returns:
            info: 车道线信息字典
        """
        info = {
            'lane_count': len(masks),
            'has_lane': len(masks) > 0,
        }

        # 计算车道区域
        if masks:
            combined_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                combined_mask = combined_mask | mask
            info['coverage'] = float(combined_mask.sum() / combined_mask.size)
        else:
            info['coverage'] = 0.0

        return info


# 测试代码
if __name__ == "__main__":
    # 创建车道线检测器
    detector = LaneDetector()

    # 读取测试图像
    test_image = cv2.imread("videos/test.jpg")

    if test_image is not None:
        # 执行检测
        results = detector.detect(test_image)
        masks = detector.get_lane_masks(results)

        # 绘制结果
        annotated = detector.draw_lane_regions(test_image, masks)

        # 显示统计信息
        info = detector.get_lane_info(masks)
        print(f"[INFO] 车道线信息: {info}")

        # 保存结果
        cv2.imwrite("outputs/lane_result.jpg", annotated)
        print("[INFO] 结果已保存到 outputs/lane_result.jpg")
    else:
        print("[WARNING] 未找到测试图像，请将测试图像放入 videos 目录")
