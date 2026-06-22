"""
交通流量统计模块
Traffic Statistics Module

功能：
- 车辆跟踪
- 车流量统计
- 速度估计
- 轨迹绘制
"""

import cv2
import numpy as np
from collections import defaultdict
from config.settings import TRAFFIC_STATS_CONFIG, DETECTION_CLASSES
from utils import draw_chinese_text


class VehicleTracker:
    """车辆跟踪器类"""

    def __init__(self, max_disappeared=None, max_distance=None):
        """
        初始化跟踪器

        Args:
            max_disappeared: 最大消失帧数
            max_distance: 最大跟踪距离
        """
        self.max_disappeared = max_disappeared or TRAFFIC_STATS_CONFIG["max_disappeared"]
        self.max_distance = max_distance or TRAFFIC_STATS_CONFIG["max_distance"]

        # 跟踪器状态
        self.next_id = 0
        self.objects = {}           # id -> 车辆信息
        self.disappeared = {}        # id -> 消失帧数

        # 统计数据
        self.total_count = 0
        self.count_by_class = defaultdict(int)
        self.track_history = defaultdict(list)

    def register(self, bbox, class_id, class_name):
        """
        注册新车辆

        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            class_id: 类别ID
            class_name: 类别名称

        Returns:
            object_id: 分配的对象ID
        """
        self.objects[self.next_id] = {
            'bbox': bbox,
            'class_id': class_id,
            'class_name': class_name,
            'center': self._get_center(bbox),
            'first_seen': None,
            'last_seen': None,
        }
        self.disappeared[self.next_id] = 0
        self.track_history[self.next_id] = []

        object_id = self.next_id
        self.next_id += 1

        return object_id

    def deregister(self, object_id):
        """注销车辆"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.track_history:
            del self.track_history[object_id]

    def _get_center(self, bbox):
        """计算边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calculate_distance(self, center1, center2):
        """计算两个中心点之间的欧氏距离"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def update(self, detections):
        """
        更新跟踪器

        Args:
            detections: 当前帧检测到的目标列表

        Returns:
            tracked_objects: 跟踪中的对象字典
        """
        if len(detections) == 0:
            # 没有检测到任何目标，标记所有现有对象为消失
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            # 有检测结果，尝试匹配
            if len(self.objects) == 0:
                # 第一帧，直接注册所有检测
                for det in detections:
                    obj_id = self.register(
                        det['bbox'],
                        det['class_id'],
                        det['class_name']
                    )
                    self.track_history[obj_id].append(self._get_center(det['bbox']))
            else:
                # 匹配已有对象和新检测
                object_ids = list(self.objects.keys())
                object_centers = [self.objects[oid]['center'] for oid in object_ids]

                # 计算距离矩阵
                det_centers = [self._get_center(d['bbox']) for d in detections]
                distances = np.zeros((len(object_ids), len(det_centers)))

                for i, obj_center in enumerate(object_centers):
                    for j, det_center in enumerate(det_centers):
                        distances[i, j] = self._calculate_distance(obj_center, det_center)

                # 贪婪匹配
                matched_objects = set()
                matched_detections = set()

                # 按距离排序
                distance_order = np.argsort(distances.flatten())

                for idx in distance_order:
                    obj_idx = idx // len(det_centers)
                    det_idx = idx % len(det_centers)

                    if obj_idx in matched_objects or det_idx in matched_detections:
                        continue

                    if distances[obj_idx, det_idx] <= self.max_distance:
                        object_id = object_ids[obj_idx]
                        detection = detections[det_idx]

                        # 更新对象信息
                        self.objects[object_id]['bbox'] = detection['bbox']
                        self.objects[object_id]['center'] = self._get_center(detection['bbox'])
                        self.disappeared[object_id] = 0

                        # 更新历史轨迹
                        self.track_history[object_id].append(
                            self._get_center(detection['bbox'])
                        )
                        # 限制轨迹长度
                        if len(self.track_history[object_id]) > 30:
                            self.track_history[object_id].pop(0)

                        matched_objects.add(obj_idx)
                        matched_detections.add(det_idx)

                # 处理未匹配的对象
                for i, object_id in enumerate(object_ids):
                    if i not in matched_objects:
                        self.disappeared[object_id] += 1

                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)

                # 处理未匹配的检测（注册新对象）
                for j, detection in enumerate(detections):
                    if j not in matched_detections:
                        obj_id = self.register(
                            detection['bbox'],
                            detection['class_id'],
                            detection['class_name']
                        )
                        self.track_history[obj_id].append(
                            self._get_center(detection['bbox'])
                        )

        return self.objects

    def count_vehicles(self, count_line_y):
        """
        统计通过计数线的车辆数量

        Args:
            count_line_y: 计数线 Y 坐标

        Returns:
            count: 通过的车辆数
        """
        count = 0
        for object_id, obj_data in list(self.objects.items()):
            center_y = obj_data['center'][1]

            # 检查是否通过计数线
            history = self.track_history[object_id]
            if len(history) >= 2:
                prev_y = history[-2][1]
                curr_y = history[-1][1]

                # 从上往下或从下往上通过
                if (prev_y < count_line_y <= curr_y) or (prev_y > count_line_y >= curr_y):
                    count += 1
                    self.total_count += 1
                    self.count_by_class[obj_data['class_name']] += 1

        return count

    def get_current_count(self):
        """获取当前跟踪的车辆数"""
        return len(self.objects)

    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_count': self.total_count,
            'current_tracked': len(self.objects),
            'count_by_class': dict(self.count_by_class)
        }


class TrafficStatistics:
    """交通统计类"""

    def __init__(self):
        """初始化交通统计器"""
        self.tracker = VehicleTracker()
        self.vehicle_counts = defaultdict(int)
        self.counted_objects = set()  # 已计数的车辆ID，避免重复计数
        self.frame_count = 0
        self.fps = 0
        self.fps_history = []

        # 计数线配置
        self.count_line = TRAFFIC_STATS_CONFIG["count_line"]

        # ROI 配置
        self.roi = TRAFFIC_STATS_CONFIG["roi"]

    def update(self, frame, detections):
        """
        更新统计

        Args:
            frame: 当前帧
            detections: 检测结果列表

        Returns:
            stats: 统计信息
        """
        self.frame_count += 1

        # 获取帧高度，动态计算计数线位置（画面中间偏下）
        frame_height = frame.shape[0] if frame is not None else 720
        count_line_y = int(frame_height * 0.6)  # 计数线在画面60%高度位置

        # 更新跟踪器
        tracked_objects = self.tracker.update(detections)

        # 统计当前帧各类别车辆数量
        current_counts = defaultdict(int)
        for det in detections:
            class_name = det.get('class_name', '未知')
            current_counts[class_name] += 1
        
        # 更新累计各类别计数（检测车辆是否通过计数线）
        passing_count = 0
        for obj_id, obj_data in tracked_objects.items():
            if obj_id not in self.counted_objects:
                # 检查当前车辆中心是否在计数线下方
                curr_y = obj_data['center'][1]
                
                # 获取上一帧位置（如果存在）
                history = self.tracker.track_history.get(obj_id, [])
                if len(history) >= 2:
                    prev_y = history[-2][1]
                    # 车辆从上方穿过计数线到下方
                    if prev_y < count_line_y and curr_y >= count_line_y:
                        self.vehicle_counts[obj_data['class_name']] += 1
                        self.tracker.total_count += 1
                        self.counted_objects.add(obj_id)
                        passing_count += 1

        stats = {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'current_vehicles': len(tracked_objects),
            'total_count': self.tracker.total_count,
            'count_by_class': dict(self.vehicle_counts),
            'current_by_class': dict(current_counts),
            'passing_now': passing_count,
        }

        return stats

    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_count': self.tracker.total_count,
            'current_tracked': len(self.tracker.objects),
            'count_by_class': dict(self.vehicle_counts)
        }

    def draw_overlay(self, frame, stats):
        """
        绘制统计信息覆盖层

        Args:
            frame: 输入帧
            stats: 统计信息

        Returns:
            annotated: 添加了统计信息的图像
        """
        annotated = frame.copy()

        # 创建半透明背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

        # 绘制统计信息
        y_offset = 35
        font_size = 16

        info_lines = [
            f"帧: {stats['frame_count']}",
            f"帧率: {stats['fps']:.1f}",
            f"当前车辆: {stats['current_vehicles']}",
            f"累计总数: {stats['total_count']}",
        ]

        for i, line in enumerate(info_lines):
            annotated, _, _ = draw_chinese_text(
                annotated, line, (20, y_offset + i * 28),
                font_size=font_size, text_color=(255, 255, 255), bg_color=None
            )

        # 绘制各类别统计
        y_offset = 35 + len(info_lines) * 28 + 15
        for class_name, count in stats['count_by_class'].items():
            annotated, _, _ = draw_chinese_text(
                annotated, f"  {class_name}: {count}", (20, y_offset),
                font_size=14, text_color=(200, 200, 200), bg_color=None
            )
            y_offset += 22

        # 绘制计数线（使用动态计算的位置）
        frame_height = frame.shape[0]
        count_line_y = int(frame_height * 0.6)
        frame_width = frame.shape[1]
        cv2.line(
            annotated,
            (0, count_line_y),
            (frame_width, count_line_y),
            (0, 255, 255),
            2
        )

        return annotated

    def draw_tracks(self, frame):
        """
        绘制跟踪轨迹

        Args:
            frame: 输入帧

        Returns:
            annotated: 绘制了轨迹的图像
        """
        annotated = frame.copy()

        for object_id, history in self.tracker.track_history.items():
            if len(history) < 2:
                continue

            # 绘制轨迹线
            points = np.array(history, dtype=np.int32)
            for i in range(len(points) - 1):
                cv2.line(
                    annotated,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    (0, 255, 255),
                    2
                )

            # 绘制车辆ID
            if history:
                last_point = history[-1]
                cv2.circle(annotated, tuple(last_point), 5, (0, 255, 255), -1)
                cv2.putText(
                    annotated,
                    f"ID:{object_id}",
                    tuple(last_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )

        return annotated


# 测试代码
if __name__ == "__main__":
    from object_detection import ObjectDetector

    # 创建检测器和统计器
    detector = ObjectDetector()
    traffic_stats = TrafficStatistics()

    # 读取测试图像
    test_image = cv2.imread("videos/test.jpg")

    if test_image is not None:
        # 执行检测
        results = detector.detect(test_image)
        detections = detector.filter_traffic_targets(results)

        # 更新统计
        stats = traffic_stats.update(test_image, detections)

        # 绘制结果
        annotated = traffic_stats.draw_overlay(test_image, stats)
        annotated = traffic_stats.draw_tracks(annotated)

        # 显示统计信息
        print(f"[INFO] 统计信息: {stats}")

        # 保存结果
        cv2.imwrite("outputs/traffic_stats.jpg", annotated)
        print("[INFO] 结果已保存到 outputs/traffic_stats.jpg")
    else:
        print("[WARNING] 未找到测试图像，请将测试图像放入 videos 目录")
