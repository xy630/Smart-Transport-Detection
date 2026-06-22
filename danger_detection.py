"""危险行为检测模块"""

import cv2
import numpy as np
from utils import draw_chinese_text

class DangerDetector:
    """危险行为检测器"""
    
    def __init__(self):
        self.warnings = []
        self.last_frame_warnings = []
        self.track_history = {}  
        self.frame_count = 0
    
    def detect_lane_departure(self, frame, detections, lane_info):
        """
        检测车道偏离（双向车道，每方向4个小车道）
        
        Args:
            frame: 当前帧
            detections: 目标检测结果
            lane_info: 车道线信息
            
        Returns:
            warnings: 车道偏离预警列表
        """
        warnings = []
        
        if len(detections) == 0:
            return warnings
        
        height, width = frame.shape[:2]
        
        left_lanes = [
            {'left': width * 0.02, 'right': width * 0.12},
            {'left': width * 0.12, 'right': width * 0.22},
            {'left': width * 0.22, 'right': width * 0.32},
            {'left': width * 0.32, 'right': width * 0.42},
        ]
        
        right_lanes = [
            {'left': width * 0.58, 'right': width * 0.68},
            {'left': width * 0.68, 'right': width * 0.78},
            {'left': width * 0.78, 'right': width * 0.88},
            {'left': width * 0.88, 'right': width * 0.98},
        ]
        
        all_lanes = left_lanes + right_lanes
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            car_center_x = (x1 + x2) / 2
            car_center_y = (y1 + y2) / 2
            
            if car_center_y < height * 0.4:
                continue
            
            found_lane = None
            for lane in all_lanes:
                if lane['left'] <= car_center_x <= lane['right']:
                    found_lane = lane
                    break
            
            if found_lane is None:
                continue
            
            lane_center = (found_lane['left'] + found_lane['right']) / 2
            lane_width = found_lane['right'] - found_lane['left']
            
            offset_from_center = car_center_x - lane_center
            
            if abs(offset_from_center) > lane_width * 0.3:
                if offset_from_center < 0:
                    direction = "左偏"
                else:
                    direction = "右偏"
                
                warnings.append({
                    'type': 'lane_departure',
                    'message': f"{det['class_name']}车道偏离({direction})",
                    'confidence': det['confidence'],
                    'bbox': det['bbox']
                })
        
        return warnings
    
    def detect_wrong_direction(self, frame, detections):
        """
        检测逆行（基于运动轨迹判断）
        
        Args:
            frame: 当前帧
            detections: 目标检测结果
            
        Returns:
            warnings: 逆行预警列表
        """
        warnings = []
        height, width = frame.shape[:2]
        
        road_bottom = height * 0.4
        
        self.frame_count += 1
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_y = (y1 + y2) / 2
            center_x = (x1 + x2) / 2
            
            if center_y < road_bottom:
                continue
            
            det_key = f"{det['class_name']}_{center_x:.0f}_{center_y:.0f}"
            
            if det_key not in self.track_history:
                self.track_history[det_key] = []
            
            self.track_history[det_key].append((center_x, center_y))
            
            if len(self.track_history[det_key]) >= 5:
                history = self.track_history[det_key]
                start_x = history[0][0]
                start_y = history[0][1]
                end_x = history[-1][0]
                end_y = history[-1][1]
                
                dy = end_y - start_y
                dx = end_x - start_x
                
                if dy < -20:
                    if dx > 20 and center_x > width * 0.6:
                        warnings.append({
                            'type': 'wrong_direction',
                            'message': f"{det['class_name']}疑似逆行",
                            'confidence': det['confidence'],
                            'bbox': det['bbox']
                        })
        
        return warnings
    
    def detect_red_light(self, frame, detections):
        """
        检测闯红灯（简化实现）
        
        Args:
            frame: 当前帧
            detections: 目标检测结果
            
        Returns:
            warnings: 闯红灯预警列表
        """
        warnings = []
        
        height, width = frame.shape[:2]
        roi = frame[0:int(height*0.1), :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_pixels = np.sum(mask > 0)
        
        if red_pixels > 1000:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                center_y = (y1 + y2) / 2
                
                if center_y > height * 0.3 and center_y < height * 0.6:
                    warnings.append({
                        'type': 'red_light',
                        'message': f"{det['class_name']}疑似闯红灯",
                        'confidence': det['confidence'],
                        'bbox': det['bbox']
                    })
        
        return warnings
    
    def detect_all(self, frame, detections, lane_info):
        """
        检测所有危险行为
        
        Args:
            frame: 当前帧
            detections: 目标检测结果
            lane_info: 车道线信息
            
        Returns:
            all_warnings: 所有预警列表
        """
        self.warnings = []
        
        self.warnings.extend(self.detect_lane_departure(frame, detections, lane_info))
        self.warnings.extend(self.detect_wrong_direction(frame, detections))
        self.warnings.extend(self.detect_red_light(frame, detections))
        
        return self.warnings
    
    def draw_warnings(self, frame, warnings):
        """
        在图像上绘制预警信息
        
        Args:
            frame: 输入图像
            warnings: 预警列表
            
        Returns:
            annotated_frame: 绘制了预警的图像
        """
        annotated = frame.copy()
        
        for warning in warnings:
            x1, y1, x2, y2 = warning['bbox']
            message = warning['message']
            confidence = warning['confidence']
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            label = f"⚠ {message} {confidence:.2f}"
            annotated, _, _ = draw_chinese_text(annotated, label, (x1, y1 - 15), 
                                               font_size=14, text_color=(255, 255, 255), 
                                               bg_color=(0, 0, 255))
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        return annotated