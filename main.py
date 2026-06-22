"""
智慧交通系统主程序
Smart Transportation System - Main Program

基于计算机视觉的智慧交通监测与辅助驾驶系统

功能：
- 车道线检测与偏离预警
- 车辆、行人、非机动车实时检测
- 交通流量统计
- 实时显示、预警、数据统计

作者：Smart Transportation Team
版本：1.0.0
"""

import cv2
import time
import numpy as np
from datetime import datetime

from object_detection import ObjectDetector
from lane_detection import LaneDetector
from traffic_stats import TrafficStatistics
from config.settings import (
    SYSTEM_NAME, VERSION, VIDEO_CONFIG, MODEL_CONFIG
)
from utils import draw_chinese_text


class SmartTransportationSystem:
    """智慧交通系统主类"""

    def __init__(self, video_source=0):
        """
        初始化智慧交通系统

        Args:
            video_source: 视频源 (0=摄像头，字符串=视频文件路径)
        """
        self.video_source = video_source
        self.cap = None
        self.writer = None

        # 初始化组件
        self.detector = None
        self.lane_detector = None
        self.traffic_stats = None

        # 系统状态
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.fps = 0

        # 配置
        self.show_display = VIDEO_CONFIG["show_display"]
        self.window_name = VIDEO_CONFIG["window_name"]
        self.save_output = VIDEO_CONFIG["save_output"]

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化系统组件"""
        print("=" * 60)
        print(f"  {SYSTEM_NAME}")
        print(f"  Version: {VERSION}")
        print("=" * 60)

        # 初始化目标检测器
        print("\n[INFO] 初始化目标检测模块...")
        det_config = MODEL_CONFIG["detection_model"]
        self.detector = ObjectDetector(
            model_name=det_config["model_name"],
            conf_threshold=det_config["conf_threshold"],
            device=det_config["device"]
        )

        # 初始化车道线检测器
        print("[INFO] 初始化车道线检测模块...")
        seg_config = MODEL_CONFIG["segmentation_model"]
        self.lane_detector = LaneDetector(
            model_name=seg_config["model_name"],
            conf_threshold=seg_config["conf_threshold"],
            device=seg_config["device"]
        )

        # 初始化交通统计器
        print("[INFO] 初始化交通统计模块...")
        self.traffic_stats = TrafficStatistics()

        # 初始化视频源
        print("[INFO] 初始化视频源...")
        self._init_video_source()

        print("\n[INFO] 系统初始化完成！")
        print("-" * 60)

    def _init_video_source(self):
        """初始化视频源"""
        if isinstance(self.video_source, int):
            # 摄像头
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            print(f"[INFO] 已连接到摄像头 {self.video_source}")
        else:
            # 视频文件
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {self.video_source}")
            print(f"[INFO] 已加载视频文件: {self.video_source}")

        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[INFO] 视频分辨率: {self.frame_width}x{self.frame_height}")
        print(f"[INFO] 视频帧率: {self.video_fps}")

        # 初始化视频写入器
        if self.save_output:
            output_path = VIDEO_CONFIG["output_path"]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                output_path, fourcc,
                VIDEO_CONFIG["fps"],
                (self.frame_width, self.frame_height)
            )
            print(f"[INFO] 输出视频将保存到: {output_path}")

    def _calculate_fps(self):
        """计算FPS"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed

    def process_frame(self, frame):
        """
        处理单帧图像

        Args:
            frame: 输入帧

        Returns:
            processed_frame: 处理后的帧
            results: 检测和统计结果
        """
        # 1. 车道线检测
        lane_results = self.lane_detector.detect(frame)
        lane_masks = self.lane_detector.get_lane_masks(lane_results)

        # 绘制车道线
        frame = self.lane_detector.draw_lane_regions(frame, lane_masks)

        # 获取车道线信息
        lane_info = self.lane_detector.get_lane_info(lane_masks)

        # 2. 目标检测
        det_results = self.detector.detect(frame)
        detections = self.detector.filter_traffic_targets(det_results)

        # 绘制检测结果
        frame = self.detector.draw_detections(frame, detections)

        # 3. 交通流量统计
        traffic_results = self.traffic_stats.update(frame, detections)

        # 更新FPS
        traffic_results['fps'] = self.fps

        # 4. 绘制统计信息
        frame = self.traffic_stats.draw_overlay(frame, traffic_results)

        # 5. 绘制跟踪轨迹
        frame = self.traffic_stats.draw_tracks(frame)

        # 6. 绘制车道线检测信息
        frame, _, _ = draw_chinese_text(
            frame, f"车道覆盖率: {lane_info['coverage']:.2%}",
            (self.frame_width - 180, 30),
            font_size=14, text_color=(0, 255, 0), bg_color=None
        )

        # 7. 车道偏离预警
        if lane_info['has_lane'] and detections:
            # 检查第一个车辆（通常是最靠近的）
            for det in detections[:1]:
                is_departing, deviation = self.lane_detector.check_lane_departure(
                    frame, det['bbox']
                )
                if is_departing:
                    frame, _, _ = draw_chinese_text(
                        frame, "车道偏离警告！",
                        (self.frame_width // 2 - 80, 60),
                        font_size=22, text_color=(0, 0, 255), bg_color=None
                    )

        return frame, {
            'detections': detections,
            'lane_info': lane_info,
            'traffic_stats': traffic_results
        }

    def _draw_info_panel(self, frame):
        """绘制信息面板"""
        # 系统信息
        panel_height = 60
        panel = np.zeros((panel_height, self.frame_width, 3), dtype=np.uint8)

        # 标题
        title = f"{SYSTEM_NAME} | 帧率: {self.fps:.1f} | 帧数: {self.frame_count}"
        panel, _, _ = draw_chinese_text(
            panel, title, (10, 40),
            font_size=16, text_color=(255, 255, 255), bg_color=None
        )

        # 时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        panel, _, _ = draw_chinese_text(
            panel, current_time, (self.frame_width - 180, 40),
            font_size=14, text_color=(255, 255, 255), bg_color=None
        )

        # 合并到主帧
        frame = np.vstack([panel, frame])

        return frame

    def run(self):
        """运行主循环"""
        print("\n[INFO] 开始处理视频...")
        print("[INFO] 按 'q' 键退出，按 's' 键截图")

        self.is_running = True
        self.start_time = time.time()

        while self.is_running:
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] 视频播放完毕或读取错误")
                break

            self.frame_count += 1
            self._calculate_fps()

            # 处理帧
            processed_frame, results = self.process_frame(frame)

            # 绘制信息面板
            display_frame = self._draw_info_panel(processed_frame)

            # 保存帧
            if self.save_output and self.writer is not None:
                self.writer.write(processed_frame)

            # 显示帧
            if self.show_display:
                # 调整显示大小
                display = cv2.resize(display_frame, (960, 540))
                cv2.imshow(self.window_name, display)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] 用户按下 'q' 键，退出系统")
                    break
                elif key == ord('s'):
                    # 截图
                    screenshot_path = f"outputs/screenshot_{self.frame_count}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"[INFO] 截图已保存: {screenshot_path}")

            # 每30帧打印一次状态
            if self.frame_count % 30 == 0:
                stats = results['traffic_stats']
                print(f"[INFO] Frame {self.frame_count} | "
                      f"Vehicles: {stats['current_vehicles']} | "
                      f"Total Count: {stats['total_count']} | "
                      f"FPS: {self.fps:.1f}")

        self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n[INFO] 正在清理资源...")

        if self.cap is not None:
            self.cap.release()

        if self.writer is not None:
            self.writer.release()

        if self.show_display:
            cv2.destroyAllWindows()

        # 打印最终统计
        final_stats = self.traffic_stats.get_statistics()
        print("\n" + "=" * 60)
        print("  最终统计报告")
        print("=" * 60)
        print(f"  总帧数: {self.frame_count}")
        print(f"  平均FPS: {self.fps:.2f}")
        print(f"  总车流量: {final_stats['total_count']}")
        print(f"  各类别统计:")
        for cls, count in final_stats['count_by_class'].items():
            print(f"    - {cls}: {count}")
        print("=" * 60)
        print("\n[INFO] 系统已退出，感谢使用！")

    def stop(self):
        """停止系统"""
        self.is_running = False


def main():
    """主函数"""
    # 可以修改视频源：
    # - 0: 使用默认摄像头
    # - "videos/traffic.mp4": 使用视频文件
    video_source = "videos/traffic_rideo.mp4"  # 使用视频文件

    # 创建并运行系统
    system = SmartTransportationSystem(video_source=video_source)
    system.run()


if __name__ == "__main__":
    main()
