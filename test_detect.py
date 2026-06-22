"""测试检测流程"""
import cv2
import sys
sys.path.insert(0, 'D:\\项目\\Smart Transportation')

from object_detection import ObjectDetector
from lane_detection import LaneDetector
from traffic_stats import TrafficStatistics

# 初始化模块
detector = ObjectDetector()
lane_detector = LaneDetector()
traffic_stats = TrafficStatistics()

# 测试视频路径
test_video = 'D:\\项目\\Smart Transportation\\videos\\traffic_rideo.mp4'

# 打开视频
cap = cv2.VideoCapture(test_video)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {test_video}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")

# 设置输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_output.mp4', fourcc, fps, (width, height))

current_frame = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\r处理帧: {current_frame+1}/{frame_count}", end='')
        
        # 检测目标
        results = detector.detect(frame)
        
        # 过滤交通目标
        detections = detector.filter_traffic_targets(results)
        print(f" | 检测到 {len(detections)} 个目标")
        
        # 车道线检测
        lane_results = lane_detector.detect(frame)
        
        # 获取车道线掩码
        lane_info = lane_detector.get_lane_masks(lane_results)
        
        # 绘制结果
        frame = detector.draw_detections(frame, detections)
        
        if len(lane_info) > 0:
            frame = lane_detector.draw_lane_regions(frame, lane_info)
        
        # 更新统计并绘制
        current_stats = traffic_stats.update(frame, detections)
        stats_frame = traffic_stats.draw_overlay(frame.copy(), current_stats)
        out.write(stats_frame)
        
        current_frame += 1
        
except Exception as e:
    print(f"\n错误发生在第 {current_frame} 帧:")
    import traceback
    traceback.print_exc()
    
finally:
    cap.release()
    out.release()
    print("\n处理完成！")

# 获取统计
final_stats = traffic_stats.get_statistics()
print(f"统计结果: {final_stats}")