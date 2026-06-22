"""简单测试目标检测"""
import cv2
import sys
sys.path.insert(0, 'D:\\项目\\Smart Transportation')

from object_detection import ObjectDetector

# 初始化检测器
detector = ObjectDetector()
print(f"模型加载: {detector.model_name}")
print(f"类别: {detector.class_names}")

# 测试图片
test_image = cv2.imread('test.jpg')
if test_image is None:
    # 使用视频帧测试
    cap = cv2.VideoCapture('videos/traffic_rideo.mp4')
    ret, test_image = cap.read()
    cap.release()

if test_image is not None:
    print(f"输入图像: {test_image.shape}")
    
    # 执行检测
    results = detector.detect(test_image)
    print(f"检测结果类型: {type(results)}")
    print(f"结果数量: {len(results)}")
    
    # 过滤交通目标
    detections = detector.filter_traffic_targets(results)
    print(f"过滤后检测到 {len(detections)} 个目标")
    
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
else:
    print("无法加载测试图像")