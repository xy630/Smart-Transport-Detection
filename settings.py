# 智慧交通系统配置文件
# Smart Transportation System Configuration

import cv2

# ========== 系统设置 ==========
SYSTEM_NAME = "基于计算机视觉的智慧交通监测与辅助驾驶系统"
VERSION = "1.0.0"

# ========== 模型配置 ==========
MODEL_CONFIG = {
    # YOLOv8 目标检测模型配置
    "detection_model": {
        "model_name": "models/yolov8n.pt",  # 本地模型路径
        "conf_threshold": 0.30,       # 置信度阈值（提高以减少误检）
        "iou_threshold": 0.45,       # NMS IOU阈值
        "device": "cpu",             # 运行设备: cpu/cuda
    },

    # YOLOv8-seg 车道线分割模型配置
    "segmentation_model": {
        "model_name": "models/yolov8n-seg.pt",  # 本地模型路径
        "conf_threshold": 0.30,
        "iou_threshold": 0.45,
        "device": "cpu",
    }
}

# ========== 目标检测类别 ==========
# 注意：YOLOv8 COCO数据集类别编号
DETECTION_CLASSES = {
    0: "行人",      # person (0)
    1: "自行车",    # bicycle (1)
    2: "汽车",      # car (2)
    3: "摩托车",    # motorcycle (3)
    5: "公交车",    # bus (5, 注意：不是4!)
    7: "卡车",      # truck (7, 注意：不是5!)
}

# 需要统计的交通目标 (对应正确的类别编号)
TRAFFIC_TARGETS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ========== 车道线检测配置 ==========
LANE_CONFIG = {
    # 车道线颜色 (BGR)
    "lane_color": (0, 255, 0),        # 绿色
    "lane_alpha": 0.3,                # 透明度

    # 车道线区域填充
    "fill_lane": True,

    # 车道偏离预警
    "lane_departure_warning": True,
    "departure_threshold": 0.15,      # 偏离阈值（占图像宽度的比例）
}

# ========== 交通流量统计配置 ==========
TRAFFIC_STATS_CONFIG = {
    # 统计区域（ROI）设置
    "roi": {
        "x1": 0, "y1": 400,
        "x2": 640, "y2": 720,
    },

    # 车辆计数线（设置在画面中间偏下位置）
    "count_line": {
        "x1": 0, "y1": 250,
        "x2": 640, "y2": 250,
    },

    # 跟踪配置
    "max_disappeared": 30,            # 最大消失帧数
    "max_distance": 50,              # 最大跟踪距离
}

# ========== 视频配置 ==========
VIDEO_CONFIG = {
    # 输入源：0 表示摄像头，字符串表示视频文件路径
    "source": 0,

    # 视频保存配置
    "save_output": True,
    "output_path": "outputs/result.mp4",
    "fps": 30,

    # 显示配置
    "show_display": True,
    "window_name": "Smart Transportation System",
    "resize_scale": 1.0,              # 显示缩放
}

# ========== 界面配置 ==========
UI_CONFIG = {
    # 边框颜色 (BGR)
    "colors": {
        "person": (255, 0, 0),        # 蓝色
        "bicycle": (0, 255, 255),    # 黄色
        "car": (0, 165, 255),         # 橙色
        "motorcycle": (0, 255, 0),   # 绿色
        "bus": (128, 0, 128),         # 紫色
        "truck": (0, 0, 139),         # 深红色
    },

    # 字体
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.6,
    "font_thickness": 2,
}

# ========== 性能配置 ==========
PERFORMANCE_CONFIG = {
    "skip_frames": 0,                # 跳帧处理（0=处理所有帧）
    "threaded": False,               # 多线程处理
    "batch_size": 1,
}
