"""
智慧交通系统 - Flask 后端服务
提供视频上传、检测处理、结果返回等 API
"""

import sys
import io

# 设置编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import cv2
import time
import json
import uuid
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime

# 导入检测模块
from object_detection import ObjectDetector
from lane_detection import LaneDetector
from traffic_stats import TrafficStatistics
from danger_detection import DangerDetector

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 初始化检测模块
detector = ObjectDetector()
lane_detector = LaneDetector()
danger_detector = DangerDetector()
traffic_stats = TrafficStatistics()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """上传视频文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存文件
        file.save(filepath)
        
        # 获取视频信息
        video = cv2.VideoCapture(filepath)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'size': os.path.getsize(filepath),
            'duration': round(duration, 2),
            'fps': round(fps, 2),
            'width': width,
            'height': height,
            'frame_count': frame_count
        })
    
    return jsonify({'error': '不支持的文件格式'}), 400

@app.route('/api/detect', methods=['POST'])
def detect_video():
    """处理视频检测"""
    data = request.json
    file_id = data.get('file_id')
    options = data.get('options', {})
    
    if not file_id:
        return jsonify({'error': '缺少 file_id'}), 400
    
    # 查找上传的文件
    upload_files = os.listdir(UPLOAD_FOLDER)
    input_file = None
    for f in upload_files:
        if f.startswith(file_id):
            input_file = f
            break
    
    if not input_file:
        return jsonify({'error': '文件不存在'}), 404
    
    input_path = os.path.join(UPLOAD_FOLDER, input_file)
    
    # 重置统计数据（每次检测从零开始）
    global traffic_stats
    traffic_stats = TrafficStatistics()
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 设置视频输出（使用 H264 编码，浏览器支持最好）
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264 编码
    output_filename = f"{file_id}_result.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 检查 VideoWriter 是否成功打开
    if not out.isOpened():
        cap.release()
        return jsonify({'error': '无法创建输出视频文件'}), 500
    
    # 统计数据
    stats = {
        'total_frames': frame_count,
        'total_vehicles': 0,
        'count_by_class': {},
        'process_time': 0
    }
    
    start_time = time.time()
    current_frame = 0
    frames_written = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = []
            lane_info = []
            
            # 根据用户选择执行检测
            if options.get('detection', False) or options.get('alert', False) or options.get('stats', False):
                # 目标检测（危险预警和流量统计也需要目标检测结果）
                results = detector.detect(frame)
                detections = detector.filter_traffic_targets(results)
            
            if options.get('lane', False) or options.get('alert', False):
                # 车道线检测（危险预警需要车道线信息）
                lane_results = lane_detector.detect(frame)
                lane_info = lane_detector.get_lane_masks(lane_results)
            
            # 绘制结果（根据用户选择的功能）
            stats_frame = frame.copy()
            
            if options.get('detection', False):
                stats_frame = detector.draw_detections(stats_frame, detections)
            
            if options.get('lane', False) and len(lane_info) > 0:
                stats_frame = lane_detector.draw_lane_regions(stats_frame, lane_info)
            
            if options.get('stats', False):
                # 获取当前统计
                current_stats = traffic_stats.update(stats_frame, detections)
                # 绘制统计信息
                stats_frame = traffic_stats.draw_overlay(stats_frame, current_stats)
            
            # 危险预警检测
            if options.get('alert', False):
                warnings = danger_detector.detect_all(stats_frame, detections, lane_info)
                if len(warnings) > 0:
                    stats_frame = danger_detector.draw_warnings(stats_frame, warnings)
            
            # 写入帧
            if out.write(stats_frame):
                frames_written += 1
            
            current_frame += 1
            
            # 发送进度
            progress = int((current_frame / frame_count) * 100)
            
        # 获取最终统计
        final_stats = traffic_stats.get_statistics()
        stats['total_vehicles'] = final_stats['total_count']
        stats['count_by_class'] = final_stats['count_by_class']
        stats['process_time'] = round(time.time() - start_time, 2)
        
        cap.release()
        out.release()
        
        # 保存统计数据
        stats_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'stats': stats
        })
    
    except Exception as e:
        cap.release()
        if out is not None:
            out.release()
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<file_id>', methods=['GET'])
def get_results(file_id):
    """获取检测结果"""
    stats_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_stats.json")
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return jsonify(stats)
    
    return jsonify({'error': '结果不存在'}), 404

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载结果文件"""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    if os.path.exists(filepath):
        response = send_file(filepath, as_attachment=False)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    return jsonify({'error': '文件不存在'}), 404

@app.route('/api/stream/<file_id>')
def stream_video(file_id):
    """MJPEG 流式传输检测结果"""
    input_file = None
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(file_id):
            input_file = f
            break
    
    if not input_file:
        return jsonify({'error': '文件不存在'}), 404
    
    input_path = os.path.join(UPLOAD_FOLDER, input_file)
    
    # 重置统计数据（流式传输也需要重置）
    global traffic_stats
    traffic_stats = TrafficStatistics()
    
    def generate():
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测目标
                results = detector.detect(frame)
                detections = detector.filter_traffic_targets(results)
                
                # 车道线检测
                lane_results = lane_detector.detect(frame)
                lane_info = lane_detector.get_lane_masks(lane_results)
                
                # 绘制结果
                frame = detector.draw_detections(frame, detections)
                if len(lane_info) > 0:
                    frame = lane_detector.draw_lane_regions(frame, lane_info)
                
                # 更新统计
                traffic_stats.update(frame, detections)
                stats = traffic_stats.get_statistics()
                
                # 绘制统计信息
                frame = traffic_stats.draw_overlay(frame.copy(), {
                    'frame_count': traffic_stats.frame_count,
                    'fps': traffic_stats.fps,
                    'current_vehicles': stats['current_tracked'],
                    'total_count': stats['total_count'],
                    'count_by_class': stats['count_by_class']
                })
                
                # 编码为 JPEG
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # 控制帧率
                import time
                time.sleep(1/fps)
                
        finally:
            cap.release()
    
    from flask import Response
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取模型状态"""
    return jsonify({
        'detection_model': 'YOLOv8n',
        'segmentation_model': 'YOLOv8n-seg',
        'status': 'ready'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)