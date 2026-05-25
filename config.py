"""配置文件 - 跌倒检测训练数据生成"""
import os
from pathlib import Path

# 路径配置
VIDEO_DIR = Path(r"D:\IPC\IPC_video_data_mp4\跌倒\data_positive_0512")
ANNOTATION_DIR = Path(r"D:\IPC\IPC_video_data_annotation\跌倒_v2")
FRAME_OUTPUT_DIR = Path(r"D:\IPC\IPC_data_clip_photo\跌倒")
MODEL_PATH = Path(r"D:\pycharm\工作脚本\跌倒检测\跌倒训练数据生成\yolo26l-pose.pt")

# 视频信息 (实际FPS=20, 总帧数=662)
VIDEO_NAME = "2026-03-03_10-18-25"  # 默认视频名
VIDEO_FPS = 20.0
VIDEO_TOTAL_FRAMES = 662

# 帧率配置
FRAME_INTERVALS_MS = [250, 300, 350, 400, 450, 500]
SAMPLE_FRAMES = 11

# 关键点配置 (COCO 17点)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
NUM_KEYPOINTS = 17
HIP_CENTER_IDX = (11 + 12) // 2  # 11=left_hip, 12=right_hip

# Step1 FFmpeg配置
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
STEP1_FFMPEG_WORKERS = 1
STEP1_DECODE_MODE = "auto"  # 可选: auto / ffmpeg / ffmpeg_batch / ffmpeg_gpu
STEP1_BATCH_SEEK_TOLERANCE_MS = 120
STEP1_ENABLE_TIMING_LOG = True

# Step2 Pose检测配置
POSE_INFER_MODE = "single_gpu_batch"  # 可选: multiprocess / single_gpu_batch
POSE_BATCH_SIZE = 8
POSE_NUM_WORKERS = 4
POSE_DEVICE = "auto"  # 可选: auto / cpu / cuda
POSE_AUTO_MULTIPROCESS_MIN_FRAMES = 0

# 特征维度
FEATURE_DIM = 101
POS_DIM = 34  # 17 * 2
INTERVAL_DIM = 10
BBOX_WH_DIM = 22  # 11帧 * 2
REL_POS_DIM = 34
ANGLE_SPINE_LEG = 100
HEIGHT_CHANGE = 101
BODY_ORIENTATION = 102
LABEL_DIM = 103