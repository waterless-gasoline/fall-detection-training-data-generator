"""Step 5: 样本提取模块"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import config
from utils import AnnotationParser
import config


class SampleExtractor:
    """样本提取器 - 11帧/样本，6种帧率，滑动窗口"""

    def __init__(self, frame_dir: Path, annotation_path: Path, detection_path: Path):
        self.frame_dir = frame_dir
        self.annotation_path = annotation_path
        self.detection_path = detection_path
        self.fps = config.VIDEO_FPS  # 20.0 fps
        self.intervals_ms = config.FRAME_INTERVALS_MS
        self.sample_frames = config.SAMPLE_FRAMES

    def load_detections(self) -> List[dict]:
        """加载检测结果"""
        with open(self.detection_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_frame_indices(self, start_frame: int, interval_ms: float) -> List[int]:
        """
        根据起始帧和间隔计算11帧的帧索引
        interval_ms: 毫秒间隔
        """
        indices = []
        current = start_frame
        for i in range(self.sample_frames):
            indices.append(current)
            current += int(interval_ms / 1000 * self.fps)
        return indices

    def extract_samples(self) -> List[Dict]:
        """提取所有样本"""
        detections = self.load_detections()
        total_frames = len(detections)

        # 获取实际的帧文件范围
        frame_files = sorted(self.frame_dir.glob('*.jpg'))
        actual_frame_count = len(frame_files)
        logger.info(f"  实际帧文件数: {actual_frame_count}, 检测结果数: {total_frames}")

        samples = []
        # 对每种帧率间隔
        for interval_ms in self.intervals_ms:
            # 从第一帧开始滑动
            frame_idx = 0
            while True:
                # 计算这11帧的索引
                indices = self.get_frame_indices(frame_idx, interval_ms)

                # 检查是否越界(使用实际帧数而不是检测结果总数)
                if indices[-1] >= actual_frame_count:
                    break

                # 提取这11帧的检测数据
                sample = {
                    'interval_ms': interval_ms,
                    'start_frame': frame_idx,
                    'indices': indices,
                    'detections': [detections[i] for i in indices],
                    'start_time': frame_idx / self.fps,
                    'end_time': indices[-1] / self.fps
                }
                samples.append(sample)

                # 滑动1帧
                frame_idx += 1

        return samples

    def run(self) -> List[Dict]:
        """执行提取"""
        logger.info(f"[Step5] 开始样本提取: {self.frame_dir.name}")
        logger.info(f"  帧率间隔: {self.intervals_ms}")
        logger.info(f"  样本帧数: {self.sample_frames}")

        samples = self.extract_samples()
        logger.info(f"  提取样本数: {len(samples)}")

        return samples


def run(frame_dir: Path = None) -> List[Dict]:
    """执行提取"""
    if frame_dir is None:
        dirs = sorted(config.FRAME_OUTPUT_DIR.glob(f"{config.VIDEO_NAME}_*"))
        frame_dir = dirs[0] if dirs else config.FRAME_OUTPUT_DIR / config.VIDEO_NAME

    # 找到检测结果文件
    detection_path = Path(__file__).parent.parent / f"{config.VIDEO_NAME}_detection.json"
    annotation_path = frame_dir / "annotation.txt"

    extractor = SampleExtractor(frame_dir, annotation_path, detection_path)
    return extractor.extract_samples()


if __name__ == '__main__':
    samples = run()
    logger.info(f"共提取 {len(samples)} 个样本")