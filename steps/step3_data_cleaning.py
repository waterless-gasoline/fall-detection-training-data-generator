"""Step 3: 数据清洗模块 - 按无人物区间分割连续帧"""
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import config
from utils import FileUtils, AnnotationParser


class DataCleaner:
    """数据清洗器 - 按无人物区间分割"""

    def __init__(self, detection_path: Path, frame_dir: Path, annotation_path: Path):
        self.detection_path = detection_path
        self.frame_dir = frame_dir
        self.annotation_path = annotation_path

    def load_detections(self) -> List[dict]:
        """加载检测结果"""
        with open(self.detection_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_person_gaps(self, detections: List[dict], gap_threshold: int = 20) -> List[Tuple[int, int]]:
        """
        找出无人物的帧区间
        gap_threshold: 连续多少帧无人就判定为gap
        返回: [(gap_start, gap_end), ...]
        """
        person_flags = []
        for det in detections:
            num_det = det.get('num_detected', 0)
            person_flags.append(num_det > 0)

        gaps = []
        in_gap = False
        gap_start = 0

        for i, has_person in enumerate(person_flags):
            if not has_person and not in_gap:
                in_gap = True
                gap_start = i
            elif has_person and in_gap:
                in_gap = False
                gap_len = i - gap_start
                if gap_len >= gap_threshold:
                    gaps.append((gap_start, i))

        return gaps

    def split_by_gaps(self, frame_dir: Path, gaps: List[Tuple[int, int]], video_name: str) -> List[Path]:
        """
        按gap分割文件夹
        gaps: [(gap_start, gap_end), ...]
        """
        if not gaps:
            # 没有gap，返回原目录
            return [frame_dir]

        # 构建分割点
        split_points = []
        for gap_start, gap_end in gaps:
            split_points.append(gap_start)

        frames = sorted(frame_dir.glob('*.jpg'))
        splits = []
        start_idx = 0
        split_num = 1

        for gap_start in split_points:
            if gap_start <= start_idx:
                continue

            dst_folder = config.FRAME_OUTPUT_DIR / f"{video_name}_{split_num}"
            dst_folder.mkdir(parents=True, exist_ok=True)

            for i in range(start_idx, gap_start):
                shutil.copy2(frames[i], dst_folder / frames[i].name)
            splits.append(dst_folder)
            start_idx = gap_start + (gaps[len([s for s in split_points if s < gap_start])][1] - gaps[len([s for s in split_points if s < gap_start])][0])
            split_num += 1

        # 处理最后一段
        if start_idx < len(frames):
            dst_folder = config.FRAME_OUTPUT_DIR / f"{video_name}_{split_num}"
            dst_folder.mkdir(parents=True, exist_ok=True)
            for i in range(start_idx, len(frames)):
                shutil.copy2(frames[i], dst_folder / frames[i].name)
            splits.append(dst_folder)

        return splits

    def run(self) -> List[Path]:
        """执行清洗"""
        logger.info(f"[Step3] 开始数据清洗...")

        # 加载检测结果
        detections = self.load_detections()
        logger.info(f"  总帧数: {len(detections)}")

        # 找无人物区间
        gaps = self.find_person_gaps(detections)
        logger.info(f"  发现 {len(gaps)} 个无人物区间: {gaps}")

        if not gaps:
            logger.info("[Step3] 没有需要分割的区间")
            return [self.frame_dir]

        # 分割文件夹
        splits = self.split_by_gaps(self.frame_dir, gaps, self.frame_dir.name)

        # 复制标注文件
        if self.annotation_path.exists():
            for split_dir in splits:
                dst_annotation = split_dir / "annotation.txt"
                shutil.copy2(self.annotation_path, dst_annotation)

        logger.info(f"[Step3] 分割完成: {len(splits)} 个子文件夹")
        for s in splits:
            logger.info(f"    - {s.name}")
        return splits


def run(video_name: str = config.VIDEO_NAME) -> List[Path]:
    """执行清洗"""
    detection_path = Path(__file__).parent.parent / f"{video_name}_detection.json"
    frame_dir = config.FRAME_OUTPUT_DIR / video_name
    annotation_path = config.ANNOTATION_DIR / video_name / "annotation.txt"

    cleaner = DataCleaner(detection_path, frame_dir, annotation_path)
    return cleaner.run()


if __name__ == '__main__':
    run()