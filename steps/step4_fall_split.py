"""Step 4: 按跌倒次数分割模块"""
import shutil
from pathlib import Path
from typing import List, Tuple
import config
from utils import FileUtils, AnnotationParser


class FallSegmenter:
    """跌倒分割器 - 按跌倒事件分割连续帧"""

    def __init__(self, frame_dir: Path, annotation_path: Path):
        self.frame_dir = frame_dir
        self.annotation_path = annotation_path

    def load_intervals(self) -> List[Tuple[float, float]]:
        """加载跌倒区间"""
        parser = AnnotationParser()
        return parser.parse(self.annotation_path)

    def split_by_falls(self, intervals: List[Tuple[float, float]], fps: float) -> List[Tuple[int, int]]:
        """
        按跌倒区间分割帧范围
        intervals: [(start_ms, end_ms), ...]
        fps: 帧率
        返回: [(start_frame, end_frame), ...]
        """
        if not intervals:
            # 没有跌倒区间，返回完整范围
            frames = sorted(self.frame_dir.glob('*.jpg'))
            return [(0, len(frames) - 1)]

        # 按起始时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        splits = []
        prev_end = 0  # 上一页的结束位置

        for i, (start_ms, end_ms) in enumerate(sorted_intervals):
            start_frame = prev_end
            end_frame = int(start_ms / 1000 * fps)
            if end_frame > start_frame:
                splits.append((start_frame, end_frame - 1))

            # 跌倒区间单独成一段
            fall_start_frame = int(start_ms / 1000 * fps)
            fall_end_frame = int(end_ms / 1000 * fps)
            splits.append((fall_start_frame, fall_end_frame))
            prev_end = fall_end_frame + 1

        # 最后一段：从最后一个跌倒终点到视频结尾
        frames = sorted(self.frame_dir.glob('*.jpg'))
        total_frames = len(frames)
        last_end = int(sorted_intervals[-1][1] / 1000 * fps)
        if last_end < total_frames - 1:
            splits.append((last_end + 1, total_frames - 1))

        return splits

    def run(self) -> List[Path]:
        """执行分割"""
        logger.info(f"[Step4] 开始跌倒分割: {self.frame_dir.name}")

        # 加载跌倒区间
        intervals = self.load_intervals()
        logger.info(f"  跌倒区间数: {len(intervals)}")

        if len(intervals) <= 1:
            logger.info("[Step4] 只有一次或没有跌倒，无需分割")
            return [self.frame_dir]

        # 获取fps
        fps = config.VIDEO_FPS  # 20.0 fps

        # 分割
        splits = self.split_by_falls(intervals, fps)
        logger.info(f"  分割为 {len(splits)} 段")

        # 创建子文件夹
        result_dirs = []
        frames = sorted(self.frame_dir.glob('*.jpg'))

        for i, (start_f, end_f) in enumerate(splits):
            dst_dir = self.frame_dir.parent / f"{self.frame_dir.name}_fall{i+1}"
            dst_dir.mkdir(parents=True, exist_ok=True)

            for f in frames[start_f:end_f+1]:
                shutil.copy2(f, dst_dir / f.name)
            result_dirs.append(dst_dir)

        # 复制标注文件
        if self.annotation_path.exists():
            for d in result_dirs:
                dst_ann = d / "annotation.txt"
                shutil.copy2(self.annotation_path, dst_ann)

        logger.info(f"[Step4] 分割完成: {len(result_dirs)} 个目录")
        return result_dirs


def run(frame_dir: Path = None) -> List[Path]:
    """执行分割"""
    if frame_dir is None:
        # 默认处理第一个清洗后的目录
        base_dir = config.FRAME_OUTPUT_DIR
        dirs = sorted(base_dir.glob(f"{config.VIDEO_NAME}_*"))
        if dirs:
            frame_dir = dirs[0]
        else:
            frame_dir = base_dir / config.VIDEO_NAME

    annotation_path = frame_dir / "annotation.txt"
    if not annotation_path.exists():
        annotation_path = config.ANNOTATION_DIR / frame_dir.name / "annotation.txt"

    segmenter = FallSegmenter(frame_dir, annotation_path)
    return segmenter.run()


if __name__ == '__main__':
    run()