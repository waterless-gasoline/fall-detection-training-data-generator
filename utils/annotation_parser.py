"""标注文件解析工具 - 支持用户格式: 1;8-10 表示8-10秒的跌倒"""
from pathlib import Path
from typing import List, Tuple


class AnnotationParser:
    """解析annotation.txt跌倒区间"""

    @staticmethod
    def parse(file_path: Path) -> List[Tuple[float, float]]:
        """
        解析标注文件
        支持格式:
        - "1;8-10" (序号;起始-结束，秒)
        - "8000 10000" (起始 结束，毫秒)
        - "8000,10000" (起始,结束，毫秒)
        返回: [(start_ms, end_ms), ...]
        """
        intervals = []
        if not file_path.exists():
            print(f"  [AnnotationParser] 文件不存在: {file_path}")
            return intervals

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 尝试格式1: "序号;起始-结束" (如 "1;8-10")
                if ';' in line:
                    parts = line.split(';')
                    if len(parts) >= 2:
                        time_part = parts[1]
                        if '-' in time_part:
                            try:
                                start_str, end_str = time_part.split('-')
                                start_sec = float(start_str)
                                end_sec = float(end_str)
                                intervals.append((start_sec * 1000, end_sec * 1000))  # 转换为毫秒
                                continue
                            except ValueError:
                                pass

                # 尝试格式2: "起始 结束" 或 "起始,结束" (毫秒)
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        # 如果值 > 1000，认为是毫秒；否则是秒
                        if start > 1000 or end > 1000:
                            intervals.append((start, end))
                        else:
                            intervals.append((start * 1000, end * 1000))
                    except ValueError:
                        continue

        return intervals

    @staticmethod
    def get_frame_intervals(intervals_ms: List[Tuple[float, float]], fps: float) -> List[Tuple[int, int]]:
        """将毫秒区间转换为帧区间"""
        frame_intervals = []
        for start_ms, end_ms in intervals_ms:
            start_frame = int(start_ms / 1000 * fps)
            end_frame = int(end_ms / 1000 * fps)
            frame_intervals.append((start_frame, end_frame))
        return frame_intervals

    @staticmethod
    def filter_intervals_in_range(intervals: List[Tuple[float, float]], start_ms: float, end_ms: float) -> List[Tuple[float, float]]:
        """过滤出完全落在指定范围内的跌倒区间"""
        filtered = []
        for start, end in intervals:
            if start >= start_ms and end <= end_ms:
                filtered.append((start, end))
        return filtered


if __name__ == '__main__':
    # 测试
    test_file = Path("D:/IPC/IPC_video_data_annotation/跌倒/2026-03-03_10-18-25/annotation.txt")
    intervals = AnnotationParser.parse(test_file)
    print(f"解析结果: {intervals}")
    for start, end in intervals:
        print(f"  跌倒: {start/1000:.1f}s - {end/1000:.1f}s")