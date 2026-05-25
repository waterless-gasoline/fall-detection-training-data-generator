"""文件操作工具"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any


class FileUtils:
    """文件操作工具"""

    @staticmethod
    def ensure_dir(path: Path) -> None:
        """确保目录存在"""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_json(data: Any, file_path: Path) -> None:
        """保存JSON文件"""
        FileUtils.ensure_dir(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(file_path: Path) -> Any:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def copy_annotation(src: Path, dst: Path) -> None:
        """复制标注文件到目标目录"""
        if src.exists():
            FileUtils.ensure_dir(dst.parent)
            shutil.copy2(src, dst)

    @staticmethod
    def get_frame_files(dir_path: Path, ext: str = '.jpg') -> List[Path]:
        """获取目录中所有帧文件(按序号排序)"""
        if not dir_path.exists():
            return []
        frames = sorted(dir_path.glob(f'*{ext}'))
        return frames

    @staticmethod
    def split_folder(src_folder: Path, split_indices: List[int], dst_base: Path, suffix: str) -> List[Path]:
        """
        按帧索引分割文件夹
        split_indices: 分段边界帧索引
        """
        frames = FileUtils.get_frame_files(src_folder)
        if not frames:
            return []

        splits = []
        start_idx = 0
        split_num = 1

        for end_idx in split_indices:
            if end_idx <= start_idx:
                continue
            dst_folder = dst_base.parent / f"{dst_base.name}_{suffix}{split_num}"
            FileUtils.ensure_dir(dst_folder)

            for i in range(start_idx, min(end_idx, len(frames))):
                shutil.copy2(frames[i], dst_folder / frames[i].name)
            splits.append(dst_folder)
            start_idx = end_idx
            split_num += 1

        # 处理最后一段
        if start_idx < len(frames):
            dst_folder = dst_base.parent / f"{dst_base.name}_{suffix}{split_num}"
            FileUtils.ensure_dir(dst_folder)
            for i in range(start_idx, len(frames)):
                shutil.copy2(frames[i], dst_folder / frames[i].name)
            splits.append(dst_folder)

        return splits