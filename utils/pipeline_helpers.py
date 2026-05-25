"""Shared pipeline helpers for fall-detection training data generation."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .annotation_parser import AnnotationParser

logger = logging.getLogger(__name__)


def _elapsed(start_time: float) -> float:
    return time.perf_counter() - start_time


def get_runtime_snapshot() -> dict:
    snapshot = {'time': time.perf_counter()}
    try:
        import psutil
        process = psutil.Process()
        snapshot['rss_mb'] = round(process.memory_info().rss / 1024**2, 2)
    except Exception:
        snapshot['rss_mb'] = None
    try:
        import torch
        if torch.cuda.is_available():
            snapshot['cuda_devices'] = torch.cuda.device_count()
            snapshot['cuda_allocated_mb'] = round(torch.cuda.memory_allocated() / 1024**2, 2)
        else:
            snapshot['cuda_devices'] = 0
            snapshot['cuda_allocated_mb'] = 0.0
    except Exception:
        snapshot['cuda_devices'] = None
        snapshot['cuda_allocated_mb'] = None
    return snapshot



def _log_timing(stage: str, start_time: float, **stats):
    elapsed = _elapsed(start_time)
    stats_text = ", ".join(f"{k}={v}" for k, v in stats.items() if v is not None)
    if stats_text:
        logger.info(f"[Timing] {stage}: {elapsed:.2f}s | {stats_text}")
    else:
        logger.info(f"[Timing] {stage}: {elapsed:.2f}s")
    return elapsed


def load_pose_results(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    return data['results']


def find_annotation_path(annotation_dir: Path, video_name: str) -> Path | None:
    candidate_paths = [
        annotation_dir / video_name / "annotation.txt",
        annotation_dir / f"{video_name}.txt",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    for sub_dir in annotation_dir.iterdir():
        if sub_dir.is_dir() and sub_dir.name == video_name:
            candidate = sub_dir / "annotation.txt"
            if candidate.exists():
                return candidate

    recursive_matches = list(annotation_dir.glob(f"**/{video_name}/annotation.txt"))
    if recursive_matches:
        return recursive_matches[0]

    return None


def resolve_annotation_state(annotation_dir: Path, video_name: str) -> tuple[Path | None, str, list]:
    annotation_path = find_annotation_path(annotation_dir, video_name)
    if annotation_path is None:
        logger.warning(f"[Annotation] 标注缺失，按无跌倒处理: video={video_name}, annotation_dir={annotation_dir}")
        return None, "missing", []

    content = annotation_path.read_text(encoding='utf-8').strip()
    if not content:
        logger.warning(f"[Annotation] 标注为空，按无跌倒处理: {annotation_path}")
        return annotation_path, "empty", []

    if content.lower() == "discard":
        logger.info(f"[Annotation] 视频标记为 discard，跳过全部流程: {video_name}")
        return annotation_path, "discard", []

    intervals_ms = AnnotationParser.parse(annotation_path)
    return annotation_path, "normal", intervals_ms


def iter_video_files(video_dir: Path) -> list[Path]:
    videos = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        videos.extend(video_dir.glob(f"**/{ext}"))
    return sorted(videos)


def find_video_path(video_dir: Path, video_name: str) -> Path | None:
    for ext in ['.mp4', '.avi', '.mov']:
        path = video_dir / f"{video_name}{ext}"
        if path.exists():
            return path
        for sub_dir in video_dir.rglob('*'):
            if sub_dir.is_dir():
                path = sub_dir / f"{video_name}{ext}"
                if path.exists():
                    return path
    return None


def is_pose_round_complete(round_dir: Path) -> bool:
    pose_results_path = round_dir / "pose_results.npz"
    frame_files = sorted(round_dir.glob('frame_*.jpg'))
    if not frame_files or not pose_results_path.exists() or pose_results_path.stat().st_size == 0:
        return False
    try:
        data = np.load(pose_results_path, allow_pickle=True)
        results = data['results']
    except Exception:
        return False
    return len(results) == len(frame_files)


def is_round_feature_complete(output_dir: Path, round_idx: int) -> bool:
    features_csv = output_dir / f"round_{round_idx}_output" / f"round_{round_idx}_features.csv"
    if not features_csv.exists() or features_csv.stat().st_size == 0:
        return False
    try:
        pd.read_csv(features_csv)
        return True
    except Exception:
        return False


def collect_existing_round_samples(output_dir: Path, round_idx: int) -> list[dict]:
    features_csv = output_dir / f"round_{round_idx}_output" / f"round_{round_idx}_features.csv"
    if not features_csv.exists() or features_csv.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(features_csv)
    except Exception:
        return []
    return df.to_dict('records')


def is_video_results_bundle_complete(video_frame_dir: Path) -> bool:
    results_dir = video_frame_dir / "results"
    bundle_files = [
        results_dir / "all_features.csv",
        results_dir / "samples.npz",
        results_dir / "sample_details.txt",
        results_dir / "samples_with_details.csv",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in bundle_files)


def _snapshot_video_file(video_path: Path) -> tuple[int, float] | None:
    try:
        stat = video_path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime


def _collect_ready_videos(
    video_dir: Path,
    frame_output_dir: Path,
    resume: bool,
    file_snapshots: dict[str, tuple[int, float]],
    stable_seconds: int,
    in_progress: set[str] | None = None,
    cooldown_until: dict[str, float] | None = None,
) -> tuple[list[Path], dict[str, tuple[int, float]]]:
    now = time.time()
    updated_snapshots: dict[str, tuple[int, float]] = {}
    ready_videos: list[Path] = []
    in_progress = in_progress or set()
    cooldown_until = cooldown_until or {}

    for video_path in iter_video_files(video_dir):
        video_key = str(video_path.resolve())
        if video_key in in_progress:
            continue

        snapshot = _snapshot_video_file(video_path)
        if snapshot is None:
            continue
        updated_snapshots[video_key] = snapshot

        previous_snapshot = file_snapshots.get(video_key)
        size, mtime = snapshot

        if video_key in cooldown_until and cooldown_until[video_key] > now:
            continue
        if resume and is_video_results_bundle_complete(frame_output_dir / video_path.stem):
            continue
        if now - mtime < stable_seconds:
            continue
        if previous_snapshot != snapshot:
            continue

        ready_videos.append(video_path)

    return sorted(ready_videos), updated_snapshots
