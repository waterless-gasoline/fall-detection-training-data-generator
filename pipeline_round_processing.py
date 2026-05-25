"""Round processing and NPZ generation helpers for the main pipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import random
import shutil

from utils import AnnotationParser
from utils.pipeline_helpers import _elapsed, _log_timing, load_pose_results
from utils.pipeline_features import (
    compute_features_batch,
    compute_sample_debug_details,
    compute_sample_features_batch_for_npz,
    extract_samples_parallel,
)


def _mp():
    module = sys.modules.get('main_pipeline') or sys.modules.get('__main__')
    if module is None:
        import main_pipeline as module  # type: ignore
    return module


def _logger():
    return _mp().logger


def load_frame_timestamps(round_dir: Path):
    return _mp().load_frame_timestamps(round_dir)


def _read_image(image_path: Path):
    if not image_path.exists() or image_path.stat().st_size == 0:
        return None
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def clean_no_person_frames(frames: list, detections: np.ndarray, gap_threshold: int = 20) -> list:
    """清洗逻辑：删除所有无人帧，以及无人区间前后的短暂有人帧"""
    logger = _logger()
    person_flags = np.array([det.get('has_person', False) for det in detections])
    n = len(person_flags)

    if n == 0:
        logger.info("  清洗: 检测结果为空，保留全部帧")
        return list(range(len(frames)))

    no_person_regions = []
    in_region = False
    start = 0

    for i in range(n):
        if not person_flags[i] and not in_region:
            in_region = True
            start = i
        elif person_flags[i] and in_region:
            in_region = False
            no_person_regions.append((start, i - 1))

    if not person_flags[-1]:
        no_person_regions.append((start, n - 1))

    logger.info(f"  无人区间: {[(f'{s}-{e}', e-s+1) for s,e in no_person_regions]}")

    to_delete = set()
    for start, end in no_person_regions:
        for i in range(start, end + 1):
            to_delete.add(i)

        if start > 0 and not person_flags[start - 1]:
            pre_start = start - 1
            while pre_start > 0 and not person_flags[pre_start - 1]:
                pre_start -= 1
            if start - pre_start <= gap_threshold:
                for i in range(pre_start, start):
                    to_delete.add(i)

        if end < n - 1 and not person_flags[end + 1]:
            post_end = end + 1
            while post_end < n - 1 and not person_flags[post_end + 1]:
                post_end += 1
            if post_end - end <= gap_threshold:
                for i in range(end + 1, post_end + 1):
                    to_delete.add(i)

    kept_indices = [i for i in range(n) if i not in to_delete]
    logger.info(f"  清洗: 删除{len(to_delete)}帧, 保留{len(kept_indices)}帧")

    return kept_indices


def clip_video_from_frames(frame_paths: list, output_path: Path, fps: float = 20.0):
    """将帧序列合成为视频"""
    if not frame_paths:
        return False

    first_frame = _read_image(frame_paths[0])
    if first_frame is None:
        return False

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_path in frame_paths:
        frame = _read_image(frame_path)
        if frame is not None:
            out.write(frame)

    out.release()
    return True


def create_fall_video_clips(
    round_dir: Path,
    segment_output_dir: Path,
    samples: list,
    round_idx: int,
    timestamps: list,
    video_path: Path,
    fps: float = 20.0,
):
    """为跌倒样本创建视频剪辑"""
    logger = _logger()
    clips_dir = segment_output_dir / "fall_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    fall_clips_paths = []
    frame_files = sorted(round_dir.glob('*.jpg'))
    if not frame_files:
        logger.warning(f"  警告: {round_dir} 中没有找到帧文件")
        return fall_clips_paths

    for i, sample in enumerate(samples):
        if sample.get('label', 0) != 1:
            continue

        start_frame = sample['start_frame']
        sample_timestamps = sample['timestamps']
        start_time = sample_timestamps[0]
        end_time = sample_timestamps[-1]

        clip_frame_paths = []
        for offset in range(11):
            frame_idx = start_frame + offset
            if frame_idx < len(frame_files):
                clip_frame_paths.append(frame_files[frame_idx])

        if len(clip_frame_paths) >= 2:
            output_path = clips_dir / f"round{round_idx}_fall_{i}_T{start_time:.2f}-{end_time:.2f}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            first_frame = _read_image(clip_frame_paths[0])
            if first_frame is None:
                continue
            h, w = first_frame.shape[:2]
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            for frame_path in clip_frame_paths:
                frame = _read_image(frame_path)
                if frame is not None:
                    out.write(frame)
            out.release()
            fall_clips_paths.append(str(output_path))

    logger.info(f"  跌倒视频剪辑: {len(fall_clips_paths)} 个 -> {clips_dir}")
    return fall_clips_paths


def process_single_round(
    round_dir: Path,
    round_idx: int,
    pose_results_path: Path,
    annotation_path: Path,
    output_dir: Path,
    video_path: Path = None,
) -> Tuple[str, List[Dict]]:
    """处理单个round的数据"""
    logger = _logger()
    round_start = time.perf_counter()
    logger.info(f"\n{'='*50}")
    logger.info(f"处理 Round {round_idx}: {round_dir.name}")
    logger.info(f"{'='*50}")

    segment_output_dir = output_dir / f"round_{round_idx}_output"
    segment_output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    detections = load_pose_results(pose_results_path)
    timestamps = load_frame_timestamps(round_dir)
    _log_timing(f"Round {round_idx} load inputs", load_start, detections=len(detections), has_timestamps=timestamps is not None)

    logger.info(f"  总帧数: {len(detections)}")

    if timestamps is None:
        fps = 20.0
        total_frames = len(detections)
        timestamps = [i / fps for i in range(total_frames)]
        logger.info(f"  警告: 无时间戳文件，使用估算: FPS={fps}")

    frames = sorted(round_dir.glob('*.jpg'))
    clean_start = time.perf_counter()
    kept_indices = clean_no_person_frames(frames, detections, gap_threshold=20)
    _log_timing(
        f"Round {round_idx} clean no-person frames",
        clean_start,
        input_frames=len(frames),
        kept_frames=len(kept_indices),
        removed_frames=len(frames) - len(kept_indices),
    )

    if len(kept_indices) == len(detections):
        logger.info("  没有需要清洗的区间")
        segments = [round_dir]
        kept_timestamps = timestamps
        cleaned_detections = detections
    else:
        cleaned_dir = segment_output_dir / f"{round_dir.name}_cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        kept_timestamps = [timestamps[i] for i in kept_indices if i < len(timestamps)]
        cleaned_detections = [detections[i] for i in kept_indices if i < len(detections)]

        copy_start = time.perf_counter()
        for orig_idx in kept_indices:
            shutil.copy2(frames[orig_idx], cleaned_dir / frames[orig_idx].name)
        _log_timing(f"Round {round_idx} copy cleaned frames", copy_start, copied_frames=len(kept_indices))

        logger.info(f"  清洗完成: {len(kept_indices)} 帧 -> {cleaned_dir.name}")
        segments = [cleaned_dir]

        dst_ann = cleaned_dir / "annotation.txt"
        if annotation_path is not None and not dst_ann.exists() and annotation_path.exists():
            shutil.copy2(annotation_path, dst_ann)

    if annotation_path is None:
        intervals_ms = []
    else:
        parser = AnnotationParser()
        intervals_ms = parser.parse(annotation_path)
    logger.info(f"  跌倒区间: {[(s / 1000, e / 1000) for s, e in intervals_ms]}")
    logger.info(
        f"  [DEBUG][Round {round_idx}] detections={len(detections)}, frames={len(frames)}, "
        f"kept_indices={len(kept_indices)}, kept_timestamps={len(kept_timestamps)}, intervals={len(intervals_ms)}"
    )

    sample_frames = 11

    for seg_dir in sorted(segments):
        segment_start = time.perf_counter()
        logger.info(f"\n  处理片段: {seg_dir.name}")

        seg_detections = cleaned_detections
        cleaned_timestamps = kept_timestamps

        sample_extract_start = time.perf_counter()
        samples = extract_samples_parallel(
            seg_dir, seg_detections, cleaned_timestamps, intervals_ms,
            sample_frames,
        )
        _log_timing(f"Round {round_idx} sample extraction", sample_extract_start, samples=len(samples), frames=len(cleaned_timestamps))
        logger.info(f"  提取样本数: {len(samples)}")
        logger.info(
            f"  [DEBUG][Round {round_idx}][{seg_dir.name}] seg_detections={len(seg_detections)}, "
            f"cleaned_timestamps={len(cleaned_timestamps)}, sample_frames={sample_frames}, intervals={len(intervals_ms)}"
        )

        if not samples:
            debug_last_timestamp = cleaned_timestamps[-1] if cleaned_timestamps else None
            debug_first_timestamp = cleaned_timestamps[0] if cleaned_timestamps else None
            logger.warning(
                f"  [DEBUG][Round {round_idx}][{seg_dir.name}] zero samples. "
                f"first_ts={debug_first_timestamp}, last_ts={debug_last_timestamp}, "
                f"duration={(debug_last_timestamp - debug_first_timestamp) if len(cleaned_timestamps) >= 2 else None}, "
                f"annotation_intervals={[(s / 1000, e / 1000) for s, e in intervals_ms[:10]]}"
            )
            logger.info("  跳过 - 此段无样本")
            _log_timing(f"Round {round_idx} segment total", segment_start, segment=seg_dir.name, samples=0)
            continue

        logger.info("  开始特征计算(并行)...")

        feature_start = time.perf_counter()
        all_sample_features, all_relationships = compute_features_batch(samples, intervals_ms)
        feature_elapsed = _elapsed(feature_start)
        _log_timing(
            f"Round {round_idx} feature computation",
            feature_start,
            samples=len(samples),
            per_sample_s=f"{feature_elapsed / len(samples):.4f}" if len(samples) else None,
        )
        all_labels = [int(feat[-1]) for feat in all_sample_features]

        for i, label in enumerate(all_labels):
            samples[i]['label'] = label

        if video_path and video_path.exists():
            clip_start = time.perf_counter()
            create_fall_video_clips(
                seg_dir, segment_output_dir, samples,
                round_idx, cleaned_timestamps, video_path, fps=20.0,
            )
            _log_timing(f"Round {round_idx} fall clip generation", clip_start)

        columns = [f'feat_{i}' for i in range(139)]
        columns.append('label')
        df = pd.DataFrame(all_sample_features, columns=columns)
        df['start_time'] = [s['timestamps'][0] for s in samples]
        df['end_time'] = [s['timestamps'][-1] for s in samples]
        df['fall_relationship'] = all_relationships

        features_csv = segment_output_dir / f"round_{round_idx}_features.csv"
        csv_start = time.perf_counter()
        df.to_csv(features_csv, index=False)
        _log_timing(f"Round {round_idx} save csv", csv_start, rows=len(df), output=features_csv.name)
        logger.info(f"  特征保存到: {features_csv}")
        _log_timing(f"Round {round_idx} segment total", segment_start, segment=seg_dir.name, samples=len(samples))
        _log_timing(f"Round {round_idx} total", round_start, samples=len(samples))

        return str(features_csv), samples

    _log_timing(f"Round {round_idx} total", round_start, samples=0)
    return "", []


def generate_npz_and_details(video_frame_dir: Path, all_samples: List[Dict], intervals_ms: List[Tuple[float, float]]):
    """生成NPZ文件samples.npz、详细CSV和sample_details.txt"""
    logger = _logger()
    total_start = time.perf_counter()

    results_dir = video_frame_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 50)
    logger.info("生成NPZ和详细文档...")
    logger.info("=" * 50)

    n_samples = len(all_samples)
    n_frames = 10
    feat_dim = 121

    X = np.zeros((n_samples, n_frames, feat_dim))
    y = np.zeros(n_samples, dtype=np.int32)

    all_sample_data = []

    logger.info(f"  计算 {n_samples} 个样本的NPZ特征(并行)...")
    npz_feature_start = time.perf_counter()
    X = compute_sample_features_batch_for_npz(all_samples)
    _log_timing("NPZ feature computation", npz_feature_start, samples=n_samples)

    y = np.array([s.get('label', 0) for s in all_samples], dtype=np.int32)
    all_sample_data = [
        {
            'sample_idx': i,
            'label': s.get('label', 0),
            'start_time': s['timestamps'][0],
            'end_time': s['timestamps'][-1],
            'intervals_ms': s.get('interval_ms', []),
        }
        for i, s in enumerate(all_samples)
    ]

    npz_save_start = time.perf_counter()
    npz_path = results_dir / "samples.npz"
    np.savez(npz_path, X=X, y=y)
    _log_timing("NPZ save", npz_save_start, output=npz_path.name, shape=X.shape)
    logger.info(f"  NPZ已保存: {npz_path}")
    logger.info(f"    X.shape = {X.shape}")
    logger.info(f"    y.shape = {y.shape}")
    logger.info(f"    FALL (label=1) count: {np.sum(y == 1)}")
    logger.info(f"    NOFALL (label=0) count: {np.sum(y == 0)}")

    fall_indices = [i for i, s in enumerate(all_sample_data) if s['label'] == 1]
    nofall_indices = [i for i, s in enumerate(all_sample_data) if s['label'] == 0]

    n_sample = min(25, len(fall_indices), len(nofall_indices))
    selected_fall = random.sample(fall_indices, n_sample) if fall_indices else []
    selected_nofall = random.sample(nofall_indices, n_sample) if nofall_indices else []
    selected_indices = selected_fall + selected_nofall
    random.shuffle(selected_indices)

    logger.info(f"\n  生成50个随机抽样样本的详细计算过程...")
    logger.info(f"    跌倒样本: {len(selected_fall)} 个")
    logger.info(f"    非跌倒样本: {len(selected_nofall)} 个")

    details_start = time.perf_counter()
    details_path = results_dir / "sample_details.txt"
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("跌倒检测训练数据 - 样本详细计算过程\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总样本数: {n_samples}\n")
        f.write(f"跌倒样本数: {len(fall_indices)}\n")
        f.write(f"非跌倒样本数: {len(nofall_indices)}\n")
        f.write(f"抽样数: {len(selected_indices)} (25 FALL + 25 NOFALL)\n\n")

        for idx in selected_indices:
            sample = all_samples[idx]
            details = compute_sample_debug_details(
                sample['detections'],
                sample['timestamps'],
                idx,
                intervals_ms,
            )
            f.write(details)
    _log_timing("Sample details generation", details_start, selected=len(selected_indices), output=details_path.name)

    logger.info(f"  详细计算文档已保存: {details_path}")

    csv_start = time.perf_counter()
    csv_data = []
    for i, sample in enumerate(all_samples):
        start_time = sample['timestamps'][0]
        end_time = sample['timestamps'][-1]

        csv_data.append({
            'sample_idx': i,
            'label': sample.get('label', 0),
            'start_time': start_time,
            'end_time': end_time,
            'label_name': 'FALL' if sample.get('label', 0) == 1 else 'NOFALL',
            'num_frames': len(sample['detections']),
        })

    df_details = pd.DataFrame(csv_data)
    csv_path = results_dir / "samples_with_details.csv"
    df_details.to_csv(csv_path, index=False)
    _log_timing("Sample details CSV save", csv_start, rows=len(df_details), output=csv_path.name)
    logger.info(f"  样本信息CSV已保存: {csv_path}")

    logger.info("\n" + "=" * 50)
    logger.info("NPZ和详细文档生成完成!")
    logger.info("=" * 50)
    _log_timing("Generate NPZ and details total", total_start, samples=n_samples)
