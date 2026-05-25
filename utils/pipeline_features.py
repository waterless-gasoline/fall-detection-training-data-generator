"""Pipeline feature and sample helpers."""
from __future__ import annotations

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .pose_utils import compute_body_orientation, compute_spine_leg_angle

# Existing behavior depends on these module-level globals in main_pipeline.py.
# The orchestrator will inject them after import.
logger = None
_DISABLE_INTERNAL_PARALLELISM = False


def set_pipeline_runtime(logger_obj, disable_internal_parallelism: bool) -> None:
    global logger, _DISABLE_INTERNAL_PARALLELISM
    logger = logger_obj
    _DISABLE_INTERNAL_PARALLELISM = disable_internal_parallelism


def _compute_features_wrapper(args):
    sample_detections, sample_timestamps, interval_ms, intervals_ms = args
    return compute_features_for_sample(sample_detections, sample_timestamps, interval_ms, intervals_ms)


def extract_keypoints_flat(det: dict) -> np.ndarray:
    """从检测结果提取并展平关键点"""
    keypoints = det.get('keypoints', [])
    if isinstance(keypoints, np.ndarray) and keypoints.shape[0] >= 17:
        flat = keypoints[:17, :2].flatten()
        return flat.astype(np.float64)
    return np.zeros(34, dtype=np.float64)


def compute_bounding_box(positions_seq: np.ndarray) -> tuple:
    """
    计算11帧的最小外接矩形（跳过无人帧，即全0坐标帧）
    返回: (left, top, right, bottom, bbox_w, bbox_h)
    """
    valid_frames = []
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        if np.sum(frame_x) > 0 or np.sum(frame_y) > 0:
            valid_frames.append(i)

    if len(valid_frames) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    all_x = positions_seq[valid_frames][:, 0::2]
    all_y = positions_seq[valid_frames][:, 1::2]

    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)

    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    return min_x, min_y, max_x, max_y, bbox_w, bbox_h


def compute_features_for_sample(sample_detections: list, sample_timestamps: list,
                                 interval_ms: list, intervals_ms: list) -> tuple:
    """计算单个样本的139维特征"""
    positions_list = [extract_keypoints_flat(d) for d in sample_detections]
    positions_seq = np.array(positions_list)

    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    positions_norm = positions_seq.copy()
    positions_norm[:, 0::2] = np.clip((positions_norm[:, 0::2] - min_x) / (bbox_w if bbox_w > 0 else 1), 0, 1)
    positions_norm[:, 1::2] = np.clip((positions_norm[:, 1::2] - min_y) / (bbox_h if bbox_h > 0 else 1), 0, 1)

    velocities = []
    for i in range(1, len(positions_seq)):
        dt = sample_timestamps[i] - sample_timestamps[i-1]
        if dt > 0:
            v = (positions_norm[i] - positions_norm[i-1]) / dt
        else:
            v = np.zeros(34)
        velocities.append(v)
    velocities_seq = np.array(velocities) if velocities else np.zeros((0, 34))

    accelerations = []
    for i in range(1, len(velocities_seq)):
        dt_prev = sample_timestamps[i] - sample_timestamps[i-1]
        dt_curr = sample_timestamps[i+1] - sample_timestamps[i]
        dt_avg = (dt_prev + dt_curr) / 2 if (dt_prev + dt_curr) > 0 else 1
        if dt_avg > 0:
            a = (velocities_seq[i] - velocities_seq[i-1]) / dt_avg
        else:
            a = np.zeros(34)
        accelerations.append(a)
    accelerations_seq = np.array(accelerations) if accelerations else np.zeros((0, 34))

    rel_positions = []
    for positions in positions_seq:
        left_hip = positions[22:24]
        right_hip = positions[24:26]
        hip_center = (left_hip + right_hip) / 2
        hip_center_flat = np.tile(hip_center, 17)
        rel = positions - hip_center_flat
        rel_positions.append(rel)
    rel_positions_seq = np.array(rel_positions)

    spine_leg_angles = [compute_spine_leg_angle(p) for p in positions_seq]

    hip_height_changes = []
    for i in range(1, len(positions_seq)):
        left_hip = positions_seq[i, 22:24]
        right_hip = positions_seq[i, 24:26]
        hip_curr = (left_hip + right_hip) / 2
        left_hip_prev = positions_seq[i-1, 22:24]
        right_hip_prev = positions_seq[i-1, 24:26]
        hip_prev = (left_hip_prev + right_hip_prev) / 2
        hip_height_changes.append(hip_curr[1] - hip_prev[1])
    hip_height_changes = np.array(hip_height_changes)

    body_orientations = [compute_body_orientation(p) for p in positions_seq]

    sample_start_s = sample_timestamps[0]
    sample_end_s = sample_timestamps[-1]
    sample_start_ms = sample_start_s * 1000
    sample_end_ms = sample_end_s * 1000

    label = 0
    fall_relationship = ""

    for s_ms, e_ms in intervals_ms:
        if s_ms >= sample_start_ms and e_ms <= sample_end_ms:
            label = 1
            fall_relationship = f"fall_in_sample:{s_ms:.0f}-{e_ms:.0f}"
            break
        elif s_ms <= sample_end_ms and e_ms >= sample_start_ms:
            fall_relationship = f"overlap:{s_ms:.0f}-{e_ms:.0f}"

    features = []
    features.extend(positions_norm[-1].tolist())
    if len(velocities_seq) > 0:
        features.extend(velocities_seq[-1].tolist())
    else:
        features.extend([0] * 34)
    if len(accelerations_seq) > 0:
        features.extend(accelerations_seq[-1].tolist())
    else:
        features.extend([0] * 34)
    features.extend(rel_positions_seq[-1].tolist())
    features.append(spine_leg_angles[-1])
    features.append(hip_height_changes[-1] if len(hip_height_changes) > 0 else 0)
    features.append(body_orientations[-1])
    features.append(label)

    return np.array(features), fall_relationship


def compute_sample_features_for_npz(sample_detections: list, sample_timestamps: list) -> np.ndarray:
    """
    计算单个样本所有11帧的121维特征，用于NPZ输出
    返回: shape (10, 121) - 10帧有效数据(跳过第一帧)
    """
    positions_list = [extract_keypoints_flat(d) for d in sample_detections]
    positions_seq = np.array(positions_list)

    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    positions_norm = positions_seq.copy()
    positions_norm[:, 0::2] = (positions_norm[:, 0::2] - min_x) / (bbox_w if bbox_w > 0 else 1)
    positions_norm[:, 1::2] = (positions_norm[:, 1::2] - min_y) / (bbox_h if bbox_h > 0 else 1)

    rel_positions = []
    for i in range(len(positions_seq)):
        left_hip = positions_seq[i, 22:24]
        right_hip = positions_seq[i, 24:26]
        hip_center = (left_hip + right_hip) / 2
        hip_center_norm = np.array([
            (hip_center[0] - min_x) / (bbox_w if bbox_w > 0 else 1),
            (hip_center[1] - min_y) / (bbox_h if bbox_h > 0 else 1)
        ])
        hip_center_norm_flat = np.tile(hip_center_norm, 17)
        rel = positions_norm[i] - hip_center_norm_flat
        rel_positions.append(rel)
    rel_positions_seq = np.array(rel_positions)

    bbox_widths = []
    bbox_heights = []
    bbox_ratios = []
    bbox_areas = []
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        if np.sum(frame_x) > 0 or np.sum(frame_y) > 0:
            w = (np.max(frame_x) - np.min(frame_x)) / (bbox_w if bbox_w > 0 else 1)
            h = (np.max(frame_y) - np.min(frame_y)) / (bbox_h if bbox_h > 0 else 1)
            ratio = w / h if h > 0 else 0
            area = w * h
        else:
            w, h, ratio, area = 0, 0, 0, 0
        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_ratios.append(ratio)
        bbox_areas.append(area)

    spine_leg_angles = np.array([compute_spine_leg_angle(p) for p in positions_seq])
    hip_height_changes = np.array([
        ((positions_seq[i, 22:24] + positions_seq[i, 24:26]) / 2)[1] -
        ((positions_seq[i-1, 22:24] + positions_seq[i-1, 24:26]) / 2)[1]
        for i in range(1, len(positions_seq))
    ])
    body_orientations = np.array([compute_body_orientation(p) for p in positions_seq])

    sample_features = []
    n_frames = len(positions_seq)

    for frame_idx in range(n_frames):
        features = []
        features.extend(positions_norm[frame_idx].tolist())

        if frame_idx > 0:
            dt = sample_timestamps[frame_idx] - sample_timestamps[frame_idx - 1]
            features.extend([dt] * 10)
        else:
            features.extend([0] * 10)

        features.extend(bbox_widths[1:])
        features.extend(bbox_heights[1:])
        features.extend(bbox_ratios[1:])
        features.extend(bbox_areas[1:])

        features.extend(rel_positions_seq[frame_idx].tolist())

        features.append(spine_leg_angles[frame_idx])

        if frame_idx > 0:
            features.append(hip_height_changes[frame_idx - 1])
        else:
            features.append(0)

        features.append(body_orientations[frame_idx])

        sample_features.append(features)

    return np.array(sample_features)[1:]


def compute_sample_debug_details(sample_detections: list, sample_timestamps: list,
                                 sample_idx: int, intervals_ms: list) -> str:
    """
    生成单个样本的123维详细计算过程
    """
    positions_list = [extract_keypoints_flat(d) for d in sample_detections]
    positions_seq = np.array(positions_list)

    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    positions_norm = positions_seq.copy()
    positions_norm[:, 0::2] = np.clip((positions_norm[:, 0::2] - min_x) / (bbox_w if bbox_w > 0 else 1), 0, 1)
    positions_norm[:, 1::2] = np.clip((positions_norm[:, 1::2] - min_y) / (bbox_h if bbox_h > 0 else 1), 0, 1)

    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Sample #{sample_idx}")
    lines.append(f"{'='*60}")
    lines.append(f"时间范围: {sample_timestamps[0]:.3f}s - {sample_timestamps[-1]:.3f}s")
    lines.append(f"帧间隔: {[f'{t:.0f}ms' for t in [(sample_timestamps[i]-sample_timestamps[i-1])*1000 for i in range(1, len(sample_timestamps))]]}")
    lines.append(f"总帧数: {len(sample_detections)}")

    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    lines.append("")
    lines.append("-" * 60)
    lines.append("外接矩形计算 (Bounding Box)")
    lines.append("-" * 60)
    lines.append(f"  各帧bbox范围:")
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        f_min_x, f_max_x = np.min(frame_x), np.max(frame_x)
        f_min_y, f_max_y = np.min(frame_y), np.max(frame_y)
        lines.append(f"    帧{i}: x=[{f_min_x:.1f}, {f_max_x:.1f}], y=[{f_min_y:.1f}, {f_max_y:.1f}]")
    lines.append(f"  合并后外接矩形: left={min_x:.1f}, top={min_y:.1f}, right={max_x:.1f}, bottom={max_y:.1f}")
    lines.append(f"  外接矩形宽度: bbox_w = {max_x:.1f} - {min_x:.1f} = {bbox_w:.1f}")
    lines.append(f"  外接矩形高度: bbox_h = {max_y:.1f} - {min_y:.1f} = {bbox_h:.1f}")

    sample_start_ms = sample_timestamps[0] * 1000
    sample_end_ms = sample_timestamps[-1] * 1000
    label = 0
    for s_ms, e_ms in intervals_ms:
        if s_ms >= sample_start_ms and e_ms <= sample_end_ms:
            label = 1
            lines.append(f"标签: FALL (跌倒区间 {s_ms:.0f}-{e_ms:.0f}ms 完全在样本范围内)")
            break
        elif s_ms <= sample_end_ms and e_ms >= sample_start_ms:
            lines.append(f"标签: NOFALL (与跌倒区间 {s_ms:.0f}-{e_ms:.0f}ms 有重叠)")

    lines.append("")

    velocities = []
    for i in range(1, len(positions_seq)):
        dt = sample_timestamps[i] - sample_timestamps[i-1]
        if dt > 0:
            v = (positions_norm[i] - positions_norm[i-1]) / dt
        else:
            v = np.zeros(34)
        velocities.append(v)
    velocities_seq = np.array(velocities) if velocities else np.zeros((0, 34))

    accelerations = []
    for i in range(1, len(velocities_seq)):
        dt_prev = sample_timestamps[i] - sample_timestamps[i-1]
        dt_curr = sample_timestamps[i+1] - sample_timestamps[i]
        dt_avg = (dt_prev + dt_curr) / 2 if (dt_prev + dt_curr) > 0 else 1
        if dt_avg > 0:
            a = (velocities_seq[i] - velocities_seq[i-1]) / dt_avg
        else:
            a = np.zeros(34)
        accelerations.append(a)
    accelerations_seq = np.array(accelerations) if accelerations else np.zeros((0, 34))

    if len(velocities_seq) > 0:
        vel_min = np.min(velocities_seq)
        vel_max = np.max(velocities_seq)
        if vel_max != vel_min:
            velocities_normed = (velocities_seq - vel_min) / (vel_max - vel_min)
        else:
            velocities_normed = np.zeros_like(velocities_seq)
    else:
        velocities_normed = np.zeros((1, 34))

    if len(accelerations_seq) > 0:
        acc_min = np.min(accelerations_seq)
        acc_max = np.max(accelerations_seq)
        if acc_max != acc_min:
            accelerations_normed = (accelerations_seq - acc_min) / (acc_max - acc_min)
        else:
            accelerations_normed = np.zeros_like(accelerations_seq)
    else:
        accelerations_normed = np.zeros((1, 34))

    rel_positions = []
    for positions in positions_seq:
        left_hip = positions[22:24]
        right_hip = positions[24:26]
        hip_center = (left_hip + right_hip) / 2
        hip_center_flat = np.tile(hip_center, 17)
        rel = positions - hip_center_flat
        rel[0::2] /= bbox_w if bbox_w > 0 else 1
        rel[1::2] /= bbox_h if bbox_h > 0 else 1
        rel_positions.append(rel)
    rel_positions_seq = np.array(rel_positions)

    spine_leg_angles = [compute_spine_leg_angle(p) for p in positions_seq]
    hip_height_changes = np.array([
        ((positions_seq[i, 22:24] + positions_seq[i, 24:26]) / 2)[1] -
        ((positions_seq[i-1, 22:24] + positions_seq[i-1, 24:26]) / 2)[1]
        for i in range(1, len(positions_seq))
    ])
    body_orientations = [compute_body_orientation(p) for p in positions_seq]

    bbox_widths = []
    bbox_heights = []
    bbox_ratios = []
    bbox_areas = []
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        if np.sum(frame_x) > 0 or np.sum(frame_y) > 0:
            w = (np.max(frame_x) - np.min(frame_x)) / (bbox_w if bbox_w > 0 else 1)
            h = (np.max(frame_y) - np.min(frame_y)) / (bbox_h if bbox_h > 0 else 1)
            ratio = w / h if h > 0 else 0
            area = w * h
        else:
            w, h, ratio, area = 0, 0, 0, 0
        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_ratios.append(ratio)
        bbox_areas.append(area)

    lines.append("-" * 60)
    lines.append("特征0-33 (positions 外接矩形相对坐标归一化): 所有11帧")
    lines.append("-" * 60)
    lines.append("  归一化方式: (x - min_x) / bbox_w, (y - min_y) / bbox_h")
    for frame_idx in range(11):
        lines.append(f"  --- Frame {frame_idx} ---")
        for i in range(17):
            kp = keypoint_names[i]
            x_raw = positions_seq[frame_idx][i*2]
            y_raw = positions_seq[frame_idx][i*2+1]
            x_norm = positions_norm[frame_idx][i*2]
            y_norm = positions_norm[frame_idx][i*2+1]
            lines.append(f"    feat_{i*2}: {kp}_x: raw={x_raw:.2f}, norm={x_norm:.6f}")
            lines.append(f"    feat_{i*2+1}: {kp}_y: raw={y_raw:.2f}, norm={y_norm:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征34-43 (采样间隔): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 时间差 (秒)")
    lines.append("  计算方式: sample_timestamps[i] - sample_timestamps[i-1]")
    for frame_idx in range(11):
        if frame_idx == 0:
            lines.append(f"  feat_34: frame_{frame_idx} = 0 (无前一帧)")
        else:
            dt = sample_timestamps[frame_idx] - sample_timestamps[frame_idx - 1]
            lines.append(f"  feat_34: frame_{frame_idx} = {sample_timestamps[frame_idx]:.6f} - {sample_timestamps[frame_idx-1]:.6f} = {dt:.6f}s")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征44-63 (bbox归一化宽高): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 归一化宽高")
    lines.append("  计算方式: frame_bbox_w / bbox_w, frame_bbox_h / bbox_h")
    for frame_idx in range(10):
        lines.append(f"  feat_{44 + frame_idx}: frame_{frame_idx + 1}_w = {bbox_widths[frame_idx + 1]:.6f}")
    for frame_idx in range(10):
        lines.append(f"  feat_{54 + frame_idx}: frame_{frame_idx + 1}_h = {bbox_heights[frame_idx + 1]:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征64-73 (bbox宽高比): 所有11帧")
    lines.append("-" * 60)
    lines.append("  计算方式: frame_bbox_w / frame_bbox_h，高度为0时取0")
    for frame_idx in range(10):
        lines.append(f"  feat_{64 + frame_idx}: frame_{frame_idx + 1}_ratio = {bbox_ratios[frame_idx + 1]:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征74-83 (bbox面积): 所有11帧")
    lines.append("-" * 60)
    lines.append("  计算方式: frame_bbox_w * frame_bbox_h")
    for frame_idx in range(10):
        lines.append(f"  feat_{74 + frame_idx}: frame_{frame_idx + 1}_area = {bbox_areas[frame_idx + 1]:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征84-117 (relative_positions 外接矩形相对坐标归一化): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 相对坐标 (基于外接矩形)")
    lines.append("  计算方式: (point - hip_center) / bbox_w, (point - hip_center) / bbox_h")
    for frame_idx in range(1, 11):
        lines.append(f"  --- Frame {frame_idx} ---")
        left_hip = positions_seq[frame_idx][22:24]
        right_hip = positions_seq[frame_idx][24:26]
        hip_center = (left_hip + right_hip) / 2
        for i in range(17):
            kp = keypoint_names[i]
            raw_x = positions_seq[frame_idx][i*2]
            raw_y = positions_seq[frame_idx][i*2+1]
            rx = rel_positions_seq[frame_idx][i*2]
            ry = rel_positions_seq[frame_idx][i*2+1]
            lines.append(f"    feat_{84+i*2}: rel_pos_{kp}_x:")
            lines.append(f"      原始坐标: {raw_x:.2f}, hip_center_x: {hip_center[0]:.2f}")
            lines.append(f"      相对坐标: {raw_x:.2f} - {hip_center[0]:.2f} = {raw_x - hip_center[0]:.2f}")
            lines.append(f"      归一化: {raw_x - hip_center[0]:.2f} / {bbox_w:.1f} = {rx:.6f}")
            lines.append(f"    feat_{84+i*2+1}: rel_pos_{kp}_y:")
            lines.append(f"      原始坐标: {raw_y:.2f}, hip_center_y: {hip_center[1]:.2f}")
            lines.append(f"      相对坐标: {raw_y:.2f} - {hip_center[1]:.2f} = {raw_y - hip_center[1]:.2f}")
            lines.append(f"      归一化: {raw_y - hip_center[1]:.2f} / {bbox_h:.1f} = {ry:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征118 (spine_leg_angle): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 角度值 (无坐标类型)")
    for frame_idx in range(1, 11):
        lines.append(f"  feat_118: frame_{frame_idx} = {spine_leg_angles[frame_idx]:.6f} rad")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征119 (hip_height_change 外接矩形归一化): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 高度差值 (y方向)")
    lines.append("  归一化方式: (y差值) / bbox_h")
    for frame_idx in range(1, 11):
        if frame_idx == 1:
            lines.append(f"  feat_119: frame_{frame_idx} = 0 (无前一帧)")
        else:
            h_change = hip_height_changes[frame_idx-1] if frame_idx-1 < len(hip_height_changes) else 0
            lines.append(f"  feat_119: frame_{frame_idx} = ({positions_norm[frame_idx][1]:.4f} - {positions_norm[frame_idx-1][1]:.4f}) = {h_change:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征120 (body_orientation): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 角度值 (无坐标类型)")
    for frame_idx in range(1, 11):
        lines.append(f"  feat_120: frame_{frame_idx} = {body_orientations[frame_idx]:.6f} rad")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def extract_samples(frame_dir: Path, detections: np.ndarray, timestamps: list,
                    intervals_ms: list, sample_frames: int = 11) -> list:
    """提取所有样本 - 每个样本的11帧间隔独立随机从配置中选取"""
    total_frames = len(timestamps)
    if logger:
        logger.info(f"  提取样本: 总帧数={total_frames}")

    interval_choices_ms = [250, 300, 350, 400, 450, 500]

    samples = []

    frame_idx = 0
    while True:
        random_intervals = [random.choice(interval_choices_ms) for _ in range(sample_frames)]

        sample_timestamps = []
        start_time = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
        sample_timestamps.append(start_time)

        current_time = start_time
        for i in range(1, sample_frames):
            current_time += random_intervals[i] / 1000.0
            sample_timestamps.append(current_time)

        if sample_timestamps[-1] > timestamps[total_frames - 1]:
            break

        sample_detections = []
        for seq_idx in range(frame_idx, frame_idx + sample_frames):
            if seq_idx < len(detections):
                sample_detections.append(detections[seq_idx])
            else:
                sample_detections.append({'keypoints': np.zeros((17, 3)), 'has_person': False})

        has_no_person = False
        for det in sample_detections:
            keypoints = det.get('keypoints', np.zeros((17, 3)))
            if isinstance(keypoints, np.ndarray):
                kp_x = keypoints[:17, 0]
                if np.all(kp_x == 0):
                    has_no_person = True
                    break

        if has_no_person:
            frame_idx += 1
            continue

        sample = {
            'interval_ms': random_intervals,
            'start_frame': frame_idx,
            'timestamps': sample_timestamps,
            'detections': sample_detections
        }
        samples.append(sample)

        frame_idx += 1

    return samples


def _extract_single_sample(args) -> dict | None:
    """从单个起始帧提取单个样本（用于并行）"""
    frame_idx, detections, timestamps, sample_frames, interval_choices_ms = args

    total_frames = len(timestamps)
    debug_info = {
        'frame_idx': frame_idx,
        'reason': None,
        'sample_end': None,
        'last_timestamp': timestamps[total_frames - 1] if total_frames else None,
        'no_person_offset': None,
    }

    random_intervals = [random.choice(interval_choices_ms) for _ in range(sample_frames)]

    sample_timestamps = []
    start_time = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
    sample_timestamps.append(start_time)

    current_time = start_time
    for i in range(1, sample_frames):
        current_time += random_intervals[i] / 1000.0
        sample_timestamps.append(current_time)

    debug_info['sample_end'] = sample_timestamps[-1]

    if sample_timestamps[-1] > timestamps[total_frames - 1]:
        debug_info['reason'] = 'sample_end_exceeds_last_timestamp'
        return {'_debug_skip': debug_info}

    sample_detections = []
    for seq_idx in range(frame_idx, frame_idx + sample_frames):
        if seq_idx < len(detections):
            sample_detections.append(detections[seq_idx])
        else:
            sample_detections.append({'keypoints': np.zeros((17, 3)), 'has_person': False})

    has_no_person = False
    for offset, det in enumerate(sample_detections):
        keypoints = det.get('keypoints', np.zeros((17, 3)))
        if isinstance(keypoints, np.ndarray):
            kp_x = keypoints[:17, 0]
            if np.all(kp_x == 0):
                has_no_person = True
                debug_info['no_person_offset'] = offset
                break

    if has_no_person:
        debug_info['reason'] = 'contains_no_person_frame'
        return {'_debug_skip': debug_info}

    return {
        'interval_ms': random_intervals,
        'start_frame': frame_idx,
        'timestamps': sample_timestamps,
        'detections': sample_detections
    }


def extract_samples_parallel(frame_dir: Path, detections: np.ndarray, timestamps: list,
                            intervals_ms: list, sample_frames: int = 11,
                            num_workers: int = None) -> list:
    """使用多进程并行提取所有样本"""
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    total_frames = len(timestamps)
    if logger:
        logger.info(f"  [并行] 提取样本: 总帧数={total_frames}, 工作进程数={num_workers}")

    interval_choices_ms = [250, 300, 350, 400, 450, 500]

    frame_args = []
    for frame_idx in range(total_frames):
        frame_args.append((
            frame_idx, detections, timestamps, sample_frames, interval_choices_ms
        ))

    if _DISABLE_INTERNAL_PARALLELISM:
        samples = []
        skip_counts = {'sample_end_exceeds_last_timestamp': 0, 'contains_no_person_frame': 0, 'other': 0}
        skip_examples = []
        for arg in frame_args:
            result = _extract_single_sample(arg)
            if result is None:
                skip_counts['other'] += 1
                continue
            if '_debug_skip' in result:
                debug_skip = result['_debug_skip']
                reason = debug_skip.get('reason') or 'other'
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                if len(skip_examples) < 5:
                    skip_examples.append(debug_skip)
                continue
            samples.append(result)
        samples.sort(key=lambda x: x['start_frame'])
        if logger:
            logger.info(f"  [单进程] 提取样本数: {len(samples)}")
            if not samples:
                logger.warning(
                    f"  [DEBUG][sample-extract][single] zero samples. total_frames={total_frames}, "
                    f"skip_counts={skip_counts}, examples={skip_examples}"
                )
        return samples

    samples = []
    skip_counts = {'sample_end_exceeds_last_timestamp': 0, 'contains_no_person_frame': 0, 'other': 0}
    skip_examples = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_extract_single_sample, arg): arg[0] for arg in frame_args}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    skip_counts['other'] += 1
                    continue
                if '_debug_skip' in result:
                    debug_skip = result['_debug_skip']
                    reason = debug_skip.get('reason') or 'other'
                    skip_counts[reason] = skip_counts.get(reason, 0) + 1
                    if len(skip_examples) < 5:
                        skip_examples.append(debug_skip)
                    continue
                samples.append(result)
            except Exception as e:
                if logger:
                    frame_idx = futures[future]
                    logger.warning(f"  帧{frame_idx}处理异常: {e}")

    samples.sort(key=lambda x: x['start_frame'])

    if logger:
        logger.info(f"  [并行] 提取样本数: {len(samples)}")
        if not samples:
            logger.warning(
                f"  [DEBUG][sample-extract][parallel] zero samples. total_frames={total_frames}, "
                f"skip_counts={skip_counts}, examples={skip_examples}"
            )
    return samples


def compute_features_batch(samples: list, intervals_ms: list, max_workers: int = None):
    if _DISABLE_INTERNAL_PARALLELISM:
        all_sample_features = []
        all_relationships = []
        for s in samples:
            feat, fall_rel = compute_features_for_sample(s['detections'], s['timestamps'], s['interval_ms'], intervals_ms)
            all_sample_features.append(feat)
            all_relationships.append(fall_rel)
        return all_sample_features, all_relationships

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    all_sample_features = []
    all_relationships = []

    args_list = [
        (s['detections'], s['timestamps'], s['interval_ms'], intervals_ms)
        for s in samples
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_compute_features_wrapper, args) for args in args_list]
        for future in as_completed(futures):
            feat, fall_rel = future.result()
            all_sample_features.append(feat)
            all_relationships.append(fall_rel)

    return all_sample_features, all_relationships


def _compute_sample_features_npz_wrapper(args):
    """包装compute_sample_features_for_npz用于并行"""
    sample_detections, sample_timestamps = args
    return compute_sample_features_for_npz(sample_detections, sample_timestamps)


def compute_sample_features_batch_for_npz(samples: list, max_workers: int = None) -> np.ndarray:
    """
    使用进程池并行计算多个样本的NPZ特征
    返回: shape (n_samples, 10, 139)
    """
    if _DISABLE_INTERNAL_PARALLELISM:
        results = []
        for s in samples:
            result = compute_sample_features_for_npz(s['detections'], s['timestamps'])
            results.append(result)
        return np.array(results)

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    args_list = [(s['detections'], s['timestamps']) for s in samples]

    results = [None] * len(samples)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_sample_features_npz_wrapper, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return np.array(results)
