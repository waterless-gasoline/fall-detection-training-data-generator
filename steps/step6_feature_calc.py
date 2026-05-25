"""Step 6: 特征计算模块"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import config
from utils import PoseUtils, AnnotationParser

logger = logging.getLogger(__name__)


def normalize_sample_feature(values: np.ndarray, name: str = "") -> Tuple[np.ndarray, Dict]:
    """
    对单个样本的特征值进行样本级归一化
    公式: (current_value - sample_min) / (sample_max - sample_min)
    如果 max == min, 则该特征归一化结果设为0

    Args:
        values: 特征值数组 (N, 34) 所有帧
        name: 特征名称，用于日志

    Returns:
        normalized: 归一化后的特征值数组 (N, 34)
        stats: {'min': min_v, 'max': max_v}
    """
    # values shape: (N, 34)
    min_v = np.min(values)
    max_v = np.max(values)

    stats = {'min': min_v, 'max': max_v, 'name': name}

    if max_v == min_v:
        # max == min 时，所有值归一化为0
        normalized = np.zeros_like(values)
    else:
        normalized = (values - min_v) / (max_v - min_v)

    return normalized, stats


class FeatureCalculator:
    """特征计算器 - 139维特征"""

    def __init__(self, detection_path: Path, annotation_path: Path):
        self.detection_path = detection_path
        self.annotation_path = annotation_path

    def load_detections(self) -> List[dict]:
        """加载检测结果"""
        with open(self.detection_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_annotations(self) -> List[Tuple[float, float]]:
        """加载跌倒区间(毫秒)"""
        parser = AnnotationParser()
        return parser.parse(self.annotation_path)

    def flatten_keypoints(self, keypoints: List[dict]) -> np.ndarray:
        """
        将关键点展平为34维向量 (17点 * 2坐标)
        keypoints: [{'x', 'y', 'name'}, ...]
        """
        positions = []
        for kp in keypoints:
            positions.append(kp['x'])
            positions.append(kp['y'])
        return np.array(positions)

    def normalize_positions(self, positions: np.ndarray, img_width: int = 1920, img_height: int = 1080) -> np.ndarray:
        """归一化关键点坐标到[0,1]"""
        normalized = positions.copy()
        normalized[0::2] /= img_width   # x坐标
        normalized[1::2] /= img_height  # y坐标
        return normalized

    def compute_velocities(self, positions_seq: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        计算速度 (34维)
        positions_seq: (N, 34) 各帧的positions
        dt: 时间间隔(秒)
        """
        velocities = []
        for i in range(1, len(positions_seq)):
            v = (positions_seq[i] - positions_seq[i-1]) / dt
            velocities.append(v)
        return np.array(velocities)

    def compute_accelerations(self, velocities: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        计算加速度 (34维)
        velocities: (N-1, 34)
        """
        accelerations = []
        for i in range(1, len(velocities)):
            a = (velocities[i] - velocities[i-1]) / dt
            accelerations.append(a)
        return np.array(accelerations)

    def compute_relative_positions(self, positions_seq: np.ndarray) -> np.ndarray:
        """
        计算相对位置(减去髋中心)
        positions_seq: (N, 34)
        """
        rel_pos = []
        for positions in positions_seq:
            # 计算髋中心
            left_hip = positions[22:24]
            right_hip = positions[24:26]
            hip_center = (left_hip + right_hip) / 2
            # 减去髋中心
            hip_center_flat = np.tile(hip_center, 17)  # (34,)
            rel = positions - hip_center_flat
            rel_pos.append(rel)
        return np.array(rel_pos)

    def compute_spine_leg_angles(self, positions_seq: np.ndarray) -> np.ndarray:
        """
        计算躯干与腿夹角 (N帧)
        """
        angles = []
        for positions in positions_seq:
            angle = PoseUtils.compute_spine_leg_angle(positions)
            angles.append(angle)
        return np.array(angles)

    def compute_hip_height_changes(self, positions_seq: np.ndarray) -> np.ndarray:
        """
        计算髋部高度变化 - 髋中心的y坐标差分 (归一化)
        positions_seq: (N, 34)
        """
        changes = []
        for i in range(1, len(positions_seq)):
            # 髋中心: 左髋(索引22,23) + 右髋(索引24,25) / 2
            left_hip = positions_seq[i, 22:24]
            right_hip = positions_seq[i, 24:26]
            hip_curr = (left_hip + right_hip) / 2
            left_hip_prev = positions_seq[i-1, 22:24]
            right_hip_prev = positions_seq[i-1, 24:26]
            hip_prev = (left_hip_prev + right_hip_prev) / 2
            changes.append(hip_curr[1] - hip_prev[1])
        return np.array(changes)

    def compute_body_orientations(self, positions_seq: np.ndarray) -> np.ndarray:
        """
        计算身体朝向 (躯干与竖直方向夹角)
        """
        orientations = []
        for positions in positions_seq:
            ori = PoseUtils.compute_body_orientation(positions)
            orientations.append(ori)
        return np.array(orientations)

    def is_fall_sample(self, start_frame: int, end_frame: int, intervals: List[Tuple[float, float]], fps: float) -> int:
        """
        判断样本是否为跌倒样本
        条件: 跌倒区间的所有帧都落在样本范围内
        start_frame, end_frame: 样本帧范围
        intervals: [(start_ms, end_ms), ...]
        """
        sample_start_ms = start_frame / fps * 1000
        sample_end_ms = end_frame / fps * 1000

        for start_ms, end_ms in intervals:
            # 跌倒区间完全在样本范围内
            if start_ms >= sample_start_ms and end_ms <= sample_end_ms:
                return 1
        return 0

    def compute_sample_features(self, sample: Dict, fps: float = 10) -> np.ndarray:
        """
        计算单个样本的139维特征
        sample: {
            'indices': [帧索引列表],
            'detections': [检测结果列表],
            'interval_ms': 帧间隔,
            'start_frame': 起始帧
        }
        """
        detections = sample['detections']
        indices = sample['indices']

        # 提取并归一化positions (N, 34)
        positions_list = []
        for det in detections:
            keypoints = det.get('keypoints', [])
            if len(keypoints) >= 17:
                flat = self.flatten_keypoints(keypoints[:17])
            else:
                flat = np.zeros(34)
            positions_list.append(flat)

        positions_seq = np.array(positions_list)
        positions_normalized = self.normalize_positions(positions_seq)

        # 计算velocities (N-1, 34)
        dt = sample['interval_ms'] / 1000.0
        velocities_seq = self.compute_velocities(positions_seq, dt)

        # 计算accelerations (N-2, 34)
        accelerations_seq = self.compute_accelerations(velocities_seq, dt)

        # 计算relative_positions (N, 34)
        relative_positions_seq = self.compute_relative_positions(positions_seq)

        # 计算spine_leg_angles (N,)
        spine_leg_angles = self.compute_spine_leg_angles(positions_seq)

        # 计算hip_height_changes (N-1,)
        hip_height_changes = self.compute_hip_height_changes(positions_seq)

        # 计算body_orientations (N,)
        body_orientations = self.compute_body_orientations(positions_seq)

        # 计算跌倒区间
        intervals = self.load_annotations()

        # 计算标签
        label = self.is_fall_sample(
            indices[0], indices[-1], intervals, fps
        )

        # 对velocities(34-67)和accelerations(68-101)进行样本级归一化
        # 使用所有帧的值计算min/max，然后对每帧归一化
        if len(velocities_seq) > 0:
            velocities_normalized, vel_stats = normalize_sample_feature(velocities_seq, "velocities")
        else:
            velocities_normalized = np.zeros((1, 34))
            vel_stats = {'min': 0, 'max': 0, 'name': 'velocities'}

        if len(accelerations_seq) > 0:
            accelerations_normalized, acc_stats = normalize_sample_feature(accelerations_seq, "accelerations")
        else:
            accelerations_normalized = np.zeros((1, 34))
            acc_stats = {'min': 0, 'max': 0, 'name': 'accelerations'}

        logger.info(f"[Step6] Sample {sample.get('start_frame', '?')}: "
                    f"vel_min={vel_stats['min']:.4f}, vel_max={vel_stats['max']:.4f}, "
                    f"acc_min={acc_stats['min']:.4f}, acc_max={acc_stats['max']:.4f}, "
                    f"vel_normalized shape={velocities_normalized.shape}, "
                    f"acc_normalized shape={accelerations_normalized.shape}")

        # 拼接139维特征
        features = []

        # 0-33: positions (最后一帧)
        features.extend(positions_normalized[-1].tolist())

        # 34-67: velocities (用0替代)
        features.extend([0] * 34)

        # 68-101: accelerations (用0替代)
        features.extend([0] * 34)

        # 102-135: relative_positions (最后一帧)
        features.extend(relative_positions_seq[-1].tolist())

        # 136: spine_leg_angle (最后一帧)
        features.append(spine_leg_angles[-1] if len(spine_leg_angles) > 0 else 0)

        # 137: height_change (最后一帧)
        features.append(hip_height_changes[-1] if len(hip_height_changes) > 0 else 0)

        # 138: body_orientation (最后一帧)
        features.append(body_orientations[-1] if len(body_orientations) > 0 else 0)

        # 139: label
        features.append(label)

        return np.array(features)

    def compute_sample_features_debug(self, sample: Dict, fps: float = 10) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        计算单个样本的139维特征，并返回详细计算过程
        返回: (features, dim_details, calculation_steps)
        - features: 139维特征向量
        - dim_details: 每个维度的计算说明
        - calculation_steps: 每一步的详细计算过程
        """
        detections = sample['detections']
        indices = sample['indices']
        interval_ms = sample.get('interval_ms', 300)

        img_width, img_height = 1920, 1080

        # 提取并归一化positions (N, 34)
        positions_list = []
        for det in detections:
            keypoints = det.get('keypoints', [])
            if len(keypoints) >= 17:
                flat = self.flatten_keypoints(keypoints[:17])
            else:
                flat = np.zeros(34)
            positions_list.append(flat)

        positions_seq = np.array(positions_list)
        positions_normalized = self.normalize_positions(positions_seq)

        # 计算velocities (N-1, 34)
        dt = interval_ms / 1000.0
        velocities_seq = self.compute_velocities(positions_seq, dt)

        # 计算accelerations (N-2, 34)
        accelerations_seq = self.compute_accelerations(velocities_seq, dt)

        # 计算relative_positions (N, 34)
        relative_positions_seq = self.compute_relative_positions(positions_seq)

        # 计算spine_leg_angles (N,)
        spine_leg_angles = self.compute_spine_leg_angles(positions_seq)

        # 计算hip_height_changes (N-1,)
        hip_height_changes = self.compute_hip_height_changes(positions_seq)

        # 计算body_orientations (N,)
        body_orientations = self.compute_body_orientations(positions_seq)

        # 加载跌倒区间
        intervals = self.load_annotations()

        # 计算标签
        label = self.is_fall_sample(
            indices[0], indices[-1], intervals, fps
        )

        # 记录计算过程
        calculation_steps = []
        dim_details = []

        # 对velocities和accelerations进行归一化
        if len(velocities_seq) > 0:
            vel_min = np.min(velocities_seq)
            vel_max = np.max(velocities_seq)
            if vel_max != vel_min:
                velocities_normed = (velocities_seq - vel_min) / (vel_max - vel_min)
            else:
                velocities_normed = np.zeros_like(velocities_seq)
        else:
            vel_min = vel_max = 0
            velocities_normed = np.zeros((1, 34))

        if len(accelerations_seq) > 0:
            acc_min = np.min(accelerations_seq)
            acc_max = np.max(accelerations_seq)
            if acc_max != acc_min:
                accelerations_normed = (accelerations_seq - acc_min) / (acc_max - acc_min)
            else:
                accelerations_normed = np.zeros_like(accelerations_seq)
        else:
            acc_min = acc_max = 0
            accelerations_normed = np.zeros((1, 34))

        # 归一化统计
        calculation_steps.append(f"[Normalization] velocities: min={vel_min:.6f}, max={vel_max:.6f}")
        calculation_steps.append(f"[Normalization] accelerations: min={acc_min:.6f}, max={acc_max:.6f}")

        # 0-33: positions (最后一帧)
        last_pos = positions_normalized[-1]
        last_pos_raw = positions_seq[-1]
        for i in range(17):
            x_raw = last_pos_raw[i*2]
            y_raw = last_pos_raw[i*2+1]
            x_norm = last_pos[i*2]
            y_norm = last_pos[i*2+1]
            kp_name = config.KEYPOINT_NAMES[i]
            dim_details.append(f"feat_{i}: {kp_name}_x normalized = {x_raw}/{img_width} = {x_norm:.6f}")
            calculation_steps.append(f"  Position {kp_name}: raw=({x_raw:.2f},{y_raw:.2f}) -> normalized=({x_norm:.6f},{y_norm:.6f})")
            dim_details.append(f"feat_{i+1}: {kp_name}_y normalized = {y_raw}/{img_height} = {last_pos[i*2+1]:.6f}")
            calculation_steps.append(f"  Position {kp_name}: raw=({x_raw:.2f},{y_raw:.2f}) -> normalized=({x_norm:.6f},{last_pos[i*2+1]:.6f})")

        # 34-67: velocities (最后一帧，归一化后)
        # 归一化公式: (v - vel_min) / (vel_max - vel_min)
        if len(velocities_normed) > 0:
            last_vel = velocities_normed[-1]
            for i in range(17):
                vx_norm = last_vel[i*2]
                vy_norm = last_vel[i*2+1]
                kp_name = config.KEYPOINT_NAMES[i]
                dim_details.append(f"feat_{34+i*2}: velocity_{kp_name}_x normalized = {vx_norm:.6f}")
                calculation_steps.append(f"  Velocity {kp_name}_x: normalized = {vx_norm:.6f}")
                dim_details.append(f"feat_{34+i*2+1}: velocity_{kp_name}_y normalized = {vy_norm:.6f}")
                calculation_steps.append(f"  Velocity {kp_name}_y: normalized = {vy_norm:.6f}")
        else:
            for i in range(17):
                dim_details.append(f"feat_{34+i*2}: velocity_{config.KEYPOINT_NAMES[i]}_x = 0 (no data)")
                calculation_steps.append(f"  Velocity {config.KEYPOINT_NAMES[i]}_x: no data")
                dim_details.append(f"feat_{34+i*2+1}: velocity_{config.KEYPOINT_NAMES[i]}_y = 0 (no data)")
                calculation_steps.append(f"  Velocity {config.KEYPOINT_NAMES[i]}_y: no data")

        # 68-101: accelerations (最后一帧，归一化后)
        # 归一化公式: (a - acc_min) / (acc_max - acc_min)
        if len(accelerations_normed) > 0:
            last_acc = accelerations_normed[-1]
            for i in range(17):
                ax_norm = last_acc[i*2]
                ay_norm = last_acc[i*2+1]
                kp_name = config.KEYPOINT_NAMES[i]
                dim_details.append(f"feat_{68+i*2}: accel_{kp_name}_x normalized = {ax_norm:.6f}")
                calculation_steps.append(f"  Acceleration {kp_name}_x: normalized = {ax_norm:.6f}")
                dim_details.append(f"feat_{68+i*2+1}: accel_{kp_name}_y normalized = {ay_norm:.6f}")
                calculation_steps.append(f"  Acceleration {kp_name}_y: normalized = {ay_norm:.6f}")
        else:
            for i in range(17):
                dim_details.append(f"feat_{68+i*2}: accel_{config.KEYPOINT_NAMES[i]}_x = 0 (no data)")
                calculation_steps.append(f"  Acceleration {config.KEYPOINT_NAMES[i]}_x: no data")
                dim_details.append(f"feat_{68+i*2+1}: accel_{config.KEYPOINT_NAMES[i]}_y = 0 (no data)")
                calculation_steps.append(f"  Acceleration {config.KEYPOINT_NAMES[i]}_y: no data")

        # 102-135: relative_positions (最后一帧)
        last_rel = relative_positions_seq[-1]
        left_hip = positions_seq[-1][22:24]
        right_hip = positions_seq[-1][24:26]
        hip_center = (left_hip + right_hip) / 2
        for i in range(17):
            x = last_rel[i*2]
            y = last_rel[i*2+1]
            kp_name = config.KEYPOINT_NAMES[i]
            dim_details.append(f"feat_{102+i*2}: rel_pos_{kp_name}_x = {positions_seq[-1][i*2]:.2f} - {hip_center[0]:.2f} = {x:.6f}")
            calculation_steps.append(f"  Relative Pos {kp_name}: ({positions_seq[-1][i*2]:.2f} - hip_center_x{hip_center[0]:.2f}) = {x:.6f}")
            dim_details.append(f"feat_{102+i*2+1}: rel_pos_{kp_name}_y = {positions_seq[-1][i*2+1]:.2f} - {hip_center[1]:.2f} = {y:.6f}")
            calculation_steps.append(f"  Relative Pos {kp_name}: ({positions_seq[-1][i*2+1]:.2f} - hip_center_y{hip_center[1]:.2f}) = {y:.6f}")

        # 136: spine_leg_angle
        sla = spine_leg_angles[-1] if len(spine_leg_angles) > 0 else 0
        dim_details.append(f"feat_136: spine_leg_angle = {sla:.6f} rad")
        calculation_steps.append(f"  Spine-Leg Angle: angle_between(spine_vector, leg_vector) = {sla:.6f} rad")

        # 137: height_change
        hc = hip_height_changes[-1] if len(hip_height_changes) > 0 else 0
        dim_details.append(f"feat_137: hip_height_change = hip_y[-1] - hip_y[-2] = {hc:.6f}")
        calculation_steps.append(f"  Hip Height Change: hip_y[last] - hip_y[prev] = {hc:.6f}")

        # 138: body_orientation
        bo = body_orientations[-1] if len(body_orientations) > 0 else 0
        dim_details.append(f"feat_138: body_orientation = {bo:.6f} rad")
        calculation_steps.append(f"  Body Orientation: angle_between(spine_vector, vertical) = {bo:.6f} rad")

        # 拼接139维特征
        features = []
        features.extend(positions_normalized[-1].tolist())
        if len(velocities_normed) > 0:
            features.extend(velocities_normed[-1].tolist())
        else:
            features.extend([0] * 34)
        if len(accelerations_normed) > 0:
            features.extend(accelerations_normed[-1].tolist())
        else:
            features.extend([0] * 34)
        features.extend(relative_positions_seq[-1].tolist())
        features.append(spine_leg_angles[-1] if len(spine_leg_angles) > 0 else 0)
        features.append(hip_height_changes[-1] if len(hip_height_changes) > 0 else 0)
        features.append(body_orientations[-1] if len(body_orientations) > 0 else 0)
        features.append(label)

        return np.array(features), dim_details, calculation_steps

    def compute_all_features(self, samples: List[Dict], fps: float = 10) -> pd.DataFrame:
        """计算所有样本的特征"""
        logger.info(f"[Step6] 计算 {len(samples)} 个样本的特征...")

        all_features = []
        for i, sample in enumerate(samples):
            feat = self.compute_sample_features(sample, fps)
            all_features.append(feat)

            if (i + 1) % 100 == 0:
                logger.info(f"  已处理 {i+1}/{len(samples)}")

        # 创建DataFrame
        columns = [f'feat_{i}' for i in range(139)]
        columns.append('label')
        columns.append('start_time')
        columns.append('end_time')

        # 添加 start_time 和 end_time
        all_times = []
        for sample in samples:
            all_times.append([sample.get('start_time', 0), sample.get('end_time', 0)])

        df = pd.DataFrame(all_features, columns=columns[:-3])
        df['label'] = [f[-3] for f in all_features]
        df['start_time'] = [t[0] for t in all_times]
        df['end_time'] = [t[1] for t in all_times]

        return df

    def run(self, samples: List[Dict], output_path: Path, fps: float = 10) -> pd.DataFrame:
        """执行特征计算"""
        logger.info(f"[Step6] 开始特征计算...")

        df = self.compute_all_features(samples, fps)

        # 保存CSV
        df.to_csv(output_path, index=False)
        logger.info(f"[Step6] 特征保存到: {output_path}")

        return df


def run(samples: List[Dict] = None, output_path: Path = None) -> pd.DataFrame:
    """执行特征计算"""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "samples.csv"

    detection_path = Path(__file__).parent.parent / f"{config.VIDEO_NAME}_detection.json"

    # 找到annotation
    dirs = sorted(config.FRAME_OUTPUT_DIR.glob(f"{config.VIDEO_NAME}_*"))
    if dirs:
        annotation_path = dirs[0] / "annotation.txt"
    else:
        annotation_path = config.ANNOTATION_DIR / config.VIDEO_NAME / "annotation.txt"

    if not annotation_path.exists():
        annotation_path = config.ANNOTATION_DIR / config.VIDEO_NAME / "annotation.txt"

    calculator = FeatureCalculator(detection_path, annotation_path)

    if samples is None:
        from steps import step5_sample_extract
        samples = step5_sample_extract.run()

    return calculator.run(samples, output_path)


if __name__ == '__main__':
    df = run()
    logger.info(f"特征矩阵形状: {df.shape}")