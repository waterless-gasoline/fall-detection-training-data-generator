"""关键点处理工具"""
import numpy as np
from typing import List, Tuple


class PoseUtils:
    """关键点相关计算工具"""

    # 关键点索引 (COCO顺序)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    @staticmethod
    def get_hip_center(positions: np.ndarray) -> np.ndarray:
        """
        计算髋中心
        positions: shape (17, 2) 或 (34,) flatten 或 (N, 17, 2)
        返回:髋中心坐标
        """
        if positions.shape[-1] == 2:
            if len(positions.shape) == 2:
                # (17, 2)
                return (positions[11] + positions[12]) / 2
            elif len(positions.shape) == 3:
                # (N, 17, 2)
                return (positions[:, 11] + positions[:, 12]) / 2
        elif len(positions.shape) == 1:
            # (34,) flatten
            left_hip = positions[22:24]  # 11*2, 11*2+1
            right_hip = positions[24:26]  # 12*2
            return (left_hip + right_hip) / 2
        raise ValueError(f"Unsupported positions shape: {positions.shape}")

    @staticmethod
    def get_shoulder_center(positions: np.ndarray) -> np.ndarray:
        """计算肩中心"""
        if len(positions.shape) == 2 and positions.shape[0] == 17:
            return (positions[5] + positions[6]) / 2
        elif len(positions.shape) == 1 and len(positions) == 34:
            left_shoulder = positions[10:12]
            right_shoulder = positions[12:14]
            return (left_shoulder + right_shoulder) / 2
        raise ValueError(f"Unsupported positions shape: {positions.shape}")

    @staticmethod
    def compute_spine_vector(positions: np.ndarray) -> np.ndarray:
        """计算躯干向量 (肩中心 - 髋中心)"""
        shoulder_center = PoseUtils.get_shoulder_center(positions)
        hip_center = PoseUtils.get_hip_center(positions)
        return shoulder_center - hip_center

    @staticmethod
    def compute_leg_vector(positions: np.ndarray) -> np.ndarray:
        """计算腿向量 (左膝-左踝 + 右膝-右踝)"""
        if len(positions.shape) == 2 and positions.shape[0] == 17:
            left_leg = positions[13] - positions[15]  # left_knee - left_ankle
            right_leg = positions[14] - positions[16]  # right_knee - right_ankle
            return (left_leg + right_leg) / 2
        elif len(positions.shape) == 1 and len(positions) == 34:
            # flatten: 13*2=26, 15*2=30, 14*2=28, 16*2=32
            left_leg = positions[26:28] - positions[30:32]
            right_leg = positions[28:30] - positions[32:34]
            return (left_leg + right_leg) / 2
        raise ValueError(f"Unsupported positions shape: {positions.shape}")

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量之间的夹角(弧度)"""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    @staticmethod
    def compute_spine_leg_angle(positions: np.ndarray) -> float:
        """计算躯干与腿的夹角"""
        spine = PoseUtils.compute_spine_vector(positions)
        leg = PoseUtils.compute_leg_vector(positions)
        return PoseUtils.angle_between_vectors(spine, leg)

    @staticmethod
    def compute_body_orientation(positions: np.ndarray) -> float:
        """计算身体朝向(躯干与竖直方向夹角)"""
        spine = PoseUtils.compute_spine_vector(positions)
        # 竖直向上向量 (0, -1) 因为y轴向下
        vertical = np.array([0, -1])
        return PoseUtils.angle_between_vectors(spine, vertical)

    @staticmethod
    def compute_relative_positions(positions: np.ndarray) -> np.ndarray:
        """计算相对位置(减去髋中心)"""
        if len(positions.shape) == 1 and len(positions) == 34:
            hip_center = PoseUtils.get_hip_center(positions)  # (2,)
            # Create a (34,) array with hip_center repeated for each point (x,y for each of 17 points)
            hip_center_flat = np.tile(hip_center, 17)  # (34,)
            return positions - hip_center_flat
        raise ValueError(f"Unsupported positions shape: {positions.shape}")


def compute_spine_leg_angle(positions: np.ndarray) -> float:
    return PoseUtils.compute_spine_leg_angle(positions)


def compute_body_orientation(positions: np.ndarray) -> float:
    return PoseUtils.compute_body_orientation(positions)
