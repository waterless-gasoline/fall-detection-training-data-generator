"""验证速度和加速度计算结果 - 可视化11帧的运动学数据"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
DETECTION_PATH = Path(__file__).parent / "2026-03-03_10-18-25_detection.json"
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 采样间隔配置（毫秒）
INTERVAL_MS = 300
FPS = 20
SAMPLE_FRAMES = 11


def load_detections():
    """加载检测结果"""
    with open(DETECTION_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_keypoints(keypoints):
    """将关键点展平为34维向量 (17点 * 2坐标)"""
    positions = []
    for kp in keypoints:
        positions.append(kp['x'])
        positions.append(kp['y'])
    return np.array(positions)


def compute_velocities(positions_seq, dt):
    """计算速度 (N-1, 34)"""
    velocities = []
    for i in range(1, len(positions_seq)):
        v = (positions_seq[i] - positions_seq[i-1]) / dt
        velocities.append(v)
    return np.array(velocities)


def compute_accelerations(velocities, dt):
    """计算加速度 (N-2, 34)"""
    accelerations = []
    for i in range(1, len(velocities)):
        a = (velocities[i] - velocities[i-1]) / dt
        accelerations.append(a)
    return np.array(accelerations)


def get_frame_indices(start_frame, interval_ms):
    """根据起始帧和间隔计算11帧的帧索引"""
    indices = []
    current = start_frame
    for i in range(SAMPLE_FRAMES):
        indices.append(current)
        current += int(interval_ms / 1000 * FPS)
    return indices


def visualize_sample(detections, start_frame=0, interval_ms=300, keypoint_indices=None):
    """
    可视化单个样本的速度和加速度

    参数:
        detections: 检测结果列表
        start_frame: 起始帧
        interval_ms: 帧间隔（毫秒）
        keypoint_indices: 要可视化的关键点索引列表，None则显示所有17点
    """
    dt = interval_ms / 1000.0

    # 获取帧索引
    indices = get_frame_indices(start_frame, interval_ms)
    print(f"Frame indices: {indices}")

    # 提取positions (N, 34)
    positions_list = []
    for idx in indices:
        if idx < len(detections):
            keypoints = detections[idx].get('keypoints', [])
            if len(keypoints) >= 17:
                flat = flatten_keypoints(keypoints[:17])
            else:
                flat = np.zeros(34)
            positions_list.append(flat)
        else:
            positions_list.append(np.zeros(34))

    positions_seq = np.array(positions_list)

    # 计算速度和加速度
    velocities_seq = compute_velocities(positions_seq, dt)
    accelerations_seq = compute_accelerations(velocities_seq, dt)

    print(f"\nData shape:")
    print(f"  positions: {positions_seq.shape}")
    print(f"  velocities: {velocities_seq.shape}")
    print(f"  accelerations: {accelerations_seq.shape}")

    # 时间轴（用于绘图）
    # positions: 11个时间点, velocities: 10个时间点(帧间), accelerations: 9个时间点
    t_pos = np.arange(len(positions_seq)) * dt
    t_vel = (np.arange(len(velocities_seq)) + 0.5) * dt  # 取中点
    t_acc = (np.arange(len(accelerations_seq)) + 1.0) * dt  # 取中点

    # 选择要显示的关键点
    if keypoint_indices is None:
        keypoint_indices = [0, 5, 6, 11, 12]  # 默认: nose, left_shoulder, right_shoulder, left_hip, right_hip

    n_kps = len(keypoint_indices)

    # 创建图形
    fig, axes = plt.subplots(n_kps, 3, figsize=(15, 4 * n_kps))
    if n_kps == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Velocity & Acceleration Verification (start={start_frame}, interval={interval_ms}ms, dt={dt:.3f}s)', fontsize=14)

    colors = {'x': 'b', 'y': 'r'}

    for row, kp_idx in enumerate(keypoint_indices):
        kp_name = KEYPOINT_NAMES[kp_idx]

        # Velocity X
        axes[row, 0].plot(t_vel, velocities_seq[:, kp_idx*2], 'b-', marker='o', label=f'{kp_name}_vx')
        axes[row, 0].set_ylabel('pixel/s')
        axes[row, 0].set_title(f'{kp_name} Velocity X')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Velocity Y
        axes[row, 1].plot(t_vel, velocities_seq[:, kp_idx*2+1], 'r-', marker='o', label=f'{kp_name}_vy')
        axes[row, 1].set_ylabel('pixel/s')
        axes[row, 1].set_title(f'{kp_name} Velocity Y')
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Acceleration X
        if len(accelerations_seq) > 0:
            axes[row, 2].plot(t_acc, accelerations_seq[:, kp_idx*2], 'b--', marker='s', label=f'{kp_name}_ax')
            axes[row, 2].plot(t_acc, accelerations_seq[:, kp_idx*2+1], 'r--', marker='s', label=f'{kp_name}_ay')
        axes[row, 2].set_ylabel('pixel/s^2')
        axes[row, 2].set_title(f'{kp_name} Acceleration (X:Blue, Y:Red)')
        axes[row, 2].grid(True, alpha=0.3)
        axes[row, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[row, 2].legend(fontsize=8)

        # 只在最后一行添加x轴标签
        if row == n_kps - 1:
            for col in range(3):
                axes[row, col].set_xlabel('Time (s)')

    plt.tight_layout()
    output_path = Path(__file__).parent / f'velocity_accel_visualization_start{start_frame}_int{interval_ms}ms.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nImage saved to: {output_path}")
    plt.close()

    # 打印数值统计
    print(f"\n=== {kp_name} Velocity/Acceleration Stats ===")
    for kp_idx in keypoint_indices:
        kp_name = KEYPOINT_NAMES[kp_idx]
        print(f"\n{kp_name}:")
        print(f"  Velocity X (pixel/s): mean={velocities_seq[:, kp_idx*2].mean():.2f}, std={velocities_seq[:, kp_idx*2].std():.2f}, range=[{velocities_seq[:, kp_idx*2].min():.2f}, {velocities_seq[:, kp_idx*2].max():.2f}]")
        print(f"  Velocity Y (pixel/s): mean={velocities_seq[:, kp_idx*2+1].mean():.2f}, std={velocities_seq[:, kp_idx*2+1].std():.2f}, range=[{velocities_seq[:, kp_idx*2+1].min():.2f}, {velocities_seq[:, kp_idx*2+1].max():.2f}]")
        if len(accelerations_seq) > 0:
            print(f"  Accel X (pixel/s^2): mean={accelerations_seq[:, kp_idx*2].mean():.2f}, std={accelerations_seq[:, kp_idx*2].std():.2f}, range=[{accelerations_seq[:, kp_idx*2].min():.2f}, {accelerations_seq[:, kp_idx*2].max():.2f}]")
            print(f"  Accel Y (pixel/s^2): mean={accelerations_seq[:, kp_idx*2+1].mean():.2f}, std={accelerations_seq[:, kp_idx*2+1].std():.2f}, range=[{accelerations_seq[:, kp_idx*2+1].min():.2f}, {accelerations_seq[:, kp_idx*2+1].max():.2f}]")

    return positions_seq, velocities_seq, accelerations_seq


def verify_calculation_details(detections, start_frame=0, interval_ms=300):
    """详细验证计算过程，打印前几帧的具体计算"""
    dt = interval_ms / 1000.0
    indices = get_frame_indices(start_frame, interval_ms)

    print(f"\n=== 详细计算验证 (dt={dt:.3f}s) ===")

    # 提取positions
    positions_list = []
    for idx in indices:
        keypoints = detections[idx].get('keypoints', [])[:17]
        positions_list.append(flatten_keypoints(keypoints))
    positions_seq = np.array(positions_list)

    # 只打印nose的前几帧
    print(f"\nNose 位置 (前3帧):")
    for i in range(min(3, len(positions_seq))):
        print(f"  帧{i}: x={positions_seq[i][0]:.2f}, y={positions_seq[i][1]:.2f}")

    print(f"\n速度计算 (基于相邻帧差分):")
    for i in range(min(3, len(positions_seq)-1)):
        vx = (positions_seq[i+1][0] - positions_seq[i][0]) / dt
        vy = (positions_seq[i+1][1] - positions_seq[i][1]) / dt
        print(f"  frame{i}->{i+1}: vx = ({positions_seq[i+1][0]:.2f} - {positions_seq[i][0]:.2f}) / {dt:.3f} = {vx:.4f} pixel/s")
        print(f"           vy = ({positions_seq[i+1][1]:.2f} - {positions_seq[i][1]:.2f}) / {dt:.3f} = {vy:.4f} pixel/s")

    # 计算完整的速度序列
    velocities_seq = compute_velocities(positions_seq, dt)

    print(f"\n加速度计算 (基于相邻速度差分):")
    for i in range(min(2, len(velocities_seq)-1)):
        ax = (velocities_seq[i+1][0] - velocities_seq[i][0]) / dt
        ay = (velocities_seq[i+1][1] - velocities_seq[i][1]) / dt
        print(f"  v{i}->v{i+1}: ax = ({velocities_seq[i+1][0]:.4f} - {velocities_seq[i][0]:.4f}) / {dt:.3f} = {ax:.4f} pixel/s^2")
        print(f"           ay = ({velocities_seq[i+1][1]:.4f} - {velocities_seq[i][1]:.4f}) / {dt:.3f} = {ay:.4f} pixel/s^2")


def main():
    """主函数"""
    print("加载检测结果...")
    detections = load_detections()
    print(f"共 {len(detections)} 帧检测结果")

    # 使用第一个样本进行验证
    start_frame = 0
    interval_ms = 300

    # 详细计算验证
    verify_calculation_details(detections, start_frame, interval_ms)

    # 可视化
    print("\n生成可视化图片...")
    visualize_sample(detections, start_frame, interval_ms)

    # 额外测试：不同起始帧
    print("\n" + "="*50)
    print("测试不同起始帧:")
    for sf in [0, 50, 100]:
        print(f"\n--- 起始帧 {sf} ---")
        visualize_sample(detections, start_frame=sf, interval_ms=interval_ms)


if __name__ == '__main__':
    main()