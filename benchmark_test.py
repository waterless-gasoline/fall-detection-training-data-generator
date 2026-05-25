"""基准测试脚本 - 测试视频处理效率"""
import time
import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from main_pipeline import run_pipeline

# 直接定义路径配置
VIDEO_DIR = Path(r"D:\IPC\IPC_video_data_mp4\跌倒\data_positive_0512\normal")
ANNOTATION_DIR = Path(r"D:\IPC\IPC_video_data_annotation\跌倒")
FRAME_OUTPUT_DIR = Path(r"D:\IPC\IPC_data_clip_photo\跌倒\data_positive_0512")
MODEL_PATH = Path(r"D:\pycharm\工作脚本\跌倒检测\跌倒训练数据生成\yolo26l-pose.pt")
VIDEO_FPS = 20.0

def benchmark_videos(video_names, video_dir, annotation_dir, frame_output_dir, model_path, fps=20.0, num_rounds=6):
    """运行基准测试，返回每个视频的处理时间"""
    results = []

    for i, video_name in enumerate(video_names):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(video_names)}] 测试视频: {video_name}")
        print('='*60)

        start_time = time.time()

        try:
            result = run_pipeline(
                video_name=video_name,
                video_dir=video_dir,
                annotation_dir=annotation_dir,
                frame_output_dir=frame_output_dir,
                model_path=model_path,
                fps=fps,
                num_rounds=num_rounds,
                skip_existing=True,
                skip_pose_detection=True  # 跳过pose检测，复用现有结果
            )

            elapsed = time.time() - start_time

            samples_count = 0
            if 'final' in result:
                import pandas as pd
                df = pd.read_csv(result['final'])
                samples_count = len(df)

            results.append({
                'video_name': video_name,
                'elapsed_seconds': elapsed,
                'samples': samples_count,
                'success': True,
                'error': None
            })

            print(f"  完成! 耗时: {elapsed:.1f}秒, 样本数: {samples_count}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  失败: {e}")
            results.append({
                'video_name': video_name,
                'elapsed_seconds': elapsed,
                'samples': 0,
                'success': False,
                'error': str(e)
            })

    return results

if __name__ == "__main__":
    # 测试前5个视频 (使用ir目录，已有处理数据)
    video_names = [
        "0407add_IR_positive_0004",
        "0407add_IR_positive_0006",
        "0407add_IR_positive_0008",
        "0407add_IR_positive_0011",
        "0407add_IR_positive_0014",
    ]

    # ir视频对应的视频路径和标注目录
    video_dir = Path(r"D:\IPC\IPC_video_data_mp4\跌倒\data_positive_0512\ir")

    print("="*60)
    print("基准测试 - 视频处理效率测试")
    print("="*60)
    print(f"视频目录: {video_dir}")
    print(f"测试视频数: {len(video_names)}")
    print(f"每视频Round数: 6")
    print("="*60)

    results = benchmark_videos(
        video_names=video_names,
        video_dir=video_dir,
        annotation_dir=ANNOTATION_DIR,
        frame_output_dir=FRAME_OUTPUT_DIR,
        model_path=MODEL_PATH,
        fps=VIDEO_FPS,
        num_rounds=6
    )

    # 输出汇总
    print("\n" + "="*60)
    print("基准测试结果汇总")
    print("="*60)
    print(f"{'视频名':<35} {'耗时(秒)':<12} {'样本数':<10} {'状态'}")
    print("-"*60)

    total_time = 0
    success_count = 0

    for r in results:
        status = "成功" if r['success'] else f"失败({r['error']})"
        print(f"{r['video_name']:<35} {r['elapsed_seconds']:<12.1f} {r['samples']:<10} {status}")
        total_time += r['elapsed_seconds']
        if r['success']:
            success_count += 1

    print("-"*60)
    print(f"总耗时: {total_time:.1f}秒")
    print(f"平均每视频: {total_time/len(results):.1f}秒")
    print(f"成功: {success_count}/{len(results)}")

    # 保存结果
    import json
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'total_time': total_time,
            'avg_time': total_time / len(results),
            'success_count': success_count
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")