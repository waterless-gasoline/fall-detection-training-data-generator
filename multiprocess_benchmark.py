"""多进程vs单进程性能测试脚本"""
import time
import sys
import os
import json
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 配置
VIDEO_DIR = Path(r"D:\IPC\IPC_video_data_mp4\跌倒\data_positive_0512\ir")
ANNOTATION_DIR = Path(r"D:\IPC\IPC_video_data_annotation\跌倒")
FRAME_OUTPUT_DIR = Path(r"D:\IPC\IPC_data_clip_photo\跌倒\data_positive_0512")
MODEL_PATH = Path(r"D:\pycharm\工作脚本\跌倒检测\跌倒训练数据生成\yolo26l-pose.pt")
VIDEO_FPS = 20.0

# 测试视频（使用已有完整数据的视频）
TEST_VIDEOS = [
    "0407add_IR_positive_0004",
    "0407add_IR_positive_0006",
]


def run_single_video(video_name, skip_pose_detection=True):
    """运行单个视频处理"""
    from main_pipeline import run_pipeline

    start = time.time()
    try:
        result = run_pipeline(
            video_name=video_name,
            video_dir=VIDEO_DIR,
            annotation_dir=ANNOTATION_DIR,
            frame_output_dir=FRAME_OUTPUT_DIR,
            model_path=MODEL_PATH,
            fps=VIDEO_FPS,
            num_rounds=6,
            skip_existing=True,
            skip_pose_detection=skip_pose_detection
        )

        samples = 0
        if 'final' in result:
            import pandas as pd
            df = pd.read_csv(result['final'])
            samples = len(df)

        return {
            'video_name': video_name,
            'elapsed': time.time() - start,
            'samples': samples,
            'error': None
        }
    except Exception as e:
        return {
            'video_name': video_name,
            'elapsed': time.time() - start,
            'samples': 0,
            'error': str(e)
        }


def run_batch_sequential(videos):
    """顺序执行（模拟单进程）"""
    results = []
    total_start = time.time()

    for v in videos:
        print(f"  处理: {v}...")
        r = run_single_video(v)
        results.append(r)
        print(f"    完成: {r['elapsed']:.1f}秒, {r['samples']}样本")

    total_time = time.time() - total_start
    return results, total_time


def run_batch_parallel(videos, num_workers):
    """并行执行（多进程）"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    total_start = time.time()

    print(f"  启动 {num_workers} 个worker进程...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_video, v): v for v in videos}

        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(f"    完成: {r['video_name']} - {r['elapsed']:.1f}秒")

    total_time = time.time() - total_start
    return results, total_time


def print_summary(results, total_time, config_name):
    """打印结果汇总"""
    successful = [r for r in results if r['error'] is None]
    total_samples = sum(r['samples'] for r in successful)

    print(f"\n  {config_name}:")
    print(f"    总耗时: {total_time:.1f}秒")
    if successful:
        print(f"    平均每视频: {total_time/len(successful):.1f}秒")
    print(f"    总样本数: {total_samples}")
    print(f"    成功: {len(successful)}/{len(results)}")

    return total_time, len(successful), total_samples


def main():
    print("=" * 70)
    print("多进程性能对比测试")
    print("=" * 70)
    print(f"CPU核心数: {multiprocessing.cpu_count()}")
    print(f"测试视频: {TEST_VIDEOS}")
    print("=" * 70)

    all_results = {}

    # 测试1: 单进程（顺序）
    print("\n" + "-" * 70)
    print("测试1: 单进程（顺序执行）")
    print("-" * 70)
    results_seq, time_seq = run_batch_sequential(TEST_VIDEOS)
    all_results['单进程'] = {'time': time_seq, 'results': results_seq}

    # 测试2: 2进程并行
    print("\n" + "-" * 70)
    print("测试2: 2进程并行")
    print("-" * 70)
    results_2p, time_2p = run_batch_parallel(TEST_VIDEOS, num_workers=2)
    all_results['2进程'] = {'time': time_2p, 'results': results_2p}

    # 测试3: 4进程并行
    print("\n" + "-" * 70)
    print("测试3: 4进程并行")
    print("-" * 70)
    results_4p, time_4p = run_batch_parallel(TEST_VIDEOS, num_workers=4)
    all_results['4进程'] = {'time': time_4p, 'results': results_4p}

    # 汇总输出
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print()
    print("| 配置 | 耗时 | 加速比 | 备注 |")
    print("|------|------|--------|------|")

    baseline = time_seq
    configs = [
        ('单进程', time_seq, '顺序执行'),
        ('2进程', time_2p, '并行2进程'),
        ('4进程', time_4p, '并行4进程'),
    ]

    for name, t, note in configs:
        speedup = baseline / t if t > 0 else 0
        print(f"| {name} | {t:.1f}秒 | {speedup:.2f}x | {note} |")

    print()
    print("-" * 70)
    print("各视频详细结果:")
    print("-" * 70)

    for video in TEST_VIDEOS:
        print(f"\n{video}:")
        for config_name, data in all_results.items():
            for r in data['results']:
                if r['video_name'] == video:
                    status = "成功" if r['error'] is None else f"失败: {r['error']}"
                    print(f"  {config_name}: {r['elapsed']:.1f}秒 ({r['samples']}样本) - {status}")

    # 保存结果
    output_path = Path(__file__).parent / "multiprocess_benchmark_results.json"
    save_data = {
        'config': {
            'cpu_count': multiprocessing.cpu_count(),
            'test_videos': TEST_VIDEOS,
        },
        'results': {k: {'time': v['time'], 'details': v['results']} for k, v in all_results.items()},
        'summary': [
            {'config': '单进程', 'time': time_seq, 'speedup': 1.0},
            {'config': '2进程', 'time': time_2p, 'speedup': round(time_seq/time_2p, 2) if time_2p > 0 else 0},
            {'config': '4进程', 'time': time_4p, 'speedup': round(time_seq/time_4p, 2) if time_4p > 0 else 0},
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()