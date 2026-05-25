"""多进程批量运行脚本 - 独立进程隔离
支持优雅退出和强制终止，带进度条
"""
import atexit
import subprocess
import multiprocessing
import sys
import os
import signal
import time
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm


# 全局进程列表，用于清理
_active_processes = []
_cleanup_started = False
_shutdown_requested = False


def _wait_process_exit(proc, timeout: float) -> bool:
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def kill_process_tree(pid):
    """杀死整个进程树"""
    try:
        if sys.platform == 'win32':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True, timeout=5)
        else:
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"[警告] 无法杀死进程树 PID={pid}: {e}")


def terminate_process_tree(pid):
    """优雅终止整个进程树"""
    try:
        if sys.platform == 'win32':
            subprocess.run(['taskkill', '/T', '/PID', str(pid)], capture_output=True, timeout=5)
        else:
            os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"[警告] 无法终止进程树 PID={pid}: {e}")


def cleanup_processes():
    """清理所有活跃的子进程"""
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True

    print("\n[清理] 开始清理子进程...")
    for p_info in list(_active_processes):
        proc = p_info.get('process')
        if not proc or proc.poll() is not None:
            continue

        pid = p_info['pid']
        try:
            print(f"[清理] 终止进程树 PID={pid}")
            terminate_process_tree(pid)
            if not _wait_process_exit(proc, timeout=3):
                print(f"[清理] PID={pid} 未响应，强制Kill进程树")
                kill_process_tree(pid)
                _wait_process_exit(proc, timeout=2)
        except Exception as e:
            print(f"[清理] PID={pid} 终止失败: {e}")

    _active_processes.clear()
    print("[清理] 清理完成")


def signal_handler(signum, frame):
    """处理中断信号"""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[信号] 收到信号 {signum}，正在清理...")
    cleanup_processes()
    raise SystemExit(1)


atexit.register(cleanup_processes)


def get_video_list(video_dir: Path) -> list:
    """获取所有视频文件"""
    videos = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        videos.extend(video_dir.glob(f"**/{ext}"))
    return sorted(videos)



def run_parallel_batch(video_dir: Path, num_workers: int = None,
                       max_total_videos: int = None,
                       timeout_per_video: int = 300,
                       resume: bool = False,
                       reuse_frames: bool = False,
                       reuse_split_frames: bool = False,
                       watch: bool = False,
                       poll_interval: int = 30,
                       stable_seconds: int = 60,
                       idle_exit_seconds: int | None = None,
                       retry_cooldown_seconds: int = 60):
    """
    多进程并行处理视频

    Args:
        video_dir: 视频目录
        num_workers: 并行进程数，默认 CPU核心数-1
        max_total_videos: 最多处理的总视频数
        timeout_per_video: 每个视频超时时间（秒）
        resume: 是否启用断点续跑（跳过已完成的步骤和round）
        reuse_frames: 是否复用现有切帧和pose结果，只重新计算后续流程
        reuse_split_frames: 是否跳过切帧但重新跑pose和后续流程
        watch: 是否持续监控并处理新上传视频
    """
    global _active_processes

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    videos = get_video_list(video_dir)
    total_videos = len(videos)

    if total_videos == 0:
        print("[ERROR] 没有找到视频文件")
        return

    # 限制总视频数
    if max_total_videos:
        videos = videos[:max_total_videos]
        total_videos = len(videos)

    # 注册信号处理
    if sys.platform != 'win32':
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Windows 上使用轮询检测 Ctrl+C
        pass

    print(f"=" * 60)
    print(f"多进程批量运行")
    print(f"=" * 60)
    print(f"CPU核心数: {multiprocessing.cpu_count()}")
    print(f"启动进程数: {num_workers}")
    print(f"总视频数: {total_videos}")
    print(f"每视频超时: {timeout_per_video}秒")
    print(f"=" * 60)

    # 计算每个worker处理的视频数
    videos_per_worker = (total_videos + num_workers - 1) // num_workers

    # 启动所有进程
    processes = []
    start_time = time.time()

    try:
        for worker_idx in range(num_workers):
            offset = worker_idx * videos_per_worker
            count = min(videos_per_worker, total_videos - offset)

            if count <= 0:
                continue

            print(f"\n启动 Worker {worker_idx + 1}/{num_workers}:")
            print(f"  Offset: {offset}, Count: {count}")

            # 构建命令
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "main_pipeline.py"),
                "--batch",
                f"--offset={offset}",
                f"--max-videos={count}",
                "--workers=1",
            ]
            if resume:
                cmd.append("--resume")
            if reuse_frames:
                cmd.append("--reuse-frames")
            if reuse_split_frames:
                cmd.append("--reuse-split-frames")
            if watch:
                cmd.extend([
                    "--watch",
                    f"--poll-interval={poll_interval}",
                    f"--stable-seconds={stable_seconds}",
                    f"--retry-cooldown-seconds={retry_cooldown_seconds}",
                ])
                if idle_exit_seconds is not None:
                    cmd.append(f"--idle-exit-seconds={idle_exit_seconds}")
            cmd.append("--disable-internal-parallel")

            # 启动进程 - 输出到管道以解析进度
            log_dir = Path(__file__).parent / "logs" / f"worker_{worker_idx}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"worker_{worker_idx}.log"

            env = os.environ.copy()
            env['DISABLE_INTERNAL_PARALLEL'] = '1'
            # 使用 PIPE 而不是文件，这样我们可以实时读取进度
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
                env=env,
                text=True,
                bufsize=1,
                start_new_session=(sys.platform != 'win32')
            )

            p_info = {
                'worker_idx': worker_idx,
                'pid': process.pid,
                'offset': offset,
                'count': count,
                'process': process,
                'start_time': time.time(),
                'last_line_time': time.time(),
                'line_count': 0
            }
            processes.append(p_info)
            _active_processes.append(p_info)

            print(f"  PID: {process.pid}")
            print(f"  命令: {' '.join(cmd)}")

        # 监控进度
        print(f"\n{'=' * 60}")
        print("开始处理... (按 Ctrl+C 可以安全退出)")
        print(f"{'=' * 60}\n")

        # 创建进度条 - 视频级
        total_videos = sum(p['count'] for p in processes if p)
        pbar = tqdm(total=total_videos, desc="处理视频", unit="个", initial=0)
        pbar.set_postfix_str(f"0/{total_videos} 完成")

        completed = 0

        while completed < len(processes):
            # 读取子进程输出，解析进度标记
            for p_info in processes:
                if p_info is None:
                    continue
                proc = p_info['process']
                if proc.stdout:
                    # 读取可用行（非阻塞）
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        line = line.rstrip()
                        if '[PROGRESS] video_completed:' in line:
                            pbar.update(1)
                        elif line:
                            # 只显示重要日志，不显示每个视频的详细信息
                            if '完成:' in line or '失败' in line or '错误' in line:
                                print(f"[W{p_info['worker_idx']+1}] {line}")

            # 检查是否有进程结束
            for i, p_info in enumerate(processes):
                if p_info is None:
                    continue
                proc = p_info['process']
                poll = proc.poll()

                if poll is not None:
                    # 进程已结束
                    elapsed = time.time() - p_info['start_time']
                    status = "成功" if poll == 0 else "失败"
                    print(f"\n[{elapsed:.1f}s] Worker {p_info['worker_idx']+1} 完成: {status}, "
                          f"PID={p_info['pid']}, 退出码={poll}")

                    completed += 1
                    if p_info in _active_processes:
                        _active_processes.remove(p_info)

                    # 移除已完成的进程引用
                    processes[i] = None

            # 检查是否还有活着的进程
            alive_processes = [p for p in processes if p is not None]
            if not alive_processes:
                break

            # 检查超时
            for p_info in alive_processes:
                elapsed = time.time() - p_info['start_time']
                if elapsed > p_info['count'] * timeout_per_video:
                    print(f"\n[超时] Worker {p_info['worker_idx']+1} 超时，强制终止...")
                    kill_process_tree(p_info['pid'])
                    completed += 1

            # 短暂休息
            time.sleep(0.1)

        pbar.close()

    except KeyboardInterrupt:
        print("\n[中断] 收到用户中断，正在清理...")
    finally:
        cleanup_processes()

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("所有进程已完成")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    import config

    parser = argparse.ArgumentParser(description="多进程批量运行跌倒检测pipeline")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"并行进程数 (默认: CPU核心数-1)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="最多处理的总视频数")
    parser.add_argument("--timeout", type=int, default=300,
                        help="每个视频超时时间(秒)")
    parser.add_argument("--resume", action="store_true",
                        help="启用断点续跑：自动跳过已完成的步骤和round")
    parser.add_argument("--reuse-frames", action="store_true",
                        help="复用现有切帧和pose结果，只重新计算后续流程")
    parser.add_argument("--reuse-split-frames", action="store_true",
                        help="跳过切帧，但重新执行pose检测和后续流程")
    parser.add_argument("--watch", action="store_true",
                        help="持续轮询目录，自动处理新上传且已稳定的视频")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="watch模式下每次扫描间隔秒数")
    parser.add_argument("--stable-seconds", type=int, default=60,
                        help="视频文件保持不变多少秒后才视为上传完成")
    parser.add_argument("--idle-exit-seconds", type=int, default=None,
                        help="watch模式空闲多久后自动退出")
    parser.add_argument("--retry-cooldown-seconds", type=int, default=60,
                        help="watch模式失败视频的重试冷却时间")

    args = parser.parse_args()

    run_parallel_batch(
        video_dir=config.VIDEO_DIR,
        num_workers=args.workers,
        max_total_videos=args.max_videos,
        timeout_per_video=args.timeout,
        resume=args.resume,
        reuse_frames=args.reuse_frames,
        reuse_split_frames=args.reuse_split_frames,
        watch=args.watch,
        poll_interval=args.poll_interval,
        stable_seconds=args.stable_seconds,
        idle_exit_seconds=args.idle_exit_seconds,
        retry_cooldown_seconds=args.retry_cooldown_seconds,
    )
