"""Step 1: 视频切帧模块 - 随机间隔切帧 (ffmpeg版)"""
import atexit
import json
import random
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
import sys, os, logging
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# 简单日志配置
logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
logger = logging.getLogger(__name__)


def _resolve_binary(config_value: str, binary_name: str) -> str:
    """解析ffmpeg/ffprobe配置，兼容绝对路径和PATH命令"""
    if not config_value:
        return binary_name

    candidate = Path(config_value)
    if candidate.exists():
        return str(candidate)

    return config_value


FFMPEG_BIN = _resolve_binary(getattr(config, 'FFMPEG_BIN', 'ffmpeg'), 'ffmpeg')
FFPROBE_BIN = _resolve_binary(getattr(config, 'FFPROBE_BIN', 'ffprobe'), 'ffprobe')
PLANNED_TIMESTAMPS_FILE = 'planned_timestamps.json'
FRAME_TIMESTAMPS_FILE = 'frame_timestamps.json'
ROUND_DONE_MARKER = 'round_done.marker'

_ACTIVE_EXTRACTORS = set()
_ACTIVE_EXTRACTORS_LOCK = threading.Lock()
_SIGNAL_HANDLERS_INSTALLED = False
_PREVIOUS_SIGNAL_HANDLERS = {}


def _register_active_extractor(extractor) -> None:
    with _ACTIVE_EXTRACTORS_LOCK:
        _ACTIVE_EXTRACTORS.add(extractor)


def _unregister_active_extractor(extractor) -> None:
    with _ACTIVE_EXTRACTORS_LOCK:
        _ACTIVE_EXTRACTORS.discard(extractor)


def cleanup_active_ffmpeg_extractors() -> None:
    with _ACTIVE_EXTRACTORS_LOCK:
        extractors = list(_ACTIVE_EXTRACTORS)
    for extractor in extractors:
        try:
            extractor.cleanup_ffmpeg_processes()
        except Exception as exc:
            logger.warning(f"[Step1] 清理ffmpeg进程失败: {exc}")


def _chain_signal_handler(signum, frame):
    cleanup_active_ffmpeg_extractors()
    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum)
    if callable(previous):
        previous(signum, frame)
        return
    if previous == signal.SIG_DFL:
        raise KeyboardInterrupt() if signum == getattr(signal, 'SIGINT', None) else SystemExit(1)


def ensure_ffmpeg_cleanup_hooks() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    atexit.register(cleanup_active_ffmpeg_extractors)
    for signame in ('SIGINT', 'SIGTERM'):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            _PREVIOUS_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
            signal.signal(signum, _chain_signal_handler)
        except (ValueError, OSError):
            continue
    _SIGNAL_HANDLERS_INSTALLED = True


class VideoFrameExtractor:
    """视频切帧器 - 随机间隔切帧 (真实ffmpeg抽帧)"""

    def __init__(self, video_path: Path, output_dir: Path):
        ensure_ffmpeg_cleanup_hooks()
        self.video_path = video_path
        self.output_dir = output_dir
        self.source_fps = None
        self.total_frames = None
        self.frame_intervals_ms = config.FRAME_INTERVALS_MS  # [250, 300, 350, 400, 450, 500]
        self.ffmpeg_workers = max(1, int(getattr(config, 'STEP1_FFMPEG_WORKERS', 1)))
        self._ffmpeg_lock = threading.Lock()
        self._active_ffmpeg_processes: dict[int, subprocess.Popen] = {}
        self._cleanup_started = False

    def _terminate_ffmpeg_process(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is not None:
                return
            if sys.platform == 'win32':
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                pass
            if sys.platform == 'win32':
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)
        except ProcessLookupError:
            pass

    def cleanup_ffmpeg_processes(self) -> None:
        with self._ffmpeg_lock:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            processes = list(self._active_ffmpeg_processes.values())
        for proc in processes:
            self._terminate_ffmpeg_process(proc)
        with self._ffmpeg_lock:
            self._active_ffmpeg_processes.clear()

    def _register_ffmpeg_process(self, proc: subprocess.Popen) -> None:
        with self._ffmpeg_lock:
            self._active_ffmpeg_processes[proc.pid] = proc
            self._cleanup_started = False
        _register_active_extractor(self)

    def _unregister_ffmpeg_process(self, proc: subprocess.Popen) -> None:
        with self._ffmpeg_lock:
            self._active_ffmpeg_processes.pop(proc.pid, None)
            has_active = bool(self._active_ffmpeg_processes)
        if not has_active:
            _unregister_active_extractor(self)

    def get_video_info(self) -> tuple:
        """用ffprobe获取视频信息"""
        # 帧率
        fps_cmd = [
            FFPROBE_BIN, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames',
            '-of', 'default=noprint_wrappers=1', str(self.video_path)
        ]
        try:
            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ffprobe不存在或不可执行: {FFPROBE_BIN}") from e

        fps_str = None
        frames_str = None
        for line in fps_result.stdout.splitlines():
            if line.startswith('r_frame_rate='):
                fps_str = line.split('=', 1)[1]
            elif line.startswith('nb_frames='):
                frames_str = line.split('=', 1)[1]

        if fps_str and '/' in fps_str:
            num, den = fps_str.split('/')
            self.source_fps = float(num) / float(den) if float(den) != 0 else 20.0
        else:
            self.source_fps = float(fps_str) if fps_str else 20.0

        self.total_frames = int(frames_str) if frames_str and frames_str.isdigit() else 0
        return self.source_fps, self.total_frames

    def get_video_duration(self) -> float:
        """用ffprobe获取视频时长（秒）"""
        cmd = [
            FFPROBE_BIN, '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(self.video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ffprobe不存在或不可执行: {FFPROBE_BIN}") from e
        duration_text = result.stdout.strip()
        if not duration_text:
            raise ValueError(f"无法获取视频时长: {self.video_path}")
        return float(duration_text)

    def _build_timestamp_list(self, video_duration_s: float) -> tuple:
        """按现有随机间隔逻辑生成本轮所有目标时间戳"""
        intervals_s = [ms / 1000.0 for ms in self.frame_intervals_ms]
        target_timestamps = []
        interval_ms_list = []
        current_time_s = 0.0

        while current_time_s < video_duration_s:
            target_timestamps.append(current_time_s)
            interval_s = random.choice(intervals_s)
            interval_ms_list.append(int(interval_s * 1000))
            current_time_s += interval_s

        return target_timestamps, interval_ms_list

    def _round_paths(self, round_dir: Path) -> tuple[Path, Path, Path]:
        """返回round相关状态文件路径"""
        return (
            round_dir / PLANNED_TIMESTAMPS_FILE,
            round_dir / FRAME_TIMESTAMPS_FILE,
            round_dir / ROUND_DONE_MARKER,
        )

    def _load_planned_timestamps(self, planned_path: Path) -> list[float] | None:
        """加载已保存的目标时间戳计划"""
        if not planned_path.exists():
            return None
        try:
            data = json.loads(planned_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        return [float(item) for item in data]

    def _is_valid_frame_file(self, frame_path: Path) -> bool:
        """判断单帧文件是否有效"""
        return frame_path.exists() and frame_path.stat().st_size > 0

    def _load_or_create_round_plan(self, round_dir: Path, video_duration_s: float) -> tuple[list[float], list[int]]:
        """加载或创建round的目标时间戳计划"""
        planned_path, _, done_marker = self._round_paths(round_dir)
        planned_timestamps = self._load_planned_timestamps(planned_path)
        if planned_timestamps is None:
            planned_timestamps, interval_ms_list = self._build_timestamp_list(video_duration_s)
            planned_path.write_text(json.dumps(planned_timestamps), encoding='utf-8')
            if done_marker.exists():
                done_marker.unlink()
            return planned_timestamps, interval_ms_list

        interval_ms_list = []
        for idx, timestamp in enumerate(planned_timestamps[:-1]):
            next_timestamp = planned_timestamps[idx + 1]
            interval_ms_list.append(int(round((next_timestamp - timestamp) * 1000)))
        return planned_timestamps, interval_ms_list

    def is_round_complete(self, round_dir: Path) -> bool:
        """检查round切帧结果是否完整"""
        planned_path, timestamps_path, done_marker = self._round_paths(round_dir)
        if not (planned_path.exists() and timestamps_path.exists() and done_marker.exists()):
            logger.info(
                "[ReuseDebug][step1] round=%s complete=False reason=missing_marker planned=%s timestamps=%s done=%s",
                round_dir.name,
                planned_path.exists(),
                timestamps_path.exists(),
                done_marker.exists(),
            )
            return False

        planned_timestamps = self._load_planned_timestamps(planned_path)
        if not planned_timestamps:
            logger.info("[ReuseDebug][step1] round=%s complete=False reason=empty_planned", round_dir.name)
            return False

        try:
            frame_timestamps = json.loads(timestamps_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            logger.info("[ReuseDebug][step1] round=%s complete=False reason=bad_timestamps_json", round_dir.name)
            return False

        if len(frame_timestamps) != len(planned_timestamps):
            logger.info(
                "[ReuseDebug][step1] round=%s complete=False reason=timestamp_length_mismatch planned=%s actual=%s",
                round_dir.name,
                len(planned_timestamps),
                len(frame_timestamps),
            )
            return False

        for index in range(len(planned_timestamps)):
            frame_path = round_dir / f'frame_{index:06d}.jpg'
            if not self._is_valid_frame_file(frame_path):
                logger.info(
                    "[ReuseDebug][step1] round=%s complete=False reason=missing_frame index=%s path=%s",
                    round_dir.name,
                    index,
                    frame_path,
                )
                return False

        logger.info("[ReuseDebug][step1] round=%s complete=True", round_dir.name)
        return True

    def _extract_frame_with_ffmpeg(self, timestamp: float, output_path: Path) -> bool:
        """用ffmpeg在指定时间点提取单帧"""
        cmd = [
            FFMPEG_BIN, '-y', '-ss', f'{timestamp:.3f}',
            '-i', str(self.video_path), '-vframes', '1', '-q:v', '2', str(output_path)
        ]
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
            'encoding': 'utf-8',
            'errors': 'ignore',
        }
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs['start_new_session'] = True
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ffmpeg不存在或不可执行: {FFMPEG_BIN}") from e

        self._register_ffmpeg_process(proc)
        try:
            proc.communicate()
            return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
        finally:
            self._unregister_ffmpeg_process(proc)

    def _extract_frame_task(self, index: int, timestamp: float, round_dir: Path) -> tuple[int, float, bool]:
        """执行单个时间戳抽帧任务"""
        output_path = round_dir / f'frame_{index:06d}.jpg'
        ok = self._extract_frame_with_ffmpeg(timestamp, output_path)
        return index, timestamp, ok

    def _extract_round_frames_batched(self, round_dir: Path, target_timestamps: list[float]) -> list[bool]:
        """用单个ffmpeg进程批量抽帧，减少每帧子进程开销"""
        total_targets = len(target_timestamps)
        extraction_results = [False] * total_targets
        pending_tasks: list[tuple[int, float]] = []

        for index, timestamp in enumerate(target_timestamps):
            frame_path = round_dir / f'frame_{index:06d}.jpg'
            if self._is_valid_frame_file(frame_path):
                extraction_results[index] = True
            else:
                pending_tasks.append((index, timestamp))

        if not pending_tasks:
            logger.info(f'[Step1] 当前round所有帧已存在，无需补切: {total_targets}/{total_targets}')
            return extraction_results

        seek_start = max(0.0, pending_tasks[0][1] - (config.STEP1_BATCH_SEEK_TOLERANCE_MS / 1000.0))
        logger.info(
            f'[Step1] 需要补切 {len(pending_tasks)}/{total_targets} 帧, mode=batched, seek_start={seek_start:.3f}s'
        )

        output_dir = round_dir
        temp_pattern = output_dir / 'ffmpeg_batch_%06d.jpg'
        cmd = [
            FFMPEG_BIN, '-y', '-ss', f'{seek_start:.3f}', '-i', str(self.video_path),
            '-vsync', '0', '-q:v', '2', str(temp_pattern)
        ]

        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
            'encoding': 'utf-8',
            'errors': 'ignore',
        }
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs['start_new_session'] = True

        proc = None
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            self._register_ffmpeg_process(proc)
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.warning(f'[Step1] batched ffmpeg 抽帧失败，回退到逐帧模式: {stderr[-500:]}')
                return self._extract_round_frames_fallback(round_dir, target_timestamps, extraction_results, pending_tasks)

            generated_frames = sorted(round_dir.glob('ffmpeg_batch_*.jpg'))
            if len(generated_frames) < len(pending_tasks):
                logger.warning(
                    f'[Step1] batched ffmpeg 生成帧数不足，回退到逐帧模式: generated={len(generated_frames)}, '
                    f'expected={len(pending_tasks)}'
                )
                for frame_path in generated_frames:
                    frame_path.unlink(missing_ok=True)
                return self._extract_round_frames_fallback(round_dir, target_timestamps, extraction_results, pending_tasks)

            for (index, timestamp), generated_path in zip(pending_tasks, generated_frames):
                output_path = round_dir / f'frame_{index:06d}.jpg'
                generated_path.replace(output_path)
                extraction_results[index] = True
            return extraction_results
        finally:
            if proc is not None:
                self._unregister_ffmpeg_process(proc)
            for frame_path in round_dir.glob('ffmpeg_batch_*.jpg'):
                if frame_path.exists():
                    frame_path.unlink()

    def _extract_round_frames_fallback(
        self,
        round_dir: Path,
        target_timestamps: list[float],
        extraction_results: list[bool],
        pending_tasks: list[tuple[int, float]],
    ) -> list[bool]:
        """回退到逐帧抽帧"""
        if not pending_tasks:
            return extraction_results

        logger.info(f'[Step1] 回退逐帧模式: ffmpeg_workers={self.ffmpeg_workers}')
        last_pct = -1
        processed = len(extraction_results) - len(pending_tasks)

        try:
            with ThreadPoolExecutor(max_workers=self.ffmpeg_workers) as executor:
                futures = [
                    executor.submit(self._extract_frame_task, index, timestamp, round_dir)
                    for index, timestamp in pending_tasks
                ]
                for future in as_completed(futures):
                    index, timestamp, ok = future.result()
                    extraction_results[index] = ok
                    processed += 1

                    if processed % 50 == 0 or processed == len(target_timestamps):
                        pct = min(100, int(processed / len(target_timestamps) * 100)) if target_timestamps else 100
                        if pct != last_pct:
                            success_count = sum(extraction_results)
                            logger.info(f'[Step1] 提取进度: {processed}/{len(target_timestamps)} 帧 ({pct}%), 成功 {success_count} 帧')
                            last_pct = pct
        except BaseException:
            self.cleanup_ffmpeg_processes()
            raise

        return extraction_results

    def _extract_round_frames(self, round_dir: Path, target_timestamps: list[float]) -> list[bool]:
        """按计划时间戳补齐round帧文件"""
        decode_mode = getattr(config, 'STEP1_DECODE_MODE', 'auto')
        if decode_mode in {'auto', 'ffmpeg_batch'} and len(target_timestamps) >= 8:
            return self._extract_round_frames_batched(round_dir, target_timestamps)
        return self._extract_round_frames_fallback(round_dir, target_timestamps, [False] * len(target_timestamps), [(index, timestamp) for index, timestamp in enumerate(target_timestamps)])


    def extract_one_round(self, round_idx: int, force: bool = False) -> tuple:
        """
        执行一轮切帧：随机间隔遍历整个视频（ffmpeg逐时间点抽帧）

        Returns:
            (frame_count, interval_ms_list)
        """
        round_start = time.perf_counter()
        round_dir = self.output_dir / f'round_{round_idx}'
        planned_path, timestamps_path, done_marker = self._round_paths(round_dir)

        if force and round_dir.exists():
            for frame_path in round_dir.glob('frame_*.jpg'):
                frame_path.unlink()
            for path in [planned_path, timestamps_path, done_marker]:
                if path.exists():
                    path.unlink()

        if not force and self.is_round_complete(round_dir):
            logger.info(f'[Step1] 跳过round_{round_idx}，已有完整数据')
            planned_timestamps = self._load_planned_timestamps(planned_path) or []
            interval_ms_list = []
            for idx, timestamp in enumerate(planned_timestamps[:-1]):
                interval_ms_list.append(int(round((planned_timestamps[idx + 1] - timestamp) * 1000)))
            return len(planned_timestamps), interval_ms_list

        round_dir.mkdir(parents=True, exist_ok=True)
        if done_marker.exists():
            done_marker.unlink()

        video_duration_s = self.get_video_duration()
        source_fps = self.source_fps if self.source_fps else 20.0
        logger.info(f'[Step1] Round {round_idx}: 视频时长={video_duration_s:.2f}s, 源FPS={source_fps}')

        target_timestamps, interval_ms_list = self._load_or_create_round_plan(round_dir, video_duration_s)
        total_targets = len(target_timestamps)
        decode_mode = getattr(config, 'STEP1_DECODE_MODE', 'auto')
        logger.info(f'[Step1] round={round_idx} decode_mode={decode_mode}, ffmpeg_workers={self.ffmpeg_workers}')


        try:
            extraction_results = self._extract_round_frames(round_dir, target_timestamps)

            frame_timestamps = [
                timestamp for timestamp, ok in zip(target_timestamps, extraction_results) if ok
            ]
            extracted_idx = len(frame_timestamps)

            with open(timestamps_path, 'w', encoding='utf-8') as f:
                json.dump(frame_timestamps, f)

            if extracted_idx == total_targets:
                done_marker.write_text('done\n', encoding='utf-8')

            elapsed = time.perf_counter() - round_start
            avg_frame_s = elapsed / extracted_idx if extracted_idx else 0.0
            sampled_fps = extracted_idx / video_duration_s if video_duration_s > 0 else 0.0
            logger.info(f'[Step1] Round {round_idx}: 提取 {extracted_idx} 帧 -> {round_dir}')
            logger.info(f'[Step1] 计划时间戳保存到: {planned_path}')
            logger.info(f'[Step1] 时间戳保存到: {timestamps_path}')
            logger.info(
                f'[Timing] Step1 round {round_idx}: {elapsed:.2f}s | '
                f'extracted_frames={extracted_idx}, avg_frame_s={avg_frame_s:.4f}, sampled_fps={sampled_fps:.2f}, '
                f'workers={self.ffmpeg_workers}, decode_mode={getattr(config, "STEP1_DECODE_MODE", "auto")}'
            )

            return extracted_idx, interval_ms_list
        finally:
            self.cleanup_ffmpeg_processes()

    def extract_frames(self, force: bool = False, num_rounds: int = 6) -> List[Path]:
        """
        执行多轮切帧

        Args:
            force: 是否强制重新切帧
            num_rounds: 轮数，默认6次
        """
        total_start = time.perf_counter()
        self.get_video_info()
        logger.info(f"[Step1] 开始切帧: {self.video_path.name}")
        logger.info(f"  源视频: FPS={self.source_fps}, 总帧数={self.total_frames}")
        logger.info(f"  随机间隔: {self.frame_intervals_ms} ms")
        logger.info(f"  切帧轮数: {num_rounds}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        try:
            for round_idx in range(1, num_rounds + 1):
                logger.info(f"\n--- Round {round_idx}/{num_rounds} ---")
                frame_count, intervals = self.extract_one_round(round_idx, force)
                results.append(frame_count)

            total_elapsed = time.perf_counter() - total_start
            total_extracted = sum(count for count in results if count and count > 0)
            logger.info(f"\n[Step1] 切帧完成: 共 {num_rounds} 轮")
            for i, count in enumerate(results):
                logger.info(f"  Round {i+1}: {count} 帧")
            logger.info(
                f"[Timing] Step1 total: {total_elapsed:.2f}s | "
                f"rounds={num_rounds}, total_extracted_frames={total_extracted}"
            )
            return results
        finally:
            self.cleanup_ffmpeg_processes()
            _unregister_active_extractor(self)


def extract_random_intervals(video_path: Path, output_dir: Path, force: bool = False, num_rounds: int = 6) -> List[Path]:
    """为指定视频执行随机间隔切帧"""
    extractor = VideoFrameExtractor(video_path, output_dir)
    return extractor.extract_frames(force=force, num_rounds=num_rounds)


def run(video_name: str = config.VIDEO_NAME, force: bool = False, num_rounds: int = 6) -> List[Path]:
    """执行切帧"""
    video_path = config.VIDEO_DIR / f"{video_name}.mp4"
    output_dir = config.FRAME_OUTPUT_DIR / video_name

    extractor = VideoFrameExtractor(video_path, output_dir)
    return extractor.extract_frames(force=force, num_rounds=num_rounds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='视频切帧 - 随机间隔')
    parser.add_argument('--video', type=str, default=config.VIDEO_NAME, help='视频名')
    parser.add_argument('--force', action='store_true', help='强制重新切帧')
    parser.add_argument('--rounds', type=int, default=6, help='切帧轮数')
    args = parser.parse_args()

    run(args.video, args.force, args.rounds)
