"""Step 2: 人体关键点检测模块"""
import logging
import multiprocessing as mp
import os
import time
import subprocess
from pathlib import Path

import numpy as np

import config

logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
logger = logging.getLogger(__name__)

VALID_INFER_MODES = {"multiprocess", "single_gpu_batch"}
VALID_DEVICES = {"auto", "cpu", "cuda"}
POSE_CACHE_DIR = "pose_cache"
POSE_DONE_MARKER = "pose_done.marker"


def _get_cuda_runtime_context() -> str:
    try:
        import torch
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
        pytorch_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')
        if not torch.cuda.is_available():
            return (
                f"CUDA_VISIBLE_DEVICES={cuda_visible}, PYTORCH_CUDA_ALLOC_CONF={pytorch_alloc_conf}, "
                f"cuda_available=False"
            )

        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device() if device_count > 0 else None
        current_name = torch.cuda.get_device_name(current_device) if current_device is not None else '<none>'
        parts = [
            f"CUDA_VISIBLE_DEVICES={cuda_visible}",
            f"PYTORCH_CUDA_ALLOC_CONF={pytorch_alloc_conf}",
            f"cuda_available=True",
            f"device_count={device_count}",
            f"current_device={current_device}",
            f"current_name={current_name}",
        ]
        for idx in range(device_count):
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            parts.append(
                f"gpu{idx}: free={free_bytes / 1024**3:.2f}GB, total={total_bytes / 1024**3:.2f}GB, "
                f"allocated={allocated / 1024**3:.2f}GB, reserved={reserved / 1024**3:.2f}GB, "
                f"name={torch.cuda.get_device_name(idx)}"
            )
        return " | ".join(parts)
    except Exception as exc:
        return f"cuda_context_failed: {exc}"


def _get_nvidia_smi_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return 'nvidia-smi:no_gpu_rows'
        return ' || '.join(lines)
    except Exception as exc:
        return f'nvidia-smi_failed: {exc}'


def _get_process_gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return 'nvidia-smi:no_compute_apps'
        return ' || '.join(lines[:20])
    except Exception as exc:
        return f'compute_apps_failed: {exc}'


def _log_runtime_context(prefix: str) -> None:
    logger.info(
        f"{prefix} pid={os.getpid()}, parent_pid={os.getppid()}, "
        f"process={mp.current_process().name}, cwd={Path.cwd()}, cuda={_get_cuda_runtime_context()}, "
        f"nvidia_smi={_get_nvidia_smi_snapshot()}, compute_apps={_get_process_gpu_snapshot()}"
    )


    """从YOLO结果中提取统一输出格式"""
    keypoints_array = np.zeros((17, 3), dtype=np.float32)
    keypoints_list = []

    if result.keypoints is not None and len(result.keypoints) > 0:
        kp = result.keypoints.xy[0]
        conf = result.keypoints.conf[0] if result.keypoints.conf is not None else None

        for i in range(min(17, len(kp))):
            x, y = float(kp[i][0].item()), float(kp[i][1].item())
            c = float(conf[i].item()) if conf is not None else 1.0
            keypoints_array[i] = [x, y, c]
            keypoints_list.append({'x': x, 'y': y, 'name': keypoint_names[i]})

    return {
        'frame': result.path.split('/')[-1].split('\\')[-1] if result.path else 'unknown.jpg',
        'keypoints': keypoints_array,
        'has_person': len(keypoints_list) > 0
    }


def _process_batch_worker(args):
    """Worker进程：处理一个batch的检测"""
    batch_file_strs, model_path, keypoint_names, device = args
    from ultralytics import YOLO

    model = YOLO(model_path)
    model.to(device)
    batch_files = [Path(f) for f in batch_file_strs]
    results = model(batch_files, verbose=False)
    return [_extract_result_record(result, keypoint_names) for result in results]


class PoseDetector:
    """人体关键点检测器"""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.keypoint_names = config.KEYPOINT_NAMES

    def _cache_dir(self, frame_dir: Path) -> Path:
        return frame_dir / POSE_CACHE_DIR

    def _done_marker_path(self, frame_dir: Path) -> Path:
        return frame_dir / POSE_DONE_MARKER

    def _frame_cache_path(self, frame_dir: Path, frame_name: str) -> Path:
        return self._cache_dir(frame_dir) / f"{Path(frame_name).stem}.npz"

    def _is_valid_output_file(self, output_path: Path, expected_count: int) -> bool:
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False
        try:
            with np.load(output_path, allow_pickle=True) as data:
                results = data['results']
        except Exception:
            return False
        return len(results) == expected_count

    def _load_cached_result(self, cache_path: Path) -> dict | None:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return None
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                result = data['result'].item()
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        return result

    def _save_cached_result(self, cache_path: Path, result: dict) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, result=result)

    def _save_cached_result_for_frame(self, frame_dir: Path, frame_name: str, result: dict) -> None:
        normalized = dict(result)
        normalized['frame'] = frame_name
        self._save_cached_result(self._frame_cache_path(frame_dir, frame_name), normalized)

    def _write_final_results(self, output_path: Path, results: list[dict]) -> None:
        np.savez_compressed(output_path, results=results)

    def _collect_frame_state(self, frame_dir: Path) -> tuple[list[Path], list[dict | None], list[Path]]:
        frame_files = sorted(frame_dir.glob('frame_*.jpg'))
        cached_results = []
        missing_frames = []
        for frame_path in frame_files:
            if not frame_path.exists() or frame_path.stat().st_size == 0:
                continue
            cached = self._load_cached_result(self._frame_cache_path(frame_dir, frame_path.name))
            cached_results.append(cached)
            if cached is None:
                missing_frames.append(frame_path)
        return frame_files, cached_results, missing_frames

    def is_round_complete(self, frame_dir: Path, output_path: Path | None = None) -> bool:
        frame_files, _, _ = self._collect_frame_state(frame_dir)
        if not frame_files:
            return False
        output_path = output_path or (frame_dir / 'pose_results.npz')
        done_marker = self._done_marker_path(frame_dir)
        if not done_marker.exists():
            return False
        return self._is_valid_output_file(output_path, len(frame_files))

    def load_model(self, device: str = 'cpu'):
        """加载模型"""
        from ultralytics import YOLO
        logger.info(f"[Step2] 加载模型: {self.model_path} -> {device}")
        logger.info(
            f"[Step2][ModelLoad][Before] pid={os.getpid()}, parent_pid={os.getppid()}, "
            f"process={mp.current_process().name}, cwd={Path.cwd()}, cuda={_get_cuda_runtime_context()}"
        )
        try:
            self.model = YOLO(str(self.model_path))
            logger.info(
                f"[Step2][ModelLoad][AfterInit] pid={os.getpid()}, parent_pid={os.getppid()}, "
                f"process={mp.current_process().name}, cuda={_get_cuda_runtime_context()}"
            )
            self.model.to(device)
            logger.info(
                f"[Step2][ModelLoad][AfterTo] pid={os.getpid()}, parent_pid={os.getppid()}, "
                f"process={mp.current_process().name}, target_device={device}, cuda={_get_cuda_runtime_context()}"
            )
        except Exception as exc:
            logger.error(f"[Step2][ModelLoad] 失败: device={device}, model={self.model_path}, error={exc}")
            logger.error(f"[Step2][ModelLoad] traceback:\n{traceback.format_exc()}")
            logger.error(f"[Step2][ModelLoad][ErrorContext] cuda={_get_cuda_runtime_context()}")
            raise

    def _resolve_device(self) -> str:
        configured = getattr(config, 'POSE_DEVICE', 'auto')
        if configured not in VALID_DEVICES:
            raise ValueError(f"无效的 POSE_DEVICE: {configured}")

        if configured in {'cpu', 'cuda'}:
            return configured

        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    def _resolve_mode(self, total_frames: int | None = None, device: str | None = None) -> str:
        mode = getattr(config, 'POSE_INFER_MODE', 'multiprocess')
        if mode not in VALID_INFER_MODES:
            raise ValueError(f"无效的 POSE_INFER_MODE: {mode}")

        resolved_device = device or self._resolve_device()
        if mode == 'multiprocess' and resolved_device == 'cuda':
            auto_threshold = max(0, int(getattr(config, 'POSE_AUTO_MULTIPROCESS_MIN_FRAMES', 0)))
            if auto_threshold <= 0 or (total_frames is not None and total_frames < auto_threshold):
                logger.info(
                    f"[Step2] 检测到单卡/小批量GPU场景，自动将模式从 multiprocess 调整为 single_gpu_batch | "
                    f"frames={total_frames}, threshold={auto_threshold}"
                )
                return 'single_gpu_batch'
        return mode

    def detect_frame(self, frame_path: Path) -> dict:
        """检测单帧"""
        results = self.model(frame_path, verbose=False)
        return _extract_result_record(results[0], self.keypoint_names)

    def _detect_batches_single_process(self, frame_files: list[Path], batch_size: int, device: str) -> list:
        """单进程批量推理，适合GPU大batch"""
        if self.model is None:
            self.load_model(device)

        all_results = []
        total_frames = len(frame_files)

        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_files = frame_files[batch_start:batch_end]
            results = self.model(batch_files, verbose=False)
            all_results.extend(_extract_result_record(result, self.keypoint_names) for result in results)

            processed = batch_end
            if processed % 100 == 0 or processed == total_frames:
                logger.info(f"  已检测 {processed}/{total_frames} 帧")

        return all_results

    def _detect_batches_with_cache(self, frame_dir: Path, missing_frames: list[Path], batch_size: int, num_workers: int, device: str, mode: str) -> None:
        """仅对缺失帧执行检测，并写入逐帧cache"""
        if not missing_frames:
            return

        total_missing = len(missing_frames)
        logger.info(f"[Step2] 需要补跑 {total_missing} 帧")
        _log_runtime_context(f"[Step2][CacheDetect] frame_dir={frame_dir.name}, mode={mode}, missing={total_missing}")

        if mode == 'multiprocess':
            batches = []
            for batch_start in range(0, total_missing, batch_size):
                batch_end = min(batch_start + batch_size, total_missing)
                batches.append(missing_frames[batch_start:batch_end])

            worker_args = [
                ([str(f) for f in batch], str(self.model_path), self.keypoint_names, device)
                for batch in batches
            ]

            processed = 0
            with mp.Pool(processes=num_workers) as pool:
                for batch, batch_results in zip(batches, pool.imap(_process_batch_worker, worker_args)):
                    for frame_path, result in zip(batch, batch_results):
                        self._save_cached_result_for_frame(
                            frame_dir,
                            frame_path.name,
                            result,
                        )
                    processed += len(batch_results)
                    if processed % 100 == 0 or processed == total_missing:
                        logger.info(f"  已补跑 {processed}/{total_missing} 帧")
            return

        if self.model is None:
            self.load_model(device)

        processed = 0
        for batch_start in range(0, total_missing, batch_size):
            batch_end = min(batch_start + batch_size, total_missing)
            batch_files = missing_frames[batch_start:batch_end]
            results = self.model(batch_files, verbose=False)
            batch_records = [_extract_result_record(result, self.keypoint_names) for result in results]
            for frame_path, result in zip(batch_files, batch_records):
                self._save_cached_result_for_frame(
                    frame_dir,
                    frame_path.name,
                    result,
                )
            processed += len(batch_records)
            if processed % 100 == 0 or processed == total_missing:
                logger.info(f"  已补跑 {processed}/{total_missing} 帧")

    def _rebuild_final_from_cache(self, frame_dir: Path, output_path: Path, frame_files: list[Path]) -> tuple[list[dict], list[str]]:
        """从逐帧cache重建最终pose结果"""
        all_results = []
        missing_names = []
        for frame_path in frame_files:
            cached = self._load_cached_result(self._frame_cache_path(frame_dir, frame_path.name))
            if cached is None:
                missing_names.append(frame_path.name)
                continue
            all_results.append(cached)

        if not missing_names:
            self._write_final_results(output_path, all_results)
            self._done_marker_path(frame_dir).write_text('done\n', encoding='utf-8')

        return all_results, missing_names

    def _clear_round_state(self, frame_dir: Path, output_path: Path) -> None:
        done_marker = self._done_marker_path(frame_dir)
        if done_marker.exists():
            done_marker.unlink()
        if output_path.exists():
            output_path.unlink()

    def detect_folder(self, frame_dir: Path, output_path: Path, force: bool = False, batch_size: int = None, num_workers: int = None) -> Path:
        """
        检测整个文件夹的帧

        Args:
            frame_dir: 帧文件目录
            output_path: 输出路径
            force: 是否强制重新检测
            batch_size: 批量大小，默认读配置
            num_workers: 并行worker数量，默认读配置
        """
        detect_start = time.perf_counter()
        frame_files, _, missing_frames = self._collect_frame_state(frame_dir)
        total_frames = len(frame_files)
        if total_frames == 0:
            raise ValueError(f"未找到可检测的帧文件: {frame_dir}")

        device = self._resolve_device()
        mode = self._resolve_mode(total_frames=total_frames, device=device)
        batch_size = batch_size or getattr(config, 'POSE_BATCH_SIZE', 32)
        num_workers = num_workers or getattr(config, 'POSE_NUM_WORKERS', min(mp.cpu_count(), 8))

        if mode == 'single_gpu_batch' and device != 'cuda':
            logger.warning(f"[Step2] single_gpu_batch 模式下未检测到CUDA，改为单进程 {device} 推理")

        if mode == 'multiprocess' and device == 'cuda':
            logger.info(f"[Step2] CUDA场景下优先单进程大batch推理以减少进程争用: batch_size={batch_size}")

        if force:
            self._clear_round_state(frame_dir, output_path)
            missing_frames = frame_files
            cached_results = [None] * total_frames

        if not force and self.is_round_complete(frame_dir, output_path):
            logger.info(f"[Step2] 跳过检测，已有完整结果: {output_path}")
            return output_path

        logger.info(
            f"[Step2][BatchPlan] frame_dir={frame_dir.name}, mode={mode}, device={device}, "
            f"batch_size={batch_size}, num_workers={num_workers}, total_missing={len(missing_frames)}, "
            f"cuda={_get_cuda_runtime_context()}"
        )
        logger.info(
            f"[Step2][DetectFolder][Runtime] pid={os.getpid()}, parent_pid={os.getppid()}, "
            f"process={mp.current_process().name}, frame_dir={frame_dir.name}, cuda={_get_cuda_runtime_context()}"
        )

        if not missing_frames and not self._is_valid_output_file(output_path, total_frames):
            logger.info("[Step2] 检测cache完整，开始重建最终pose_results.npz")

        try:
            self._detect_batches_with_cache(frame_dir, missing_frames, batch_size, num_workers, device, mode)
        except Exception as exc:
            logger.error(
                f"[Step2][DetectFolder][Error] frame_dir={frame_dir.name}, device={device}, mode={mode}, "
                f"batch_size={batch_size}, num_workers={num_workers}, error={exc}, cuda={_get_cuda_runtime_context()}"
            )
            logger.error(f"[Step2][DetectFolder][Error] traceback:\n{traceback.format_exc()}")
            raise
        all_results, still_missing = self._rebuild_final_from_cache(frame_dir, output_path, frame_files)

        if still_missing:
            preview = ', '.join(still_missing[:5])
            raise RuntimeError(f"Pose cache重建失败，仍缺少 {len(still_missing)} 帧: {preview}")

        total_elapsed = time.perf_counter() - detect_start
        total_batches = (total_frames + batch_size - 1) // batch_size if total_frames else 0
        avg_frame_s = total_elapsed / total_frames if total_frames else 0.0
        avg_batch_s = total_elapsed / total_batches if total_batches else 0.0
        logger.info(f"[Step2] 检测完成: {len(all_results)} 帧 -> {output_path}")
        logger.info(
            f"[Timing] Step2 detect_folder: {total_elapsed:.2f}s | "
            f"frames={total_frames}, batches={total_batches}, workers={num_workers}, "
            f"mode={mode}, device={device}, avg_frame_s={avg_frame_s:.4f}, "
            f"avg_batch_s={avg_batch_s:.4f}, cache_hits={total_frames - len(missing_frames)}, "
            f"cache_misses={len(missing_frames)}"
        )
        return output_path


def run(video_name: str = config.VIDEO_NAME, force: bool = False) -> Path:
    """执行检测"""
    frame_dir = config.FRAME_OUTPUT_DIR / video_name
    detection_path = Path(__file__).parent.parent / f"{video_name}_detection.npz"

    detector = PoseDetector(config.MODEL_PATH)
    return detector.detect_folder(frame_dir, detection_path, force=force)


if __name__ == '__main__':
    run()
