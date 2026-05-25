"""主流程脚本 - 跌倒检测训练数据生成"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from pipeline_orchestrator import run_batch_pipeline, run_pipeline
from pipeline_round_processing import clean_no_person_frames, generate_npz_and_details, process_single_round
from utils.pipeline_features import compute_features_batch, extract_samples_parallel, set_pipeline_runtime
from utils.pipeline_helpers import load_pose_results


# 全局标志：是否使用内部并行（当外部run_parallel.py提供进程隔离时设为False）
# 通过环境变量设置，避免进程间传递变量的复杂性
_DISABLE_INTERNAL_PARALLELISM = os.environ.get('DISABLE_INTERNAL_PARALLEL', '0') == '1'


# 配置日志 - 只写文件，不输出到终端
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


set_pipeline_runtime(logger, _DISABLE_INTERNAL_PARALLELISM)


def load_frame_timestamps(round_dir: Path):
    timestamps_file = round_dir / "frame_timestamps.json"
    if not timestamps_file.exists():
        return None

    with open(timestamps_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        timestamps = data.get('timestamps')
        if timestamps is None:
            return None
    else:
        timestamps = data

    timestamps = [float(ts) for ts in timestamps]
    if not timestamps:
        return timestamps

    max_timestamp = max(timestamps)
    if max_timestamp > 1000:
        logger.info(f"[时间戳] 检测到毫秒时间戳，自动转换为秒: {timestamps_file}")
        return [ts / 1000.0 for ts in timestamps]

    return timestamps


if __name__ == "__main__":
    import config
    from steps.step1_video_split import cleanup_active_ffmpeg_extractors, ensure_ffmpeg_cleanup_hooks

    ensure_ffmpeg_cleanup_hooks()

    parser = argparse.ArgumentParser(description="跌倒检测训练数据生成流程")
    parser.add_argument("--video-name", type=str, help="视频名称（单视频模式）")
    parser.add_argument("--batch", action="store_true", help="批量处理所有视频")
    parser.add_argument("--max-videos", type=int, default=None, help="最多处理视频数")
    parser.add_argument("--fps", type=float, default=config.VIDEO_FPS, help="视频帧率")
    parser.add_argument("--rounds", type=int, default=6, help="切帧轮数")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--recompute", action="store_true", help="强制重新计算（包括切帧和pose检测）")
    mode_group.add_argument("--reuse-frames", action="store_true", help="复用现有切帧和pose结果，只重新计算特征")
    mode_group.add_argument("--reuse-split-frames", action="store_true", help="复用现有切帧结果，但重新执行pose检测和后续流程")
    parser.add_argument("--workers", type=int, default=1, help="并行视频处理数 (A100建议4-8)")
    parser.add_argument("--offset", type=int, default=0, help="视频列表起始偏移量")
    parser.add_argument("--watch", action="store_true", help="持续轮询目录，自动处理新上传且已稳定的视频")
    parser.add_argument("--poll-interval", type=int, default=30, help="watch模式下每次扫描间隔秒数")
    parser.add_argument("--stable-seconds", type=int, default=60, help="视频文件保持不变多少秒后才视为上传完成")
    parser.add_argument("--idle-exit-seconds", type=int, default=None, help="watch模式空闲多久后自动退出")
    parser.add_argument("--retry-cooldown-seconds", type=int, default=60, help="watch模式失败视频的重试冷却时间")
    parser.add_argument("--resume", action="store_true",
                        help="启用严格断点续跑：整视频结果完整则直接跳过，否则仅从缺失的切帧/pose/round特征继续补跑")
    parser.add_argument("--disable-internal-parallel", action="store_true",
                        help="禁用内部并行（当外部已提供进程隔离时使用）")

    args = parser.parse_args()

    logger.info(
        "[CLI] internal_parallel_disabled=%s env_DISABLE_INTERNAL_PARALLEL=%s",
        _DISABLE_INTERNAL_PARALLELISM,
        os.environ.get('DISABLE_INTERNAL_PARALLEL', '<unset>'),
    )

    # 设置全局并行标志
    if args.disable_internal_parallel:
        logger.info("[配置] 内部并行已禁用")

    try:
        if args.batch or args.video_name is None:
            logger.info("[CLI] resolved_mode=batch")
            # 批量模式 - 使用config中的视频目录
            results = run_batch_pipeline(
                video_dir=config.VIDEO_DIR,
                annotation_dir=config.ANNOTATION_DIR,
                frame_output_dir=config.FRAME_OUTPUT_DIR,
                model_path=config.MODEL_PATH,
                fps=args.fps,
                num_rounds=args.rounds,
                resume=args.resume,
                max_videos=args.max_videos,
                num_workers=args.workers,
                offset=args.offset,
                skip_pose_detection=args.reuse_frames,
                skip_frame_extraction=args.reuse_frames or args.reuse_split_frames,
                watch=args.watch,
                poll_interval=args.poll_interval,
                stable_seconds=args.stable_seconds,
                idle_exit_seconds=args.idle_exit_seconds,
                retry_cooldown_seconds=args.retry_cooldown_seconds,
            )
        else:
            logger.info("[CLI] resolved_mode=single_video")
            # 单视频模式
            results = run_pipeline(
                video_name=args.video_name,
                video_dir=config.VIDEO_DIR,
                annotation_dir=config.ANNOTATION_DIR,
                frame_output_dir=config.FRAME_OUTPUT_DIR,
                model_path=config.MODEL_PATH,
                fps=args.fps,
                num_rounds=args.rounds,
                resume=args.resume,
                skip_pose_detection=args.reuse_frames,
                skip_frame_extraction=args.reuse_frames or args.reuse_split_frames
            )
    except KeyboardInterrupt:
        cleanup_active_ffmpeg_extractors()
        raise
    finally:
        cleanup_active_ffmpeg_extractors()

    logger.info("\n结果汇总:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
