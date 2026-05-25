"""Pipeline orchestration helpers for fall-detection data generation."""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.pipeline_helpers import (
    _collect_ready_videos,
    _log_timing,
    find_video_path,
    get_runtime_snapshot,
    is_pose_round_complete,
    is_round_feature_complete,
    is_video_results_bundle_complete,
    iter_video_files,
    resolve_annotation_state,
)


def _get_step2_runtime_snapshot() -> dict:
    snapshot = get_runtime_snapshot()
    snapshot['pid'] = os.getpid()
    snapshot['parent_pid'] = os.getppid()
    snapshot['disable_internal_parallelism'] = _disable_internal_parallelism()
    snapshot['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
    snapshot['pytorch_cuda_alloc_conf'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')
    try:
        import torch
        if torch.cuda.is_available():
            snapshot['cuda_current_device'] = torch.cuda.current_device()
            snapshot['cuda_device_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            snapshot['cuda_current_device'] = None
            snapshot['cuda_device_names'] = []
    except Exception as exc:
        snapshot['cuda_current_device'] = None
        snapshot['cuda_device_names'] = [f'error: {exc}']
    return snapshot


def _mp():
    module = sys.modules.get('main_pipeline') or sys.modules.get('__main__')
    if module is None:
        import main_pipeline as module  # type: ignore
    return module


def _logger():
    return _mp().logger


def _disable_internal_parallelism() -> bool:
    return _mp()._DISABLE_INTERNAL_PARALLELISM


def detect_single_round(round_dir: Path, round_idx: int, model_path: Path, use_resume: bool) -> tuple:
    """检测单个round的pose（用于并行）"""
    logger = _logger()
    pose_results_path = round_dir / "pose_results.npz"
    logger.info(
        f"[Batch][Step2][RoundStart] round={round_idx}, round_dir={round_dir}, pid={os.getpid()}, "
        f"parent_pid={os.getppid()}, use_resume={use_resume}, runtime={_get_step2_runtime_snapshot()}"
    )
    if not round_dir.exists():
        logger.info(f"[Batch][Step2][RoundSkip] round={round_idx}, reason=round_dir_missing")
        return round_idx, True
    from steps.step2_pose_detection import PoseDetector
    detector = PoseDetector(model_path)
    if use_resume and detector.is_round_complete(round_dir, pose_results_path):
        logger.info(f"[Batch][Step2][RoundSkip] round={round_idx}, reason=pose_complete")
        return round_idx, True
    try:
        detector.detect_folder(round_dir, pose_results_path, force=not use_resume)
        logger.info(f"[Batch][Step2][RoundDone] round={round_idx}, pid={os.getpid()}")
    except Exception as exc:
        logger.error(
            f"[Batch][Step2][RoundError] round={round_idx}, round_dir={round_dir}, pid={os.getpid()}, error={exc}, runtime={_get_step2_runtime_snapshot()}"
        )
        logger.error(f"[Batch][Step2][RoundError] traceback:\n{traceback.format_exc()}")
        raise
    return round_idx, False


def _detect_round_wrapper(args):
    """包装detect_single_round用于并行"""
    round_dir, round_idx, model_path, use_resume = args
    return detect_single_round(round_dir, round_idx, model_path, use_resume)


def run_pipeline(
    video_name: str,
    video_dir: Path,
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float = 20.0,
    num_rounds: int = 6,
    skip_existing: bool = True,
    skip_pose_detection: bool = True,
    skip_frame_extraction: bool = False,
    resume: bool = False
) -> dict:
    """运行完整流程

    Args:
        skip_pose_detection: True则跳过pose检测，直接使用现有的pose_results.npz
        skip_frame_extraction: True则复用现有切帧结果，但仍可重新执行pose检测
    """
    logger = _logger()
    pipeline_start = time.perf_counter()
    from steps.step2_pose_detection import PoseDetector

    video_path = find_video_path(video_dir, video_name)
    annotation_path, annotation_state, intervals_ms = resolve_annotation_state(annotation_dir, video_name)
    video_frame_dir = frame_output_dir / video_name
    from steps.step1_video_split import VideoFrameExtractor
    frame_extractor = VideoFrameExtractor(video_path, video_frame_dir) if video_path else None

    logger.info(f"视频路径: {video_path}")
    logger.info(f"帧输出目录: {video_frame_dir}")
    logger.info(f"标注路径: {annotation_path}")
    logger.info(
        f"[Stats] video={video_name}, rounds={num_rounds}, skip_existing={skip_existing}, "
        f"skip_pose_detection={skip_pose_detection}, skip_frame_extraction={skip_frame_extraction}"
    )

    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_name}（在 {video_dir} 中递归查找）")
    if annotation_state == "discard":
        logger.info(f"[Skip] 视频 {video_name} 标记为 discard，跳过全部流程")
        return {}

    if resume and is_video_results_bundle_complete(video_frame_dir):
        logger.info(f"[Resume][video] 完整结果已存在，整视频跳过: {video_name}")
        final_output = video_frame_dir / "results" / "all_features.csv"
        results = {'final': str(final_output)}
        _log_timing("Single video pipeline total", pipeline_start, video=video_name, total_samples=0, fall_samples=0)
        return results

    logger.info(f"  跌倒区间: {[(s / 1000, e / 1000) for s, e in intervals_ms]}")
    logger.info(
        "[ReuseMode][single] video=%s reuse_frames=%s reuse_split_frames=%s resume=%s",
        video_name,
        skip_pose_detection,
        skip_frame_extraction and skip_pose_detection,
        resume,
    )
    logger.info(
        "[ReuseMode][single] 语义: reuse_frames=复用切帧+优先复用pose(缺失则补跑), reuse_split_frames=仅复用切帧并重跑pose"
    )

    results = {}
    all_features_dfs = []
    all_samples = []

    round1_dir = video_frame_dir / "round_1"
    round1_exists = round1_dir.exists()
    round1_complete = frame_extractor.is_round_complete(round1_dir) if frame_extractor and round1_exists else False
    need_frame_extraction = (
        frame_extractor is None or
        not skip_frame_extraction or
        not round1_complete
    )
    logger.info(
        "[ReuseDebug][single] video=%s resume=%s skip_frame_extraction=%s skip_pose_detection=%s round1_dir=%s round1_exists=%s round1_complete=%s need_frame_extraction=%s",
        video_name,
        resume,
        skip_frame_extraction,
        skip_pose_detection,
        round1_dir,
        round1_exists,
        round1_complete,
        need_frame_extraction,
    )

    if need_frame_extraction:
        logger.info("[ReuseDebug][single] entering Step1 because frame_extractor_none=%s skip_frame_extraction=%s round1_complete=%s", frame_extractor is None, skip_frame_extraction, round1_complete)
        logger.info("开始切帧...")
        logger.info(f"[Runtime][Before Step1] {get_runtime_snapshot()}")
        from steps.step1_video_split import extract_random_intervals
        step1_start = time.perf_counter()
        extract_random_intervals(video_path, video_frame_dir, force=not skip_frame_extraction, num_rounds=num_rounds)
        _log_timing("Step1 extract random intervals", step1_start, video=video_name, rounds=num_rounds)
        logger.info(f"[Runtime][After Step1] {get_runtime_snapshot()}")
    else:
        logger.info("跳过切帧，复用现有完整 round_* 结果")

    step2_start = None
    detector = None
    output_dir = video_frame_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    for round_idx in range(1, num_rounds + 1):
        round_dir = video_frame_dir / f"round_{round_idx}"
        pose_results_path = round_dir / "pose_results.npz"
        if not round_dir.exists():
            logger.info(f"[Round {round_idx}] 帧目录不存在，跳过Pose检测")
            continue

        if resume and is_round_feature_complete(output_dir, round_idx):
            logger.info(f"[Resume][Round {round_idx}] 复用已有特征CSV，跳过Pose和后处理")
            continue

        pose_complete = is_pose_round_complete(round_dir)
        need_pose_detection = (
            need_frame_extraction
            or not pose_complete
            or (resume and not pose_complete)
        )

        if not need_pose_detection:
            if resume:
                logger.info(f"[Resume][Round {round_idx}] 特征缺失但Pose完整，复用Pose并继续后处理")
            else:
                logger.info(f"[Round {round_idx}] 跳过Pose检测，已有完整结果")
            continue

        if detector is None:
            logger.info("开始Pose检测...")
            detector = PoseDetector(model_path)
            step2_start = time.perf_counter()

        reason = "[Resume][Round {round_idx}] Pose缺失，补跑Pose后继续后处理" if resume else (
            "reuse模式补齐缺失pose" if skip_pose_detection else "正常执行pose检测"
        )
        logger.info(reason.format(round_idx=round_idx))
        pose_start = time.perf_counter()
        detector.detect_folder(round_dir, pose_results_path, force=True)
        _log_timing(f"Step2 pose detection round {round_idx}", pose_start, round=round_idx)

    if step2_start is not None:
        _log_timing("Step2 pose detection total", step2_start, video=video_name, rounds=num_rounds)
        logger.info("切帧和Pose检测阶段完成")

    for round_idx in tqdm(range(1, num_rounds + 1), desc="  Round", leave=False):
        round_start = time.perf_counter()
        round_dir = video_frame_dir / f"round_{round_idx}"
        pose_results_path = round_dir / "pose_results.npz"

        if not round_dir.exists():
            logger.info(f"\n[Round {round_idx}] 目录不存在，跳过")
            continue

        if resume and is_round_feature_complete(output_dir, round_idx):
            logger.info(f"[Resume][Round {round_idx}] 复用已有特征CSV，跳过Pose和后处理")
            result_path = output_dir / f"round_{round_idx}_output" / f"round_{round_idx}_features.csv"
            df = pd.read_csv(result_path)
            if len(df) > 0:
                all_features_dfs.append(df)
            results[round_idx] = str(result_path)
            _log_timing(f"Round {round_idx} pipeline post-processing", round_start, samples=0)
            continue

        pose_complete = is_pose_round_complete(round_dir)
        if not pose_complete:
            if skip_pose_detection:
                logger.info(f"\n[Round {round_idx}] Pose结果不完整且当前配置跳过Pose检测，跳过此round")
                continue
            logger.info(f"[Resume][Round {round_idx}] Pose缺失，补跑Pose后继续后处理" if resume else f"\n[Round {round_idx}] 开始补跑Pose检测...")
            detector = PoseDetector(model_path)
            pose_start = time.perf_counter()
            detector.detect_folder(round_dir, pose_results_path, force=True)
            _log_timing(f"Step2 pose detection round {round_idx}", pose_start, round=round_idx)
            pose_complete = is_pose_round_complete(round_dir)
        elif resume:
            logger.info(f"[Resume][Round {round_idx}] 特征缺失但Pose完整，复用Pose并继续后处理")
        else:
            logger.info(f"\n[Round {round_idx}] 跳过Pose检测，已有完整结果")

        if not pose_complete:
            logger.info(f"[Round {round_idx}] Pose结果仍不完整，跳过后处理")
            continue

        result, samples = _mp().process_single_round(
            round_dir,
            round_idx,
            pose_results_path,
            annotation_path,
            output_dir,
            video_path=video_path,
        )

        if result and Path(result).exists():
            df = pd.read_csv(result)
            if len(df) > 0:
                all_features_dfs.append(df)
            results[round_idx] = result

        if samples:
            all_samples.extend(samples)

        _log_timing(f"Round {round_idx} pipeline post-processing", round_start, samples=len(samples))

    output_dir = video_frame_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_rounds = [
        round_idx for round_idx in range(1, num_rounds + 1)
        if (video_frame_dir / f"round_{round_idx}").exists()
    ]

    if resume and existing_rounds and all(
        is_round_feature_complete(output_dir, round_idx)
        for round_idx in existing_rounds
    ) and is_video_results_bundle_complete(video_frame_dir):
        logger.info("跳过视频级汇总，复用现有完整 results 产物")
        final_output = output_dir / "all_features.csv"
        results['final'] = str(final_output)
    elif all_features_dfs:
        merge_start = time.perf_counter()
        all_features_df = pd.concat(all_features_dfs, ignore_index=True)
        final_output = video_frame_dir / "results" / "all_features.csv"
        all_features_df.to_csv(final_output, index=False)
        _log_timing("Merge round feature CSVs", merge_start, rows=len(all_features_df), output=final_output.name)

        logger.info("\n" + "=" * 50)
        logger.info(f"流程完成! 最终特征文件: {final_output}")
        logger.info(f"总样本数: {len(all_features_df)}")
        if 'label' in all_features_df.columns:
            logger.info(f"跌倒样本数: {(all_features_df['label'] > 0).sum():.0f}")
        logger.info("=" * 50)

        results['final'] = str(final_output)

        if all_samples:
            npz_start = time.perf_counter()
            _mp().generate_npz_and_details(video_frame_dir, all_samples, intervals_ms)
            _log_timing("Generate NPZ and details", npz_start, samples=len(all_samples))

    total_samples = len(all_samples)
    fall_samples = sum(1 for s in all_samples if s.get('label', 0) == 1)
    _log_timing("Single video pipeline total", pipeline_start, video=video_name, total_samples=total_samples, fall_samples=fall_samples)
    return results


def _process_video_wrapper(args):
    """包装process_single_video用于并行"""
    video_path, video_name, video_frame_dir, annotation_dir, frame_output_dir, model_path, fps, num_rounds, resume, skip_pose_detection, skip_frame_extraction = args
    return process_single_video(
        video_path, video_name, video_frame_dir,
        annotation_dir, frame_output_dir, model_path,
        fps, num_rounds, resume, skip_pose_detection, skip_frame_extraction,
    )


def process_single_video(
    video_path: Path,
    video_name: str,
    video_frame_dir: Path,
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float,
    num_rounds: int,
    resume: bool,
    skip_pose_detection: bool = False,
    skip_frame_extraction: bool = False,
) -> dict:
    """处理单个视频 - 供并行调用"""
    logger = _logger()
    video_start = time.perf_counter()
    result = {
        'video_name': video_name,
        'features_df': None,
        'samples': [],
        'fall': 0,
        'nofall': 0,
        'error': None,
    }

    try:
        annotation_path, annotation_state, intervals_ms = resolve_annotation_state(annotation_dir, video_name)
        if annotation_state == "discard":
            logger.info(f"[Batch] 视频标记为 discard，跳过全部流程: {video_name}")
            return result

        if resume and is_video_results_bundle_complete(video_frame_dir):
            logger.info(f"[Resume][video] 完整结果已存在，整视频跳过: {video_name}")
            final_output = video_frame_dir / "results" / "all_features.csv"
            video_df = pd.read_csv(final_output)
            result['features_df'] = video_df
            if 'label' in video_df.columns:
                result['fall'] = int((video_df['label'] > 0).sum())
                result['nofall'] = int((video_df['label'] == 0).sum())
            _log_timing(
                "Batch single video total",
                video_start,
                video=video_name,
                samples=len(video_df),
                fall=result['fall'],
                nofall=result['nofall'],
            )
            return result

        logger.info(
            f"[Stats] batch video={video_name}, rounds={num_rounds}, resume={resume}, "
            f"skip_pose_detection={skip_pose_detection}, skip_frame_extraction={skip_frame_extraction}, annotation_state={annotation_state}"
        )

        logger.info(
            "[ReuseMode][batch] video=%s reuse_frames=%s reuse_split_frames=%s resume=%s",
            video_name,
            skip_pose_detection,
            skip_frame_extraction and skip_pose_detection,
            resume,
        )
        logger.info(
            "[ReuseMode][batch] 语义: reuse_frames=复用切帧+优先复用pose(缺失则补跑), reuse_split_frames=仅复用切帧并重跑pose"
        )

        from steps.step1_video_split import VideoFrameExtractor
        frame_extractor = VideoFrameExtractor(video_path, video_frame_dir)

        round1_dir = video_frame_dir / "round_1"
        round1_exists = round1_dir.exists()
        round1_complete = frame_extractor.is_round_complete(round1_dir) if round1_exists else False
        need_frame_extraction = (
            not skip_frame_extraction or not round1_complete
        )
        logger.info(
            "[ReuseDebug][batch] video=%s resume=%s skip_pose_detection=%s skip_frame_extraction=%s round1_dir=%s round1_exists=%s round1_complete=%s need_frame_extraction=%s",
            video_name,
            resume,
            skip_pose_detection,
            skip_frame_extraction,
            round1_dir,
            round1_exists,
            round1_complete,
            need_frame_extraction,
        )

        if need_frame_extraction:
            logger.info("[ReuseDebug][batch] entering Step1 because skip_frame_extraction=%s round1_complete=%s", skip_frame_extraction, round1_complete)
            logger.info(f"[Runtime][Before Batch Step1] {get_runtime_snapshot()}")
            from steps.step1_video_split import extract_random_intervals
            step1_start = time.perf_counter()
            extract_random_intervals(video_path, video_frame_dir, force=not skip_frame_extraction, num_rounds=num_rounds)
            _log_timing("Batch Step1 extract random intervals", step1_start, video=video_name, rounds=num_rounds)
            logger.info(f"[Runtime][After Batch Step1] {get_runtime_snapshot()}")
        else:
            logger.info(f"[Batch] 跳过切帧，复用现有完整 round_* 结果: {video_name}")

        output_dir = video_frame_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_rounds = [
            round_idx for round_idx in range(1, num_rounds + 1)
            if (video_frame_dir / f"round_{round_idx}").exists()
        ]

        if resume and existing_rounds and all(
            is_round_feature_complete(output_dir, round_idx)
            for round_idx in existing_rounds
        ) and is_video_results_bundle_complete(video_frame_dir):
            _log_timing(
                "Batch single video total",
                video_start,
                video=video_name,
                samples=len(video_df),
                fall=result['fall'],
                nofall=result['nofall'],
            )
            return result

        step2_start = time.perf_counter()
        rounds_skipping_features = []
        rounds_requiring_pose = []
        rounds_reusing_pose = []
        for round_idx in range(1, num_rounds + 1):
            round_dir = video_frame_dir / f"round_{round_idx}"
            if not round_dir.exists():
                continue
            if resume and is_round_feature_complete(output_dir, round_idx):
                rounds_skipping_features.append(round_idx)
                continue
            if is_pose_round_complete(round_dir):
                rounds_reusing_pose.append(round_idx)
            else:
                rounds_requiring_pose.append(round_idx)

        if rounds_skipping_features:
            logger.info(f"[Batch][Resume] 复用已有特征CSV，跳过完整round: {video_name} | rounds={rounds_skipping_features}")

        if rounds_reusing_pose:
            logger.info(f"[Batch][Step2] 复用已有Pose结果: {video_name} | rounds={rounds_reusing_pose}")

        if rounds_requiring_pose:
            logger.info(f"[Runtime][Before Batch Step2] {get_runtime_snapshot()}")
            if not _disable_internal_parallelism() and rounds_requiring_pose:
                logger.warning(
                    f"[Batch][Step2] 检测到GPU场景下的round级内部并行，自动降级为串行执行以避免fork-CUDA冲突: "
                    f"video={video_name}, target_rounds={rounds_requiring_pose}, runtime={_get_step2_runtime_snapshot()}"
                )
            action = "[Resume] Pose缺失，补跑Pose后继续后处理" if resume else ("reuse模式补跑缺失Pose" if skip_pose_detection else "按当前配置执行Pose检测")
            args_list = [(video_frame_dir / f"round_{i}", i, model_path, resume) for i in rounds_requiring_pose]
            logger.info(
                f"[Batch][Step2] video={video_name}, action={action}, internal_parallel_disabled={_disable_internal_parallelism()}, "
                f"round_workers={len(args_list)}, pid={os.getpid()}, target_rounds={rounds_requiring_pose}, runtime={_get_step2_runtime_snapshot()}"
            )
            for args in args_list:
                _detect_round_wrapper(args)
            _log_timing("Batch Step2 pose detection total", step2_start, video=video_name, rounds=len(rounds_requiring_pose))
            logger.info(f"[Runtime][After Batch Step2] {get_runtime_snapshot()}")
        else:
            logger.info(f"[Batch][Step2] 所有round均已有完整Pose结果: {video_name}")

        video_features_dfs = []
        video_samples = []

        for round_idx in range(1, num_rounds + 1):
            round_start = time.perf_counter()
            round_dir = video_frame_dir / f"round_{round_idx}"
            pose_results_path = round_dir / "pose_results.npz"
            output_dir = video_frame_dir / "results"
            output_dir.mkdir(parents=True, exist_ok=True)

            if not round_dir.exists():
                continue

            if resume and is_round_feature_complete(output_dir, round_idx):
                logger.info(f"[Resume][Round {round_idx}] 复用已有特征CSV，跳过Pose和后处理")
                feat_result = output_dir / f"round_{round_idx}_output" / f"round_{round_idx}_features.csv"
                df = pd.read_csv(feat_result)
                if len(df) > 0:
                    video_features_dfs.append(df)
                _log_timing(f"Batch round {round_idx} post-processing", round_start, video=video_name, samples=0)
                continue

            pose_complete = is_pose_round_complete(round_dir)
            if not pose_complete:
                if skip_pose_detection:
                    logger.info(f"[Batch][Round {round_idx}] Pose结果不完整且当前配置跳过Pose检测，跳过后处理")
                    continue
                logger.info(f"[Resume][Round {round_idx}] Pose缺失，补跑Pose后继续后处理" if resume else f"[Batch][Round {round_idx}] Pose结果不完整，开始补跑")
                round_detector = PoseDetector(model_path)
                pose_start = time.perf_counter()
                round_detector.detect_folder(round_dir, pose_results_path, force=False)
                _log_timing(f"Batch Step2 pose detection round {round_idx}", pose_start, video=video_name, round=round_idx)
                pose_complete = is_pose_round_complete(round_dir)
            elif resume:
                logger.info(f"[Resume][Round {round_idx}] 特征缺失但Pose完整，复用Pose并继续后处理")

            if not pose_complete:
                logger.info(f"[Batch][Round {round_idx}] Pose结果仍不完整，跳过后处理")
                continue

            feat_result, samples = _mp().process_single_round(
                round_dir,
                round_idx,
                pose_results_path,
                annotation_path,
                output_dir,
                video_path=video_path,
            )

            if feat_result and Path(feat_result).exists():
                df = pd.read_csv(feat_result)
                if len(df) > 0:
                    video_features_dfs.append(df)
                    video_samples.extend(samples)

            _log_timing(f"Batch round {round_idx} post-processing", round_start, video=video_name, samples=len(samples))

        if video_features_dfs:
            video_df = pd.concat(video_features_dfs, ignore_index=True)
            result['features_df'] = video_df
            result['samples'] = video_samples
            if 'label' in video_df.columns:
                result['fall'] = int((video_df['label'] > 0).sum())
                result['nofall'] = int((video_df['label'] == 0).sum())

        _log_timing(
            "Batch single video total",
            video_start,
            video=video_name,
            samples=len(result['samples']),
            fall=result['fall'],
            nofall=result['nofall'],
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[Batch] 视频处理失败: {video_name} | {e}")
        result['error'] = str(e)

    return result


def _process_video_batch(
    video_files: list[Path],
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float,
    num_rounds: int,
    resume: bool,
    num_workers: int,
    skip_pose_detection: bool,
    skip_frame_extraction: bool,
) -> dict:
    """处理一批已选中的视频"""
    logger = _logger()
    all_video_features = []
    all_video_samples = []
    total_fall = 0
    total_nofall = 0
    errors = []

    video_args = [
        (vp, vp.stem, frame_output_dir / vp.stem, annotation_dir, frame_output_dir, model_path, fps, num_rounds, resume, skip_pose_detection, skip_frame_extraction)
        for vp in video_files
    ]

    with tqdm(total=len(video_files), desc="处理视频") as pbar:
        if num_workers <= 1:
            logger.info("[Batch] num_workers<=1，使用主进程顺序处理视频，避免fork-CUDA冲突")
            for va in video_args:
                res = _process_video_wrapper(va)
                pbar.update(1)

                if res['error']:
                    error_text = f"{res['video_name']}: {res['error']}"
                    errors.append(error_text)
                    logger.error(f"[Batch] {error_text}")

                if res['features_df'] is not None:
                    all_video_features.append(res['features_df'])
                    all_video_samples.extend(res['samples'])
                    total_fall += res['fall']
                    total_nofall += res['nofall']

                logger.info(f"  {res['video_name']} 完成: {len(res['samples'])} 样本")
                print(f"[PROGRESS] video_completed: {res['video_name']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_video_wrapper, va): va for va in video_args}
                for future in as_completed(futures):
                    res = future.result()
                    pbar.update(1)

                    if res['error']:
                        error_text = f"{res['video_name']}: {res['error']}"
                        errors.append(error_text)
                        logger.error(f"[Batch] {error_text}")

                    if res['features_df'] is not None:
                        all_video_features.append(res['features_df'])
                        all_video_samples.extend(res['samples'])
                        total_fall += res['fall']
                        total_nofall += res['nofall']

                    logger.info(f"  {res['video_name']} 完成: {len(res['samples'])} 样本")
                    print(f"[PROGRESS] video_completed: {res['video_name']}", flush=True)

    return {
        'all_video_features': all_video_features,
        'all_video_samples': all_video_samples,
        'total_fall': total_fall,
        'total_nofall': total_nofall,
        'errors': errors,
    }


def run_batch_pipeline(
    video_dir: Path,
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float = 20.0,
    num_rounds: int = 6,
    resume: bool = False,
    max_videos: int = None,
    num_workers: int = 1,
    offset: int = 0,
    skip_pose_detection: bool = False,
    skip_frame_extraction: bool = False,
    watch: bool = False,
    poll_interval: int = 30,
    stable_seconds: int = 60,
    idle_exit_seconds: int | None = None,
    retry_cooldown_seconds: int = 60,
) -> dict:
    """批量运行完整流程 - 支持一次性处理和轮询增量处理"""
    logger = _logger()
    batch_start = time.perf_counter()

    logger.info(
        f"[Stats] batch resume={resume}, rounds={num_rounds}, "
        f"skip_pose_detection={skip_pose_detection}, skip_frame_extraction={skip_frame_extraction}, offset={offset}, max_videos={max_videos}, "
        f"watch={watch}, poll_interval={poll_interval}, stable_seconds={stable_seconds}, "
        f"idle_exit_seconds={idle_exit_seconds}, retry_cooldown_seconds={retry_cooldown_seconds}"
    )

    all_video_features = []
    all_video_samples = []
    total_fall = 0
    total_nofall = 0
    errors = []

    if not watch:
        video_files = iter_video_files(video_dir)
        if max_videos:
            video_files = video_files[offset:offset + max_videos]
        elif offset:
            video_files = video_files[offset:]

        logger.info(f"找到 {len(video_files)} 个视频，启用了 {num_workers} 个并行worker")

        batch_result = _process_video_batch(
            video_files=video_files,
            annotation_dir=annotation_dir,
            frame_output_dir=frame_output_dir,
            model_path=model_path,
            fps=fps,
            num_rounds=num_rounds,
            resume=resume,
            num_workers=num_workers,
            skip_pose_detection=skip_pose_detection,
            skip_frame_extraction=skip_frame_extraction,
        )
        all_video_features.extend(batch_result['all_video_features'])
        all_video_samples.extend(batch_result['all_video_samples'])
        total_fall += batch_result['total_fall']
        total_nofall += batch_result['total_nofall']
        errors.extend(batch_result['errors'])
    else:
        logger.info("启动连续监控模式，等待新视频上传完成后自动处理")
        file_snapshots: dict[str, tuple[int, float]] = {}
        cooldown_until: dict[str, float] = {}
        processed_in_watch = set()
        idle_since = time.time()

        while True:
            ready_videos, file_snapshots = _collect_ready_videos(
                video_dir=video_dir,
                frame_output_dir=frame_output_dir,
                resume=resume,
                file_snapshots=file_snapshots,
                stable_seconds=stable_seconds,
                cooldown_until=cooldown_until,
            )

            if offset:
                ready_videos = ready_videos[offset:]
            if max_videos:
                remaining = max(0, max_videos - len(processed_in_watch))
                ready_videos = ready_videos[:remaining]

            ready_videos = [vp for vp in ready_videos if str(vp.resolve()) not in processed_in_watch]

            if ready_videos:
                idle_since = time.time()
                logger.info(f"本轮发现 {len(ready_videos)} 个可处理视频")
                batch_result = _process_video_batch(
                    video_files=ready_videos,
                    annotation_dir=annotation_dir,
                    frame_output_dir=frame_output_dir,
                    model_path=model_path,
                    fps=fps,
                    num_rounds=num_rounds,
                    resume=resume,
                    num_workers=num_workers,
                    skip_pose_detection=skip_pose_detection,
                    skip_frame_extraction=skip_frame_extraction,
                )
                all_video_features.extend(batch_result['all_video_features'])
                all_video_samples.extend(batch_result['all_video_samples'])
                total_fall += batch_result['total_fall']
                total_nofall += batch_result['total_nofall']
                errors.extend(batch_result['errors'])

                failed_names = {error.split(':', 1)[0] for error in batch_result['errors']}
                now = time.time()
                for video_path in ready_videos:
                    video_key = str(video_path.resolve())
                    if video_path.stem in failed_names:
                        cooldown_until[video_key] = now + retry_cooldown_seconds
                    else:
                        processed_in_watch.add(video_key)

                if max_videos and len(processed_in_watch) >= max_videos:
                    logger.info(f"已达到 max_videos={max_videos}，结束连续监控")
                    break
            else:
                if idle_exit_seconds is not None and (time.time() - idle_since) >= idle_exit_seconds:
                    logger.info(f"空闲超过 {idle_exit_seconds} 秒，结束连续监控")
                    break
                logger.info(f"当前没有就绪视频，{poll_interval} 秒后重试")
                time.sleep(poll_interval)

    if all_video_features:
        merge_start = time.perf_counter()
        final_df = pd.concat(all_video_features, ignore_index=True)
        results_dir = frame_output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        final_output = results_dir / "all_features.csv"
        final_df.to_csv(final_output, index=False)
        _log_timing("Batch merge all features", merge_start, rows=len(final_df), output=final_output.name)

        logger.info(f"\n{'='*60}")
        logger.info("批量处理完成!")
        logger.info(f"总样本数: {len(final_df)}")
        logger.info(f"跌倒样本数: {total_fall}")
        logger.info(f"非跌倒样本数: {total_nofall}")
        logger.info(f"最终特征文件: {final_output}")
        if errors:
            logger.info(f"失败视频: {len(errors)}")
            for error in errors:
                logger.error(f"  {error}")
        logger.info(f"{'='*60}")

        if all_video_samples:
            npz_start = time.perf_counter()
            _mp().generate_npz_and_details(frame_output_dir, all_video_samples, [])
            _log_timing("Batch generate NPZ and details", npz_start, samples=len(all_video_samples))

        _log_timing("Batch pipeline total", batch_start, total_samples=len(final_df), errors=len(errors))
        return {'final': str(final_output), 'total_samples': len(final_df), 'fall': total_fall, 'nofall': total_nofall}

    _log_timing("Batch pipeline total", batch_start, total_samples=0, errors=len(errors))
    if errors:
        logger.error("批量处理失败详情:")
        for error in errors:
            logger.error(f"  {error}")
    return {}
