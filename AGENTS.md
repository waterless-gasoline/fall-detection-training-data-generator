# Fall Detection Training Data Generation - AGENTS.md

## Project Overview

跌倒检测训练数据生成项目，处理视频提取姿态关键点，生成 101 维特征，并导出 CSV/NPZ 训练数据。

**工作目录**: `D:\pycharm\工作脚本\跌倒检测\跌倒训练数据生成`

## Current Execution Notes

- 当前主要直接运行 `main_pipeline.py`
- `main_pipeline.py` 现为兼容壳层，负责日志初始化、运行时注入、CLI 参数解析，并委托 `pipeline_orchestrator.py`
- `run_parallel.py` 是外层包装器，不应默认假设用户在使用它
- `--resume` 现在是**显式开关**，只有命令行传入时才启用
- GPU 推理阶段对嵌套并行敏感，推荐优先使用 `--disable-internal-parallel`

## Key Files

| 文件 | 说明 |
|------|------|
| `main_pipeline.py` | CLI/兼容壳层：日志初始化、运行时注入、参数解析，并调用 orchestrator |
| `pipeline_orchestrator.py` | 单视频 / 批量 / watch 编排主逻辑 |
| `pipeline_round_processing.py` | round 清洗、样本后处理、fall clip、NPZ/detail 生成 |
| `run_parallel.py` | 外层多进程包装入口 |
| `config.py` | 配置文件：路径、帧率、推理模式、设备等 |
| `steps/step1_video_split.py` | Step1：视频切帧，支持 round 级补切 |
| `steps/step2_pose_detection.py` | Step2：Pose检测，支持逐帧 cache 补跑 |
| `utils/pipeline_features.py` | 特征计算、样本提取、调试明细生成 |
| `utils/pipeline_helpers.py` | 标注解析入口、状态检查、结果加载、watch 辅助逻辑 |
| `utils/pose_utils.py` | 关键点计算工具 |

## Feature Dimensions (121维)

| 特征索引 | 名称 | 计算方式 |
|---------|------|----------|
| 0-33 | positions | 外接矩形相对坐标归一化，可为负值 |
| 34-43 | 采样间隔 | 10个时间差 |
| 44-53 | bbox宽 | 10帧的bbox宽归一化值 |
| 54-63 | bbox高 | 10帧的bbox高归一化值 |
| 64-73 | bbox宽高比 | 10帧的bbox宽/高，高为0时取0 |
| 74-83 | bbox面积 | 10帧的bbox宽×高 |
| 84-117 | relative_positions | 相对髋中心的位置 |
| 118 | spine_leg_angle | spine=左肩-左髋, leg=左膝-左髋, `np.arctan2` |
| 119 | height_change | 高度变化 |
| 120 | body_orientation | 身体朝向，`np.arctan2` |

## Resume / Reuse Behavior

### Resume 仅在 `--resume` 时启用

```bash
python main_pipeline.py --batch --resume
```

启用后会按严格续跑顺序检查：
- 视频级输出：`results/all_features.csv` / `samples.npz` / `sample_details.txt` / `samples_with_details.csv` 全部完整时，整视频直接跳过
- Round输出：`results/round_{idx}_output/round_{idx}_features.csv` 完整时，跳过该 round 的 pose 和后处理
- Step2：仅对缺失 round 特征的 round 检查 `pose_cache/*.npz` / `pose_results.npz` / `pose_done.marker`，已有完整 pose 则复用，否则补跑
- Step1：仅在切帧结果不完整时补缺失帧
- `pipeline_orchestrator.py` 会输出 `[Resume][video]` / `[Resume][Round X]` 日志，明确显示当前是在整视频跳过、复用特征CSV、复用 pose，还是补跑 pose

### 复用模式
```bash
python main_pipeline.py --batch --reuse-frames
python main_pipeline.py --batch --reuse-split-frames
```

- `--reuse-frames`: 复用现有切帧；若 round 下已有完整 `pose_results.npz` 则直接复用，否则自动补跑缺失 pose，再执行后续流程
- `--reuse-split-frames`: 复用现有切帧，但重新跑 pose 和后续
- `pipeline_orchestrator.py` 现在会输出 `[ReuseMode]` / `[ReuseDebug]` 日志，明确显示当前是在复用切帧、复用 pose，还是补跑 pose

## Important Implementation Notes

### 1. positions 不使用 clip
- 避免将负值强行裁到 0

### 2. spine_leg_angle / body_orientation
- spine = 左肩 - 左髋
- leg = 左膝 - 左髋
- 使用 `np.arctan2`，范围 `(-π, π)`

### 3. 无人帧过滤
- 在样本提取阶段，跳过包含无人帧的样本

### 4. 并行与稳定性
- `main_pipeline.py --batch --workers N`：视频级并行
- 视频级 / 批量 / watch 编排当前主要看 `pipeline_orchestrator.py`
- `--disable-internal-parallel`：关闭内部并行，避免嵌套并行
- GPU 推理容易因嵌套多进程导致 `cuDNN_STATUS_INTERNAL_ERROR`
- 多卡可见不等于自动按卡分配；当前代码没有完善的 per-worker GPU 绑定

### 5. 标注处理
- 缺失标注：按无跌倒处理
- 空标注：按无跌倒处理
- `discard`：整视频跳过
- `process_single_round()` 当前位于 `pipeline_round_processing.py`
- `load_pose_results`、标注状态解析、round 完整性检查、watch 辅助逻辑来自 `utils/pipeline_helpers.py`

## Common Commands

```bash
# 默认整批重跑（不启用resume）
python main_pipeline.py --batch --workers 1

# 显式启用resume
python main_pipeline.py --batch --workers 1 --resume

# 禁用内部并行，降低嵌套并发问题
python main_pipeline.py --batch --workers 4 --disable-internal-parallel

# 复用切帧，重跑pose和后续
python main_pipeline.py --batch --workers 1 --reuse-split-frames
```
