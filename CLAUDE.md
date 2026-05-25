# 跌倒检测训练数据生成 - 项目文档

## 项目概述

本项目用于从视频中提取跌倒检测训练数据，通过多步骤管道处理视频，最终生成 101 维特征样本，并可导出 NPZ 训练数据。

**核心流程**：视频 → 切帧 → Pose检测 → 数据清洗 → 跌倒分割/样本提取 → 特征计算 → CSV/NPZ输出

**特征维度**：121维
- 0-33: 关键点位置 (positions)
- 34-43: 采样间隔 (10个时间差)
- 44-53: bbox归一化宽
- 54-63: bbox归一化高
- 64-73: bbox宽高比（宽/高）
- 74-83: bbox面积（宽×高）
- 84-117: 相对位置 (relative_positions)
- 118: 脊柱-腿部角度 (spine_leg_angle)
- 119: 高度变化 (hip_height_change)
- 120: 身体朝向 (body_orientation)

**输出格式**：NPZ格式，`X.shape=(N, 10, 121)`, `y.shape=(N,)`，标签 `1=FALL`, `0=NOFALL`

## 目录结构

```text
跌倒训练数据生成/
├── main_pipeline.py           # 主流程兼容入口 / 薄壳CLI
├── pipeline_orchestrator.py   # 单视频 / 批量 / watch 编排
├── pipeline_round_processing.py # round清洗、样本后处理、NPZ/detail生成
├── run_parallel.py            # 外层多进程包装入口
├── config.py                  # 配置文件
├── ARCHITECTURE.md            # 架构文档
├── OPTIMIZATION_REPORT.md
├── steps/                     # 步骤模块
│   ├── step1_video_split.py    # 视频切帧
│   ├── step2_pose_detection.py # Pose检测
│   ├── step3_data_cleaning.py  # 数据清洗
│   ├── step4_fall_split.py     # 跌倒分割
│   ├── step5_sample_extract.py # 样本提取
│   └── step6_feature_calc.py   # 特征计算
├── utils/                     # 工具模块
│   ├── pose_utils.py          # 关键点工具
│   ├── annotation_parser.py   # 标注解析
│   ├── pipeline_features.py   # 特征计算/样本提取实现
│   ├── pipeline_helpers.py    # 标注、状态检查、结果加载等帮助函数
│   └── file_utils.py          # 文件工具
├── logs/                      # 日志目录
└── *.csv/*.json               # 数据文件
```

## 主要入口和运行方式

### 推荐入口：main_pipeline.py

```bash
python main_pipeline.py --batch --workers 1
```

说明：
- 当前用户主要直接使用 `main_pipeline.py`
- `main_pipeline.py` 现在是兼容入口 / 薄壳CLI：负责日志、运行时注入、参数解析，并调用 `pipeline_orchestrator.py`
- `--workers` 在 batch 模式下控制视频级并行
- 如需避免嵌套并行，使用：

```bash
python main_pipeline.py --batch --workers 4 --disable-internal-parallel
```

### 可选入口：run_parallel.py

```bash
python run_parallel.py --workers 8 --max-videos 1000 --timeout 300
```

说明：该脚本是外层包装器，会启动多个 `main_pipeline.py` 子进程；当前不应假设用户一定使用它。

## CLI 关键参数

### Resume
默认**不启用** resume，只有显式传参才启用：

```bash
python main_pipeline.py --batch --resume
```

含义（严格续跑）：
- 视频级结果包完整：若 `results/all_features.csv`、`samples.npz`、`sample_details.txt`、`samples_with_details.csv` 都完整，则整视频直接跳过
- Round输出完整：若 `results/round_{idx}_output/round_{idx}_features.csv` 已完整，则跳过该 round 的 pose 检测和后处理
- Step2：仅对缺失 round 特征的 round 检查 `pose_cache/*.npz` / `pose_results.npz` / `pose_done.marker`，已有完整 pose 则复用，否则补跑缺失 pose
- Step1：仅在切帧结果不完整时补缺失帧
- 当前 `pipeline_orchestrator.py` 会输出 `[Resume][video]` / `[Resume][Round X]` 日志，用于区分“整视频跳过”“复用特征CSV”“复用pose”“补跑pose”

### 复用模式
```bash
python main_pipeline.py --batch --reuse-frames
python main_pipeline.py --batch --reuse-split-frames
```

- `--reuse-frames`: 复用现有切帧；若已有完整 `pose_results.npz` 则直接复用，否则自动补跑缺失 pose，再执行后续特征流程
- `--reuse-split-frames`: 复用现有切帧，但重新执行 pose 和后续流程
- 当前 `pipeline_orchestrator.py` 会输出 `[ReuseMode]` / `[ReuseDebug]` 日志，用于区分“复用切帧”“复用pose”“补跑pose”三类行为

### Watch 模式
```bash
python main_pipeline.py --batch --watch --stable-seconds 60
```

用于边上传边处理视频；会轮询目录，只处理稳定完成上传的视频。

## 完整流程（步骤）

1. **step1_video_split.py** - 视频切帧，按 round 分目录
2. **step2_pose_detection.py** - 使用 YOLO-pose 模型检测关键点
3. **step3_data_cleaning.py** - 过滤无人帧等无效数据
4. **step4_fall_split.py** - 基于标注分割跌倒片段
5. **step5_sample_extract.py** - 提取样本（10帧/样本，间隔250-500ms）
6. **step6_feature_calc.py** - 计算 101 维特征

当前生产路径里，视频级 / 批量 / watch 编排主要在 `pipeline_orchestrator.py`，round 级清洗、样本后处理与 NPZ/detail 导出主要在 `pipeline_round_processing.py` 和 `utils/pipeline_features.py`。

## 开发注意事项

### 1. 并行策略
- `main_pipeline.py --batch --workers N`：视频级并行
- `pipeline_orchestrator.py` 承接单视频 / 批量 / watch 编排
- `--disable-internal-parallel`：禁用内部并行，避免嵌套并行和 GPU / 进程管理问题
- 环境变量 `DISABLE_INTERNAL_PARALLEL=1` 也可控制内部并行关闭
- 当前 GPU 推理更适合降低嵌套并行，避免 `cuDNN_STATUS_INTERNAL_ERROR`

### 2. Resume 状态文件
- Step1:
  - `planned_timestamps.json`
  - `frame_timestamps.json`
  - `round_done.marker`
- Step2:
  - `pose_cache/*.npz`
  - `pose_results.npz`
  - `pose_done.marker`

### 3. 关键点配置
- COCO 17点格式
- 脊柱向量：左肩 - 左髋
- 腿部向量：左膝 - 左髋
- 使用 `np.arctan2` 计算角度（范围 -π 到 π）

### 4. 特征计算
- positions 不使用 clip，容许负值
- velocities 使用实际时间戳差分
- velocities 按样本(10帧)独立归一化，公式：`(current - min) / (max - min)`

### 5. 无人帧处理
- 跳过包含无人帧（x全为0）的样本
- 视频级分割后需重新检查

### 6. 标注处理
- 缺失标注：按无跌倒处理
- 空标注：按无跌倒处理
- `discard`：整视频跳过

### 7. 依赖
- PyTorch, OpenCV, NumPy, Pandas
- YOLO-pose 模型权重: `yolo26l-pose.pt`
