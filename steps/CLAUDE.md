# Steps 模块文档

跌倒检测训练数据生成的流水线模块，按执行顺序分为 6 个步骤。

## 步骤概览

| 步骤 | 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|------|
| 1 | `step1_video_split.py` | 视频切帧 | 视频文件 | `round_N/frame_*.jpg` + 时间戳/状态文件 |
| 2 | `step2_pose_detection.py` | 姿态检测 | JPG帧图像 | `pose_results.npz` + `pose_cache/*.npz` |
| 3 | `step3_data_cleaning.py` | 数据清洗 | Pose结果/帧目录 | 清洗后帧目录 |
| 4 | `step4_fall_split.py` | 跌倒分割 | 清洗后帧目录 | 跌倒片段子目录 |
| 5 | `step5_sample_extract.py` | 样本提取 | 分割后帧目录 | 样本列表 |
| 6 | `step6_feature_calc.py` | 特征计算 | 样本列表 | CSV特征矩阵 |

## 数据依赖关系

```text
视频(.mp4/.avi/.mov)
    ↓
step1: 帧图像(.jpg) + frame_timestamps.json + round状态文件
    ↓
step2: pose_cache/*.npz + pose_results.npz
    ↓
step3/4/5: 清洗/分割/样本提取（当前主要在 main_pipeline.py 中按 round 驱动）
    ↓
step6: round_{idx}_features.csv
    ↓
视频级汇总: all_features.csv + samples.npz
```

## 关键现状说明

- 当前用户主要使用 `main_pipeline.py` 驱动整条流程
- Step3~Step6 很多逻辑已由 `main_pipeline.py::process_single_round()` 串起来，不完全依赖各 step 文件的 `run()`
- `--resume` 是显式开关；不加时不应假设启用断点续跑

## 步骤详解

### Step 1: `step1_video_split.py` - 视频切帧

**功能**: 将视频按随机间隔切帧，生成多轮 `round_*` 结果。

**关键类**: `VideoFrameExtractor`

**关键参数**:
- `frame_intervals_ms`: 随机帧间隔列表，默认 `[250, 300, 350, 400, 450, 500]` ms
- `num_rounds`: 切帧轮数，默认 6 轮
- `STEP1_FFMPEG_WORKERS`: ffmpeg 并行数

**输出**:
- `round_N/frame_XXXXXX.jpg`
- `round_N/frame_timestamps.json`
- `round_N/planned_timestamps.json`
- `round_N/round_done.marker`

**Resume相关**:
- 已有 `planned_timestamps.json` 时，不重新随机计划
- 只补缺失/损坏的帧
- 全部完成后写 `round_done.marker`

---

### Step 2: `step2_pose_detection.py` - 姿态检测

**功能**: 使用 YOLO-Pose 模型检测人体 17 个关键点。

**关键类**: `PoseDetector`

**输入**: `round_N/frame_*.jpg`

**输出**:
- `round_N/pose_cache/frame_XXXXXX.npz`
- `round_N/pose_results.npz`
- `round_N/pose_done.marker`

**结果结构**:
- 每帧记录: `{'frame': str, 'keypoints': np.ndarray(17,3), 'has_person': bool}`
- keypoints 格式: 每点 `[x, y, confidence]`

**Resume相关**:
- 仅在 `--resume` 下使用 cache 补缺失帧
- cache 完整后重建 `pose_results.npz`
- 完整时写 `pose_done.marker`

**并行注意**:
- GPU 场景下嵌套多进程容易触发 `cuDNN_STATUS_INTERNAL_ERROR`
- 推荐结合 `--disable-internal-parallel` 使用
- 多卡可见不等于自动分卡；当前代码未完善做 per-worker GPU 绑定

---

### Step 3: `step3_data_cleaning.py` - 数据清洗

**功能**: 过滤无人帧或按无人区间分割有效片段。

**关键类**: `DataCleaner`

**关键参数**:
- `gap_threshold`: 连续无人帧阈值，默认 20 帧

**说明**:
- 当前主流程中，清洗逻辑更多直接在 `main_pipeline.py` 中完成
- 该 step 文件仍可作为独立逻辑参考，但不要默认它是唯一执行路径

---

### Step 4: `step4_fall_split.py` - 跌倒分割

**功能**: 按跌倒区间分割视频片段。

**关键类**: `FallSegmenter`

**输入**:
- 清洗后的帧目录
- `annotation.txt`

**说明**:
- 当前生产路径中，跌倒分割和样本提取多由 `process_single_round()` 统一驱动
- 这里的文件更像参考/独立执行版本

---

### Step 5: `step5_sample_extract.py` - 样本提取

**功能**: 基于时间间隔提取样本。

**关键参数**:
- `intervals_ms`: `[250, 300, 350, 400, 450, 500]`
- `sample_frames`: 当前独立模块描述是 11 帧，但主流程最终训练数据输出是 `X.shape=(N, 10, 121)`，以主流程实际实现为准

**说明**:
- 这里的独立实现和 `main_pipeline.py` 主流程实现存在历史差异
- 修改样本抽取逻辑时，优先核对 `main_pipeline.py` 当前实际行为

---

### Step 6: `step6_feature_calc.py` - 特征计算

**功能**: 计算特征并生成 round 级 CSV。

**当前主流程实际目标**:
- 核心特征维度：101 维
- 视频级导出：`results/all_features.csv`、`samples.npz`

**注意**:
- `step6_feature_calc.py` 中存在历史上的 139 维描述
- 当前项目文档与主流程以 **121维** 为准
- 如果文档、step 文件、主流程不一致，优先相信 `main_pipeline.py` 当前输出行为

## Step级状态与恢复

### Step1 状态文件
- `planned_timestamps.json`
- `frame_timestamps.json`
- `round_done.marker`

### Step2 状态文件
- `pose_cache/*.npz`
- `pose_results.npz`
- `pose_done.marker`

### Round级输出
- `results/round_{idx}_output/round_{idx}_features.csv`

### 视频级输出
- `results/all_features.csv`
- `results/samples.npz`
- `results/sample_details.txt`
- `results/samples_with_details.csv`

## 常见建议

```bash
# 默认整批跑（不启用resume）
python main_pipeline.py --batch --workers 1

# 显式启用resume
python main_pipeline.py --batch --workers 1 --resume

# 禁掉内部并行，减少GPU/进程问题
python main_pipeline.py --batch --workers 4 --disable-internal-parallel
```
