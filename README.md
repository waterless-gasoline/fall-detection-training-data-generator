# 跌倒检测训练数据生成

从视频中提取跌倒检测训练数据，通过多步骤管道处理视频，最终生成103维特征样本用于模型训练。

## 核心流程

```
视频 → 切帧 → Pose检测 → 数据清洗 → 跌倒分割 → 样本提取 → 特征计算 → NPZ输出
```

## 特征维度

**121维特征**（X.shape=(N, 10, 121), y.shape=(N,)）：

| 索引范围 | 特征名称 | 描述 |
|---------|---------|------|
| 0-33 | positions | 关键点位置（17点×2坐标，外接矩形归一化）|
| 34-43 | 采样间隔 | 10个时间差（秒）|
| 44-53 | bbox宽 | 10帧的归一化宽 |
| 54-63 | bbox高 | 10帧的归一化高 |
| 64-73 | bbox宽高比 | 10帧的宽/高 |
| 74-83 | bbox面积 | 10帧的宽×高 |
| 84-117 | relative_positions | 相对于髋中心的位置 |
| 118 | spine_leg_angle | 脊柱-腿部夹角 |
| 119 | height_change | 高度变化 |
| 120 | body_orientation | 身体朝向 |

标签：1=FALL（跌倒），0=NOFALL（非跌倒）

## 目录结构

```
跌倒训练数据生成/
├── main_pipeline.py      # 主流程入口（单进程）
├── run_parallel.py       # 并行入口（推荐，多进程）
├── config.py             # 配置文件
├── steps/                # 步骤模块
│   ├── step1_video_split.py    # 视频切帧
│   ├── step2_pose_detection.py # Pose检测（YOLO-pose）
│   ├── step3_data_cleaning.py  # 数据清洗
│   ├── step4_fall_split.py     # 跌倒分割
│   ├── step5_sample_extract.py # 样本提取
│   └── step6_feature_calc.py   # 特征计算
├── utils/                # 工具模块
│   ├── pose_utils.py          # 关键点工具
│   ├── annotation_parser.py   # 标注解析
│   └── file_utils.py          # 文件工具
└── logs/                 # 日志目录
```

## 快速开始

### 推荐方式：run_parallel.py（多进程）

```bash
# 处理1000个视频，使用8个worker
python run_parallel.py --workers 8 --max-videos 1000 --timeout 300
```

参数说明：
- `--workers`: 并行进程数（默认: CPU核心数-1）
- `--max-videos`: 最多处理的总视频数
- `--timeout`: 每个视频超时时间（秒）
- `--reuse-frames`: 跳过切帧和pose检测，只重新计算特征

### 备选方式：main_pipeline.py（单进程）

```bash
# 批量模式
python main_pipeline.py --batch --max-videos 100 --workers 4

# 单视频模式
python main_pipeline.py --video-name 2026-03-03_10-18-25
```

注意：`--workers` 参数在 main_pipeline.py 中实际无效（硬编码为1），请使用 run_parallel.py 实现真正的并行。

### 参数说明

| 参数 | 说明 |
|------|------|
| `--video-name` | 视频名称（单视频模式）|
| `--batch` | 批量处理所有视频 |
| `--max-videos` | 最多处理视频数 |
| `--fps` | 视频帧率（默认20）|
| `--rounds` | 切帧轮数（默认6）|
| `--recompute` | 强制重新计算（包括切帧和pose检测）|
| `--reuse-frames` | 复用现有切帧和pose结果 |
| `--workers` | 并行视频处理数 |
| `--offset` | 视频列表起始偏移量 |

## 处理流程

### Step 1: 视频切帧
按round分目录存储帧图像，每轮使用随机时间间隔（250-500ms）。

### Step 2: Pose检测
使用 YOLO-pose 模型检测17个COCO关键点：
- 头部：nose, left_eye, right_eye, left_ear, right_ear
- 上身：left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
- 下身：left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

### Step 3: 数据清洗
- 过滤无效数据（置信度低、可见性差）
- 删除无人帧及无人区间前后的短暂帧

### Step 4: 跌倒分割
根据标注文件（annotation.txt）解析跌倒时间段。

### Step 5: 样本提取
每10帧为一个样本，帧间隔250-500ms随机选取。

### Step 6: 特征计算
计算103维特征，归一化后输出。

## 输出格式

### NPZ文件
```
samples.npz
├── X: shape (N, 10, 101)  # 特征矩阵
└── y: shape (N,)          # 标签
```

### CSV文件
- `all_features.csv`: 所有样本的121维特征
- `samples_with_details.csv`: 样本详细信息

## 配置

编辑 `config.py` 修改路径配置：

```python
VIDEO_DIR = Path(r"D:\IPC\IPC_video_data_mp4\跌倒\data_positive_0512")
ANNOTATION_DIR = Path(r"D:\IPC\IPC_video_data_annotation\跌倒")
FRAME_OUTPUT_DIR = Path(r"D:\IPC\IPC_data_clip_photo\跌倒\data_positive_0512")
MODEL_PATH = Path(r"yolo26l-pose.pt")
```

## 依赖

- Python 3.8+
- PyTorch
- OpenCV (cv2)
- NumPy
- Pandas
- tqdm
- YOLO-pose 模型权重：`yolo26l-pose.pt`

## 并行策略

| 层级 | 方式 | 状态 | 效果 |
|------|------|------|------|
| 进程级 | run_parallel.py | 正常 | **最佳** - 充分利用多核 |
| 视频级 | main_pipeline.py | 失效 | num_workers=1硬编码 |
| Round级 | main_pipeline.py | 正常 | 一般 |
| 样本级 | main_pipeline.py | 正常（已禁用）| 收益小 |

## 标注格式

annotation.txt 格式（时间单位：毫秒）：
```
fall_start,fall_end
10000,15000
30000,35000
```

表示视频中 10-15秒 和 30-35秒 为跌倒时间段。