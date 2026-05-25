# Utils 模块文档

工具模块目录，提供文件操作、姿态计算、标注解析等基础功能。

---

## file_utils.py - 文件操作工具

### 功能说明
提供文件/目录操作的常用工具函数。

### 主要类和函数

| 函数 | 说明 |
|------|------|
| `ensure_dir(path)` | 确保目录存在，不存在则创建 |
| `save_json(data, file_path)` | 保存数据为JSON文件 |
| `load_json(file_path)` | 加载JSON文件 |
| `copy_annotation(src, dst)` | 复制标注文件到目标目录 |
| `get_frame_files(dir_path, ext='.jpg')` | 获取目录下所有帧文件，按序号排序 |
| `split_folder(src_folder, split_indices, dst_base, suffix)` | 按帧索引分割文件夹 |

### 说明
- 当前主流程更常直接用 `pathlib` / `shutil` / `json`
- `file_utils.py` 更偏工具层，不一定是每条生产路径都会显式调用

---

## pose_utils.py - 姿态相关工具

### 功能说明
基于 COCO 17 关键点格式的姿态计算工具。

### 关键点索引 (COCO顺序)

| 索引 | 部位 | 索引 | 部位 |
|------|------|------|------|
| 0 | NOSE | 9 | LEFT_WRIST |
| 1 | LEFT_EYE | 10 | RIGHT_WRIST |
| 2 | RIGHT_EYE | 11 | LEFT_HIP |
| 3 | LEFT_EAR | 12 | RIGHT_HIP |
| 4 | RIGHT_EAR | 13 | LEFT_KNEE |
| 5 | LEFT_SHOULDER | 14 | RIGHT_KNEE |
| 6 | RIGHT_SHOULDER | 15 | LEFT_ANKLE |
| 7 | LEFT_ELBOW | 16 | RIGHT_ANKLE |
| 8 | RIGHT_ELBOW | | |

### 主要函数

| 函数 | 说明 | 输入shape |
|------|------|----------|
| `get_hip_center(positions)` | 计算髋中心 | `(17,2)` / `(34,)` / `(N,17,2)` |
| `get_shoulder_center(positions)` | 计算肩中心 | `(17,2)` / `(34,)` |
| `compute_spine_vector(positions)` | 计算躯干向量 | 同上 |
| `compute_leg_vector(positions)` | 计算腿向量 | 同上 |
| `angle_between_vectors(v1, v2)` | 计算两向量夹角 | `(2,)` |
| `compute_spine_leg_angle(positions)` | 计算躯干与腿夹角 | `(17,2)` / `(34,)` |
| `compute_body_orientation(positions)` | 计算身体朝向 | 同上 |
| `compute_relative_positions(positions)` | 计算相对位置 | `(34,)` |

### 当前项目约定
- spine = 左肩 - 左髋
- leg = 左膝 - 左髋
- `body_orientation` 与 `spine_leg_angle` 使用 `np.arctan2`
- 这些约定优先于旧版本里“肩中心/髋中心平均”的写法

### 依赖
- `numpy`

---

## annotation_parser.py - 标注解析工具

### 功能说明
解析跌倒检测标注文件，支持多种格式。

### 支持格式

1. **格式1**: `序号;起始-结束` (秒)
   - 示例: `1;0.65-2.35`

2. **格式2**: `起始 结束` 或 `起始,结束`
   - 可按秒或毫秒解析

### 主要函数

| 函数 | 说明 |
|------|------|
| `parse(file_path)` | 解析标注文件，返回 `[(start_ms, end_ms), ...]` |
| `get_frame_intervals(intervals_ms, fps)` | 将毫秒区间转换为帧区间 |
| `filter_intervals_in_range(intervals, start_ms, end_ms)` | 过滤完全落在指定范围内的区间 |

### 当前主流程标注策略
- 缺失标注：按无跌倒处理
- 空标注：按无跌倒处理
- `discard`：整视频跳过
- 主流程处理入口在 `main_pipeline.py::resolve_annotation_state()`

### 使用示例

```python
from utils.annotation_parser import AnnotationParser

intervals = AnnotationParser.parse(Path("annotation.txt"))
# 返回: [(650.0, 2350.0), ...] 或毫秒级区间
```

---

## 模块依赖关系

```text
annotation_parser.py  (独立)
pose_utils.py        (依赖 numpy)
file_utils.py        (依赖 json, shutil)
```

## 额外说明

- 如果 `utils` 文档与 `main_pipeline.py` 当前行为不一致，优先相信主流程实际实现
- 当前项目核心输出维度为 **121维**，不要沿用旧的 71维/139维描述
