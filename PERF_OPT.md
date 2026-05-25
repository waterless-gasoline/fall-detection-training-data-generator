# 跌倒检测Pipeline性能优化

## 项目背景

**工作目录**: `D:\pycharm\工作脚本\跌倒检测\跌倒训练数据生成`

**问题**: Windows上cv2不支持fork模式多进程，会导致DLL加载失败。原有`--workers N`参数不稳定。

## 解决方案

### 独立进程隔离
使用`subprocess.Popen`启动多个独立Python进程，通过`--offset`和`--max-videos`参数分配视频批次。

### 新增参数

```python
# main_pipeline.py
parser.add_argument("--offset", type=int, default=0, help="视频列表起始偏移量")
```

### 新增文件

**run_parallel.py** - 多进程批量运行脚本
- 自动检测CPU核心数
- 分配视频批次
- 启动多进程
- 优雅退出机制（收到Ctrl+C会清理子进程）
- 超时控制

## 测试结果

### 仅特征计算（跳过切帧和Pose检测）

| 配置 | 总耗时 | 加速比 |
|------|--------|--------|
| 单进程 | 197.3秒 | 1.00x |
| 2进程并行 | 77.0秒 | 2.56x |
| 4进程并行 | 68.2秒 | 2.89x |

### 完整流程（含切帧+Pose检测+特征计算）

| 配置 | 总耗时 | 加速比 |
|------|--------|--------|
| 单进程 | 65.1秒 | 1.00x |
| 2进程并行 | 55.0秒 | 1.19x |
| 4进程并行 | 62.6秒 | 1.04x |

## 关键发现

1. **完整流程瓶颈在GPU**：多进程竞争同一GPU，收益有限
2. **仅特征计算瓶颈在CPU**：多进程并行有效
3. **进程数增加反而变慢**：YOLO模型加载开销 + GPU资源竞争 + 进程调度开销

## 使用建议

### 场景1：仅更新特征（跳过切帧和Pose检测）
```bash
# 推荐4进程
python main_pipeline.py --batch --workers 4 --reuse-frames --max-videos 20
```

### 场景2：完整流程
```bash
# 推荐2进程或单进程
python main_pipeline.py --batch --workers 2 --max-videos 20
```

### 场景3：大规模批量处理
```bash
# 使用run_parallel.py（自动进程隔离）
python run_parallel.py --workers 4 --max-videos 100

# 或手动开多个终端
python main_pipeline.py --batch --max-videos 5 --offset 0 --workers 1
python main_pipeline.py --batch --max-videos 5 --offset 5 --workers 1
python main_pipeline.py --batch --max-videos 5 --offset 10 --workers 1
python main_pipeline.py --batch --max-videos 5 --offset 15 --workers 1
```

## run_parallel.py 进程清理机制

```python
# Ctrl+C 时会触发清理
cleanup_processes()  # 先terminate，3秒后未响应则kill

# 超时控制
timeout_per_video = 300  # 每视频默认300秒超时
```

## 进一步优化方向

1. **多GPU并行**：当前是单GPU
2. **共享YOLO模型**：减少每个进程的模型加载开销
3. **增大batch_size**：A100上可设为64-128