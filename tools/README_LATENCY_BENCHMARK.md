# CUDA Latency Benchmark

这套工具用于将 Flow Matching 项目复制到新电脑后，使用真实
`resolved_config.yaml` 和 checkpoint 标定 DINO、Memory、Policy 及并发推理延迟。

工具只调用仓库现有的 `FMInferenceRuntime`、DINO、MemoryEncoder 和
FlowMatchingPolicy，不会复制模型实现或修改 checkpoint。

## 两种运行模式

默认是 checkpoint 模式，必须提供兼容的 Flow Matching `--run-dir`，配置、
normalizer 和权重均来自真实训练目录。

尚无带 Memory 的 checkpoint 时可使用 `--architecture-only`。该模式不需要
`--run-dir`，直接实例化项目现有的 DINO、Memory、UNet 和 Policy；模型权重为随机
初始化，输入为合成数据：

```bash
python -m tools.bench_policy_latency \
    --architecture-only \
    --num-views 3 \
    --device cuda:0
```

输出会明确记录 `benchmark_mode=architecture_only`、
`checkpoint_loaded=false` 和 `task_quality_not_measured=true`。结果只用于标定当前
结构的计算 latency 和显存，不能衡量任务效果，也不能称为 trained-checkpoint
latency。`--num-views` 支持2或3；architecture-only 中省略时默认3视角。

## 前置条件

checkpoint 模式的 RUN_DIR 必须至少包含：

```text
RUN_DIR/
├── resolved_config.yaml
└── checkpoints/
    └── latest.pt
```

也可以通过 `--checkpoint` 指定其他 checkpoint。模型结构、视角数、状态维度、
action horizon 和 solver steps 均从 config/checkpoint 读取。权重不兼容时会直接打印
missing keys、unexpected keys 和 size mismatches，不会使用随机初始化层继续测试。

## 一键标定

```bash
bash tools/run_latency_calibration.sh \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --num-views 3 \
    --warmup-iterations 20 \
    --iterations 200 \
    --output-dir latency_results
```

默认顺序：

```text
DINO isolated
Memory isolated
Policy isolated
Full pipeline single-process
Concurrent realistic
Concurrent three-process stress
```

跳过三进程压力测试：

```bash
bash tools/run_latency_calibration.sh \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --skip-stress
```

只运行某一项：

```bash
bash tools/run_latency_calibration.sh \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --only memory
```

`--only` 支持 `dino` / `memory` / `policy` / `full` / `concurrent`。任一子测试
失败时，脚本会继续保留已生成结果，最终返回非零退出码。

## 独立脚本

### DINO

```bash
python -m tools.bench_dino_latency \
    --run-dir /path/to/run_dir \
    --device cuda:0
```

测量：

- 图像归一化；
- 每个视角的 backbone forward；
- 全视角串行 forward；
- patch token 提取；
- CLS 提取或 patch average pooling；
- view stack；
- `AsyncDinoBuffer._run_dino()` 完整单采样时刻。

脚本会将 AsyncDinoBuffer 的真实输出与 DINO CLS/patch average 进行数值比较，
再输出 `global_feature_source`，不从 README 猜测。如果当前使用 CLS，
`pooling_ms` 表示 global feature 生成/索引时间，不代表实际做了 average pooling。

### Memory

```bash
python -m tools.bench_memory_latency \
    --run-dir /path/to/run_dir \
    --device cuda:0
```

分别测量 checkpoint 中的：

- shared DINO projection；
- VisualTemporalMemoryEncoder（包含时间池化）；
- view fusion；
- state Conv1d encoder；
- visual/state fusion MLP；
- 完整 `policy._build_memory()`。

同时打印实际中间 shape、`temporal_pooling`、`view_processing` 和每个 Tensor
的理论显存。理论 Tensor MiB 与 CUDA allocator 的 allocated/reserved 分开报告。

### Policy

```bash
python -m tools.bench_policy_latency \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --num-inference-steps 16
```

省略 `--num-inference-steps` 时使用 config/checkpoint 中的 solver steps。

分别报告：

- current observation encoder；
- Memory encoder；
- condition fusion；
- 单次 UNet/DiT velocity forward；
- 原始当前图像 + condition + Memory + 单次 velocity forward；
- `one_forward_cached_current_DINO_ms`：缓存的当前 DINO feature + 状态条件 +
  Memory + condition fusion + 单次 velocity forward（不含 DINO backbone 和 solver）；
- 完整 solver；
- `policy.predict_action()` 总时间。

单次 velocity forward 不等于完整动作生成。完整 Flow Matching 会按 checkpoint
中的 solver step 数多次调用 velocity model。`solver_total_ms` 复用真实
`conditional_sample()` solver，但将 condition 替换为预先计算的结果，因此不包含
condition/Memory 构造；`predict_action_total_ms` 包含完整路径。

### 单进程完整流水线

```bash
python -m tools.bench_full_pipeline_latency \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --camera-period-ms 33 \
    --dino-sample-interval-frames 4
```

该测试使用真实 `FMInferenceRuntime`、`submit_async_dino_frame()` 和
`predict_rot6d_abs()`。先用真实 DINO 填满16个历史时刻，再 warmup。正式阶段的
相机 feeder 按约30 Hz 持续提交，不会等待 DINO 完成，因此会测到真实
GPU 竞争、latest-only dropped samples 和 deadline misses。

### 多进程并发

```bash
python -m tools.bench_concurrent_latency \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --scenario realistic \
    --duration-seconds 30 \
    --dino-rate-hz 7.5 \
    --policy-rate-hz 10
```

压力测试：

```bash
python -m tools.bench_concurrent_latency \
    --run-dir /path/to/run_dir \
    --device cuda:0 \
    --scenario stress \
    --duration-seconds 30
```

多进程强制使用 `spawn`，不使用 CUDA + `fork`。每个进程独立加载所需
checkpoint/module，因此显存会明显高于单进程。`stress` 是三进程压力测试，
不代表默认真机架构。

## Warmup 和首帧

每个独立测试都会：

```text
加载 config/checkpoint
→ 预分配固定 shape 输入
→ 执行一次正确性检查
→ warmup N 次
→ CUDA synchronize
→ reset peak memory
→ 正式统计
```

首次 CUDA context、cuDNN/cuBLAS kernel 初始化、allocator 增长和 warmup 全部不进入
正式结果。

## CUDA latency 与 wall latency

- `cuda_ms`：由 CUDA Event 计时，表示两个 Event 之间的 GPU 时间。
- `wall_ms`：由 `perf_counter()` 计时，包含 Python 调度、CUDA launch、同步等待
  和当时的 GPU 排队。

真机 deadline 应优先看 wall p95/p99，而不是只看平均 CUDA kernel 时间。

## Isolated 和 concurrent

- `isolated`：当前模块单独使用 GPU。
- `concurrent`：DINO 与 Policy，或 DINO/Memory/Policy 同时使用同一 GPU。

`p95_slowdown_percent` 反映 GPU 竞争导致的尾延迟增幅。如果平均值变化小但
p95/p99 明显上升，说明系统存在偶发排队、allocator 或调度抖动。

## 输出文件

每个脚本创建：

```text
latency_results/<timestamp>_<hostname>_<tag>/
├── metadata.json
├── summary.json
├── raw_samples.csv
├── console_summary.txt
└── config_snapshot.yaml
```

并发测试额外生成 `process_dino.csv`、`process_memory.csv`、
`process_policy.csv` 和 `concurrent_summary.json`。

CUDA 显存字段：

- `allocated_mib`：当前活跃 Tensor 占用；
- `reserved_mib`：PyTorch CUDA 缓存池保留；
- `peak_allocated_mib`：测试阶段活跃显存峰值；
- `peak_reserved_mib`：缓存池峰值。

`reserved_mib` 不能当作 Tensor 本身大小。

优先比较 `summary.json` 中的 wall p50/p95/p99，再结合 metadata 中的 PyTorch、
CUDA、cuDNN 和 GPU 信息解释差异。

## 边界

这些结果不等于完整真机延迟。随机图像输入不包含真实相机驱动、ROS
序列化/网络传输、硬件时间戳转换和机器人控制器延迟。它们主要用于定位
DINO、Memory、Policy 和 GPU 竞争瓶颈。
