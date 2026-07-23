# Async DINO History Memory 数据流

本文档说明真机推理时，相机图像如何经过 `AsyncDinoBuffer` 生成历史视觉
Memory，以及 Memory 如何注入 UNet 或 DiT。本文只描述当前的 `fusion` Memory 路径。

## 1. 符号

| 符号 | 含义 | 当前值 |
|---|---|---|
| `B` | batch size | 真机通常为 1 |
| `T` | DINO 历史采样数 | 64 |
| `V` | 相机视角数 | 2 或 3 |
| `C` | DINOv2-S backbone 特征维度 | 384 |
| `D` | Policy Memory 维度 | 256 |
| `Ds` | 机器人状态维度 | 由数据配置决定 |
| `H` | action horizon | 当前配置为 64 |
| `A` | action dimension | 由机器人配置决定 |

## 2. 完整数据流

```text
V 路相机图像：每路 [B,3,224,224]
        ↓ 每 8 个相机帧接收一次（约 0.264 s）
AsyncDinoBuffer
        ↓ 每个视角独立执行共享 DINOv2 backbone
单视角 DINO CLS token [B,1,384]
        ↓ squeeze + stack V 个视角
单个采样时刻 [B,V,384]
        ↓ deque(maxlen=64)
历史窗口 [B,64,V,384]（约 16.896 s）
        ↓ infer/runtime.py
Policy Memory 输入
        ↓
每个视角独立时间建模
        ↓
visual_global [B,256]
state_global  [B,64]
        ↓ concat + fusion MLP
memory_global [B,256]
        ↓
UNet global condition 或 DiT condition
```

`local patches [B,N,384]` 只在 `store_local_features=True` 时额外保存。当前
Policy Memory 使用每张图像自己的 DINO CLS token，不是 patch-average feature。
名义视觉历史范围为 `64 × 8 × 0.033 = 16.896 s`；状态 Memory 仍是独立的
64个高频状态帧，没有扩大为512帧。

## 3. AsyncDinoBuffer 数据结构

`submit_async_dino_frame()` 会将同步的 2 或 3 路图像交给
`tools/async_dino_buffer.py`。只处理满足下面条件的帧：

```python
frame_id % 8 == 0
```

单个 Buffer entry 是字典：

```python
{
    "frame_id": int,
    "feature": Tensor[B, V, 384],        # 兼容字段，即 global_feature
    "global_feature": Tensor[B, V, 384],
    "local_feature": Tensor[B, V, N, 384] | None,
    "capture_time": float,
    "ready_time": float,
    "stage_wall_ms": float,
    "end_to_end_ms": float,
}
```

第一次真实 DINO 特征产生后：

```python
feature_window = buffer.get_feature_window()
# [B,64,V,384]
```

Buffer 为空时返回 `None`。Buffer 有1到63个真实采样时，缺失的较早位置使用
first-frame CLS 在窗口左侧补齐；deque 本身仍只保存真实采样。例如三个真实采样
`[f1,f2,f3]` 返回 `[f1×61,f1,f2,f3]`。这些 padding token 参与 Temporal
Transformer，不使用零向量，也不重新运行 DINO。

## 4. infer 交给 Policy 的数据

`infer/runtime.py` 生成下列字段：

| key | shape | 说明 |
|---|---|---|
| `memory_image_backbone_feat` | `[B,64,V,384]` | DINO CLS 历史 |
| `memory_state` | `[B,64,Ds]` | 归一化的状态历史 |
| `memory_visual_offsets` | `[64]` | `[-504,-496,...,-8,0]` |

offsets 不扩展成 `64*V`。V 个视角作为 batch 独立处理，共享同一组
64 个时间 offsets。

## 5. 每视角独立时间建模

```text
feature_window                    [B,64,V,384]
permute(0,2,1,3)                 [B,V,64,384]
reshape                           [B*V,64,384]
shared DINO projection           [B*V,64,256]
shared Temporal Transformer      [B*V,64,256]
mean over 64 time outputs        [B*V,256]
reshape                           [B,V,256]
concat views                     [B,V*256]
view_fusion_proj                 [B,256]
```

DINO projection 只有一套参数：

```text
LayerNorm(384) → Linear(384,256) → SiLU
```

Temporal Transformer 也只有一套参数。它处理 `B*V` 个长度为64的序列，
而不是一个长度为 `64*V` 的序列。这里不会额外添加第65个 temporal CLS；
Transformer 输出的64个时间 token 直接做 mean pooling。训练开始处被 clamp
为 first frame 的视觉 token 使用 `valid=True`，因此会参与计算。

view fusion 的参数维度取决于训练时的视角数：

```text
V=2: [B,512] → [B,256]
V=3: [B,768] → [B,256]
```

运行时会从 Tensor shape 读取 `V`，但 `V` 必须与 checkpoint/config 中的
`n_image_views` 一致，因为 `view_fusion_proj` 是需要训练的参数层。

## 6. 状态 Memory 和最终 Memory

```text
memory_state                     [B,64,Ds]
StateConvMemoryEncoder           [B,64]

visual_global                    [B,256]
state_global                     [B,64]
concat                           [B,320]
fusion MLP                       [B,256]
memory_global                    [B,256]
memory_tokens = unsqueeze(1)     [B,1,256]
```

`memory_global` 用于 global-condition 注入；`memory_tokens` 用于 DiT cross-attention 注入。

## 7. 注入 UNet

UNet 只支持 `concat_global_cond`：

```yaml
models:
  fm:
    velocity_model: unet
  memory:
    injection: concat_global_cond
```

数据流：

```text
当前观测 ConditionEncoder
obs_cond                         [B,256]

MemoryEncoder
memory_global                    [B,256]

concat                           [B,512]
memory_cond_fusion               [B,256]
global_cond                      [B,256]

Flow Matching noisy action       [B,H,A]
diffusion/flow timestep embedding[B,256]
concat(timestep, global_cond)    [B,512]
ConditionalUnet1D residual blocks
predicted velocity               [B,H,A]
```

UNet 中的 `global_cond` 通过各个 conditional residual block 的 scale/bias 调制特征。
UNet 不接收 `condition_tokens`，也不执行 Memory cross-attention。

## 8. 注入 DiT

DiT 有两种可选方式。

### 8.1 concat_global_cond

```yaml
models:
  fm:
    velocity_model: dit
  memory:
    injection: concat_global_cond
```

```text
obs_cond [B,256] + memory_global [B,256]
        ↓ memory_cond_fusion
global_cond [B,256]
        ↓ cond_proj
DiT condition [B,hidden_dim]
        ↓ AdaLN modulation
DiT blocks
```

此模式不使用 cross-attention。

### 8.2 cross_attn

```yaml
models:
  fm:
    velocity_model: dit
  memory:
    injection: cross_attn
    cross_attn_layers: [3, 7, 11]
```

```text
obs_cond                           [B,256]     → DiT global_cond / AdaLN
memory_tokens                      [B,1,256]
context_proj                       [B,1,hidden_dim]
DiT cross-attention layers         3, 7, 11
predicted velocity                 [B,H,A]
```

在 cross-attention 模式中，Memory 只通过 `condition_tokens` 注入；`obs_cond`
仍然是 DiT 的 global condition。`cross_attn` 不允许与 UNet 组合。

## 9. 真机调用顺序

```python
runtime.start_async_dino()

# 相机循环：capture_time 应尽量来自相机驱动/ROS 消息。
capture_time = time.perf_counter()
frame = read_synchronized_cameras()
runtime.submit_async_dino_frame(frame_id, frame, capture_time=capture_time)

# 第一次 DINO 采样完成后即可调用；缺失历史会 repeat-first。
action_chunk = runtime.predict_rot6d_abs(
    current_obs,
    state_raw=current_state_window,
    memory_state_raw=state_history_64,
)

runtime.stop_async_dino()
```

如果使用硬件时间戳计算 `end_to_end_ms`，它必须与 `ready_time` 使用同一时钟域；
否则应同时保留硬件时间戳和本机 `perf_counter` 时间戳。

## 10. 主要代码位置

| 功能 | 文件 |
|---|---|
| 异步 DINO 和64采样 Buffer | `tools/async_dino_buffer.py` |
| 相机提交、Buffer 读取 | `infer/runtime.py` |
| shape 检查和调试中间量 | `infer/preprocess.py` |
| 共享 DINO projection | `models/fm/encoders/dino_v2.py` |
| Temporal Transformer、view/state fusion | `models/fm/memory/fusion.py` |
| Memory 构造与 UNet/DiT 分流 | `models/fm/flow_policy.py` |
| UNet global-condition 注入 | `models/diffusion/conditional_unet1d.py` |
| DiT AdaLN/cross-attention 注入 | `models/fm/action_dit.py` |

CUDA shape smoke test：

```bash
python -m tools.test_async_dino_memory --num-views 2
python -m tools.test_async_dino_memory --num-views 3
```
