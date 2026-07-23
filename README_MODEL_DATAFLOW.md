# 模型数据流总览：UNet / DiT × 有无 Memory

本文档基于当前仓库真实代码（`configs/train/config.yaml` + `models/` + `datasets/`

+ `infer/` + `tools/`）整理，覆盖四种组合的训练与推理数据流、Tensor 维度和数据结构。
  所有 shape 均按代码验证，未验证的内容在第 9 节单独列出。

> 与旧模板的三点差异（按当前代码为准）：
>
> 1. 当前观测三视角是 **先逐视角投影 384→256、再 concat 成 3×256=768**，不是 3×384=1152。
> 2. `concat_global_cond` 已**移除 512→256 的 `memory_cond_fusion` 压缩层**，直接使用拼接后的 512 维。
> 3. DiT 的 Memory（目标 `concat_global_cond`）经 **AdaLN 全局条件**注入，不是额外 token / cross-attention。

---

## 1. 总体配置（真实值）

| 符号               | 含义                   | 值                                                     | 来源                                                                      |
| ------------------ | ---------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| `B`              | batch size             | 训练 1024 / 部署 1                                     | `config.yaml train.batch_size` / 真机                                   |
| `V`              | 视角数                 | **3**（`base_0, left_wrist_0, right_wrist_0`） | `data.camera_views` / `models.fm.n_image_views`                       |
| `D`              | DINOv2-Small CLS 维度  | **384**                                          | `dino_v2.py` backbone `num_features`                                  |
| `cond_dim`       | 条件维度               | 256                                                    | `models.fm.cond_dim`                                                    |
| `image_feat_dim` | 单视觉分支输出         | 256                                                    | `models.fm.image_feat_dim`                                              |
| `T_visual`       | 视觉 Memory 采样数     | **128**                                          | `data.memory.visual_history_length`                                     |
| `visual_stride`  | 视觉采样间隔（相机帧） | **8**                                            | `data.memory.sample_stride`                                             |
| `visual_offsets` | 视觉时间偏移           | **[-1016, -1008, …, -8, 0]**（128 个）          | `zarr_dataset._build_memory_visual_offsets`                             |
| `T_state`        | 状态 Memory 帧数       | **64**                                           | `data.memory.history_frames`                                            |
| `state_recent`   | 状态 Memory 末端偏移   | 4                                                      | `data.memory.recent_frame`                                              |
| `cond_steps`     | 当前观测状态窗口       | **8**                                            | `data.window_size`（=`dataset.window_size`）                          |
| `n_image_steps`  | 当前观测图像帧数       | 1                                                      | `data.n_image_steps`                                                    |
| `action_dim`     | 动作维度               | **14**（`action_type: joint`）                 | `zarr_dataset._ROBOT_DIMS["joint"]`                                     |
| `action_horizon` | 预测动作长度           | **64**                                           | `data.action_horizon`，训练经 `sync_fm_action_horizon_from_data` 注入 |
| `n_action_steps` | 执行动作步数           | **64**（未单列 → 取 `action_horizon`）        | 同上                                                                      |
| `solver_steps`   | 采样步数               | **32**                                           | `models.fm.num_inference_steps`                                         |
| `solver`         | ODE 求解器             | euler                                                  | `models.fm.solver`                                                      |
| `velocity_model` | 速度场骨干             | unet（可选 dit）                                       | `models.fm.velocity_model`                                              |
| `view_pool`      | 视觉融合方式           | global_concat                                          | `models.fm.view_pool`                                                   |
| Memory             | 方法 / 注入            | fusion / concat_global_cond                            | `models.memory.method` / `.injection`                                 |

> 长度不要混淆：`T_visual=128`（稀疏视觉，stride 8，跨度约 33.5s）、`T_state=64`
> （高频状态）、`cond_steps=8`（当前观测状态窗）、`action_horizon=64`（预测长度）是四个独立量。
>
> 注意：`FlowMatchingPolicy.__init__` 的 `action_horizon` 默认是 32；真实训练由
> `trainers/policy_trainer.build_policy` 调 `sync_fm_action_horizon_from_data` 从
> `data.action_horizon=64` 覆盖，故训练/推理实际使用 **64**。

---

## 2. 视觉数据流（当前观测 & Memory）

DINOv2-Small 冻结，输入前做 ImageNet 归一化（`DinoV2SmallEncoder._imagenet_normalize`），
`forward_features` 输出 257 个 token（token 0 = CLS，1..256 = patch）。当前观测和 Memory
**都只取 token 0（global CLS）**。

### 2.1 当前观测（`_build_obs_condition` → `ConditionEncoder.forward`）

```text
raw image        [B,1,3,3,224,224]        # (B, n_image_steps=1, V=3, RGB=3, H, W)
                 └ 训练用缓存时不走原图，直接读 full-token 缓存（见 2.3）
reshape          [3B,3,224,224]
DINOv2-Small     [3B,257,384]             # forward_features
取 token 0       [B,1,3,384]              # CLS，per view
head 投影        [B,1,3,256]              # Linear(384→256)+SiLU，逐视角共享
concat 3 views   [B,1,768]                # 3×256（先投影后拼接）
view_proj        [B,1,256]                # LN(768)+Linear(768→256)+SiLU+Linear(256→256)
取最后一帧        [B,256]                  # encode_from_backbone_feat = encode_all[:, -1]
```

状态分支（`StateMLP`，`pool=flatten`）：

```text
state            [B,8,14]  → flatten [B,112] → MLP → [B,256]
```

融合（`ConditionEncoder.fusion`）：

```text
concat([image256, state256])  [B,512]
Linear(512→hidden)+SiLU+Dropout+Linear(hidden→256)
obs_cond         [B,256]
```

### 2.2 视觉 Memory（`_build_memory` → `MemoryEncoder`，fusion）

```text
memory_image_backbone_feat  [B,128,3,384]   # 已是 CLS（缓存侧取 token 0，见 2.3）
project_view_histories_from_backbone_feat:
  permute        [B,3,128,384]
  reshape        [3B,128,384]               # view-as-batch
  head 投影      [3B,128,256]               # 与当前观测共享同一 Linear(384→256)
VisualTemporalMemoryEncoder（每个视角独立）:
  + time_embed(|offset|)  [3B,128,256]      # Embedding(max_offset+1, 256)
  TransformerEncoder(2 层, 4 头, d=256)      [3B,128,256]
  LayerNorm + mean over 128 时间步           [3B,256]
  Linear(256→256)                            [3B,256]  # per-view summary
reshape          [B,3,256]
concat 3 views   [B,768]
view_fusion_proj [B,256]                     # LN(768)+Linear(768→256)+SiLU+Dropout+Linear(256→256)
visual_global    [B,256]
```

状态 Memory（`StateConvMemoryEncoder`，独立于视觉，保持 64 帧）：

```text
memory_state     [B,64,14]
2×[Conv1d(k=5)+GroupNorm+SiLU+Dropout]（ch=128）+ masked mean pool
Linear(128→64)
state_global     [B,64]
```

视觉/状态融合（`MemoryEncoder.fusion`）：

```text
concat([visual_global256, state_global64])  [B,320]
LN(320)+Linear(320→256)+SiLU+Dropout+Linear(256→256)
memory_global    [B,256]
memory_tokens    [B,1,256]                   # memory_global.unsqueeze(1)，供 DiT cross_attn 使用
```

`memory_token_proj` 在 `memory_dim(256)==cond_dim(256)` 时为 `nn.Identity`，故不改变维度。

### 2.3 训练缓存 vs 部署 Buffer

- **训练缓存**（`tools/precompute_policy_latents.py` → `frame_backbone.zarr`）保存**完整 token**：
  `[T_episode, V, 257, D] = [T, 3, 257, 384]`。
  Dataset 采样历史后在 `get_memory_camera_latent` 里 `feat[:, :, 0, :]` 取 CLS →
  `[128, 3, 384]`（`datasets/zarr_dataset.py`）。
- **部署 Buffer**（`tools/async_dino_buffer.py`，`store_local_features=False`）只保存 CLS：
  每个 deque 元素 `entry["global_feature"]` 形状 `[B,V,D]=[1,3,384]`；
  `get_feature_window()` = `stack(dim=1)` → `[B,128,3,384]`（repeat-first 左补齐）。

两侧最终喂给 Policy 的都是 `[B,128,3,384]` CLS，且都走同一 `project_view_histories`
→ `head(384→256)`，训练/推理一致。

---

## 3. 四种 Policy 数据流

条件构造在 `models/fm/flow_policy.py::_build_condition`，返回 `(global_cond, condition_tokens)`。
`self.global_cond_dim = cond_dim`；仅当 `memory_enabled and injection=="concat_global_cond"`
时变为 `cond_dim*2 = 512`。UNet 和 DiT 均以该 `global_cond_dim` 构建。

### 3.1 UNet without Memory

```text
current obs → obs_cond            [B,256]
global_cond = obs_cond            [B,256]    # global_cond_dim=256, condition_tokens=None
noisy action x_t                  [B,64,14]
flow timestep t                   [B]
  ConditionalUnet1D:
    time embed                    [B,256]
    concat(time, global_cond)     [B,256+256=512]
    每个残差块 FiLM: Linear(512→2*C) 调制
predicted velocity                [B,64,14]
```

### 3.2 UNet with Memory（concat_global_cond）

```text
current obs → obs_cond            [B,256]
history     → memory_global       [B,256]
global_cond = cat([obs_cond, memory_global], dim=-1)   [B,512]   # 无压缩层
                                                                  # global_cond_dim=512
noisy action x_t                  [B,64,14]
flow timestep t                   [B]
  ConditionalUnet1D:
    time embed                    [B,256]
    concat(time, global_cond)     [B,256+512=768]
    每个残差块 FiLM: Linear(768→2*C) 调制    # cond_encoder.1.in_features = 256+512
predicted velocity                [B,64,14]
```

### 3.3 DiT without Memory

```text
current obs → obs_cond            [B,256]
global_cond = obs_cond            [B,256]    # condition_tokens=None
noisy action x_t                  [B,64,14]
  ActionDiT:
    input_proj                    [B,64,hidden]
    cond = time_embed(t) + cond_proj(global_cond)   [B,hidden]   # cond_proj: Linear(256→hidden*4→hidden)
    每个 DiTBlock: adaLN_modulation(cond) → shift/scale/gate（AdaLN-Zero）
predicted velocity                [B,64,14]
```

### 3.4 DiT with Memory（concat_global_cond，当前目标）

```text
current obs → obs_cond            [B,256]
history     → memory_global       [B,256]
global_cond = cat([obs_cond, memory_global], dim=-1)   [B,512]   # condition_tokens=None
noisy action x_t                  [B,64,14]
  ActionDiT:
    cond = time_embed(t) + cond_proj(global_cond)   [B,hidden]   # cond_proj: Linear(512→hidden*4→hidden)
    每个 DiTBlock: adaLN_modulation(cond) → AdaLN 调制
predicted velocity                [B,64,14]
```

**DiT 的 Memory 注入方式（按代码确认）**：
`concat_global_cond` 下，Memory 经 **AdaLN 全局条件**进入——`memory_global` 拼进 512 维
`global_cond`，再由 `cond_proj` 投影、与时间步嵌入相加，喂入每个 `DiTBlock` 的
`adaLN_modulation`。此模式 `condition_tokens=None`，**不使用 token / cross-attention**。

> 代码另存在一条 `cross_attn` 备选注入（非当前目标）：`memory_tokens [B,1,256]` 经
> `context_proj` 成 `context`，在 `cross_attn_layers=[3,7,11]` 的 `DiTBlock` 里做
> cross-attention，`obs_cond` 仍走 AdaLN。仅当 `models.memory.injection: cross_attn`
> 且 `velocity_model: dit` 时构建（`ActionDiT(condition_token_dim=…, cross_attn_layers=…)`）。
> **DiT + Memory 是支持的**（AdaLN 或 cross-attn 两条路），并非缺失。

---

## 4. 四种模式对比表（真实 Tensor shape）

| Backbone | Memory | Current obs          | Memory input                                 | Condition injection               | Model 条件输入                                     | Output                 |
| -------- | ------ | -------------------- | -------------------------------------------- | --------------------------------- | -------------------------------------------------- | ---------------------- |
| UNet     | 无     | `obs_cond [B,256]` | —                                           | FiLM，`global_cond_dim=256`     | `global_cond [B,256]`（FiLM 实宽 256+256=512）   | `velocity [B,64,14]` |
| UNet     | 有     | `obs_cond [B,256]` | `[B,128,3,384]`→`memory_global [B,256]` | FiLM，`global_cond_dim=512`     | `global_cond [B,512]`                            | `velocity [B,64,14]` |
| DiT      | 无     | `obs_cond [B,256]` | —                                           | AdaLN，`cond_proj(256→hidden)` | `global_cond [B,256]`，`condition_tokens=None` | `velocity [B,64,14]` |
| DiT      | 有     | `obs_cond [B,256]` | `[B,128,3,384]`→`memory_global [B,256]` | AdaLN，`cond_proj(512→hidden)` | `global_cond [B,512]`，`condition_tokens=None` | `velocity [B,64,14]` |

> UNet FiLM 的实际条件宽度 = `diffusion_step_embed_dim(256) + global_cond_dim`：
> 无 Memory = 256+256 = 512，有 Memory = 256+512 = 768（`ConditionalResidualBlock1D.cond_encoder[1].in_features`）。
> `velocity` 长度为 `action_horizon=64`；`predict_action` 返回 `action_normalized = [:, :n_action_steps=64]`。

---

## 5. 数据结构与容器

| 阶段                | 变量 / 容器              | 类型与 shape                                                                          |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| 原始图像            | 相机张量                 | `uint8 [B,1,V,3,224,224]`（`ZarrDataset._process_image`）                         |
| 完整 DINO token     | 训练缓存 zarr            | `float32 [T_episode, V, 257, 384]`（`frame_backbone.zarr`）                       |
| CLS 张量            | token 0                  | `float32 [B,V,384]`（单帧）/ `[B,T,V,384]`（历史）                                |
| Async Buffer 单元素 | `deque` entry（dict）  | `entry["global_feature"] = [B,V,D] = [1,3,384]`；`local_feature=None`（默认不存） |
| Buffer 窗口         | `get_feature_window()` | `float32 [B,128,V,384] = [1,128,3,384]`                                             |
| Dataset batch       | `dict`                 | `{"obs": {...}, "action":[B,64,14], "meta":{idx,anchor_t,ep_idx}}`                  |
| Policy observation  | `obs` dict             | 见下                                                                                  |
| memory_global       | Tensor                   | `[B,256]`                                                                           |
| memory_tokens       | Tensor                   | `[B,1,256]`                                                                         |
| noisy action        | Tensor                   | `[B,64,14]`                                                                         |
| timestep            | Tensor                   | `[B]`（flow 时间 t∈[0,1]）                                                         |
| Policy output       | dict                     | `{"action_normalized":[B,64,14], "action_pred_normalized":[B,64,14]}`               |

`obs` dict 关键字段（`ZarrDataset.__getitem__` / `infer/runtime.py`）：

```text
当前观测:
  "state"                      [B,8,14]
  "image_backbone_feat"        [B,1,3,257,384]   # 训练：full token；推理当前帧同理
  （或原图 "image" [B,1,3,3,224,224]，当 use_camera_latent=false）
Memory（memory_enabled 时追加）:
  "memory_state"               [B,64,14]
  "memory_image_backbone_feat" [B,128,3,384]     # CLS
  "memory_visual_offsets"      [128]             # [-1016,…,0]
  "memory_visual_valid"        [B,128]           # 全 True（padding 也参与）
  "memory_state_valid"         [B,64]
```

- 训练缓存保存**完整 token** `[T,V,257,D]`；
- 部署 Async Buffer 默认只保存 **CLS**，单元素 `[V,D]=[3,384]`（含 batch 维即 `[1,3,384]`）；
- 固定窗口输出 `[B,128,V,D]=[1,128,3,384]`。

---

## 6. 训练与推理一致性对照

| 项                   | 训练                                                        | 推理                                                   | 一致点                                          |
| -------------------- | ----------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| 图像预处理           | resize 224 + ImageNet 归一化                                | 同                                                     | `DinoV2SmallEncoder._imagenet_normalize`      |
| DINO 权重            | `vit_small_patch14_dinov2`，冻结                          | 同                                                     | `pretrained_weights/dinov2_small.safetensors` |
| CLS token index      | `feat[:, :, 0, :]`                                        | `cls_token_from_output = tokens[:, :1]`              | 均取 token 0                                    |
| 视角顺序             | `base_0, left_wrist_0, right_wrist_0`                     | 同                                                     | `CAMERA_BUNDLE_ORDER`                         |
| 128×8 offsets       | `_build_memory_visual_offsets` → `[-1016,…,0]`        | `runtime.memory_visual_offsets` → 同                | 128 个，stride 8                                |
| repeat-first padding | 索引 clamp 到 episode 首帧；`memory_visual_valid` 全 True | `get_feature_window` 首帧左补齐                      | 语义一致                                        |
| feature dtype        | float32                                                     | float32                                                | —                                              |
| Memory input shape   | `[B,128,3,384]`                                           | `[B,128,3,384]`                                      | runtime 校验`(t,v,c)`                         |
| condition injection  | `concat_global_cond` → 512                               | 同（`predict_action` 内部同一 `_build_condition`） | 512                                             |

---

## 7. 代码位置索引

| 数据流环节           | 文件                                                                                                             | 类 / 函数                                                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| DINO token 提取      | `models/fm/encoders/dino_v2.py`                                                                                | `DinoV2SmallEncoder.forward_tokens` / `extract_backbone_feat` / `cls_token_from_output`                             |
| 当前观测视觉融合     | `models/fm/encoders/dino_v2.py`                                                                                | `DinoV2Encoder.encode_all_from_backbone_feat` / `_fuse_view_feats`                                                    |
| 当前观测条件         | `models/fm/condition_encoder.py`                                                                               | `ConditionEncoder.forward` / `StateMLP`                                                                               |
| Dataset 历史采样     | `datasets/zarr_dataset.py`                                                                                     | `_build_memory_visual_offsets` / `memory_visual_indices` / `get_memory_camera_latent`（CLS 提取） / `__getitem__` |
| 训练缓存             | `tools/precompute_policy_latents.py`                                                                           | full-token`frame_backbone.zarr`                                                                                         |
| Async Buffer         | `tools/async_dino_buffer.py`                                                                                   | `AsyncDinoBuffer.submit_frame` / `_run_dino` / `get_global_feature_window`                                          |
| 视觉 Temporal Memory | `models/fm/memory/fusion.py`                                                                                   | `MemoryEncoder` / `VisualTemporalMemoryEncoder` / `StateConvMemoryEncoder` / `encode_visual_views`                |
| Memory 工厂          | `models/fm/memory/factory.py`                                                                                  | `build_memory_encoder`                                                                                                  |
| condition fusion     | `models/fm/flow_policy.py`                                                                                     | `_build_obs_condition` / `_build_memory` / `_build_condition`                                                       |
| UNet Policy          | `models/diffusion/conditional_unet1d.py`                                                                       | `ConditionalUnet1D` / `ConditionalResidualBlock1D`(FiLM)                                                              |
| DiT Policy           | `models/fm/action_dit.py`                                                                                      | `ActionDiT` / `DiTBlock`(AdaLN, cross_attn)                                                                           |
| 采样 / 预测          | `models/fm/flow_policy.py`                                                                                     | `predict_action` / `conditional_sample` / `_model_forward`                                                          |
| 训练循环             | `trainers/policy_trainer.py`                                                                                   | `build_policy`（`sync_fm_action_horizon_from_data`）/ `main`                                                        |
| 推理运行时           | `infer/runtime.py`                                                                                             | `FMInferenceRuntime._get_async_memory_obs` / `predict_rot6d_abs`                                                      |
| benchmark            | `tools/bench_*_latency.py` / `tools/run_latency_calibration.sh` / `tools/test_dino_global_local_memory.py` | 见各文件                                                                                                                  |
| shape 测试           | `tools/test_memory_cond_dim.py`                                                                                | disabled 256 / enabled 512                                                                                                |

---

## 8. 如何启动训练

前置：数据集 zarr 已就绪，`configs/train/config.yaml` 的 `data.root_dir` 指向数据；
`pretrained_weights/dinov2_small.safetensors` 存在（`HF_HUB_OFFLINE=1` 离线加载）。

```bash
cd /home/tracy/uncertain_rl/flow_matching   # 或远端 /mnt/workspace/zyj/rl/lyc/flow_matching

# 1) 预计算 DINO full-token 缓存（frame_backbone.zarr，[T,V,257,D]）
#    换 window/stride/horizon/memory 无需重算；换数据或 DINO 模型才需 --force
./scripts/precompute.sh --config configs/train/config.yaml --gpus 0

# 2) 训练（单卡）
./scripts/train.sh --config configs/train/config.yaml --gpus 0

# 2') 多卡 DDP（>1 GPU 自动走 torchrun）
./scripts/train.sh --config configs/train/config.yaml --gpus 0,1,2,3,4,5,6,7

# 或一步完成：先 precompute 再 train
./scripts/run_all.sh --config configs/train/config.yaml --gpus 0
```

- 入口：`train.py --config <yaml>` → `trainers.policy_trainer.main`。
- 超参改 `configs/train/config.yaml`（`data / train / models / output / checkpoint`）。
- checkpoint：`{output.root_dir}/{run_name}/checkpoints/latest.pt`（每 epoch）+ `epoch_XXXX.pt`。
- 切换骨干：`models.fm.velocity_model: unet|dit`；切换 Memory 注入：`models.memory.injection: concat_global_cond|cross_attn`。
- 开启/关闭 Memory：`data.memory.enabled: true|false`。
