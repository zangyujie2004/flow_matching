# 触觉 Stage 1 / Stage 2：训练、预测、测试与数据格式

本文档描述 `/mnt/workspace/lyc/flow_matching` 当前实际实现，重点对应正在使用的 QPOS 联合动作–触觉模型：

```text
outputs/huanggua_office_0729_qpos_joint_tactile/huanggua_office_unet
```

当前模型不是“先预测 action，再单独预测 tactile”。Stage 2 的 UNet 在一次 Flow Matching 推理中联合生成未来 action 和未来 tactile latent。

---

## 1. 一句话概括

```text
Stage 1
单帧四路触觉形变图
    -> 共享权重的 Conv2D Residual Encoder
    -> 4 个 16-D token（合计 64-D）
    -> 共享权重的转置卷积 Decoder
    -> 重建四路触觉形变图

预计算
每个数据帧的触觉图 -> 冻结的 Stage 1 Encoder -> 标准化 64-D latent cache

Stage 2
视觉 + 8 帧 QPOS 状态 + 当前帧触觉 latent
    -> global condition (B,256)

高斯轨迹 (B,128,78)
    -> 条件 UNet Flow Matching
    -> 未来 128 步 [14-D QPOS action + 64-D tactile latent]
    -> action 反归一化；tactile latent 经冻结 Decoder 还原为触觉图
```

其中 QPOS 联合轨迹每个时间步是：

```text
14-D action + 64-D tactile latent = 78-D
```

EEF 模式只改变机器人状态和 action 的维度：

```text
20-D action + 64-D tactile latent = 84-D
```

Stage 1 与 QPOS/EEF 无关；它只训练触觉图的编码和重建。QPOS/EEF 的区别只出现在 Stage 2。

---

## 2. 当前数据和模型快照

### 2.1 Stage 1 / Stage 2 训练数据

```text
/mnt/workspace/lyc/data/huanggua_office/huanggua_office_0729_2248/replay_buffer.zarr
```

实际数据：

| Zarr key | shape | dtype | 含义 |
|---|---:|---|---|
| `data/camera` | `(503245,224,224,9)` | `uint8` | 3 个 RGB 相机在通道维拼接 |
| `data/state_30hz` | `(503245,62)` | `float32` | 机器人状态全集 |
| `data/action_30hz` | `(503245,62)` | `float32` | 机器人 action 全集 |
| `data/tactile` | `(503245,35,20,24)` | `float32` | 4 个触觉传感器，每个 6 通道 |
| `meta/episode_ends` | `(246,)` | integer | 246 个 episode 的累计结束位置 |

### 2.2 独立重建测试数据

```text
/mnt/workspace/lyc/data/huanggua_office/huanggua_office_0731_2151/replay_buffer.zarr
```

它有 285 个 episode、575783 帧。此前测试的原始子任务：

```text
/mnt/oss_data/arx3/huanggua_office/peel_huanggua_coffee_xin_wl_0730_65
```

在处理后数据中对应 global episode `[160,225)`，共 65 个 episode。

### 2.3 当前 checkpoint

Stage 1 最佳 checkpoint：

```text
outputs/huanggua_office_0729_tactile_ae/tactile_residual_4x16/checkpoints/best.pt
```

- 最佳 epoch：97
- 最佳验证 L1：`0.0061573484`，这是归一化到 `[-1,1]` 后的触觉图 L1
- AE 参数：4 个传感器、每个 3 通道、每个 token 16-D

Stage 2 QPOS checkpoint：

```text
outputs/huanggua_office_0729_qpos_joint_tactile/huanggua_office_unet/checkpoints/latest.pt
```

- epoch：256
- global step：228352
- velocity model：Conditional UNet1D
- 推理：32 个 Euler 积分步

---

## 3. 原始数据格式

### 3.1 QPOS 和 EEF

`state_30hz/action_30hz` 的总维度是 62：

```text
0:14   dual-arm QPOS
14:34  dual-arm EEF
34:48  effort
48:62  velocity
```

当前 QPOS Stage 2 使用 `0:14`：

```text
[left_joint_1 ... left_joint_6, left_gripper,
 right_joint_1 ... right_joint_6, right_gripper]
```

shape：

```text
单帧 QPOS: (14,)
8 帧状态历史: (8,14)
128 步 action: (128,14)
```

EEF 使用 `14:34`，每只手臂 10-D：

```text
[x, y, z, rot6d_0 ... rot6d_5, gripper]
```

左右手拼接为 20-D。

### 3.2 触觉

四个传感器的固定顺序是：

```text
0 left_wrist_0
1 left_wrist_1
2 right_wrist_0
3 right_wrist_1
```

原始单帧 shape 为 `(35,20,24)`。每个传感器占 6 个通道：

```text
[x, y, z, dx, dy, dz]
```

模型只提取每个传感器的 `dx,dy,dz`：

```text
(35,20,24)
    -> 4 x (35,20,3)
    -> 沿通道拼接
    -> (35,20,12)
```

12 通道顺序是：

```text
[left_wrist_0(dx,dy,dz),
 left_wrist_1(dx,dy,dz),
 right_wrist_0(dx,dy,dz),
 right_wrist_1(dx,dy,dz)]
```

注意：代码中的 `dx/dy/dz` 是点云形变量。除非外部已经做过力标定，否则不应直接把它们称为牛顿单位的“力”。切向量通常画为：

```text
|dxy| = sqrt(dx^2 + dy^2)
```

### 3.3 视觉

相机顺序：

```text
base_0, left_wrist_0, right_wrist_0
```

原始图像每帧为 `(224,224,9)`，拆成 3 路 `(3,224,224)`。

当前训练使用预计算 DINOv2-S/14 CLS cache：

```text
frame_image_backbone_feat: (503245,3,384)
```

DINOv2 原始 `forward_features` 的确有 257 个 token：

```text
1 CLS + 16 x 16 patch = 257
```

但当前配置是：

```yaml
precompute:
  token_mode: cls
models:
  fm:
    view_pool: global_concat
```

所以 cache 和 Stage 2 条件分支只使用每个相机的 CLS token，不把 257 个 token 全部送进 Stage 2。

---

## 4. Stage 1：触觉 Autoencoder

### 4.1 数据划分

Stage 1 先按 episode 划分，而不是从同一个 episode 随机拆帧：

```yaml
data:
  val_fraction: 0.1
seed: 42
```

当前 246 个 episode 被划分为：

```text
221 个训练 episode
25 个验证 episode
```

触觉 min/max normalizer 只用训练 episode 拟合，然后将每个 deformation 通道映射到 `[-1,1]`。验证 episode 不参与 normalizer 拟合。

### 4.2 输入 shape

Dataset 每次读取一个触觉帧：

```text
raw tactile                    (B,35,20,24)
extract dx/dy/dz               (B,35,20,12)
min-max normalize to [-1,1]    (B,35,20,12)
```

AE 内部会补一个长度为 1 的时间维：

```text
(B,35,20,12) -> (B,1,35,20,12)
```

Encoder 永远取最后一帧，因此 Stage 1 当前没有时间建模。

### 4.3 Encoder

先把 4 个传感器拆开：

```text
(B,1,35,20,12)
 -> (B,4,3,35,20)
 -> (B*4,3,35,20)
```

四个传感器共享同一套 Encoder 权重：

| 层 | 输出 shape |
|---|---|
| Conv2D + GN + SiLU + Residual | `(B*4,32,35,20)` |
| stride-2 Conv2D + Residual | `(B*4,64,18,10)` |
| stride-2 Conv2D + Residual | `(B*4,128,9,5)` |
| AdaptiveAvgPool + Linear | `(B*4,16)` |

然后恢复传感器维：

```text
sensor tokens: (B,4,16)
token grid:    (B,16,4)
latent:        (B,64)
```

实际 flatten 顺序来自 `(B,16,4)`，即先 token 维、再传感器维展开。虽然概念上是四个 16-D token，但不能假设保存的 64-D 数组是四段连续的 `[16,16,16,16]`；解码时必须使用代码中的对应 unflatten 逻辑。

### 4.4 Decoder

每个 16-D sensor token 使用同一个共享 Decoder：

```text
(B*4,16)
 -> Linear
 -> (B*4,128,9,5)
 -> Residual + ConvTranspose2D
 -> (B*4,64,18,10)
 -> Residual + ConvTranspose2D
 -> (B*4,32,35,20)
 -> Residual + Conv2D + tanh
 -> (B*4,3,35,20)
```

合并后：

```text
(B,4,3,35,20) -> (B,35,20,12)
```

### 4.5 Stage 1 loss

当前配置：

```yaml
loss:
  type: l1
```

训练目标是归一化触觉图的逐元素 L1：

```text
L_AE = mean(abs(tactile_reconstruction - tactile_target))
```

代码同时记录 L1 和 MSE，但只有 `loss.type` 指定的项用于反向传播。可以改为 `mse`，但当前 checkpoint 是用 L1 训练的。

### 4.6 Stage 1 训练参数和命令

配置文件：

```text
configs/train/tactile_ae.yaml
```

关键参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `frame_stride` | 1 | 使用每一帧触觉 |
| `token_dim` | 16 | 每个传感器 token 维度 |
| `hidden_dims` | `[32,64,128]` | Encoder/Decoder 通道数 |
| `epochs` | 100 | 训练到第 100 epoch 自动停止 |
| `batch_size` | 256 | 每个 GPU 的 batch size |
| `lr` | `3e-4` | AdamW 学习率 |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `mixed_precision` | bf16 | 混合精度 |
| `save_every` | 10 | 每 10 epoch 保存一次 |

8 卡且保持原单卡 global batch 256：

```bash
cd /mnt/workspace/lyc/flow_matching

PYTHON=/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
./scripts/train_tactile_ae.sh \
  --config configs/train/tactile_ae.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --batch-size 32 \
  --amp bf16
```

继续训练：

```bash
PYTHON=/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
./scripts/train_tactile_ae.sh \
  --config configs/train/tactile_ae.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --batch-size 32 \
  --amp bf16 \
  --resume outputs/huanggua_office_0729_tactile_ae/tactile_residual_4x16/checkpoints/latest.pt
```

---

## 5. Stage 1 和 Stage 2 之间的预计算

### 5.1 触觉 latent cache

Stage 2 训练不会重复对每个目标帧运行 AE Encoder。先离线编码全部 503245 帧：

```bash
cd /mnt/workspace/lyc/flow_matching

PYTHON=/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
./scripts/precompute_tactile_latents.sh \
  outputs/huanggua_office_0729_tactile_ae/tactile_residual_4x16/checkpoints/best.pt \
  /mnt/workspace/lyc/data/huanggua_office/huanggua_office_0729_2248/tactile_latent_4x16.zarr \
  0
```

处理顺序：

```text
raw (T,35,20,24)
 -> deformation (T,35,20,12)
 -> Stage 1 tactile min-max normalization
 -> frozen Encoder
 -> raw latent (T,64)
 -> 按 64 个维度分别做 z-score
 -> cached normalized latent (T,64)
```

当前 cache：

```text
tactile_latent_4x16.zarr/latent: (503245,64), float32
```

cache 同时保存：

- episode boundaries；
- raw latent 的 64-D mean/std；
- Stage 1 tactile normalizer scale/offset；
- Stage 1 checkpoint 路径和 SHA256；
- token layout。

Stage 2 会严格检查 cache 的 episode 边界和帧数是否与 replay buffer 一致。

### 5.2 视觉 cache

```bash
cd /mnt/workspace/lyc/flow_matching

PYTHON=/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
./scripts/precompute.sh \
  --config configs/train/config_qpos_stage2_joint_tactile.yaml \
  --gpus 0,1,2,3
```

当前 `token_mode=cls` 结果：

```text
frame_backbone.zarr/data/frame_image_backbone_feat
shape = (503245,3,384)
```

`mix_base_remove_hand=true` 时还会使用 base 相机去手版本 cache；左右腕相机不替换。

---

## 6. Stage 2：联合预测 action 和 tactile latent

### 6.1 `predict_tactile` 开关

```yaml
models:
  fm:
    predict_tactile: true
```

| 设置 | global condition 中是否有触觉 | Flow Matching 是否预测未来触觉 | 是否需要 Stage 1 AE/cache |
|---|---|---|---|
| `false` | 是，8帧原始触觉经过 TemporalCNN | 否，只预测 action | 不需要联合目标 latent cache |
| `true` | 是，同样使用8帧 TemporalCNN | 是，联合预测未来 latent | 必须提供 AE checkpoint 和 latent cache |

`false` 时触觉只作为条件，QPOS 轨迹为 `(B,128,14)`。`true` 时 QPOS 联合轨迹为 `(B,128,78)`。
`predict_tactile` 只控制输出目标；输入触觉编码方式由
`tactile_condition_encoder_type` 独立控制。新配置使用 `temporal_cnn`。
旧 checkpoint 没有该字段时仍按旧逻辑使用当前帧 precomputed latent。

### 6.2 一个训练样本的时间索引

设当前 anchor 为 `t`，当前配置：

```yaml
window_size: 8
n_image_steps: 1
action_horizon: 128
tactile_obs_steps: 8
tactile_target_offset: 1
```

Dataset 取：

```text
state:          [t-7, ..., t]       -> (8,14)
vision:         [t]                 -> 1 帧 x 3 相机
tactile history:[t-7, ..., t]       -> (8,35,20,12)
action target: [t, ..., t+127]      -> (128,14)
tactile target:[t+1, ..., t+128]    -> (128,64)
```

所以联合目标的第 `k` 行是：

```text
[action(t+k), tactile_latent(t+1+k)]
```

这是当前代码的精确对齐方式。`tactile_target_offset=1` 表示用当前观测预测下一帧开始的触觉。

### 6.3 action 表示

当前 QPOS 配置：

```yaml
action_type: joint
action_representation: relative
```

训练前先相对当前 anchor QPOS 转换：

```text
relative_action(t+k) = absolute_action(t+k) - state(t)
```

再按每个 action 维度 min-max 归一化到 `[-1,1]`。推理后执行逆变换：

```text
absolute_action = predicted_relative_action + state(t)
```

EEF relative 使用 SE(3) 变换，而不是简单逐元素相减。

### 6.4 global condition 数据流

#### 状态分支

```text
normalized state history (B,8,14)
 -> flatten
 -> (B,112)
 -> MLP 112 -> 512 -> 256
 -> state feature (B,256)
```

EEF 时输入为 `(B,8,20)`，flatten 后是 `(B,160)`，输出仍为 `(B,256)`。

#### 视觉分支

```text
cached CLS features (B,1,3,384)
 -> 取最后一个图像时间步
 -> 每路 384 -> 256
 -> 三路拼接 (B,768)
 -> view projection
 -> vision feature (B,256)
```

#### 历史触觉分支

```text
normalized tactile history (B,8,35,20,12)
 -> 逐帧 Conv2D
 -> frame features (B,8,64)
 -> temporal Conv1D + temporal pooling
 -> tactile feature (B,256)
```

这里不使用 Stage 1 Encoder。Stage 1 AE 仍然逐帧训练并保持冻结，只用于定义、预计算和解码未来每帧的 64-D tactile latent。

#### 融合

```text
vision  (B,256) ┐
tactile (B,256) ├ concat -> (B,768) -> MLP 768->512->256
state   (B,256) ┘

global_cond = (B,256)
```

当前 `memory.enabled=false`，因此没有额外 memory token 注入。

### 6.5 Flow Matching 联合轨迹

QPOS：

```text
action target          (B,128,14)
tactile latent target  (B,128,64)
concat x1              (B,128,78)
```

EEF：

```text
action target          (B,128,20)
tactile latent target  (B,128,64)
concat x1              (B,128,84)
```

训练时采样：

```text
x0 ~ N(0,I)
t  ~ Uniform(0,1)
xt = (1-t) x0 + t x1
target_velocity = x1 - x0
```

Conditional UNet 输入：

```text
xt:          (B,128,78)  # QPOS
t:           (B,)
global_cond: (B,256)
```

输出：

```text
pred_velocity: (B,128,78)
```

### 6.6 Stage 2 loss

action 和 tactile 分支分别求均值 MSE：

```text
L_action  = MSE(v_pred[...,0:14],  v_target[...,0:14])
L_tactile = MSE(v_pred[...,14:78], v_target[...,14:78])

L_total = action_loss_weight * L_action
        + tactile_loss_weight * L_tactile
```

当前两个权重都是 1.0。由于两部分先分别求 mean 再加权，64-D tactile 不会仅因为维度更多就自动获得 `64/14` 倍权重。

### 6.7 冻结和训练的模块

Stage 2 中：

- Stage 1 AE Encoder/Decoder：冻结、始终处于 eval；
- DINOv2 backbone：冻结；
- DINO projection head / 三视角融合层：可训练；
- 8帧 TactileCNNEncoder：可训练；
- State MLP：可训练；
- condition fusion MLP：可训练；
- Conditional UNet：可训练；
- tactile latent 本身来自预计算 cache，不对 AE 反向传播。

### 6.8 当前 QPOS Stage 2 训练命令

首次训练前生成共享的 deformation-only float32 mmap cache：

```bash
cd /mnt/workspace/lyc/flow_matching

/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
  tools/precompute_tactile_deformation_cache.py \
  --data-root /mnt/workspace/lyc/data/huanggua_office/huanggua_office_0729_2248 \
  --batch-frames 256
```

该缓存约 15.75 GiB，8个 DDP rank 只读同一个 mmap，避免每个 rank 各自预加载约 31.5 GiB 原始24通道触觉。

```bash
cd /mnt/workspace/lyc/flow_matching

PYTHON=/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
./scripts/train.sh \
  --config configs/train/config_qpos_stage2_joint_tactile.yaml \
  --gpus 0,1,2,3 \
  --amp bf16
```

注意：`train.batch_size` 是每张 GPU 的 batch size。当前 YAML 是 256，因此 4 卡 global batch 是 1024。

当前 Stage 2 没有独立的 episode-level validation dataset。每 10 epoch 的 `open_loop` 是从同一个 `ZarrDataset` 随机抽窗口，因此它是训练分布上的监控，不是严格的 held-out 验证结果。

---

## 7. 单次预测流程

### 7.1 输入

以 QPOS、batch size 1 为例：

```text
state_raw                (8,14)
state_normalized         (1,8,14)
RGB images               (1,1,3,3,224,224)
historical tactile       (1,8,35,20,12)  # TemporalCNN 使用全部8帧
```

使用预计算视觉特征时，图像输入替换为：

```text
image_backbone_feat      (1,1,3,384)
```

### 7.2 ODE 采样

推理从随机高斯联合轨迹开始：

```text
trajectory_0 ~ N(0,I), shape=(B,128,78)
```

当前参数：

```yaml
num_inference_steps: 32
solver: euler
```

Euler 从时间 0 积分到 1，共调用 UNet 32 次。最终得到：

```text
joint prediction              (B,128,78)
action_pred_normalized        (B,128,14)
tactile_latent_pred_normalized(B,128,64)
```

### 7.3 输出后处理

action：

```text
(B,128,14) normalized relative QPOS
 -> action normalizer inverse
 -> 加回 anchor QPOS
 -> (B,128,14) absolute QPOS
```

tactile：

```text
(B,128,64) normalized latent
 -> latent std/mean inverse
 -> frozen Stage 1 Decoder
 -> normalized tactile (B,128,35,20,12)
 -> Stage 1 tactile normalizer inverse
 -> physical/raw deformation space (B,128,35,20,12)
```

也就是说，一次网络调用同时预测未来 128 步，而不是每一步单独调用网络一次。

---

## 8. action chunk、执行步数与多久重新预测

这三个概念必须分开：

| 参数 | 当前值 | 作用位置 | 含义 |
|---|---:|---|---|
| `action_horizon` | 128 | 训练和模型 | 每次生成未来 128 步 action 和 tactile |
| `n_action_steps` | 当前 run 内解析为 128 | Policy 返回短 action 字段 | 最多暴露多少 action；runtime 仍可取完整预测 |
| 部署 `inference.n_action_steps` | 示例为 32 | Prometheus adapter | 实际截取并执行前多少步后重新推理 |
| episode 测试 `chunk_stride` | 默认 30 | 可视化工具 | 每隔多少数据帧重新条件预测并拼图 |

在 30 Hz 下：

```text
模型预测 128 步 = 4.267 秒未来
部署执行 32 步 = 1.067 秒后重新观测并推理
episode 测试 stride 30 = 1.000 秒后重新条件预测
```

如果部署配置为 `n_action_steps=32`，完整闭环是：

```text
时刻 t：读取 8 帧状态、当前视觉、当前触觉
 -> 一次性预测 action[t:t+128] 和 tactile[t+1:t+129]
 -> 只执行 action 的前 32 步
 -> 到约 t+32 时重新获取状态、视觉、触觉
 -> 再预测新的未来 128 步 action/tactile
```

因此“每多少步预测一次触觉”的答案是：模型每次 policy inference 都会联合预测一次完整的 128 步 tactile；若部署执行 32 步后 replan，就是每 32 个控制步重新预测一次 tactile。

当前 `FMInferenceRuntime.infer_from_window()` 返回 action chunk，并没有默认把触觉图交给机器人调度器。需要查看触觉输出时使用 `predict_rot6d_abs_with_tactile()` 或离线测试工具。

---

## 9. 测试流程

### 9.1 训练期 open-loop

每 `open_loop_test_every=10` 个 epoch：

1. 从当前训练使用的同一个 Dataset 随机抽窗口；
2. 用 32-step Euler 从高斯噪声生成完整 128 步联合轨迹；
3. action 反归一化到绝对 QPOS/EEF；
4. 计算 action MAE/MSE；
5. 在 64-D normalized latent 空间计算 tactile latent MAE/MSE。

它不解码到 `(35,20,12)` 后再计算触觉图误差，也不是 held-out episode 验证。

### 9.2 独立 Stage 2 触觉重建测试

工具：

```text
tools/eval_tactile_reconstruction.py
```

对每个窗口：

```text
真实 state history + 真实视觉 + 8帧真实 tactile history
 -> 8帧 tactile 经 TemporalCNN 得到 256-D condition
 -> Stage 2 预测未来 128 个 tactile latent
 -> Stage 1 Decoder 解码为 (128,35,20,12)
 -> 与真实未来 tactile 比较
```

同时计算 AE oracle：

```text
未来真实 tactile -> Stage 1 Encoder -> Stage 1 Decoder -> reconstruction
```

AE oracle 只衡量 Stage 1 压缩/重建误差；Stage 2 误差还包含“未来 latent 预测误差”。因此 Stage 2 通常不可能明显优于 AE oracle。

4 卡测试命令：

```bash
cd /mnt/workspace/lyc/flow_matching

mkdir -p outputs/tactile_eval/0730_65_stride30_4gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  tools/eval_tactile_reconstruction.py \
  --run-dir outputs/huanggua_office_0729_qpos_joint_tactile/huanggua_office_unet \
  --data-root /mnt/workspace/lyc/data/huanggua_office/huanggua_office_0731_2151 \
  --subtask-path /mnt/oss_data/arx3/huanggua_office/peel_huanggua_coffee_xin_wl_0730_65 \
  --base-mode both \
  --sample-stride 30 \
  --max-windows -1 \
  --batch-size 4 \
  --decode-frame-batch-size 128 \
  --num-inference-steps 32 \
  --amp bf16 \
  --seed 42 \
  --plot-samples 8 \
  --output-dir outputs/tactile_eval/0730_65_stride30_4gpu \
  2>&1 | tee outputs/tactile_eval/0730_65_stride30_4gpu/run.log
```

当前得到的物理/raw deformation 空间汇总结果：

```text
original:
  Stage2 MAE = 0.0027333234
  Stage2 MSE = 4.7475898e-05
  AE oracle MAE = 0.00080569844
  AE oracle MSE = 5.0794112e-06

remove:
  Stage2 MAE = 0.0027374223
  Stage2 MSE = 4.7575031e-05
  AE oracle MAE = 0.00080569844
  AE oracle MSE = 5.0794112e-06
```

`sample_stride=30` 只决定隔多少 anchor 抽一个测试窗口；每个被选中的窗口仍预测完整 128 步。

### 9.3 完整 episode 滚动可视化

工具：

```text
tools/visualize_tactile_episode_prediction.py
```

它不是从 episode 开头一次预测后面两千帧。模型最大 horizon 是 128，因此采用 receding-horizon 拼接：

```text
anchor t0 -> 预测未来 128 帧 -> 保留前 chunk_stride 帧
anchor t0+chunk_stride -> 用该处真实观测重新预测 -> 再保留前 chunk_stride 帧
重复直到 episode 结束
```

episode 198、去手视觉、每 30 帧重预测：

```bash
cd /mnt/workspace/lyc/flow_matching

mkdir -p outputs/tactile_eval/full_episode_ep0198_remove

CUDA_VISIBLE_DEVICES=0 \
/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python \
  tools/visualize_tactile_episode_prediction.py \
  --run-dir outputs/huanggua_office_0729_qpos_joint_tactile/huanggua_office_unet \
  --data-root /mnt/workspace/lyc/data/huanggua_office/huanggua_office_0731_2151 \
  --episode 198 \
  --base-mode remove \
  --chunk-stride 30 \
  --batch-size 4 \
  --decode-frame-batch-size 128 \
  --num-inference-steps 32 \
  --amp bf16 \
  --seed 42 \
  --fps 30 \
  --output-dir outputs/tactile_eval/full_episode_ep0198_remove \
  2>&1 | tee outputs/tactile_eval/full_episode_ep0198_remove/run.log
```

输出：

| 文件 | 内容 |
|---|---|
| `episode_tactile_prediction.mp4` | 4 个传感器的 GT、预测和误差热图 |
| `episode_temporal_curves.png` | dx、dy、dz、切向模长的完整时间曲线 |
| `episode_error_curves.png` | Stage 2、AE oracle、persistence 的逐帧误差 |
| `frame_metrics.csv` | 每一帧的 MAE/MSE |
| `metrics.json` | 全 episode 汇总指标和覆盖率 |
| `episode_prediction.npz` | GT、Stage 2、AE oracle、persistence 原始数组 |

episode 开头 8 帧用于构建状态观测历史，因此没有预测。`--max-segments 1` 只是烟雾测试，只会预测 30 帧，不能用它判断完整 episode 曲线。

---

## 10. 关键 shape 总表

下面以当前 QPOS Stage 2 为准：

| 数据/张量 | shape |
|---|---:|
| raw tactile frame | `(B,35,20,24)` |
| deformation tactile frame | `(B,35,20,12)` |
| per-sensor maps | `(B,4,3,35,20)` |
| Stage 1 sensor tokens | `(B,4,16)` |
| Stage 1 token grid | `(B,16,4)` |
| flattened tactile latent | `(B,64)` |
| state history | `(B,8,14)` |
| cached visual CLS | `(B,1,3,384)` |
| vision feature | `(B,256)` |
| state feature | `(B,256)` |
| tactile history | `(B,8,35,20,12)` |
| temporal tactile condition | `(B,256)` |
| fusion input | `(B,768)` |
| global condition | `(B,256)` |
| future action target | `(B,128,14)` |
| future tactile target | `(B,128,64)` |
| joint FM trajectory | `(B,128,78)` |
| decoded tactile prediction | `(B,128,35,20,12)` |

---

## 11. 最容易混淆的参数

### `window_size=8`

使用8帧 QPOS 状态历史。Stage 1 AE 仍然逐帧工作；Stage 2 的 TemporalCNN 使用8帧触觉历史。

### `n_image_steps=1`

条件中只使用当前 anchor 的视觉帧。

### `tactile_obs_steps=8`

Stage 2 条件使用 `[t-7,...,t]` 共8帧原始 deformation。它不改变每个未来触觉 latent 的64维大小。

### `tactile_target_offset=1`

未来触觉 target 从 `t+1` 开始，而 action target 从 `t` 开始。

### `action_horizon=128`

每次模型生成 128 步 action，同时生成 128 步 tactile latent。

### `num_inference_steps=32`

这是 Flow ODE 的数值积分次数，不是 action 数量，也不是预测时间长度。Euler 下 UNet 调用 32 次。

### `sample_stride=30`

独立窗口测试时，每隔 30 个 anchor 选一个窗口；不会把每个窗口的 128 步预测缩短为 30。

### `chunk_stride=30`

完整 episode 拼接时保留每个 128 步预测的前 30 步，并在新 anchor 重新预测。

### `n_action_steps=32`

部署层可以只执行模型 128 步 action 中的前 32 步，然后 replan。它与测试用 `chunk_stride` 是两个独立参数。

### `predict_tactile=false`

触觉仍可进入 global condition，但联合轨迹不再包含 64-D tactile latent，也不产生未来触觉预测。

---

## 12. 关键源码位置

| 功能 | 文件 |
|---|---|
| 触觉通道提取 | `tools/tactile_feat.py` |
| Stage 1 数据集和 episode split | `datasets/tactile_ae_dataset.py` |
| Residual Encoder | `models/fm/encoders/tactile_token.py` |
| AE Decoder / flatten layout | `models/fm/encoders/tactile_autoencoder.py` |
| Stage 1 trainer | `trainers/tactile_ae_trainer.py` |
| 触觉 latent 预计算 | `tools/precompute_tactile_latents.py` |
| 8帧 deformation mmap 缓存 | `tools/precompute_tactile_deformation_cache.py` |
| Stage 2 窗口和时间对齐 | `datasets/zarr_dataset.py` |
| vision/state/tactile 条件融合 | `models/fm/condition_encoder.py` |
| 联合 Flow Matching loss 和采样 | `models/fm/flow_policy.py` |
| 训练期 open-loop | `trainers/eval_open_loop.py` |
| 在线推理 runtime | `infer/runtime.py` |
| 独立触觉重建测试 | `tools/eval_tactile_reconstruction.py` |
| 完整 episode 可视化 | `tools/visualize_tactile_episode_prediction.py` |

---

## 13. 使用前检查清单

1. Stage 1 checkpoint 与 tactile latent cache 的 SHA256/路径一致。
2. tactile cache 的 `episode_ends` 与训练 replay buffer 完全一致。
3. QPOS 使用 `action_type: joint`，不要误用 EEF config。
4. `predict_tactile=true` 时必须同时存在 AE checkpoint 和 tactile latent cache。
5. `tactile_condition_encoder_type=temporal_cnn` 时必须先生成配置指定的 deformation mmap cache。
6. 当前视觉是 3 相机、CLS-only cache；不要把 `view_pool` 改为 `local_pool/local_attn` 后继续使用 CLS-only cache。
7. 多卡训练里的 `train.batch_size` 是每卡 batch，不是 global batch。
8. 训练期 open-loop 不是独立验证集；泛化结论应以独立 episode/subtask 测试为准。
9. 完整 episode 的正式结果不要使用 `--max-segments 1` 或 `--num-inference-steps 1`。
10. 部署 QPOS checkpoint 时，机器人 action space 应配置为 `abs_qpos`。
11. 修改 `action_horizon` 后，需要保证 Dataset、Policy、测试和部署的时间长度一致。
