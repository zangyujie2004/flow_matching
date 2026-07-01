# Flow Matching Policy 开发规划

> 目标：基于已验收的 `datasets.ZarrDataset`，实现 **视觉 + 触觉 + state** 条件化的 Flow Matching policy，输出 action chunk。  
> 参考：`models/_reference/policy/`（TacForeSight 精简迁移）  
> 本阶段：**只做 policy**，不接 dynamics WM / tactile latent cache。

---

## 0. 设计范围

| 包含 | 不包含 |
|------|--------|
| 单帧 camera（anchor）→ DINOv2 视觉 token | 多帧 image history（`n_image_steps>1` 后续再加） |
| 触觉时序 flow → CNN encoder | dynamics 预训练 latent / cross-attn future tactile |
| state 历史 → MLP | force / 120Hz |
| Flow Matching 生成 `(action_horizon, action_dim)` | train/val split |
| normalizer 随 checkpoint 保存（dataset 已 fit） | preprocess `normalizer.pth` |

---

## 1. 数据契约（来自 Dataset，已验证）

当前 config（`eef + relative`, `window_size=8`, `action_horizon=32`）：

```
batch["obs"]["image"]   (B, 1, 3, 3, H, W)   uint8   # 单帧，3 视角 RGB
batch["obs"]["tactile"] (B, 8, 35, 20, 12)    float32 # deformation flow
batch["obs"]["state"]   (B, 8, 20)             float32 # proprio 历史（已 norm）
batch["action"]         (B, 32, 20)            float32 # relative + norm 的 chunk
```

固定约定：
- `action_dim = 20`（eef, arm=both），运行时推导，不写 config
- 推理还原：`tools.normalizer.unnormalize_action_np` + `tools.robot_action.transform_robot_action_to_absolute`
- anchor state 取 `obs["state"][:, -1]`（与 dataset relative 定义一致）

---

## 2. 总体架构

```
batch
  │
  ├─ obs.image  ──► DinoV2Encoder ──────────────► vis_feat      (B, D_vis)
  │
  ├─ obs.tactile ─► TactileCNNEncoder ──────────► tac_feat      (B, D_tac)
  │                  (8 帧 flow 或聚合)
  │
  └─ obs.state ───► StateMLP ───────────────────► state_feat    (B, D_state)
                          │
                          ▼
                 ConditionFusion ──────────────► global_cond   (B, D_cond)
                          │
                          ▼
              ConditionalUnet1D (Flow Matching)
                          │
                          ▼
                   action chunk (B, 32, 20)
```

**与 reference 差异**

| 模块 | reference (TacForeSight) | 我们 (fm) |
|------|--------------------------|-----------|
| 视觉 | DINOv2 small, 多视角 flatten | **DINOv2 small**（timm），单帧 3 视角，可复用 reference encoder |
| 触觉 | dynamics latent + curr/fut cross-attn | **原始 deformation CNN** |
| 低维 | force + state 拼接 MLP | **仅 state MLP** |
| 条件注入 | global_cond only | 先 global_cond only（P0） |
| action 表示 | absolute / chunk_relative | **relative**（dataset 已做） |

---

## 3. 模块拆分

目标目录：

```
policy/models/fm/
├── workload.md              # 本文件
├── condition_encoder.py     # ③ 融合入口
├── encoders/
│   ├── dino_v2.py           # ① 视觉（复用 reference DinoV2SmallEncoder）
│   ├── tactile_cnn.py       # ② 触觉
│   └── state_mlp.py         # ③ state
├── flow_policy.py           # FlowMatchingPolicy
└── __init__.py

policy/models/diffusion/     # 从 reference 迁移
├── conditional_unet1d.py
└── mask_generator.py
```

### 3.1 DinoV2Encoder（视觉）

**输入**：`image (B, 1, V, 3, H, W)`，当前 `V=3, H=W=224`  
**输出**：`vis_feat (B, D_vis)`，默认 `D_vis=256`

设计要点：
- backbone：`timm` 加载 DINOv2，默认 `vit_small_patch14_dinov2.lvd142m`（与 reference 一致）
- 默认 **freeze backbone**，只训 projection head
- 实现可直接迁移 `models/_reference/policy/condition_encoder.py` 中的 `DinoV2SmallEncoder`
- 3 视角：每视角独立 forward → `(B, V, D_backbone)` → mean / concat+Linear → `(B, D_vis)`
- 单帧：取 `image[:, -1]` 或 `image[:, 0]`（`n_image_steps=1` 时等价）
- 预处理：uint8 → ImageNet normalize

```python
class DinoV2Encoder(nn.Module):
    def forward(self, image: Tensor) -> Tensor:  # (B,1,V,3,H,W) -> (B, D_vis)
        ...
```

### 3.2 TactileCNNEncoder（触觉 flow）

**输入**：`tactile (B, T, H, W, C)`，当前 `T=8, H=35, W=20, C=12`  
**输出**：`tac_feat (B, D_tac)`，默认 `D_tac=256`

设计要点（P0 方案，简单可训）：
- 每帧：`Conv2d` stack（35×20 小图，2–3 层）→ spatial global pool → `(B, T, D')`
- 时序：对 T 维 lightweight `Conv1d` → `(B, D_tac)`
- 输入已是 normalized deformation，不再做 reduce

备选（P1）：3D CNN `(B, C, T, H, W)` 直接时空编码。

```python
class TactileCNNEncoder(nn.Module):
    def forward(self, tactile: Tensor) -> Tensor:  # (B,T,H,W,C) -> (B, D_tac)
        ...
```

### 3.3 StateMLP（本体）

**输入**：`state (B, T, D_state)`，当前 `T=8, D_state=20`  
**输出**：`state_feat (B, D_state_out)`，默认 `256`

设计要点：
- P0：flatten `T * D_state` → MLP → `state_feat`
- P1：只取 anchor `state[:, -1]` 或 attention pool over T

```python
class StateMLP(nn.Module):
    def forward(self, state: Tensor) -> Tensor:  # (B,T,D) -> (B, D_out)
        ...
```

### 3.4 ConditionEncoder（融合）

```python
class ConditionEncoder(nn.Module):
    def forward(
        self,
        image: Tensor,      # (B,1,V,3,H,W)
        tactile: Tensor | None,
        state: Tensor,
    ) -> Tensor:            # global_cond (B, D_cond)
```

融合策略（P0）：
```
global_cond = FusionMLP( concat(vis_feat, tac_feat, state_feat) )  # -> (B, D_cond)
```
- `use_tactile=False` 时跳过 tac 分支，fusion 输入维相应减少
- 不做 vision/tactile gate（reference 有 gate，P1 可加）

### 3.5 FlowMatchingPolicy

复用 reference 核心：
- `ConditionalUnet1D`：1D UNet，输入 `(B, action_horizon, action_dim)`，注入 `global_cond`
- Flow Matching loss：`x_t = (1-t)*noise + t*action`，预测 velocity `action - noise`
- 推理：Euler / Heun ODE 积分 `t: 0→1`

**简化点（相对 reference）**：
- normalizer **不在 policy 内 duplicate fit**；训练时 batch 已由 dataset normalize
- policy 可选挂载 `DatasetNormalizer` 仅用于推理 `unnormalize`（或 trainer 侧处理）
- 无 `dynamics`、无 `tactile_latent_curr/future`、无 `force`

```python
class FlowMatchingPolicy(nn.Module):
    def compute_loss(self, batch) -> dict: ...
    def predict_action(self, obs, num_inference_steps=16) -> dict: ...
```

---


## 5. 开发阶段（TODO）

### P0 — 最小可训（优先）

- [ ] **迁移 diffusion**：`conditional_unet1d.py`, `mask_generator.py` → `models/diffusion/`
- [ ] **DinoV2Encoder**：迁移 reference `DinoV2SmallEncoder`，单帧 3 视角 smoke test
- [ ] **TactileCNNEncoder**：`(B,8,35,20,12)` → `(B,256)`
- [ ] **StateMLP**：`(B,8,20)` → `(B,256)`
- [ ] **ConditionEncoder**：三路 concat + fusion MLP
- [ ] **FlowMatchingPolicy**：`compute_loss` / `predict_action`（无 policy 内 re-normalize）
- [ ] **单元测试**：`models/fm/tests/test_forward.py`（mock batch，shape + loss finite）
- [ ] **打通一步训练**：dummy loop `loss.backward()` 能跑

### P1 — 与真实数据联调

- [ ] `trainers/policy_trainer.py`：`ZarrDataset` + `build_dataloader` + optimizer
- [ ] checkpoint 保存 `model` + `normalizer.state_dict()`
- [ ] 训练 smoke：1 epoch 少量 step，loss 下降
- [ ] `tools/eval_open_loop.py`（可选）：unnorm + relative→absolute 对比 GT

### P1.5 — 性能 / 质量

- [ ] DINOv2 backbone 特征 cache（`use_camera_latent` 路径，减训练 IO）
- [ ] tactile temporal conv 替代 mean pool
- [ ] condition gate（vision ↔ tactile 自适应权重，参考 reference）

### P2 — 后续（不在本轮）

- [ ] 多帧 image history（`n_image_steps > 1`）
- [ ] local_cond（per-timestep condition 注入 UNet）
- [ ] joint action space
- [ ] EMA / mixed precision / DDP

---

## 6. Reference 迁移对照

| reference 文件 | 迁移策略 |
|------------------|----------|
| `condition_encoder.py` | **部分复用** `DinoV2SmallEncoder`；其余重写（CNN tactile / state MLP / fusion） |
| `flow_policy.py` | **改写**：`_build_condition` 对接新 encoder；去掉 dynamics 分支 |
| `conditional_unet1d.py` | **原样迁移** |
| `mask_generator.py` | **原样迁移**（`action_visible=False`） |
| `policy_trainer.py` | P1 参考改写，对接 `datasets` + `tools.normalizer` |

---

## 7. 接口约定（Trainer ↔ Model）

```python
# 训练一步
batch = next(iter(loader))
# batch["obs"]: image, tactile, state  (已 normalized，除 image uint8)
# batch["action"]: (B, 32, 20)       (已 relative + normalized)

out = policy.compute_loss(batch)
out["loss"].backward()

# 推理
pred = policy.predict_action(batch["obs"], num_inference_steps=16)
# pred["action_normalized"]: (B, 32, 20)  模型空间
# trainer 侧 unnormalize + to_absolute 后送机器人
```

**注意**：dataset 已在 `__getitem__` 内 normalize；policy P0 **不再** `_normalize_obs`，避免 double norm。  
若后续需要 policy 自持 normalizer（部署方便），用 `policy.set_normalizer(ds.normalizer)` 仅做推理 unnormalize。

---

## 8. 风险 & 决策点

| 项 | 说明 | 建议 |
|----|------|------|
| DINOv2 依赖 | 需要 `timm` | `pip install timm`；权重首次自动下载 |
| 触觉 CNN 容量 | 35×20 很小，过深易过拟合 | 2–3 层 Conv + pool 足够 |
| state 用全历史还是 anchor | 全历史 8×20=160 维 flatten 简单 | P0 flatten；relative action 已锚到 `state[:,-1]` |
| RAM 15–34GB | 大 batch 吃内存 | train `batch_size` 从 8–16 起 |
| 三视角 DINO | 3× forward 慢 | freeze + 可考虑预计算 latent（P1.5） |

---

## 9. 验收标准

**P0 完成标志**：
1. mock batch forward，输出 `loss` 有限且可 backward
2. 真实 `DataLoader` batch forward 通过
3. `predict_action` 输出 shape `(B, 32, 20)`

**P1 完成标志**：
1. 端到端训练脚本可启动
2. checkpoint 含 normalizer，加载后可推理
3. open-loop 曲线与 GT 同量级（不要求最优）

---

## 10. 建议实现顺序

```
diffusion 迁移
    → DinoV2Encoder（迁移 reference）
    → StateMLP
    → TactileCNNEncoder
    → ConditionEncoder
    → FlowMatchingPolicy
    → test_forward.py
    → policy_trainer.py
```

先从 **ConditionEncoder 三路编码 + 融合** 开始，UNet 和 FM loss 可直接复用 reference。
