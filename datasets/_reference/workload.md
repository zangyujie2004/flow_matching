# Policy Dataset 开发框架 & TODO

> **架构主文档**：[`planning/architecture.md`](./planning/architecture.md)  
> 目标：基于 preprocess replay buffer 构建 policy Dataset（30Hz / RAM / policy 自维护 normalizer）。

---

## 0. 设计范围（本轮）

| 包含 | 不包含 |
|------|--------|
| 30Hz：`camera`, `tactile`, `state_30hz`, `action_30hz` | `state_120hz` / `action_120hz` |
| 全量 preload 到 RAM | 按需 zarr 切片 IO |
| config 选择 `joint` / `eef` action space | 自由 dim slice |
| action norm：`absolute` / `relative` | dynamics / tactile WM |
| camera latent 接口（默认 `false`） | latent 预计算实现 |
| 单 episode 读取 API | train/val split 工具（后续） |

---

## 1. 模块划分（四段式）

```
┌─────────────────────────────────────────────────────────────┐
│  ① ReplayBufferStore     初始化时一次性读入 RAM              │
├─────────────────────────────────────────────────────────────┤
│  ② WindowIndex             根据 window_length/stride 建索引  │
├─────────────────────────────────────────────────────────────┤
│  ③ DatasetNormalizer       按 robot_space × action_repr 建 norm│
├─────────────────────────────────────────────────────────────┤
│  ④ PolicyDataset.__getitem__   切片 → 变换 → 归一化 → 返回  │
└─────────────────────────────────────────────────────────────┘
```

建议文件：

```
policy/datasets/
├── planning/
│   ├── architecture.md    # 架构主文档
│   └── workload.md
├── robot_layout.py        # action_type → slice / dim（arm 固定 both）
├── store.py               # ① ReplayBufferStore
├── window_index.py        # ② WindowIndex
├── normalizer.py          # ③ DatasetNormalizer
└── zarr_dataset.py        # ④ ZarrDataset + DataConfig
```

---

## 2. ① ReplayBufferStore — 数据读取

### 2.1 职责

- 打开 `replay_buffer.zarr`，读取 `meta/episode_ends`
- **初始化时**将 4 路 30Hz 数据**整表**载入 RAM
- 提供按全局时间索引 / 按 episode 切片的访问接口
- 预留 camera latent 读取接口（默认关闭）

### 2.2 RAM 缓存内容

| 字段 | zarr key | Shape | Dtype | 说明 |
|------|----------|-------|-------|------|
| `camera` | `camera` | `(T, 224, 224, 9)` | uint8 | 3 视角 channel 拼接 |
| `tactile` | `tactile` | `(T, 35, 20, 24)` | float32 | 4 bundle 拼接 |
| `state` | `state_30hz` | `(T, 62)` | float32 | 完整状态向量 |
| `action` | `action_30hz` | `(T, 62)` | float32 | 完整动作向量 |

另存 episode 索引：

```python
episode_starts[i] = 0 if i==0 else episode_ends[i-1]
episode_ends[i]   # cumulative，exclusive end
```

### 2.3 Robot Space 切片

**`arm` 固定 `both`**，由 `action_type` 决定 slice（不进 config）：

| action_type | state/action dim | slice |
|-------------|------------------|-------|
| `joint` | 14 | `[0:14]` |
| `eef` | 20 | `[14:34]` |

每臂 eef：`xyz(3) + rot6d(6) + gripper(1) = 10`；模型 I/O 用 flat `(20,)`。

### 2.4 公开 API

```python
class ReplayBufferStore:
    # --- 全局时间索引（30Hz）---
    def get_camera(self, t0: int, t1: int) -> np.ndarray       # [t0:t1]
    def get_tactile(self, t0: int, t1: int) -> np.ndarray
    def get_state(self, t0: int, t1: int) -> np.ndarray        # 已按 robot_space 切片
    def get_action(self, t0: int, t1: int) -> np.ndarray

    # --- 单 episode ---
    def get_episode(self, ep_idx: int) -> dict:
        """返回该 episode 全部帧（30Hz）"""
        return {
            "camera":  (T_ep, 224, 224, 9),
            "tactile": (T_ep, 35, 20, 24),
            "state":   (T_ep, D_robot),
            "action":  (T_ep, D_robot),
            "ep_idx":  ep_idx,
            "t_start": int,
            "t_end":   int,
        }

  # --- camera latent（预留，默认不启用）---
    def get_camera_latent(self, t0: int, t1: int) -> np.ndarray | None
    # use_camera_latent=false 时始终返回 None
    # true 时从 latent_cache.zarr 读取，cache 不存在则 raise
```

### 2.5 Camera Latent 接口（预留）

```yaml
camera:
  use_latent: false                          # 当前固定 false
  latent_cache_path: null                    # 后续: <run_dir>/policy_latent_cache.zarr
  latent_key: image_backbone_feat
```

- `use_latent: false`：`__getitem__` 返回原始 `camera`（uint8）
- `use_latent: true`：跳过 camera decode，返回 `image_backbone_feat`；接口先定义，实现后置

---

## 3. ② WindowIndex — 窗口枚举

### 3.1 参数

| 参数 | 含义 |
|------|------|
| `window_length` | state / tactile 观测历史长度（含 anchor 帧） |
| `n_image_steps` | 图像历史长度（≤ window_length，可只看最近 N 帧） |
| `action_horizon` | 从 anchor 起向后的 action chunk 长度 |
| `stride` | 锚点滑动步长 |

### 3.2 Strict Anchor 规则

对每个 episode `[ep_start, ep_end)`：

```
cond_len = max(window_length, n_image_steps)
first_anchor = ep_start + cond_len - 1
last_anchor  = ep_end - action_horizon - 1    # 保证 action chunk 不越界

for anchor_t in range(first_anchor, last_anchor + 1, stride):
    windows.append((anchor_t, ep_end, ep_idx))
```

时间轴示意：

```
        obs_start          anchor_t        action_end
           |                  |                |
  ...------[--- window_length ---]--[--- action_horizon ---]--...
           |<-- n_image_steps -->|          （image 可取子区间）
```

### 3.3 各模态切片范围（均由 anchor_t 导出）

| 模态 | 时间范围 `[t0, t1)` | 长度 |
|------|---------------------|------|
| state / tactile | `[anchor_t - window_length + 1, anchor_t + 1)` | `window_length` |
| camera | `[anchor_t - n_image_steps + 1, anchor_t + 1)` | `n_image_steps` |
| action | `[anchor_t, anchor_t + action_horizon)` | `action_horizon` |

### 3.4 输出

```python
windows: List[Tuple[anchor_t, ep_end, ep_idx]]   # len = dataset 样本数
window_lookup: Dict[(anchor_t, ep_idx), idx]      # 可选，供 latent cache 校验
```

---

## 4. ③ DatasetNormalizer — 归一化

### 4.0 统一策略（已定）

**不使用 preprocess `normalizer.pth`**。policy 侧自维护一套 normalizer：

- **构建时机**：`PolicyDataset.__init__` 时，基于**全量 windows**扫描 RAM 数据并 fit
- **构建依据**：`action_type` + `action_representation` 决定 action 变换后再统计
- **持久化**：`state_dict()` 存入 training checkpoint；推理时从 checkpoint 加载，不依赖 data 目录

```
init 流程:
  Store(RAM) → WindowIndex(全量) → 遍历 windows 收集统计量 → fit DatasetNormalizer → ready
```

### 4.1 两个正交维度（action）

```
action 处理 = action_type × action_representation
```

| | `absolute` | `relative` |
|---|-----------|-----------|
| **`joint`** | 直接 fit 绝对关节角 limits | 先 `action - state[anchor]`，再 fit delta limits |
| **`eef`** | fit 绝对 eef；**rot6d 维 identity** | 先 relative pose 变换，再 fit；rot6d/rotation 维 identity |

config：

```yaml
action_type: eef              # joint | eef
action_representation: relative  # absolute | relative
```

### 4.2 Fit 规则（各字段独立）

#### state（obs，始终 absolute）

```
收集: 所有 window 的 state_slice  # joint [0:14] 或 eef [14:34]
fit:  limits → [-1, 1]
eef 时: xyz/gripper limits，rot6d identity（每臂重复）
```

#### action

| 组合 | fit 前变换 | fit 规则 |
|------|-----------|----------|
| joint + absolute | 无 | 全维 limits |
| joint + relative | `action - state[anchor]` | 全维 limits（delta） |
| eef + absolute | 无 | xyz/gripper limits，rot6d identity |
| eef + relative | relative pose（相对 anchor） | xyz/gripper limits；rotation identity |

#### tactile

```
收集: 所有 window 的 tactile patch，或全量 tactile 数组
fit:  per-6-channel limits（4 bundle × 6 ch，规则同 preprocess）
      或简化为全通道 limits（实现时二选一）
```

#### camera

不做 norm（uint8 或 `/255` 由 `image_as_uint8` 控制）。

### 4.3 eef segment 定义（fit 时用，不读 preprocess）

双臂 eef flat `(20,)`，每臂 `(10,)` = xyz(3) + rot6d(6) + gripper(1)：

| 臂内 offset | 维 | fit mode |
|-------------|-----|----------|
| 0:3 | xyz | limits |
| 3:9 | rot6d | **identity** |
| 9 | gripper | limits |

左臂 `flat[0:10]`，右臂 `flat[10:20]`，规则相同。

### 4.4 构建流程（init 时）

```python
class DatasetNormalizer:
    @classmethod
    def build(cls, store, windows, *, action_type, action_representation, output_range=(-1, 1)):
        # 1. 遍历全量 windows，收集 state / action / tactile 样本
        # 2. action 先按 action_representation 做变换，再统计
        # 3. 按 segment 规则 fit 各字段
        # 4. 返回 frozen normalizer
        ...

    def normalize_state(self, x) -> Tensor
    def normalize_action(self, x, *, anchor_state=None) -> Tensor
    def normalize_tactile(self, x) -> Tensor
    def unnormalize_action(self, x, *, anchor_state=None) -> Tensor

    def state_dict(self) -> dict
    @classmethod
    def load_state_dict(cls, d) -> "DatasetNormalizer"
```

**`__getitem__` 时顺序**：

```
raw → (action_repr 变换，仅 action) → normalize → 返回
```

`relative` 时 `anchor_state` 用 obs 窗口最后一帧 **raw absolute state**（变换基准，normalize 之前）。

### 4.5 与 preprocess normalizer 的关系

| | preprocess `normalizer.pth` | policy `DatasetNormalizer` |
|--|----------------------------|---------------------------|
| 用途 | preprocess 阶段统计 | policy 训练/推理 |
| 数据来源 | 全量 replay buffer raw | 全量 windows + action_repr 变换后 |
| 是否读取 | **否** | init 时 fit 或 checkpoint 加载 |
| segment 规则 | 可参考实现 | 代码内维护 eef rot6d identity |

`preprocess/normalize/normalizer.py` 仅作 **segment 规则参考**，不加载其参数。

### 4.6 eef relative 实现注意

`tools/action.py` 的 D=20 布局与 preprocess `(xyz, rot6d, gripper)` 不一致，需适配后再 fit / transform（见 §8 D1）。

---

## 5. ④ `__getitem__` 数据流

### 5.1 流程图

```
idx
 │
 ▼
windows[idx] → (anchor_t, ep_end, ep_idx)
 │
 ├─ 1. 算时间范围
 │     state_range  = [anchor - window_length + 1, anchor + 1)
 │     image_range  = [anchor - n_image_steps + 1, anchor + 1)
 │     action_range = [anchor, anchor + action_horizon)
 │
 ├─ 2. 从 ReplayBufferStore 切片（RAM，无 IO）
 │     state_raw   = store.get_state(*state_range)      # (W, D_robot)
 │     tactile_raw = store.get_tactile(*state_range)    # (W, 35, 20, 24)
 │     action_raw  = store.get_action(*action_range)    # (H, D_robot)
 │
 ├─ 3. Camera 分支
 │     if use_camera_latent:
 │         image = store.get_camera_latent(*image_range)   # 预留
 │     else:
 │         image = store.get_camera(*image_range)          # (N, 224, 224, 9)
 │         image = split_views(image)                      # (N, 3, 3, 224, 224)
 │
 ├─ 4. Normalizer
 │     state   = normalizer.normalize_state(state_raw)
 │     action  = normalizer.normalize_action(
 │                 action_raw,
 │                 anchor_state=state_raw[-1],   # relative 时需要
 │               )
 │     tactile = normalizer.normalize_tactile(tactile_raw)
 │
 └─ 5. 返回
       return {"obs": {...}, "action": ..., "meta": {...}}
```

### 5.2 返回结构

```python
{
    "obs": {
        "image":   Tensor[n_image_steps, n_views, 3, H, W],  # use_latent=false
        # "image_backbone_feat": Tensor[n_image_steps, D],    # use_latent=true（预留）

        "state":   Tensor[window_length, D_robot],             # 已 normalize
        "tactile": Tensor[window_length, 35, 20, 24],          # 已 normalize
    },
    "action": Tensor[action_horizon, D_robot],                 # 已 normalize
    "meta": {
        "idx": int,
        "anchor_t": int,
        "ep_idx": int,
    },
}
```

### 5.3 `get_episode(ep_idx)` — 独立 API

供 open-loop eval / 可视化，**不走 window 索引，不做 normalize**（或可选 `normalize=False`）：

```python
dataset.get_episode(ep_idx) → {
    "camera":  (T, 224, 224, 9),
    "tactile": (T, 35, 20, 24),
    "state":   (T, D_robot),
    "action":  (T, D_robot),
}
```

---

## 6. Data Config 规范（`policy/configs/`）

> 本轮只规划 `data` 段；`model` / `train` 后续单独写。
> 原则：**config 只描述意图**，`action_dim` / `state_dim` 由 `action_type` 运行时推导，不手写。
> **约定**：`arm` 固定 `both`；normalize 在 dataset init 时 fit，**不读 data 目录下任何 normalizer 文件**。

### 6.1 文件组织

```
policy/configs/
├── config.yaml           # 主入口，include data 段
└── data/
    └── default.yaml      # 可选：data 默认值，被 config.yaml include
```

当前阶段可只维护一个 `config.yaml`，`data:` 段按下面结构写。

### 6.2 完整结构（规划稿）

```yaml
# ── 路径 ──────────────────────────────────────────
data:
  run_dir: /mnt/workspace/zyj/data/processed/peel/peel_0630_1626

  paths:
    zarr: ${data.run_dir}/replay_buffer.zarr
    meta: ${data.run_dir}/meta.json
    latent_cache: null        # use_camera_latent=true 时用

  # ── zarr key 映射（对齐 preprocess，一般不改）──
  keys:
    camera: camera
    tactile: tactile
    state: state_30hz
    action: action_30hz

  # ── 加载策略 ────────────────────────────────────
  load:
    preload_to_ram: true          # 30Hz 四 key 全量进 RAM
    # 以下 modality 控制是否读入 RAM / 是否出现在 sample 里
    modalities:
      camera: true
      tactile: true
      state: true
      action: true                # action 始终需要（作为 label）

  # ── Robot：决定 state/action 取哪段、怎么变换 ──
  # arm 固定 both，不进 config
  action_type: eef                # joint | eef
  action_representation: relative # absolute | relative

  # ── 窗口采样 ────────────────────────────────────
  window:
    length: 8                     # state/tactile 历史长度（含 anchor）
    n_image_steps: 1              # 图像历史长度（≤ length）
    action_horizon: 32            # 从 anchor 向后的 action chunk
    stride: 1

  # ── Camera ─────────────────────────────────────
  camera:
    use_latent: false             # 预留；false 时读原始 RGB
    image_size: 224               # resize 目标（与存储不一致时才生效）
    as_uint8: true                # true→uint8；false→/255 float
    # 多视角顺序（与 preprocess merge 顺序一致）
    views: [base_0, left_wrist_0, right_wrist_0]

  # ── Tactile ────────────────────────────────────
  tactile:
    shape: [35, 20, 24]           # 只作文档/校验，来自 meta.json
    # bundle 顺序（与 preprocess merge 一致）
    bundles: [left_wrist_0, left_wrist_1, right_wrist_0, right_wrist_1]

  # ── 归一化（默认 dataset 内完成，可不写）──────
  # ── 归一化（policy 自维护，init 时 fit）──────
  norm:
    output_range: [-1, 1]     # limits norm 目标范围
    # 不配置 normalizer_path；参数进 checkpoint
```

### 6.3 字段说明

#### 路径 `data.paths`

| 字段 | 必填 | 说明 |
|------|------|------|
| `run_dir` | ✅ | preprocess 单次 run 目录 |
| `paths.zarr` | ✅ | 默认 `${run_dir}/replay_buffer.zarr` |
| `paths.meta` | ✅ | 校验 schema、读 layout |
| `paths.latent_cache` | ⬜ | `use_camera_latent=true` 时才需要 |

#### Robot `data.action_type` / `data.action_representation`

| 字段 | 取值 | 说明 |
|------|------|------|
| `action_type` | `joint` \| `eef` | 控制 state/action 从 62-D 取哪段 |
| `action_representation` | `absolute` \| `relative` | action 语义 + norm 策略 |

**固定约定**：`arm = both`（不进 config）。

**运行时推导（不写进 config）**：

| action_type | `state_dim` = `action_dim` | slice |
|-------------|---------------------------|-------|
| joint | 14 | `[0:14]` |
| eef | 20 | `[14:34]` |

#### 窗口 `data.window`

| 字段 | 说明 |
|------|------|
| `length` | 原 `window_size`；state/tactile 观测历史 |
| `n_image_steps` | 图像可取更短子区间 |
| `action_horizon` | 原 `action_window_size` |
| `stride` | 锚点步长 |

#### Camera `data.camera`

| 字段 | 说明 |
|------|------|
| `use_latent` | **当前固定 false**；true 时走 `paths.latent_cache` |
| `as_uint8` | dataset 输出 dtype |
| `views` | 9 通道拆分顺序，与 meta.json 一致 |

#### Norm `data.norm`

| 字段 | 说明 |
|------|------|
| `output_range` | limits norm 输出范围，默认 `[-1, 1]` |

**不在 config 里出现**：`normalizer_path`。normalizer 在 dataset init 时从全量 windows fit，训练 checkpoint 保存/加载。

| action_type × action_representation | fit 对象 |
|-------------------------------------|----------|
| joint + absolute | absolute joint limits |
| joint + relative | relative delta limits |
| eef + absolute | absolute eef，rot6d identity |
| eef + relative | relative eef，rotation identity |
| state（obs） | absolute，规则同 action_type 对应 slice |
| tactile | per-channel limits |

### 6.4 从旧 config 的迁移对照

| 旧字段 (`config.yaml`) | 新字段 | 处理 |
|------------------------|--------|------|
| `root_dir` | `data.run_dir` | 改名 |
| `window_size` | `data.window.length` | 改名 |
| `action_window_size` | `data.window.action_horizon` | 改名 |
| `action_type: eef` | `data.robot.space: eef` | 归入 robot |
| `action_representation` | `data.robot.representation` | 归入 robot |
| `preload_to_ram: false` | `data.load.preload_to_ram: true` | **默认改 true** |
| `tactile_left/right_key` | 删除 | preprocess 已合并为 `tactile` |
| `force_key`, `*_4x_key` | 删除 | 本轮不用 120Hz |
| `state_key`, `action_key` | `data.keys.state/action` | 固定映射 |
| `latent_cache_root_dir` | `data.paths.latent_cache` | 归入 paths |
| `image_size`, `image_as_uint8` | `data.camera.*` | 归入 camera |

### 6.5 示例：peel 任务 eef relative

```yaml
data:
  run_dir: /mnt/workspace/zyj/data/processed/peel/peel_0630_1626

  paths:
    zarr: ${data.run_dir}/replay_buffer.zarr
    meta: ${data.run_dir}/meta.json
    latent_cache: null

  keys:
    camera: camera
    tactile: tactile
    state: state_30hz
    action: action_30hz

  load:
    preload_to_ram: true
    modalities:
      camera: true
      tactile: true
      state: true
      action: true

  action_type: eef
  action_representation: relative   # → action_dim=20, relative pose norm

  window:
    length: 8
    n_image_steps: 1
    action_horizon: 32
    stride: 1

  camera:
    use_latent: false
    image_size: 224
    as_uint8: true
    views: [base_0, left_wrist_0, right_wrist_0]

  tactile:
    shape: [35, 20, 24]
    bundles: [left_wrist_0, left_wrist_1, right_wrist_0, right_wrist_1]

  norm:
    output_range: [-1, 1]
```

### 6.6 Config → Dataset 初始化映射

```
config.data
    │
    ├─ paths.*          → ReplayBufferStore(zarr, meta)
    ├─ load.*           → Store 加载哪些 key 到 RAM
    ├─ action_type / action_representation → action 变换 + fit 策略
    ├─ window.*         → WindowIndex
    ├─ camera.*         → __getitem__ image 分支
    └─ norm.output_range → DatasetNormalizer.build(...)  # init 全量 fit
```

### 6.7 待确认

| # | 问题 | 建议 |
|---|------|------|
| C1 | `arm` | ✅ 固定 `both` |
| C2 | normalizer | ✅ policy 自维护，init 全量 fit，不读 data |
| C3 | split | ✅ 不做，全量数据 |

---

## 7. TODO（按实现顺序）

### Step 1 — ReplayBufferStore

- [ ] **S1-1** 打开 zarr，读 `episode_ends`，构建 `episode_starts`
- [ ] **S1-2** 初始化时 load 4 key 到 RAM：`camera`, `tactile`, `state_30hz`, `action_30hz`
- [ ] **S1-3** `robot_layout.py`：根据 `robot.space` + `robot.arm` 返回 slice
- [ ] **S1-4** `get_*` 切片 API + `get_episode(ep_idx)`
- [ ] **S1-5** `get_camera_latent` 接口 stub（`use_latent=false` 返回 None）

### Step 2 — WindowIndex

- [ ] **S2-1** 实现 strict anchor 枚举（`window_length`, `n_image_steps`, `action_horizon`, `stride`）
- [ ] **S2-2** 导出每个 window 的三段区间：`state_range`, `image_range`, `action_range`
- [ ] **S2-3** 单元测试：短 episode 跳过、边界不越界

### Step 3 — DatasetNormalizer

- [ ] **S3-1** `normalizer.py`：limits / identity segment fit（参考 preprocess segment 规则，不加载其文件）
- [ ] **S3-2** `build()`：遍历全量 windows 收集 state / action / tactile
- [ ] **S3-3** `joint + absolute` / `joint + relative` fit
- [ ] **S3-4** `eef + absolute`：xyz/gripper limits + rot6d identity
- [ ] **S3-5** `eef + relative`：pose 变换后 fit（适配 rot6d layout）
- [ ] **S3-6** tactile per-channel limits fit
- [ ] **S3-7** `state_dict` / `load_state_dict` → 写入 training checkpoint

### Step 4 — PolicyDataset

- [ ] **S4-1** 组装 Store + WindowIndex + Normalizer
- [ ] **S4-2** 实现 `__len__` / `__getitem__`（按 §5 流程）
- [ ] **S4-3** `get_episode(ep_idx)` 公开方法
- [ ] **S4-4** camera 多视角拆分 `(H,W,9) → (3,3,H,W)`
- [ ] **S4-5** `tools/inspect_dataset.py` smoke test

### Step 5 — 验证 & 对接

- [ ] **S5-1** 四种 `robot_space × action_repr` 组合各跑 1 个 sample，检查 shape/range
- [ ] **S5-2** `get_episode` 与逐帧 `__getitem__` 覆盖范围一致
- [ ] **S5-3** 对接 trainer（后续）

---

## 8. 待决策

| # | 问题 | 建议 |
|---|------|------|
| D1 | `eef + relative` 的 rot6d 处理 | 方案 A：rot6d ↔ mat ↔ relative ↔ rot6d |
| D2 | `joint + relative` delta fit | 全量 windows 上 fit delta limits |
| D3 | camera 返回 uint8 还是 float | 默认 uint8，模型侧处理；config 可切 |
| D4 | tactile 是否参与 Phase 1 训练 | 先 load + normalize，模型是否用由 trainer config 决定 |
| D5 | `zarr_dataset.py` | 骨架已建，待实现四模块逻辑 |

---

## 9. 参考

| 文件 | 用途 |
|------|------|
| `refence_zarr_dataset.py` | `_build_windows`、anchor 边界逻辑 |
| `preprocess/normalize/normalizer.py` | segment 规则参考（identity/limits），不加载参数 |
| `tools/action.py` | eef relative pose 变换（需适配 rot6d） |
| `data/processed/*/meta.json` | 62-D layout 权威定义 |
