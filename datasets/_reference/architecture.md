# Policy Dataset 架构规划

> 实现入口：`policy/datasets/`  
> Config：`policy/configs/config.yaml`  
> 参考：`refence_zarr_dataset.py`（window 逻辑）

---

## 0. 约定

| 项 | 约定 |
|----|------|
| 时间轴 | 仅 30Hz，不用 120Hz |
| RAM | 初始化时 `camera/tactile/state/action` 四 key 全量 preload |
| arm | 固定 `both`（不进 config） |
| action 形状 | dual arm eef 用 flat `(20,)`；内部可 reshape `(2, 10)` |
| normalizer | **不读** preprocess `normalizer.pth`；init 时全量 windows fit；参数进 checkpoint |
| split | 不做，真机全量训练 |
| `action_dim` / `state_dim` | 运行时推导，不写 config |

| action_type | dim | slice |
|-------------|-----|-------|
| `joint` | 14 | `[0:14]` |
| `eef` | 20 | `[14:34]` |

---

## 1. 四段式架构

```
config.yaml
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  PolicyDataset (zarr_dataset.py)                           │
│                                                              │
│  __init__:                                                   │
│    ① ReplayBufferStore   ← zarr → RAM                        │
│    ② WindowIndex         ← 全量 strict anchor windows        │
│    ③ DatasetNormalizer   ← 扫 windows fit（policy 自维护）  │
│                                                              │
│  __getitem__:                                                │
│    ④ 切片 → action 变换 → normalize → 返回                  │
└──────────────────────────────────────────────────────────────┘
```

### 文件映射

```
policy/datasets/
├── planning/
│   ├── architecture.md    # 本文档
│   └── workload.md        # TODO 清单
├── robot_layout.py        # action_type → slice / dim
├── store.py               # ① ReplayBufferStore
├── window_index.py        # ② WindowIndex
├── normalizer.py          # ③ DatasetNormalizer
└── zarr_dataset.py        # ④ PolicyDataset（对外名 ZarrDataset）
```

---

## 2. ① ReplayBufferStore (`store.py`)

### 职责

- 打开 `{root_dir}/replay_buffer.zarr`，读 `meta/episode_ends`
- init 时将 4 个 30Hz key **整表**载入 `numpy` RAM
- 按全局时间索引切片；提供单 episode 读取
- `use_camera_latent=false` 时 latent 接口返回 `None`

### RAM 内容

| 属性 | zarr key | shape | dtype |
|------|----------|-------|-------|
| camera | `camera_key` | `(T,224,224,9)` | uint8 |
| tactile | `tactile_key` | `(T,35,20,24)` | float32 |
| state | `state_key` | `(T,62)` | float32 |
| action | `action_key` | `(T,62)` | float32 |

### API

```python
class ReplayBufferStore:
    def __init__(self, root_dir, *, keys, action_type, preload_to_ram=True, ...): ...

    # 全局 30Hz 切片；state/action 返回 robot slice 后数组
    def get_camera(self, t0, t1) -> np.ndarray
    def get_tactile(self, t0, t1) -> np.ndarray
    def get_state(self, t0, t1) -> np.ndarray      # (t1-t0, D_robot)
    def get_action(self, t0, t1) -> np.ndarray

  # 单 episode，raw，不 normalize
    def get_episode(self, ep_idx: int) -> dict

  # 预留
    def get_camera_latent(self, t0, t1) -> np.ndarray | None
```

---

## 3. ② WindowIndex (`window_index.py`)

### 参数（来自 config）

| config 字段 | 变量名 |
|-------------|--------|
| `window_size` | `window_length` |
| `n_image_steps` | `n_image_steps` |
| `action_horizon` | `action_horizon` |
| `stride` | `stride` |

### Strict anchor 枚举

```python
cond_len = max(window_length, n_image_steps)
first_anchor = ep_start + cond_len - 1
last_anchor  = ep_end - action_horizon - 1

for anchor_t in range(first_anchor, last_anchor + 1, stride):
    windows.append((anchor_t, ep_end, ep_idx))
```

### 时间范围（由 `anchor_t` 导出）

| 模态 | `[t0, t1)` | 长度 |
|------|------------|------|
| state / tactile | `[anchor - window_length + 1, anchor + 1)` | `window_length` |
| camera | `[anchor - n_image_steps + 1, anchor + 1)` | `n_image_steps` |
| action | `[anchor, anchor + action_horizon)` | `action_horizon` |

### API

```python
class WindowIndex:
    windows: List[Tuple[anchor_t, ep_end, ep_idx]]

    def state_range(self, idx) -> tuple[int, int]
    def image_range(self, idx) -> tuple[int, int]
    def action_range(self, idx) -> tuple[int, int]
    def __len__(self) -> int
```

---

## 4. ③ DatasetNormalizer (`normalizer.py`)

### 4.0 统一策略

- **不读** data 目录下 `normalizer.pth`
- **构建时机**：`PolicyDataset.__init__`，扫**全量 windows**
- **构建依据**：`action_type` + `action_representation`
- **持久化**：`state_dict()` → training checkpoint

```
init:
  Store(RAM) → WindowIndex → 遍历 windows 收集样本 → fit → ready
```

### 4.1 两个正交维度（仅 action 语义）

```
action 处理 = action_type × action_representation
```

| | `absolute` | `relative` |
|---|-----------|-----------|
| **joint** | fit 绝对关节 limits | `action - state[anchor]` → fit delta limits |
| **eef** | xyz/gripper limits；rot6d **identity** | relative pose → 同 segment 规则 |

> **state（obs）始终 absolute**，再 fit；relative 只影响 action。

### 4.2 各字段 fit 规则

| 字段 | fit 前 | fit 规则 |
|------|--------|----------|
| **state** | absolute slice | joint 全 limits；eef 的 rot6d identity |
| **action** joint+abs | 无 | 全维 limits |
| **action** joint+rel | `action - state[anchor]` | delta limits |
| **action** eef+abs | 无 | 每臂 xyz/gripper limits，rot6d identity |
| **action** eef+rel | relative pose（相对 anchor） | 同上 segment |
| **tactile** | 无 | per-6-channel limits |
| **camera** | — | 不 norm |

### 4.3 eef segment（每臂 10D，双臂 flat 20D）

```
[x,y,z (3)] → limits
[rot6d (6)] → identity
[gripper (1)] → limits
```

左臂 `flat[0:10]`，右臂 `flat[10:20]`。

### 4.4 API

```python
class DatasetNormalizer:
    @classmethod
    def build(cls, store, windows, *, action_type, action_representation,
              output_range=(-1, 1)) -> "DatasetNormalizer": ...

    def normalize_state(self, x, *, anchor_state=None) -> Tensor
    def normalize_action(self, x, *, anchor_state=None) -> Tensor
    def normalize_tactile(self, x) -> Tensor
    def unnormalize_action(self, x, *, anchor_state=None) -> Tensor

    def state_dict(self) -> dict
    @classmethod
    def load_state_dict(cls, d) -> "DatasetNormalizer"
```

### 4.5 `__getitem__` 时 normalize 顺序

```
state_raw  → normalize_state
action_raw → transform_repr(anchor=state_raw[-1]) → normalize_action
tactile_raw → normalize_tactile
```

`anchor_state` 使用 obs 窗口最后一帧 **raw absolute state**。

### 4.6 待实现：eef relative + rot6d

`tools/action.py` D=20 布局与 preprocess `(xyz, rot6d, gripper)` 不一致 → 需 `robot_layout` 适配后再 fit/transform。

---

## 5. ④ PolicyDataset (`zarr_dataset.py`)

### `__init__` 流程

```python
def __init__(self, config: DataConfig):
    self.store = ReplayBufferStore(...)
    self.windows = WindowIndex(self.store.episode_ends, ...)
    self.normalizer = DatasetNormalizer.build(
        self.store, self.windows,
        action_type=config.action_type,
        action_representation=config.action_representation,
        output_range=config.norm.output_range,
    )
```

### `__getitem__(idx)` 流程

```
idx → windows[idx] → (anchor_t, ep_end, ep_idx)
  → store 切片 state/tactile/action/camera
  → split_views(camera)  # (N,H,W,9) → (N,3,3,H,W)
  → normalizer
  → return {"obs": {...}, "action": ..., "meta": {...}}
```

### 返回结构

```python
{
    "obs": {
        "image":   Tensor[n_image_steps, 3, 3, H, W],   # uint8 or float
        "state":   Tensor[window_length, D_robot],
        "tactile": Tensor[window_length, 35, 20, 24],  # use_tactile=false 时可省略
    },
    "action": Tensor[action_horizon, D_robot],
    "meta": {"idx", "anchor_t", "ep_idx"},
}
```

### `get_episode(ep_idx)`

- 不走 window 索引
- 返回 raw，**不 normalize**
- 供 eval / 可视化

---

## 6. Config ↔ 模块映射

当前 `policy/configs/config.yaml`（flat 风格）：

```yaml
data:
  root_dir: ...
  window_size: 8
  stride: 1
  n_image_steps: 1
  action_horizon: 32
  action_type: eef
  action_representation: relative
  preload_to_ram: true
  use_tactile: true
  image_size: 224
  image_as_uint8: true
  use_camera_latent: false
  latent_cache_root_dir: ${data.root_dir}
  camera_key: camera
  tactile_key: tactile
  state_key: state_30hz
  action_key: action_30hz
  norm:
    output_range: [-1, 1]    # 可选，默认 [-1,1]
```

| config 字段 | 消费模块 |
|-------------|----------|
| `root_dir`, `*_key`, `preload_to_ram` | ReplayBufferStore |
| `action_type` | robot_layout + Store slice + Normalizer |
| `window_size`, `stride`, `n_image_steps`, `action_horizon` | WindowIndex |
| `action_representation` | Normalizer fit + `__getitem__` transform |
| `use_tactile` | `__getitem__` 是否返回 tactile |
| `use_camera_latent`, `latent_cache_root_dir` | camera 分支 |
| `image_size`, `image_as_uint8` | image 后处理 |
| `norm.output_range` | Normalizer |

---

## 7. 实现顺序

```
robot_layout.py → store.py → window_index.py → normalizer.py → zarr_dataset.py
```

详见 `workload.md` §7 TODO。
