# Camera / Lighting Robustness Workload

目标：缓解 **采集环境与真机部署时光照差异** 导致的 vision 推理退化。短期方案是在 **训练侧** 加简单光度增强，不改大框架（ZarrDataset → DINO condition → Flow Matching）。

**范围**：仅 `policy` 训练代码；**不涉及 `infer/` / deploy**（推理仍只做 resize + uint8）。

---

## 背景

| 现象 | 说明 |
|------|------|
| 训练数据光照相对固定 | 单场景、单时段采集，DINO 特征对亮度/色温敏感 |
| 真机光照变化 | 自然光、顶灯开关、阴影、曝光自动调节 → 与训练分布偏移 |
| 当前 pipeline 无增强 | `ZarrDataset._process_image` 仅 resize |

**约束**：默认 `data.use_camera_latent=true` 时，DINO backbone 特征在 `precompute.sh` **离线固化**；`camera_augmentation=true` 时必须走 on-the-fly 图像 + DINO，否则增强不生效。

---

## 短期策略（P0）

### 增强内容

只做 **brightness / contrast / saturation**，固定参数，**所有 camera view 同步**（`config` 里选了几路就对几路一起 aug，同一 window 共用一组随机系数）：

| 增强 | 范围 |
|------|------|
| `brightness` | `0.6 ~ 1.4` |
| `contrast` | `0.7 ~ 1.3` |
| `saturation` | `0.7 ~ 1.3` |

不做：hue、gamma、RandomCrop、Flip、Rotate、Cutout、MixUp、per-view 消融。

### 配置（单一开关）

```yaml
data:
  camera_augmentation: false   # true = 训练时启用上述光度增强
```

行为约定（实现时写死在代码里，不再拆子配置）：

- `camera_augmentation=true` → 启动时 **强制 `use_camera_latent=false`**（fail-fast 或 auto override + log）
- `camera_augmentation=false` → 保持现状，可用 latent cache
- 固定参数见上表，不做 config 可调

### 推理侧

**不改 infer / deploy**。训练增强后的 checkpoint 仍用现有 `resize_rgb_like_training` 部署；随机性仅存在于训练 dataloader。

---

## 颜色通道（RGB，已统一）

旧 zarr 已全部删除；preprocess 已提交 `8cdb468`（`color_order="rgb"`）。**当前全链路按 RGB 处理**，训练与推理对齐。

### 全链路

| 环节 | 通道顺序 | 代码 |
|------|----------|------|
| 采集录像 | 源 `bgr8` / `bgr24` | RealSense → `raw_episode` HEVC |
| preprocess → zarr | **RGB** | `loader.py` 显式 `color_order="rgb"`；`read_video_array` 默认同为 `rgb` |
| 训练 `ZarrDataset` → DINO | **RGB** | `_process_image` 无 flip；ImageNet mean/std 按 R,G,B |
| 推理 deploy | **RGB** | `bgr8` → `numpy_buffer` / `ros_image_to_rgb` flip → `resize_rgb_like_training` |
| 可视化 | **RGB** | `preprocess/vis/video_vis.py` 已去掉 `[..., ::-1]` |

```text
采集(bgr8) → preprocess 解码 flip → zarr(RGB) → 训练 DINO(RGB)
采集(bgr8) → deploy flip → infer(RGB) → DINO(RGB)
```

**注意**：`8cdb468` 之前生成的 zarr 是 BGR 像素顺序，与推理不一致；需用新 preprocess 重跑数据后再训练。

### 历史问题（已关闭）

此前 zarr 存 BGR、训练/推理按 RGB 解读，存在 R↔B 互换。已在 preprocess 出口修复，无需在 `ZarrDataset.get_camera` 再 flip。

**P0 aug**：直接在 RGB uint8 上做 brightness / contrast / saturation，与 deploy 一致。

---

## P0 Checklist

### 1. 配置 & 校验

- [x] `configs/config.yaml` 增加 `data.camera_augmentation: false`
- [x] `policy_trainer.build_dataset_and_loader`：`camera_augmentation=true` 时强制 `use_camera_latent=false`

### 2. 代码落点

| 模块 | 改动 |
|------|------|
| `datasets/image_augment.py`（新） | `apply_photometric_augment(img_uint8, rng) -> uint8`，固定三组参数 |
| `datasets/zarr_dataset.py` | `__getitem__`：`get_camera` 后、`_process_image` 前 aug；**所有 view 同步** |
| `tools/precompute_policy_latents.py` | 不增强（`camera_augmentation=false`） |
| `infer/*` | **不改** |
| `trainers/policy_trainer.py` | eval / open-loop 时关闭增强（`dataset.set_training(False)`） |

增强顺序：

```text
zarr camera (T,H,W,C)
  → photometric augment（train + camera_augmentation=true，全 view 同步）
  → _process_image（resize 224, uint8）
  → DINO backbone（use_camera_latent=false）
  → condition → Flow Matching
```

### 3. 训练路径

| 模式 | `camera_augmentation` | `use_camera_latent` | 用途 |
|------|----------------------|---------------------|------|
| 基线 | `false` | `true` | 对照 |
| **P0 实验** | `true` | `false`（强制） | 光照鲁棒性 |

### 4. 实验

- [ ] **E0**：基线（无增强 + latent）
- [ ] **E1**：`camera_augmentation=true`（固定参数，全 view）
- 记录：open-loop `action_l1`、训练 wall time、真机光照变化下 dry_run

### 5. 验收

- [x] `test_photometric_augment_uint8_range`：输出 uint8、shape 不变
- [x] `test_camera_augmentation_disables_latent_cache`
- [ ] open-loop：暗光 / 亮光 zarr slice 上 E1 优于 E0（定性即可）

---

## 不涉及 infer 的原因

| 项目 | 说明 |
|------|------|
| 随机性 | 仅训练时需要；推理要确定性 |
| 分布对齐 | 部署输入仍是「真实光照下的原始图」；模型在增强后的分布上学习，对真实域泛化更好 |
| 代码边界 | aug 挂在 `ZarrDataset.__getitem__`；`infer/preprocess.py` 保持 resize-only |
| checkpoint | 权重与现有 deploy 路径兼容，无需改 `fm_policy` / `build_obs_from_numpy_frames` |

---

## 仍待确认 / 不确定性

### 已收敛（本轮不再讨论）

- ~~是否分 view aug~~ → **否，所有 camera 一起 aug**
- ~~config 粒度~~ → **单一 `camera_augmentation: true`**
- ~~是否改 infer~~ → **否**
- ~~增强参数~~ → **固定 brightness/contrast/saturation 三档**
- ~~zarr BGR vs 推理 RGB~~ → **已修**（preprocess `8cdb468`，旧 zarr 已删，重跑后全 RGB）

### 待验证

1. **增强后训练变慢多少**  
   `use_camera_latent=false` 需实测 GPU/IO。

2. **固定参数是否过强/过弱**  
   E0/E1 对比后再调代码常量。

3. **问题量化（E0）**  
   同姿态不同光照下的 feat 距离 / action L1。

4. **硬件曝光/补光**  
   与训练增强互补。

5. **批量训练**  
   `tasks_0707` 是否加 `camera_augmentation=true` 变体。

### 非不确定性（明确不做）

- infer / deploy 侧增强或 CLAHE
- hue / gamma / 几何增强
- per-view 增强开关
- `data.augment.*` 多级配置

---

## 启动示例

```bash
# 1. 重跑 preprocess（新 zarr 为 RGB）
# cd /mnt/workspace/zyj/preprocess && python main.py ...

# 2. 基线训练
./scripts/precompute.sh --config configs/config.yaml
# camera_augmentation: false, use_camera_latent: true
./scripts/train.sh --config configs/config.yaml

# 3. 光照增强实验（无需 precompute）
# camera_augmentation: true  → 自动 use_camera_latent: false
./scripts/train.sh --config configs/config.yaml data.camera_augmentation=true
```

---

## 文件速查

| 文件 | 角色 |
|------|------|
| `datasets/zarr_dataset.py` | `_process_image`、`__getitem__` |
| `datasets/image_augment.py` | P0 新增 |
| `tools/precompute_policy_latents.py` | 离线 DINO cache（无 aug） |
| `models/fm/encoders/dino_v2.py` | ImageNet normalize（RGB） |
| `preprocess/pipeline/loader.py` | `color_order="rgb"` |
| `preprocess/pipeline/tools/decode_lossless_video.py` | 解码默认 `rgb` |
| `preprocess/vis/video_vis.py` | 直接显示 RGB，无 flip |
| `infer/preprocess.py` | deploy RGB resize（**本 workload 不改**） |
