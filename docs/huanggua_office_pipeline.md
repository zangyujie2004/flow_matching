# 削黄瓜（huanggua_office）数据处理与训练流程

记录 `huanggua_office` 从原始数据、预处理、DINOv2 特征缓存到 Flow Matching 训练的完整流程。

```text
/mnt/oss_data/arx3/huanggua_office（原始数据）
    ↓ preprocess/main.py
/mnt/workspace/lyc/data/huanggua_office/huanggua_office_MMDD_HHMM（处理后的 Zarr）
    ↓ flow_matching/scripts/precompute.sh
数据目录下的 latent_cache（DINOv2 缓存）
    ↓ flow_matching/scripts/train.sh
flow_matching/outputs/...（日志与 checkpoint）
```

## 1. 原始数据

原始数据位于：

```text
/mnt/oss_data/arx3/huanggua_office
```

其中从0731开始采集的数据属于新场景。

## 2. 数据预处理

### 2.1 脚本和配置

预处理脚本：

```text
/mnt/workspace/lyc/preprocess/main.py
```

配置文件：

```text
/mnt/workspace/lyc/preprocess/config.yaml
```

`main.py` 当前固定读取同目录下的 `config.yaml`，运行前先修改该配置，然后执行：

```bash
cd /mnt/workspace/lyc/preprocess
python main.py
```

### 2.2 输入、子任务和输出目录

配置示例：

```yaml
io:
  input_base_path: /mnt/oss_data/arx3
  task: huanggua_office

  subtasks:
    - peel_huanggua_coffee_xin_wl_0731_25
    - peel_huanggua_coffee_xin_wl_0731_35
    - peel_huanggua_coffee_xin_wl_0803_33

  output_base_path: /mnt/workspace/lyc/data
  max_episode_length: -1
  verbose: true
```

参数含义：

- `input_base_path` 与 `task` 会组成原始数据路径 `/mnt/oss_data/arx3/huanggua_office`。
- `subtasks` 决定本次处理哪些子任务。只有列在这里的目录会被处理。
- 如果要处理任务目录下的全部子任务，可以设置 `subtasks: [all]`，但新旧场景会被一起包含，使用前应确认。
- `max_episode_length: -1` 表示每个子任务使用全部 episode；设置为正整数时，每个子任务最多取该数量的 episode。
- `output_base_path` 是处理结果的根目录。

### 2.3 `off`、`replace`、`dual` 的含义

```yaml
streams:
  camera:
    base_0:
      impainting: off       # off | replace | dual
      impainting_by_subtask:
        peel_huanggua_coffee_xin_wl_0731_25: dual
```

三种模式只作用于 `base_0` 相机：


| 模式      | 主`base_0` 图像                         | 去手图像                 | 去手文件不存在时                    |
| --------- | --------------------------------------- | ------------------------ | ----------------------------------- |
| `off`     | 使用原始`base_0_rgb.mp4`                | 不读取                   | 正常处理                            |
| `replace` | 用`base_0_rgb_remove_hand.mp4` 替换原图 | 只保留去手版本作为主图   | 当前 episode 处理失败               |
| `dual`    | 保留原始`base_0_rgb.mp4`                | 若存在则额外保存去手版本 | 正常处理，但该 episode 没有去手旁路 |

补充说明：

- `replace` 适合训练时始终只使用去手后的底座相机画面。
- `dual` 同时保留原始画面和去手画面，后续可以通过训练配置选择或混合；通常更灵活。
- `dual` 的去手图像会作为独立的 `camera_base_remove_hand` 数据保存，不会覆盖原始 `base_0`。
- 兼容旧配置：`false` 等价于 `off`，`true` 等价于 `replace`。

### 2.4 按子任务控制是否去手

`impainting` 是默认值，`impainting_by_subtask` 是针对指定子任务的覆盖值。未写入映射的子任务会继承默认值。

推荐的安全配置是默认关闭，只对确认已有去手视频的子任务开启 `dual`：

```yaml
streams:
  camera:
    base_0:
      impainting: off
      impainting_by_subtask:
        peel_huanggua_coffee_xin_wl_0731_25: dual
        peel_huanggua_coffee_xin_wl_0731_35: dual
        peel_huanggua_coffee_xin_wl_0803_33: dual

      rgb: true
      left_fish_eye: false
      right_fish_eye: false
```

如果设置全局值为 `dual`：

```yaml
impainting: dual
```

那么所有未出现在 `impainting_by_subtask` 中的子任务也会使用 `dual`。因此，仅仅把某一行注释掉并不能关闭该子任务的去手处理；需要显式写成：

```yaml
impainting_by_subtask:
  some_subtask: off
```

`subtasks` 和 `impainting_by_subtask` 分工如下：

- `io.subtasks`：决定哪些子任务进入本次数据处理。
- `impainting_by_subtask`：决定这些子任务各自使用原图、替换图还是双版本。

### 2.5 预处理并行参数

当前配置使用多进程处理 episode：

```yaml
accelerate:
  enabled: true
  num_workers: 8
  max_in_flight: 8
  max_tasks_per_child: 1
  release_memory_each_episode: true
```

增加 `num_workers` 可以提高处理速度，但也会增加 CPU、内存和视频解码压力。建议先使用当前的 `8`；如果内存充足再逐步增加。

### 2.6 预处理输出

处理结果自动写入带时间戳的目录：

```text
<output_base_path>/<task>/<task>_MMDD_HHMM
```

例如：

```text
/mnt/workspace/lyc/data/huanggua_office/huanggua_office_0809_1324
```

主要文件包括：

```text
config.yaml         本次预处理使用的配置副本
meta.json           数据集和 episode 元信息
replay_buffer.zarr  训练使用的 Zarr 数据
```

运行结束时应确认日志中的成功数量：

```text
done: 成功episode数/总episode数
```

若使用 `dual`，还应检查：

```text
base_remove_hand: present=... none=...
```

`none` 表示对应 episode 没有加载到去手视频。

## 3. 计算 DINOv2 缓存

### 3.1 设置训练数据路径

进入 Flow Matching 项目：

```bash
cd /mnt/workspace/lyc/flow_matching
```

在训练配置中把 `data.root_dir` 指向上一步生成的时间戳目录：

```yaml
data:
  root_dir: /mnt/workspace/lyc/data/huanggua_office/huanggua_office_0809_1324
  use_camera_latent: true
  latent_cache_root_dir: ${data.root_dir}/latent_cache/{auto}
```

预计算和训练必须使用兼容的视觉配置，例如 DINO 模型、图像尺寸和 `token_mode` 应保持一致。

### 3.2 预计算参数

相关参数位于训练配置的 `precompute` 区域：

```yaml
precompute:
  batch_size: 256
  num_workers: 32
  prefetch_batches: 8
  multi_gpu: true
  device: cuda
  overwrite: false
  output_path: null
  token_mode: cls
```

参数含义：

- `batch_size`：一次编码的帧数。越大速度可能越快，但 CPU 内存和 GPU 显存占用也越高。建议从 `256` 开始，根据显存逐步调整。
- `num_workers`：并行读取 Zarr 的进程数。机器资源充足时可以设置为 `32` 加快读取。
- `prefetch_batches`：提前排队的 batch 数；越大占用的 CPU 内存越多。
- `multi_gpu: true`：使用命令中 `--gpus` 暴露的全部 GPU。
- `overwrite: false`：已有且身份匹配的完整缓存会跳过。
- `token_mode: cls`：只保存每张图的 CLS 特征，缓存更小；`all` 会保存全部 257 个 token，缓存和后续资源消耗显著增大。

不要把 `train.num_workers` 和 `precompute.num_workers` 混淆：前者用于训练 DataLoader，后者只用于 DINO 缓存读取。

### 3.3 执行命令

脚本路径：

```text
/mnt/workspace/lyc/flow_matching/scripts/precompute.sh
```

默认配置是：

```text
/mnt/workspace/lyc/flow_matching/configs/train/config.yaml
```

使用默认配置和默认 GPU：

```bash
bash /mnt/workspace/lyc/flow_matching/scripts/precompute.sh
```

指定配置和 8 张 GPU：

```bash
bash /mnt/workspace/lyc/flow_matching/scripts/precompute.sh \
  --config /mnt/workspace/lyc/flow_matching/configs/train/config.yaml \
  --gpus 0,1,2,3,4,5,6,7
```

也可以在命令行临时覆盖读取进程数：

```bash
bash /mnt/workspace/lyc/flow_matching/scripts/precompute.sh \
  --config /mnt/workspace/lyc/flow_matching/configs/train/config.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 32
```

缓存默认写入 `data.latent_cache_root_dir`，主要包括：

```text
frame_backbone.zarr
frame_backbone_base_remove_hand.zarr  # dual 数据中存在去手图像时生成
```

## 4. 训练 Flow Matching 策略

### 4.1 关键配置

训练前至少确认以下配置：

```yaml
output:
  root_dir: outputs/huanggua_office_0809
  run_name: huanggua_office_unet

data:
  root_dir: /mnt/workspace/lyc/data/huanggua_office/huanggua_office_0809_1324

  action_horizon: 128
  action_type: joint
  action_representation: absolute

  use_tactile: true

models:
  fm:
    use_tactile: true

train:
  batch_size: 128
  epochs: 256
  resume_path: null

checkpoint:
  save_every: 200
```

含义如下：

- `output.root_dir`：实验输出根目录。
- `output.run_name`：本次运行目录名。以上配置最终输出到 `outputs/huanggua_office_0809/huanggua_office_unet`。
- `data.root_dir`：预处理生成的带时间戳数据目录，不要直接填写 `replay_buffer.zarr`。
- `action_horizon: 128`：模型一次预测未来 128 个 action 时间步。
- `action_type: joint`：使用关节空间状态和动作。
- `action_representation: absolute`：预测绝对关节角，而不是相对增量。
- `data.use_tactile: true` 与 `models.fm.use_tactile: true`：加载触觉数据并让模型使用触觉条件。两处应保持一致。
- `train.batch_size`：**每张 GPU 的 batch size**。8 卡、每卡 128 时，全局 batch size 为 `128 × 8 = 1024`。
- `train.epochs: 256`：训练到总 epoch 编号 256 后结束。断点恢复时不是“再训练 256 个 epoch”。
- `checkpoint.save_every: 200`：每 200 个 epoch 额外保存一个带编号的归档文件，例如 `epoch_0200.pt`。

训练器每个 epoch 都会更新：

```text
checkpoints/latest.pt
```

因此 `save_every: 200` 不代表前 199 个 epoch 没有 checkpoint。

### 4.2 从头训练

首次训练设置：

```yaml
train:
  resume_path: null
```

8 卡训练命令：

```bash
bash /mnt/workspace/lyc/flow_matching/scripts/train.sh \
  --config /mnt/workspace/lyc/flow_matching/configs/train/config.yaml \
  --gpus 0,1,2,3,4,5,6,7
```

脚本会根据 GPU 数量自动选择单卡训练或多卡 DDP。H20 建议在配置中保持：

```yaml
train:
  use_amp: true
  amp_dtype: bf16
```

### 4.3 从 checkpoint 继续训练

把 `train.resume_path` 设置为要恢复的 checkpoint：

```yaml
train:
  resume_path: /mnt/workspace/lyc/flow_matching/outputs/huanggua_office_0809/huanggua_office_unet/checkpoints/latest.pt
```

然后运行相同的 `train.sh` 命令。恢复内容包括模型、优化器、normalizer、epoch 和 global step；下一轮从 checkpoint 中记录的 epoch 加一开始。

继续同一个实验时，应保持 `output.root_dir` 和 `output.run_name` 指向原实验目录。如果想从旧 checkpoint 初始化一个新实验，应先确认是否需要保留或重置优化器和 epoch；普通 `resume_path` 是严格的断点续训语义。

### 4.4 输出目录

以上示例会生成：

```text
/mnt/workspace/lyc/flow_matching/outputs/huanggua_office_0809/huanggua_office_unet/
├── checkpoints/
│   ├── latest.pt
│   └── epoch_0200.pt
├── open_loop/
├── resolved_config.yaml
└── events.out.tfevents...
```

共享盘空间不足时，open-loop 图片保存可能失败。若不需要曲线图，可以设置：

```yaml
train:
  plot_samples: 0
```

当前代码在 PNG 保存遇到 `ENOSPC` 时会跳过本轮后续绘图并继续训练，但 checkpoint 保存仍需要可用磁盘空间。

## 6. 常用命令

```bash
# 1. 预处理（先编辑 /mnt/workspace/lyc/preprocess/config.yaml）
cd /mnt/workspace/lyc/preprocess
python main.py

# 2. 计算 DINOv2 缓存
bash /mnt/workspace/lyc/flow_matching/scripts/precompute.sh \
  --config /mnt/workspace/lyc/flow_matching/configs/train/config.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 32

# 3. 训练或断点续训
bash /mnt/workspace/lyc/flow_matching/scripts/train.sh \
  --config /mnt/workspace/lyc/flow_matching/configs/train/config.yaml \
  --gpus 0,1,2,3,4,5,6,7
```
