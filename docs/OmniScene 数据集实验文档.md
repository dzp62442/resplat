# OmniScene 数据集实验说明（实施方案）

> 本文档只描述 `comp_svfgs` 分支上的拟实现方案，当前阶段不修改训练、数据加载或评测代码。方案以 DepthSplat `comp_svfgs` 分支的 OmniScene 实现为数据侧参考，以 ReSplat `main` 分支的 RE10K 实验为模型与训练侧参考。

## 目标与实验口径

- 在 OmniScene（nuScenes 派生数据）上训练并测试 ReSplat，提供 `112x200` 和 `224x400` 两套实验。
- 训练、验证、测试的 per-device batch size 均为 `1`。
- 总训练预算固定为 `100_001` 个 optimizer steps。ReSplat 是两阶段训练，因此按原 RE10K 的 `2:1` 比例拆为：
  - Init：`66_667` steps；
  - Refine：`33_334` steps；
  - 合计：`66_667 + 33_334 = 100_001` steps。
- `trainer.val_check_interval=0.01`，即每训练 `0.01` 个 epoch 触发一次 validation。它是 epoch 比例，不应写死解释成固定的 1k steps；实际 step 间隔取决于训练集长度、GPU 数量和分布式采样。
- `train.eval_model_every_n_val=10`，即每 10 次 validation 运行一次定量监控测试。ReSplat 已有该字段和调用逻辑，不需要忽略。
- `train.use_dynamic_mask=true`，动态物体区域只在训练损失中排除；正式测试的 PSNR、SSIM、LPIPS、PCC 仍按现有对比口径在完整图像上计算。
- 主表使用完整的官方 OmniScene test split，即完整读取 `bins_val_3.2m.json`，不做 `[0::14][:2048]` 抽样，也不使用 Center150。Center150 属于基于优化的方法对比，不在本次前馈高斯主实验范围内。

### 三项硬约束自查

| 自查项 | 结论 | 实施口径 |
| --- | --- | --- |
| 100k 是否为此前约定的两阶段总和 | 是 | Init `66_667` + Refine `33_334` = `100_001`，不是两个阶段各训练 100k。 |
| 是否从头训练 | 是，按前馈 baseline 的通常含义 | 不加载作者发布的 ReSplat/RE10K 完整模型；Init 从随机初始化的任务模型开始，只加载 ReSplat 方法原生依赖的 GMDepth/预训练 depth backbone；Refine 只加载本分辨率、本次 OmniScene Init 产生的 checkpoint。 |
| 训练损失是否使用动态物体掩码 | 实施后必须是 | `train.use_dynamic_mask=true`；只用静态有效像素计算/构造 Init、intermediate 和每轮 Refine 的 target MSE/LPIPS，动态区域不参与训练监督。 |

当前文档仍处于实施方案阶段：现有 ReSplat 代码尚未完成这项接入，`training_step` 中的 `valid_depth_mask` 当前仍固定为 `None`。后续实现的验收标准不是“配置里出现开关”，而是该 mask 实际传入所有 target loss 分支并经过数值测试。

## 配置概览

### 计划新增的配置与脚本

1. `config/dataset/omniscene.yaml`：OmniScene 数据集基础配置。
2. `config/dataset/view_sampler/all.yaml`：注册 `all` sampler；OmniScene 自己确定固定的 6 个 context 和 18 个 target，sampler 主要用于保持现有配置接口完整。
3. `config/experiment/omniscene_112x200.yaml`：112x200 实验配置。
4. `config/experiment/omniscene_224x400.yaml`：224x400 实验配置。
5. 每个分辨率各提供 Init、Refine 两个脚本，命名风格沿用 ReSplat，例如：
   - `scripts/omniscene_view6_112x200_base_init.sh`
   - `scripts/omniscene_view6_112x200_base_refine.sh`
   - `scripts/omniscene_view6_224x400_base_init.sh`
   - `scripts/omniscene_view6_224x400_base_refine.sh`

Hydra 的加载顺序保持不变：先加载 `config/main.yaml`，再由 `+experiment=omniscene_*` 覆盖数据集、模型、损失与实验参数，最后由脚本中的命令行参数覆盖阶段专属参数。不能只看 `config/main.yaml` 的默认值判断最终实验配置。

### 数据集基础配置

`config/dataset/omniscene.yaml` 计划采用以下核心值：

```yaml
defaults:
  - view_sampler: all

name: omniscene
roots: [datasets/omniscene]
image_shape: [224, 400]  # 由具体 experiment 覆盖
background_color: [0.0, 0.0, 0.0]
cameras_are_circular: false

near: 0.5
far: 100.0
baseline_scale_bounds: false
make_baseline_1: false
train_times_per_scene: 1
augment: false
test_split: total
```

DepthSplat 的 OmniScene YAML 虽写有 `augment: true`，但其 `DatasetOmniScene.__getitem__` 实际没有调用 augmentation shim。这里计划显式设为 `false`，使配置值与被复用的真实数据行为一致，避免出现“配置显示已增强、运行时却未增强”的死配置。

其中 `test_split` 明确支持：

- `total`：完整 `bins_val_3.2m.json`，作为最终主表结果的默认值；
- `mini`：`bins[0::14][:2048]`，只供训练中的定量监控使用，不作为论文主结果。

这样不再采用 DepthSplat 当前“修改/注释源码来切换完整测试集”的方式，split 由配置显式控制，可复现且不容易误测。`main.py` 构造训练期 `eval_cfg` 时会把副本的 `dataset.test_split` 改为 `mini`；最终 `mode=test` 则保持 `total`。

### 两个分辨率的实验配置

两份 experiment 只在 `dataset.image_shape`、实验名称和输出标签上不同，其余核心值一致：

| 配置项 | 112x200 | 224x400 |
| --- | ---: | ---: |
| `dataset.image_shape` | `[112, 200]` | `[224, 400]` |
| `data_loader.train.batch_size` | `1` | `1` |
| `data_loader.val.batch_size` | `1` | `1` |
| `data_loader.test.batch_size` | `1` | `1` |
| `trainer.max_steps` | `100_001`（总预算标记，阶段脚本覆盖） | `100_001`（总预算标记，阶段脚本覆盖） |
| `trainer.val_check_interval` | `0.01` | `0.01` |
| `train.eval_model_every_n_val` | `10` | `10` |
| `train.eval_data_length` | `2048` | `2048` |
| `train.use_dynamic_mask` | `true` | `true` |
| `test.compute_scores` | `true` | `true` |
| `test.eval_time_skip_steps` | `5` | `5` |

ReSplat 原始 `config/experiment/re10k.yaml` 用 `trainer.max_steps=300_001` 表示整个实验预算，而 Init/Refine 脚本分别覆盖为 `200_000` 和 `100_000`。OmniScene 沿用这一组织方式：experiment 中保留总预算 `100_001`，实际运行必须从阶段脚本启动，由脚本分别覆盖成 `66_667` 和 `33_334`，不能让两个阶段各训练 `100_001`。

`batch_size=1` 是 Lightning 的 per-device batch size。若开放多张可见 GPU，global batch 会随 GPU 数量增加；正式对比必须记录实际 GPU 数量，并与其它 100k-step baseline 保持相同的 step 定义，不能只比较 `max_steps` 而忽略 global sample 数。

### 模型、解码器、损失与优化器

模型参数以 ReSplat 的 RE10K base 实验为准，而不是照搬 DepthSplat 的 encoder：

- Encoder：`model/encoder=resplat`。
  - `num_depth_candidates=128`
  - `downscale_factor=4`，`shim_patch_size=4`
  - `num_scales=1`，`upsample_factor=4`，`lowest_feature_resolution=4`
  - `depth_unet_channels=128`，`gaussian_regressor_channels=512`
  - `monodepth_vit_type=vitb`
  - 初始化 Point Transformer：`attn_proj_channels=64`、`knn_samples=16`、`num_blocks=6`
  - refinement：`state_channels=512`、`num_basic_refine_blocks=4`、`update_attn_proj_channels=64`、`refine_knn_samples=8`、`render_error_mv_attn_blocks=1`
  - Gaussian adapter：`sh_degree=3`、`gaussian_scale_min=0.5`、`gaussian_scale_max=1.0`
- Decoder：`model/decoder=splatting_cuda`，即 ReSplat 的 gsplat decoder，`scale_invariant=false`。
- Loss：`loss=[mse, lpips]`。
  - MSE/实际实现中的 L1 颜色损失权重：`1.0`；
  - LPIPS 配置严格采用 RE10K experiment：`apply_after_step=0`、`weight=0.5`、`perceptual_loss=true`；
  - 不采用 DepthSplat OmniScene 中的 `LPIPS weight=0.05`。
- Init 优化器沿用 ReSplat 默认值：`lr=2e-4`、`lr_monodepth=2e-6`、`lr_depth=0`、`weight_decay=0.01`、`warm_up_ratio=0.01`。
- Refine 优化器沿用 RE10K refine 脚本：`lr=1e-4`、`lr_monodepth=0`，并设 `train.depth_smooth_loss_weight=0`。

### Init 与 Refine 的阶段配置

| 项目 | Init | Refine |
| --- | ---: | ---: |
| `trainer.max_steps` | `66_667` | `33_334` |
| `model.encoder.num_refine` | `0` | `2` |
| `model.encoder.latent_downsample` | `4`（默认） | `2` |
| `model.encoder.fixed_latent_size` | `true`（默认） | `false` |
| `model.encoder.init_gaussian_multiple` | `16` | `4` |
| `model.encoder.refine_same_num_points` | `false`（默认） | `true` |
| 基础权重来源 | 官方 ReSplat depth base 预训练权重 | 本分辨率 Init 阶段 checkpoint |
| 可训练参数 | Init encoder | 仅参数名包含 `encoder.update` 的 recurrent update 模块 |

Refine 不是 Init 之后在同一个 Trainer 内继续累计 step，而是新建 Trainer、载入 Init checkpoint、冻结基础 encoder，再从 stage-local step 0 训练 update 模块。因此最终 checkpoint 的 Lightning `global_step` 约为 `33_334`，论文中的训练预算需要按两个阶段外部求和为 `100_001`。

这里的“从头训练”具体定义为：

- Init 必须设置 `checkpointing.load=null`、`checkpointing.pretrained_model=null`、`checkpointing.resume=false`、`checkpointing.resume_update_module=null`，不得加载作者发布的完整 ReSplat 权重或任何 RE10K 训练 checkpoint；
- 保留 `model.encoder.unimatch_weights_path` 和 `checkpointing.pretrained_depth`，因为它们是 ReSplat 原方法训练 Init 时使用的通用深度先验，而不是已经在 RE10K/OmniScene 上训练好的完整 Gaussian reconstruction model；
- Refine 必须以本次同分辨率 OmniScene Init checkpoint 为唯一完整模型来源，并保持 `checkpointing.resume=false`、`checkpointing.resume_update_module=null`；
- 112x200 与 224x400 分别独立完成 Init→Refine，不跨分辨率继承任务 checkpoint。

这里有一个需要在实现阶段先验证的上游风险：RE10K 官方 Init/Refine 脚本同时改变了 `latent_downsample`、`fixed_latent_size` 和 `init_gaussian_multiple`，但只用 `checkpointing.no_strict_load=true` 加载 Init checkpoint。PyTorch 的 `strict=False` 可以忽略缺失/多余键，却通常不能忽略“同名但尺寸不同”的 tensor。正式训练前必须做一次 Init checkpoint 到 Refine 配置的 dry-load，并核对所有基础高斯预测参数是否正确载入；若存在尺寸冲突，应先报告并单独确定权重转换方案，不能静默丢弃冲突的高斯头后继续训练。

## 数据加载流程

### 可以直接复用的部分

DepthSplat 和 ReSplat 的 `DatasetCfgCommon`、`get_dataset`、`DataModule`、`context/target` batch 协议基本一致，因此以下实现可以按 DepthSplat 的结构移植：

1. 在 `src/dataset/__init__.py` 中注册 `DatasetOmniScene` 与 `DatasetOmniSceneCfg`，把 `omniscene` 加入 `DATASETS` 和 `DatasetCfg` union。
2. 移植并精简 `src/dataset/dataset_omniscene.py` 与 `src/dataset/utils_omniscene.py`。
3. 移植 `ViewSamplerAll` 及其配置/union 注册。OmniScene loader 当前直接构造固定视图，不实际依赖 sampler 的采样结果，但保留统一入口。
4. 在 `src/dataset/types.py` 的 view 类型中加入可选的 `masks` 和 `rel_depth` 字段。
5. 在 `src/dataset/shims/patch_shim.py` 中让 image、mask、relative depth 使用完全相同的中心裁剪窗口。

`DataModule.train_dataloader/val_dataloader/test_dataloader` 不需要为 OmniScene 另写一套；仍由 `dataset.name=omniscene` 自动实例化相应数据集。

### split 与样本构造

`DatasetOmniScene` 使用 `data_version=interp_12Hz_trainval`，读取：

- train：完整 `bins_train_3.2m.json`；
- val：`bins_val_3.2m.json` 的 `[:30000:3000][:10]`，得到固定的 10-bin 候选池；现有 `DataModule` 还会用 `ValidationWrapper(dataset, 1)` 包装它，因此每次 validation event 实际随机读取候选池中的 1 个 bin，而不是遍历 10 个；
- test/total：完整 `bins_val_3.2m.json`，不抽样；
- test/mini：`bins_val_3.2m.json` 的 `[0::14][:2048]`，只用于训练中监控。

每个 bin 读取 `bin_infos_3.2m/{token}.pkl`，相机顺序固定为：

1. `CAM_FRONT`
2. `CAM_FRONT_RIGHT`
3. `CAM_FRONT_LEFT`
4. `CAM_BACK`
5. `CAM_BACK_LEFT`
6. `CAM_BACK_RIGHT`

视图构造与 DepthSplat/SVF-GS 的前馈设置保持一致：

- context：6 个相机的 key frame（frame index 0），共 6 张；
- novel target：每个相机的 frame index `[1, 2]`，共 12 张；
- 最终 target：12 张 novel target 后拼接 6 张 context，共 18 张。

训练、验证和最终测试均保持同一 6-context/18-target 结构，不在训练中随机减少 target 数量。

### RGB、内外参与坐标系

`load_conditions` 按 DepthSplat 当前实现加载：

- RGB：把原始路径映射到 `samples_small`/`sweeps_small`；
- 相机参数：映射到 `samples_param_small`/`sweeps_param_small` 下的 JSON，逐图读取 `camera_intrinsic`；
- resize 后按宽、高分别缩放 `fx/cx` 与 `fy/cy`，随后把 K 的第一行除以宽、第二行除以高，返回 ReSplat decoder 所需的归一化内参；
- 外参：直接使用 `sensor2lidar_transform` 作为 OpenCV camera-to-world，不做 `flip_yz`。这与 ReSplat README 声明的 OpenCV C2W 约定一致；
- world frame：使用 key-frame LiDAR 坐标系。因为数据本身已锚定到 key-frame LiDAR，OmniScene 不照搬 RE10K refine 脚本的 `pose_align_middle_view=true`。

每张图必须独立读取和 resize 自己的 K，不能假设 6 个相机共享内参，也不能用某一张图的 K 覆盖整组。

### 动态物体掩码

- novel target 从 `samples_mask_small`/`sweeps_mask_small` 读取 PNG mask；
- 拼入 target 的 6 张 context 使用全 1 mask；
- mask 语义沿用现有数据：`True/1` 表示有效静态像素，训练时取反得到应排除的动态区域；
- `train.use_dynamic_mask=true` 时，mask 扩展为 `[B,V,3,H,W]` 并传给每一个 target 颜色损失。

DepthSplat 的单阶段 `training_step` 只处理一个输出，而 ReSplat 会对 Init/intermediate output 以及每次 recurrent refinement output 分别计算损失，因此不能只在最终输出处接入 mask。所有 target MSE/LPIPS 分支都必须使用同一个对齐后的 mask。`loss_on_input_views=false` 保持 RE10K 默认值；若将来开启 input-view supervision，应使用 context mask，不能把 18-view target mask 传给 6-view input render。

ReSplat 当前 LPIPS mask 分支会原地修改 `prediction.color` 和 `batch["target"]["image"]`。在多次 refinement loss 中，这会污染后续输出的 GT 与指标。实现时应改为基于副本的 `masked_fill`/非原地遮罩，保持与 DepthSplat 相同的屏蔽语义，但不能直接复制其原地赋值写法。

### 112x200 的实际裁剪说明

ReSplat 默认 `downscale_factor=4`、`shim_patch_size=4`，因此 data shim 要求空间尺寸能被 `16` 整除：

- `224x400` 均可被 16 整除，进入模型后仍为 `224x400`；
- `112x200` 中宽度 200 不能被 16 整除，会被中心裁剪为 `112x192`。

这也是 DepthSplat 当前 `112x200` 配置在相同 patch shim 下的实际行为。方案仍把 loader 的 resize 分辨率设为用户指定的 `112x200`，并同步裁剪 image、mask、rel_depth 和内参；最终指标实际覆盖中心的 `112x192` 区域。若论文口径要求模型内部与最终渲染必须严格为 `112x200`，则需要另行设计 padding/网络尺寸适配，不能只关闭 shim 后假定网络仍然合法。

### 与 DepthSplat/SVF-GS loader 的差异

- 与 DepthSplat 相比：主要数据路径、6/18 视图组织、K 归一化、mask 和 relative-depth 加载可以复用；ReSplat 侧新增的工作主要是把 mask 接到多轮 refinement loss，并保证最终 refinement depth 用于 PCC。
- 与 SVF-GS 相比：ReSplat 不需要 loader 返回 Metric3D metric depth、置信度、显式 rays、FOV 或 `w2i`；ReSplat encoder/decoder 根据 RGB、C2W、归一化 K、near/far 自行完成深度预测、反投影和渲染。
- 不复用 SVF-GS 的 `flip_yz`；ReSplat 使用 OpenCV C2W，直接采用 `sensor2lidar_transform`。
- 不加入 Center150 split。本次前馈方法主表只保留官方完整 test；`mini` 仅是训练监控用的工程 split。

## 主程序调用方式

### 数据入口

调用方式仍是：

```bash
python -m src.main +experiment=omniscene_112x200 ...
python -m src.main +experiment=omniscene_224x400 ...
```

`src/main.py` 只需要增加 OmniScene 的训练期评测分支：当 `dataset.roots`/`dataset.name` 指向 OmniScene 时，不构造 RE10K 的 evaluation-index sampler，而是保留 `view_sampler=all`，复制当前 dataset 配置并把副本的 `test_split` 设为 `mini` 后交给 `run_full_test_sets_eval`。最终 `mode=test` 不走这份副本，直接使用 experiment 中的 `test_split=total`。

### Init 训练命令模板

以 112x200 为例：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m src.main +experiment=omniscene_112x200 \
    trainer.max_steps=66667 \
    model.encoder.init_gaussian_multiple=16 \
    checkpointing.load=null \
    checkpointing.pretrained_model=null \
    checkpointing.pretrained_depth=pretrained/resplat-depth-base-352x640-60be7abf.pth \
    checkpointing.resume=false \
    checkpointing.resume_update_module=null \
    train.use_dynamic_mask=true \
    trainer.val_check_interval=0.01 \
    train.eval_model_every_n_val=10 \
    output_dir=checkpoints/resplat/omniscene-view6-112x200/base-init
```

224x400 只替换 experiment 与输出路径，模型、损失、学习率和训练节奏保持一致。

### Refine 训练命令模板

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m src.main +experiment=omniscene_112x200 \
    trainer.max_steps=33334 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.load=null \
    checkpointing.pretrained_model=<init_checkpoint> \
    checkpointing.no_strict_load=true \
    checkpointing.resume=false \
    checkpointing.resume_update_module=null \
    train.use_dynamic_mask=true \
    trainer.val_check_interval=0.01 \
    train.eval_model_every_n_val=10 \
    output_dir=checkpoints/resplat/omniscene-view6-112x200/base-refine
```

ReSplat 当前 `main.py` 会在 `num_refine>0` 时自动冻结所有不含 `encoder.update` 的参数；这部分保持原项目逻辑，不照搬 DepthSplat 的单阶段训练方式。

### 完整 test 命令模板

最终主表只测试 Refine checkpoint，并显式指定完整 split：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m src.main +experiment=omniscene_112x200 \
    mode=test \
    dataset.test_split=total \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    checkpointing.pretrained_model=<refine_checkpoint> \
    test.compute_scores=true \
    output_dir=outputs/resplat-omniscene-112x200-base-refine-total
```

224x400 同理。最终结果必须核对 test dataloader 长度等于完整 `bins_val_3.2m.json` 的 bin 数，并核对每项指标 JSON 的条目数，不能用 tmux/进程退出作为结果完整性的唯一证据。

## PCC 指标补充方案

### 相对深度加载

PCC 的参考深度严格沿用 SVF-GS 与 DepthSplat：

1. 只在 `stage=test` 时加载，train/val 不读取，避免额外 I/O。
2. 从 `samples_dpt_small`/`sweeps_dpt_small` 读取 DepthAnything-v2 disparity `.npy`。
3. 若 RGB 被 resize，相同方式把 disparity resize 到 loader 分辨率。
4. 计算：

```text
ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
min_disp = disp.max() / ratio
rel_depth = 1 / max(disp, min_disp)
rel_depth = (rel_depth - rel_depth.min()) / (rel_depth.max() - rel_depth.min())
```

5. 12 张 novel target 的 relative depth 与 6 张 context 的 relative depth 按 RGB 相同顺序拼接成 `[18,H,W]`，写入 `target["rel_depth"]`。
6. patch shim 必须同步裁剪 `rel_depth`，并在 smoke test 中检查 shape 对齐、finite 和 `[0,1]` 范围。

### 测试时渲染深度

ReSplat 的 gsplat decoder 与 DepthSplat 不完全相同：它当前固定使用 `render_mode="RGB+ED"`，即使 `depth_mode=None` 也会返回 expected depth。因此数据侧不需要提供 metric depth，也不需要新增另一套深度 renderer。

接入时仍建议在“`test.compute_scores=true` 且存在 `target.rel_depth`”时令 `depth_mode="depth"`，保持接口意图清晰，并处理两个 ReSplat 特有分支：

- `test_step` 的最终 PCC 必须使用最后一次 refinement 的 `render_output[-1].depth`，不能使用 Init Gaussian 的 depth；
- 当 `test.render_chunk_size` 非空时，当前代码只拼接 chunk 的 color、丢弃 depth，需要同时拼接各 chunk 的 `curr_output.depth`。

`run_full_test_sets_eval` 同样在 refinement 后读取最终 output depth。Init 阶段的训练期监控则使用 Init output depth。

### PCC 计算位置与口径

`src/evaluation/metrics.py` 增加与 SVF-GS/DepthSplat 一致的 `get_pcc/compute_pcc`，使用 `torchmetrics.PearsonCorrCoef`。对一个 bin 的 18 个 target view 和所有像素整体 flatten 后计算一个 PCC：

```python
pcc = compute_pcc(
    rearrange(rel_depth, "b v h w -> (b v) h w"),
    rearrange(output.depth, "b v h w -> (b v) h w"),
)
```

该口径是“每个 bin 一个 PCC，再对所有 bin 算术平均”，与 SVF-GS 当前实现保持一致；不擅自改成先逐图计算再平均，也不使用动态 mask 过滤 PCC。

### PCC 统计与输出

- `ModelWrapper.test_step`：把当前 bin 的 PCC 追加到 `test_step_outputs["pcc"]`。
- `ModelWrapper.on_test_end`：复用现有汇总逻辑，输出：
  - `metrics/scores_pcc_all.json`：每个 bin 一个值；
  - `metrics/scores_all_avg.json`：包含最终平均 `pcc`，同时保留 PSNR、SSIM、LPIPS 和耗时。
- `ModelWrapper.run_full_test_sets_eval`：在 `scores_dict` 中加入与 RGB 指标同级的 `pcc/probabilistic`，用于训练期 mini split 监控。
- 只有 `target.rel_depth` 与 `output.depth` 同时存在才计算 PCC；缺失文件或 shape 不一致应直接给出带 bin token/图像路径的错误，不能悄悄少算样本。

因此，DepthSplat 的“相对深度读取—渲染深度—计算 PCC—JSON 汇总”主链路可以复用，但不能整段原样复制：ReSplat 必须选择最终 recurrent output、补齐 chunk depth 拼接，并在多轮 loss 中避免 mask 的原地污染。

## 实现文件清单

计划新增：

- `config/dataset/omniscene.yaml`
- `config/dataset/view_sampler/all.yaml`
- `config/experiment/omniscene_112x200.yaml`
- `config/experiment/omniscene_224x400.yaml`
- `src/dataset/dataset_omniscene.py`
- `src/dataset/utils_omniscene.py`
- 4 个 OmniScene Init/Refine 运行脚本

计划修改：

- `src/dataset/__init__.py`：注册 dataset 与配置类型。
- `src/dataset/view_sampler/__init__.py`：注册 `all` sampler。
- `src/dataset/types.py`：声明 `masks`、`rel_depth`。
- `src/dataset/shims/patch_shim.py`：同步裁剪 mask 与 relative depth。
- `config/main.yaml`、`src/model/model_wrapper.py`：增加 `train.use_dynamic_mask` 并覆盖所有 target loss/refinement 路径。
- `src/loss/loss_lpips.py`：改为非原地 mask。
- `src/evaluation/metrics.py`：增加 PCC。
- `src/main.py`：OmniScene 训练期 mini eval 配置，不再要求 evaluation-index JSON。
- `README.md`：实现完成并验证后补充两种分辨率、两阶段训练与完整测试命令。

## 实现后的验证清单

1. **Hydra 配置检查**：分别打印两种 experiment 的最终 resolved config，确认 dataset、encoder、decoder、loss、三个 batch size、两阶段 steps、验证/测试节奏和完整 test split。
2. **真实样本检查**：各 split 至少读取一个真实 bin，确认 context/target 为 6/18 views，逐图 K 正确，C2W 为 OpenCV convention，near/far 为 0.5/100。
3. **两种分辨率检查**：loader 输出分别为 112x200、224x400；data shim 后分别为 112x192、224x400，且 image/mask/rel_depth 完全对齐。
4. **mask 检查**：统计静态/动态像素比例，确认 `True=静态有效`，并验证 Init、intermediate、每一轮 Refine 的 target loss 都应用了 mask；LPIPS 不修改原始 GT tensor。
5. **PCC 检查**：人工构造同序、逆序 depth 验证 PCC 约为 `+1/-1`；再用真实样本验证最终 refinement depth 与 relative depth shape 一致，chunk/non-chunk 结果一致。
6. **两阶段 checkpoint 门禁**：先做 Init checkpoint 到 Refine 配置的 dry-load，列出 missing/unexpected/shape-mismatch keys；基础预测权重未正确加载时不得启动正式 Refine 训练。
7. **单步 smoke**：两种分辨率分别完成 Init train/val/test 和 Refine train/val/test 的真实数据单步前向、反向与指标落盘。
8. **完整测试完整性**：最终 test dataloader 长度等于 `bins_val_3.2m.json` 的完整长度；`scores_psnr_all.json`、`scores_ssim_all.json`、`scores_lpips_all.json`、`scores_pcc_all.json` 条目数全部一致。

## 小结

OmniScene 的配置、6/18 视图数据组织、动态 mask、relative depth 与 PCC 主链路可以以 DepthSplat 为模板迁移；ReSplat 的核心模型和训练超参数则严格保留 RE10K base 方案。真正不能直接照搬的部分有三处：两阶段 `66_667 + 33_334` 预算与 checkpoint 交接、多轮 recurrent output 的 mask 接入、以及最终 refinement/chunk 路径的 depth 与 PCC。最终前馈主表只使用完整官方 test split，不包含 Center150。
