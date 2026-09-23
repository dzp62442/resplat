# 在 OmniScene 数据集上与 SVF-GS 进行对比

数据组织、实验口径与实现细节见
[`docs/OmniScene 数据集实验文档.md`](docs/OmniScene%20数据集实验文档.md)。
两种分辨率都采用 Init `66,667` steps + Refine `33,334` steps，总计
`100,001` optimizer steps；训练损失启用动态物体掩码。
两个阶段正常完成后都会自动评估完整 mini（2,048 个样本、18 个 target
views），并把最终 PSNR、SSIM、LPIPS、PCC 写入该阶段的 W&B 记录和本地 JSON。

## 训练

### 112x200

```bash
# 第一阶段：从头训练 Init
bash scripts/omniscene_view6_112x200_base_init.sh

# 第一阶段完成后启动 Refine
bash scripts/omniscene_view6_112x200_base_refine.sh \
    checkpoints/resplat/omniscene-view6-112x200/base-init/checkpoints/final-step_66667.ckpt
```

### 224x400

```bash
# 第一阶段：从头训练 Init
bash scripts/omniscene_view6_224x400_base_init.sh

# 第一阶段完成后启动 Refine
bash scripts/omniscene_view6_224x400_base_refine.sh \
    checkpoints/resplat/omniscene-view6-224x400/base-init/checkpoints/final-step_66667.ckpt
```

### 重复实验与阶段结束评估

同一次实验的两个阶段使用同一个后缀。Refine 不传 checkpoint 时，自动加载
同分辨率、同后缀 Init 的 `final-step_66667.ckpt`。例如 `_1`：

```bash
conda activate resplat
bash scripts/omniscene_view6_112x200_base_init.sh _1 && \
bash scripts/omniscene_view6_112x200_base_refine.sh _1
```

Init 的训练和最终 mini 评估都成功后才会启动 Refine。重复下一次时把两处 `_1`
换为 `_2`；224x400 使用对应分辨率的脚本。输出目录分别为
`checkpoints/resplat/omniscene-view6-112x200/base-init_1` 和 `base-refine_1`，
W&B run 名称也分别为 `base-init_1`、`base-refine_1`。已存在的非空目录会拒绝
新训练，防止覆盖旧实验。

后缀只区分实验，不自动改变 seed；如需多种子实验，可在两阶段命令后追加
相同的 `seed=111124 data_loader.train.seed=1235`。其他 Hydra `key=value`
参数也支持透传；仍兼容 `refine.sh <init.ckpt> [_suffix]` 的显式权重用法。

最终 mini 与每 10 次 validation 的周期 mini 独立。结果分别保存在：

```text
base-init_1/mini-final-step_66667/metrics/scores_all_avg.json
base-refine_1/mini-final-step_33334/metrics/scores_all_avg.json
```

同目录的 `scores_{psnr,ssim,lpips,pcc}_all.json` 保存逐样本分数，
`evaluation.json` 记录最终 checkpoint、step、split、样本数量和场景列表。
W&B 的 `final_mini/*` 明确标识最终结果，`test/*` 也更新为最终结果。
暂停、报错或诊断保护提前停止不会触发最终 mini；断点恢复后正常到达目标步数才会触发。

Refine 脚本包含当前全部稳定性配置，新重复实验会从 Refine 第 0 步启用
有界尺度更新和 raw scale 正则（历史修复实验是在中途加入正则）。

## 完整测试

主表使用完整的官方 OmniScene test split，不使用 Center150。

### 112x200

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_112x200 \
    mode=test \
    dataset.test_split=total \
    model.encoder.num_refine=2 \
    model.encoder.refine_scale_update_mode=bounded_additive \
    model.encoder.refine_scale_delta_max=0.5 \
    model.encoder.refine_scale_max=4.0 \
    model.encoder.refine_update_head_fp32=true \
    checkpointing.pretrained_model=checkpoints/resplat/omniscene-view6-112x200/base-refine/checkpoints/final-step_33334.ckpt \
    test.compute_scores=true \
    output_dir=outputs/resplat-omniscene-112x200-base-refine-total
```

### 224x400

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_224x400 \
    mode=test \
    dataset.test_split=total \
    model.encoder.num_refine=2 \
    model.encoder.refine_scale_update_mode=bounded_additive \
    model.encoder.refine_scale_delta_max=0.5 \
    model.encoder.refine_scale_max=4.0 \
    model.encoder.refine_update_head_fp32=true \
    checkpointing.pretrained_model=checkpoints/resplat/omniscene-view6-224x400/base-refine/checkpoints/final-step_33334.ckpt \
    test.compute_scores=true \
    output_dir=outputs/resplat-omniscene-224x400-base-refine-total
```

---

<p align="center">
  <h1 align="center">ReSplat: Learning Recurrent Gaussian Splatting</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    &middot;
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a>
    &middot;
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
    &middot;
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
  </p>
  <h3 align="center">ECCV 2026 Spotlight</h3>
  <h3 align="center">
    <a href="https://arxiv.org/abs/2510.08575">Paper</a> | <a href="https://haofeixu.github.io/resplat/">Project Page</a> | <a href="MODEL_ZOO.md">Models</a>
  </h3>
</p>

<p align="center">
  <img src="https://haofeixu.github.io/resplat/assets/teaser.png" alt="ReSplat teaser" width="100%">
</p>

ReSplat is a feed-forward recurrent model for 3D Gaussian splatting that iteratively refines Gaussians using the rendering error as a gradient-free feedback signal for test-time adaptation.

**Key features:**
- **Compact initialization**: Predicts Gaussians in a subsampled space (16× fewer Gaussians than prior per-pixel methods)
- **Recurrent refinement**: Weight-sharing recurrent module that uses rendering error to predict per-Gaussian parameter updates

## Installation

This codebase is developed with Python 3.12, PyTorch 2.7.0, and CUDA 12.8.

We recommend setting up a virtual environment (e.g., [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html)) before installation:

```bash
# conda
conda create -y -n resplat python=3.12
conda activate resplat

# torch 2.7.0, cuda 12.8 改为 11.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# Install gsplat 1.5.3
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

# Install pointops (kNN)
cd src/model/encoder/pointops && python setup.py install && cd ../../../..
```

## Model Zoo

Pre-trained models are available in the [Model Zoo](MODEL_ZOO.md).

Download the weights and place (or symlink) them in the `pretrained` directory:

```bash
ln -s YOUR_MODEL_PATH pretrained
```

## Camera Conventions

The camera intrinsic matrices are normalized, with the first row divided by the image width and the second row divided by the image height.

The camera extrinsic matrices follow the OpenCV convention for camera-to-world transformation (+X right, +Y down, +Z pointing into the screen).

## Dataset Preparation

See [DATASETS.md](DATASETS.md) for detailed instructions on preparing RealEstate10K, DL3DV and ACID datasets.

Symlink the downloaded datasets to the `datasets` directory:

```bash
ln -s YOUR_DATASET_PATH datasets
```

## Demo

Check [scripts/infer_colmap.sh](scripts/infer_colmap.sh) for running our pre-trained models on COLMAP datasets.

A demo scene can be downloaded [here](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/dl3dv-colmap-demo.zip) to quickly try our method.


## Evaluation

Evaluation scripts are also provided in [scripts/](scripts) for reproducing the results in our paper.

## Training

ReSplat is trained in two stages: (1) initial Gaussian prediction and (2) recurrent refinement.

The training scripts in [scripts/](scripts) contain the exact commands and hyperparameters used for the experiments in our paper. Please refer to them for detailed configurations.

Before training, you need to download the pre-trained [depth model](MODEL_ZOO.md), and set up your [wandb account](config/main.yaml) (in particular, by setting `wandb.entity=YOUR_ACCOUNT`) for logging.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{xu2026resplat,
      title={ReSplat: Learning Recurrent Gaussian Splatting},
      author={Xu, Haofei and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
      booktitle={ECCV},
      year={2026}
    }
```

## Acknowledgements

Our codebase builds upon several excellent open-source projects: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [MVSplat360](https://github.com/donydchen/mvsplat360), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [DepthSplat](https://github.com/cvg/depthsplat), [Pointcept](https://github.com/Pointcept/Pointcept), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [gsplat](https://github.com/nerfstudio-project/gsplat), and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank all the authors for their great work.
