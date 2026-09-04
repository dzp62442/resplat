#!/usr/bin/env bash
set -euo pipefail

# Fresh OmniScene Init training: 66_667 of the total 100_001 optimizer steps.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m src.main \
    +experiment=omniscene_224x400 \
    trainer.max_steps=66667 \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.refine_same_num_points=true \
    model.encoder.num_refine=0 \
    checkpointing.load=null \
    checkpointing.pretrained_model=null \
    checkpointing.pretrained_depth=pretrained/resplat-depth-base-352x640-60be7abf.pth \
    checkpointing.resume=false \
    checkpointing.resume_update_module=null \
    train.use_dynamic_mask=true \
    trainer.val_check_interval=0.01 \
    train.eval_model_every_n_val=10 \
    wandb.project=omniscene-view6-224x400 \
    output_dir=checkpoints/resplat/omniscene-view6-224x400/base-init
