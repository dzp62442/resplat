#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <omniscene-init-checkpoint>" >&2
    exit 2
fi

init_checkpoint="$1"

# Refine trains only encoder.update* for the remaining 33_334 optimizer steps.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m src.main \
    +experiment=omniscene_224x400 \
    trainer.max_steps=33334 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    model.encoder.recurrent_use_checkpointing=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.load=null \
    checkpointing.pretrained_model="${init_checkpoint}" \
    checkpointing.no_strict_load=true \
    checkpointing.resume=false \
    checkpointing.resume_update_module=null \
    train.use_dynamic_mask=true \
    trainer.val_check_interval=0.01 \
    train.eval_model_every_n_val=10 \
    wandb.project=omniscene-view6-224x400 \
    output_dir=checkpoints/resplat/omniscene-view6-224x400/base-refine
