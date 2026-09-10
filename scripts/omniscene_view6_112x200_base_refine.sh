#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <omniscene-init-checkpoint>" >&2
    exit 2
fi

init_checkpoint="$1"

# Refine trains only encoder.update* for the remaining 33_334 optimizer steps.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m src.main \
    +experiment=omniscene_112x200 \
    trainer.max_steps=33334 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    model.encoder.recurrent_use_checkpointing=true \
    model.encoder.refine_scale_update_mode=bounded_additive \
    model.encoder.refine_scale_delta_max=0.5 \
    model.encoder.refine_scale_max=4.0 \
    model.encoder.refine_update_head_fp32=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.every_n_train_steps=500 \
    checkpointing.load=null \
    checkpointing.pretrained_model="${init_checkpoint}" \
    checkpointing.no_strict_load=true \
    checkpointing.resume=false \
    checkpointing.resume_update_module=null \
    train.use_dynamic_mask=true \
    train.refine_raw_scale_regularization_weight=0.01 \
    train.diagnostics_log_every_n_steps=10 \
    train.diagnostics_stop_on_divergence=true \
    train.diagnostics_scale_mean_threshold=2.0 \
    train.diagnostics_delta_scale_mean_threshold=0.25 \
    train.diagnostics_raw_scale_saturation_fraction_threshold=0.25 \
    train.diagnostics_divergence_patience=100 \
    trainer.val_check_interval=0.01 \
    trainer.log_every_n_steps=10 \
    train.eval_model_every_n_val=10 \
    wandb.project=omniscene-view6-112x200 \
    output_dir=checkpoints/resplat/omniscene-view6-112x200/base-refine
