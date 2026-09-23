#!/usr/bin/env bash
# Sourced by the four OmniScene stage scripts. Paths are project-root relative.
omniscene_stage="$1"
omniscene_resolution="$2"
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."

omniscene_suffix=""
omniscene_init_checkpoint=""
omniscene_overrides=()
omniscene_config_only=false
for omniscene_arg in "$@"; do
    case "$omniscene_arg" in
        *=*) omniscene_overrides+=("$omniscene_arg") ;;
        --cfg|--resolve|--help|-h)
            omniscene_config_only=true
            omniscene_overrides+=("$omniscene_arg") ;;
        job|all|hydra)
            omniscene_overrides+=("$omniscene_arg") ;;
        *.ckpt)
            if [[ "$omniscene_stage" != refine || -n "$omniscene_init_checkpoint" ]]; then
                echo "Only Refine accepts one explicit Init checkpoint." >&2
                exit 2
            fi
            omniscene_init_checkpoint="$omniscene_arg" ;;
        _*)
            if [[ -n "$omniscene_suffix" || ! "$omniscene_arg" =~ ^_[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
                echo "Use one experiment suffix such as _1 or _repeat2." >&2
                exit 2
            fi
            omniscene_suffix="$omniscene_arg" ;;
        *)
            echo "Usage: $0 [<init.ckpt> (Refine only)] [_suffix] [key=value ...]" >&2
            exit 2 ;;
    esac
done

omniscene_root="checkpoints/resplat/omniscene-view6-${omniscene_resolution}"
omniscene_output_dir="${omniscene_root}/base-${omniscene_stage}${omniscene_suffix}"
if [[ "$omniscene_stage" == refine && -z "$omniscene_init_checkpoint" ]]; then
    omniscene_init_checkpoint="${omniscene_root}/base-init${omniscene_suffix}/checkpoints/final-step_66667.ckpt"
fi

# A new repeat must not overwrite an earlier repeat. Explicit restoration can
# reuse its output directory. Hydra inspection does not start a training run.
omniscene_resume=false
omniscene_load=null
omniscene_effective_pretrained="$omniscene_init_checkpoint"
for omniscene_arg in "${omniscene_overrides[@]}"; do
    case "$omniscene_arg" in
        output_dir=*) omniscene_output_dir="${omniscene_arg#output_dir=}" ;;
        checkpointing.resume=*) omniscene_resume="${omniscene_arg#checkpointing.resume=}" ;;
        checkpointing.load=*) omniscene_load="${omniscene_arg#checkpointing.load=}" ;;
        checkpointing.pretrained_model=*)
            omniscene_effective_pretrained="${omniscene_arg#checkpointing.pretrained_model=}" ;;
    esac
done
if [[ "$omniscene_config_only" == false ]]; then
    if [[ "$omniscene_resume" != true && "$omniscene_load" == null && -d "$omniscene_output_dir" && -n "$(ls -A "$omniscene_output_dir")" ]]; then
        echo "Output already exists: $omniscene_output_dir. Use a new suffix or explicit checkpoint restoration." >&2
        exit 2
    fi
    if [[ "$omniscene_stage" == refine && "$omniscene_effective_pretrained" != null && ! -f "$omniscene_effective_pretrained" ]]; then
        echo "Init checkpoint not found: $omniscene_effective_pretrained" >&2
        exit 2
    fi
fi
