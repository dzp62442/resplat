import re
import warnings
from pathlib import Path

import torch

from collections import OrderedDict


# Function to extract the step number from the filename
def extract_step(file_name):
    match = re.search(r"(?:^|-)step_(\d+)\.ckpt$", file_name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint name: {file_name}")
    return int(match.group(1))


def checkpoint_info(path: Path) -> dict:
    """Inspect local trusted Lightning checkpoints on CPU without GPU allocation."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        state = torch.load(path, map_location="cpu", weights_only=False)
    step = state.get("global_step")
    if not isinstance(step, int) or step < 0 or not state.get("state_dict"):
        raise ValueError(f"Invalid checkpoint contents: {path}")
    if step != extract_step(path.name):
        raise ValueError(f"Checkpoint step does not match filename: {path}")
    optimizers = state.get("optimizer_states", [])
    schedulers = state.get("lr_schedulers", [])
    full = bool(optimizers and schedulers and state.get("loops", {}).get("fit_loop"))
    full = full and all(isinstance(opt.get("state"), dict) and opt.get("param_groups") for opt in optimizers)
    full = full and all(isinstance(scheduler, dict) and scheduler for scheduler in schedulers)
    return {"step": step, "resumable": full}


def find_latest_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    candidates = []
    for path in ckpt_dir.glob("*.ckpt"):
        try:
            candidates.append((extract_step(path.name), path))
        except ValueError:
            continue
    for _, path in sorted(candidates, reverse=True):
        try:
            if checkpoint_info(path)["resumable"]:
                return path
        except Exception as error:
            warnings.warn(f"Skipping unreadable checkpoint {path}: {error}")
    raise ValueError(f"No complete optimizer/scheduler/loop checkpoint found in {ckpt_dir}")


def no_resume_upsampler(pretrained_state_dict):
    new_state_dict = OrderedDict()
    for key, value in pretrained_state_dict.items():
        if 'upsampler' not in key:
            new_state_dict[key] = value

    return new_state_dict


def load_partial_state_dict(model, pretrained_state_dict):
    # Load only matching parameters
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    # for key in model_state_dict:
    #     if key not in filtered_state_dict:
    #         print(key)
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
