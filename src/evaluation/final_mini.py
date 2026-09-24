"""Persist complete stage-final mini results independently of periodic logging."""

import json
import math
from pathlib import Path


def evaluate_saved_final_mini(model, data_module, checkpoint: Path, max_steps: int,
                             output_dir: Path, logger=None, device="cuda") -> None:
    """Reuse the exact end-of-stage metric path without a fit/optimizer step."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("global_step") != max_steps or max_steps <= 0:
        raise ValueError("Final mini recovery requires the exact final-step checkpoint")
    expected = output_dir / "checkpoints" / f"final-step_{max_steps}.ckpt"
    if checkpoint.resolve() != expected.resolve():
        # The process may have exited between the last periodic save and the
        # final callback. Promote the exact completed state, never train again.
        if checkpoint.parent.resolve() not in (
            (output_dir / "checkpoints").resolve(), (output_dir / "checkpoints_backups").resolve()
        ):
            raise ValueError("Final mini recovery must use this stage's checkpoint")
        expected.parent.mkdir(parents=True, exist_ok=True)
        temporary = expected.with_suffix(".ckpt.tmp")
        torch.save({key: value for key, value in payload.items()
                    if key not in ("optimizer_states", "lr_schedulers")}, temporary)
        temporary.replace(expected)
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    model.to(device).eval()
    with torch.inference_mode():
        model.run_full_test_sets_eval(
            final_output_dir=output_dir / f"mini-final-step_{max_steps}",
            eval_data_module=data_module, eval_step=max_steps, eval_logger=logger,
        )


def save_final_mini_scores(
    output_dir: Path,
    scores: dict[str, list[float]],
    expected_samples: int,
    metadata: dict,
) -> dict[str, float]:
    required = ("psnr", "ssim", "lpips", "pcc")
    if expected_samples <= 0:
        raise ValueError("Stage-final mini evaluation requires a nonempty dataset")
    for name in required:
        values = scores.get(name, [])
        if len(values) != expected_samples or not all(map(math.isfinite, values)):
            raise RuntimeError(
                f"Incomplete/non-finite final mini {name}: "
                f"{len(values)} scores, expected {expected_samples}"
            )

    summary = {name: sum(scores[name]) / expected_samples for name in required}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        with (output_dir / f"scores_{name}_all.json").open("w") as f:
            json.dump(scores[name], f, allow_nan=False)
    with (output_dir / "evaluation.json").open("w") as f:
        json.dump(
            {**metadata, "sample_count": expected_samples, "split": "mini"},
            f,
            indent=2,
            allow_nan=False,
        )
    # Written last: a summary is only published after all four arrays validate.
    with (output_dir / "scores_all_avg.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return summary
