"""Persist complete stage-final mini results independently of periodic logging."""

import json
import math
from pathlib import Path


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
