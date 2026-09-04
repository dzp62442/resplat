from collections.abc import Mapping

from torch import Tensor, nn


def load_state_dict_with_shape_check(
    module: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    strict: bool,
    source: str,
    allowed_missing_prefixes: tuple[str, ...] | None = None,
    reject_unexpected: bool = False,
):
    """Load a checkpoint without silently accepting same-name shape changes."""
    current_state = module.state_dict()
    mismatched = [
        (key, tuple(value.shape), tuple(current_state[key].shape))
        for key, value in state_dict.items()
        if key in current_state and value.shape != current_state[key].shape
    ]
    if mismatched:
        details = "\n".join(
            f"  - {key}: checkpoint {checkpoint_shape}, model {model_shape}"
            for key, checkpoint_shape, model_shape in mismatched
        )
        raise RuntimeError(
            f"Checkpoint {source!r} has {len(mismatched)} same-name parameter "
            f"shape mismatch(es):\n{details}\n"
            "Refusing a partial load because the unmatched model parameters would "
            "remain randomly initialized."
        )

    if allowed_missing_prefixes is not None:
        missing = sorted(current_state.keys() - state_dict.keys())
        disallowed_missing = [
            key
            for key in missing
            if not key.startswith(allowed_missing_prefixes)
        ]
        unexpected = sorted(state_dict.keys() - current_state.keys())
        if disallowed_missing or (reject_unexpected and unexpected):
            details = []
            if disallowed_missing:
                details.append(
                    "disallowed missing keys:\n  - "
                    + "\n  - ".join(disallowed_missing)
                )
            if reject_unexpected and unexpected:
                details.append(
                    "unexpected keys:\n  - " + "\n  - ".join(unexpected)
                )
            raise RuntimeError(
                f"Checkpoint {source!r} failed the staged-load key audit:\n"
                + "\n".join(details)
            )

    return module.load_state_dict(state_dict, strict=strict)
