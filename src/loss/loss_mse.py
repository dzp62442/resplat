from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        global_step: int,
        clamp_large_error: float,
        valid_depth_mask: Tensor | None,
        loss_on_input_views: bool = False
    ) -> Float[Tensor, ""]:
        if loss_on_input_views:
            delta = prediction.color - batch["context"]["image"]
        else:
            delta = prediction.color - batch["target"]["image"]

        if valid_depth_mask is not None:
            if valid_depth_mask.shape != delta.shape:
                raise ValueError(
                    "Loss mask/image shape mismatch: "
                    f"mask={tuple(valid_depth_mask.shape)}, image={tuple(delta.shape)}"
                )
            static_pixels = ~valid_depth_mask.bool()
            if not static_pixels.any():
                raise ValueError("Dynamic-object mask excludes every training pixel")
            delta = delta[static_pixels]

        if clamp_large_error > 0:
            valid_mask = delta.abs() < clamp_large_error
            delta = delta[valid_mask]

        return self.cfg.weight * (delta.abs()).mean()
