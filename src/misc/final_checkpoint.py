from pathlib import Path

from pytorch_lightning import Callback, LightningModule, Trainer


def get_final_checkpoint_path(directory: Path, global_step: int) -> Path:
    return directory / f"final-step_{global_step}.ckpt"


class FinalCheckpoint(Callback):
    """Save the exact final optimizer step after a successful fit."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        if trainer.global_step <= 0:
            return
        path = get_final_checkpoint_path(self.directory, trainer.global_step)
        # Lightning requires every rank to call save_checkpoint; only rank zero
        # writes the file.
        trainer.save_checkpoint(path, weights_only=True)
        if trainer.is_global_zero:
            print(f"Saved final checkpoint: {path}")
