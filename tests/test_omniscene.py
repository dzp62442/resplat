import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.dataset.dataset_omniscene import DatasetOmniScene
from src.dataset.shims.patch_shim import apply_patch_shim_to_views
from src.evaluation.metrics import compute_pcc
from src.loss.loss_lpips import LossLpips, LossLpipsCfg
from src.loss.loss_mse import LossMse, LossMseCfg, LossMseCfgWrapper
from src.misc.checkpoint_loading import load_state_dict_with_shape_check
from src.model.decoder.decoder import DecoderOutput
from src.model.model_wrapper import _get_target_invalid_mask


class DummyPerceptualLoss(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (predicted - target).abs().mean()


class DummyStagedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2)
        self.update = nn.Linear(2, 2)


class TestOmniScene(unittest.TestCase):
    def _make_dataset_root(self, root: Path) -> None:
        data_dir = root / "interp_12Hz_trainval"
        (data_dir / "bin_infos_3.2m").mkdir(parents=True)
        bins = ["scene-test-bin000"] * 30
        for split in ("train", "val"):
            with (data_dir / f"bins_{split}_3.2m.json").open("w") as file:
                json.dump({"bins": bins}, file)

        sensor_info = {}
        for camera_index, camera in enumerate(DatasetOmniScene.camera_types):
            frames = []
            for frame_index in range(3):
                relative_path = Path("samples") / camera / f"{frame_index}.jpg"
                source_path = Path("/datasets/nuScenes") / relative_path
                frames.append(
                    {
                        "data_path": str(source_path),
                        "sensor2lidar_transform": np.array(
                            [
                                [1, 0, 0, camera_index],
                                [0, 1, 0, frame_index],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ],
                            dtype=np.float32,
                        ),
                    }
                )

                image = np.full(
                    (8, 16, 3), camera_index * 20 + frame_index, dtype=np.uint8
                )
                rgb_path = root / "samples_small" / camera / f"{frame_index}.jpg"
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(rgb_path)

                parameter_path = (
                    root / "samples_param_small" / camera / f"{frame_index}.json"
                )
                parameter_path.parent.mkdir(parents=True, exist_ok=True)
                intrinsics = [
                    [20.0 + camera_index, 0.0, 7.0],
                    [0.0, 21.0 + frame_index, 3.0],
                    [0.0, 0.0, 1.0],
                ]
                with parameter_path.open("w") as file:
                    json.dump({"camera_intrinsic": intrinsics}, file)

                disparity = np.linspace(1, 10, 8 * 16, dtype=np.float32).reshape(8, 16)
                depth_path = (
                    root / "samples_dpt_small" / camera / f"{frame_index}.npy"
                )
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(depth_path, disparity)

                if frame_index > 0:
                    mask = np.full((8, 16), 255, dtype=np.uint8)
                    mask[0, 0] = 0
                    mask_path = (
                        root
                        / "samples_mask_small"
                        / camera
                        / f"{frame_index}.png"
                    )
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(mask_path)
            sensor_info[camera] = frames

        with (data_dir / "bin_infos_3.2m" / "scene-test-bin000.pkl").open(
            "wb"
        ) as file:
            pickle.dump({"sensor_info": sensor_info}, file)

    @staticmethod
    def _dataset_cfg(root: Path, test_split: str = "total") -> SimpleNamespace:
        return SimpleNamespace(
            roots=[root],
            image_shape=[8, 16],
            near=0.5,
            far=100.0,
            test_split=test_split,
        )

    def test_dataset_builds_six_context_and_eighteen_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset_root(root)
            dataset = DatasetOmniScene(self._dataset_cfg(root), "test", None)
            example = dataset[0]

            self.assertEqual(len(dataset), 30)
            self.assertEqual(example["context"]["image"].shape, (6, 3, 8, 16))
            self.assertEqual(example["target"]["image"].shape, (18, 3, 8, 16))
            self.assertEqual(example["target"]["masks"].shape, (18, 8, 16))
            self.assertEqual(example["target"]["rel_depth"].shape, (18, 8, 16))
            self.assertFalse(example["target"]["masks"][:12, 0, 0].any())
            self.assertTrue(example["target"]["masks"][12:].all())
            self.assertFalse(
                torch.equal(
                    example["context"]["intrinsics"][0],
                    example["context"]["intrinsics"][1],
                )
            )
            self.assertTrue(torch.isfinite(example["target"]["rel_depth"]).all())

            mini_dataset = DatasetOmniScene(
                self._dataset_cfg(root, "mini"), "test", None
            )
            self.assertEqual(len(mini_dataset), 3)

            train_dataset = DatasetOmniScene(self._dataset_cfg(root), "train", None)
            self.assertNotIn("rel_depth", train_dataset[0]["target"])

    def test_patch_shim_crops_auxiliaries_and_adjusts_principal_point(self) -> None:
        image = torch.zeros(1, 2, 3, 112, 200)
        masks = torch.ones(1, 2, 112, 200, dtype=torch.bool)
        rel_depth = torch.arange(112 * 200).reshape(1, 1, 112, 200).repeat(1, 2, 1, 1)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
        intrinsics[:, :, 0, 0] = 0.3
        intrinsics[:, :, 1, 1] = 0.4
        intrinsics[:, :, 0, 2] = 0.4
        intrinsics[:, :, 1, 2] = 0.6

        updated = apply_patch_shim_to_views(
            {
                "image": image,
                "masks": masks,
                "rel_depth": rel_depth,
                "intrinsics": intrinsics,
            },
            16,
        )

        self.assertEqual(updated["image"].shape[-2:], (112, 192))
        self.assertEqual(updated["masks"].shape[-2:], (112, 192))
        self.assertEqual(updated["rel_depth"].shape[-2:], (112, 192))
        self.assertAlmostEqual(updated["intrinsics"][0, 0, 0, 0].item(), 0.3125)
        self.assertAlmostEqual(
            updated["intrinsics"][0, 0, 0, 2].item(),
            (0.4 * 200 - 4) / 192,
        )
        self.assertAlmostEqual(updated["intrinsics"][0, 0, 1, 2].item(), 0.6)

    def test_pcc_and_masked_losses(self) -> None:
        depth = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        self.assertAlmostEqual(compute_pcc(depth, depth).item(), 1.0, places=5)
        self.assertAlmostEqual(compute_pcc(depth, -depth).item(), -1.0, places=5)

        target = torch.zeros(1, 1, 3, 2, 2)
        predicted = torch.ones_like(target)
        predicted[:, :, :, 0, 0] = 100
        invalid_mask = torch.zeros_like(target, dtype=torch.bool)
        invalid_mask[:, :, :, 0, 0] = True
        static_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        static_mask[:, :, 0, 0] = False
        batch = {"target": {"image": target.clone(), "masks": static_mask}}
        generated_mask = _get_target_invalid_mask(batch, target, True)
        self.assertTrue(torch.equal(generated_mask, invalid_mask))
        output = DecoderOutput(color=predicted.clone(), depth=None)

        mse = LossMse(LossMseCfgWrapper(LossMseCfg(weight=1.0)))
        value = mse(
            output,
            batch,
            None,
            0,
            clamp_large_error=0,
            valid_depth_mask=invalid_mask,
        )
        self.assertAlmostEqual(value.item(), 1.0)

        lpips = LossLpips.__new__(LossLpips)
        nn.Module.__init__(lpips)
        lpips.cfg = LossLpipsCfg(
            weight=1.0, apply_after_step=0, perceptual_loss=True
        )
        lpips.name = "lpips"
        lpips.lpips = DummyPerceptualLoss()
        original_prediction = output.color.clone()
        original_target = batch["target"]["image"].clone()
        lpips(
            output,
            batch,
            None,
            0,
            valid_depth_mask=invalid_mask,
        )
        self.assertTrue(torch.equal(output.color, original_prediction))
        self.assertTrue(torch.equal(batch["target"]["image"], original_target))

    def test_checkpoint_shape_gate_rejects_partial_load(self) -> None:
        model = nn.Linear(3, 2)
        checkpoint = {
            "weight": torch.zeros(4, 3),
            "bias": torch.zeros(2),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "same-name parameter shape mismatch",
        ):
            load_state_dict_with_shape_check(
                model,
                checkpoint,
                strict=False,
                source="init.ckpt",
            )

        staged_model = DummyStagedModel()
        init_checkpoint = {
            key: value
            for key, value in staged_model.state_dict().items()
            if key.startswith("base.")
        }
        incompatible = load_state_dict_with_shape_check(
            staged_model,
            init_checkpoint,
            strict=False,
            source="init.ckpt",
            allowed_missing_prefixes=("update.",),
            reject_unexpected=True,
        )
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(key.startswith("update.") for key in incompatible.missing_keys)
        )

        with self.assertRaisesRegex(RuntimeError, "staged-load key audit"):
            load_state_dict_with_shape_check(
                staged_model,
                {"base.bias": staged_model.base.bias},
                strict=False,
                source="broken-init.ckpt",
                allowed_missing_prefixes=("update.",),
                reject_unexpected=True,
            )


if __name__ == "__main__":
    unittest.main()
