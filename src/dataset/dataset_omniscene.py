import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage, UnbatchedExample
from .utils_omniscene import load_conditions, load_info
from .view_sampler import ViewSampler


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = 0.5
    far: float = 100.0
    baseline_scale_bounds: bool = False
    shuffle_val: bool = False
    train_times_per_scene: int = 1
    highres: bool = False
    test_split: Literal["total", "mini"] = "total"


class DatasetOmniScene(Dataset):
    data_version = "interp_12Hz_trainval"
    dataset_prefix = "/datasets/nuScenes"
    camera_types = (
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        load_rel_depth: bool | None = None,
    ) -> None:
        super().__init__()
        if len(cfg.roots) != 1:
            raise ValueError(f"OmniScene expects exactly one root, got {cfg.roots}")
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.data_root = cfg.roots[0]
        self.load_rel_depth = (
            stage == "test" if load_rel_depth is None else load_rel_depth
        )
        if stage != "test":
            self.load_rel_depth = False

        split_path = self.data_root / self.data_version / (
            "bins_train_3.2m.json" if stage == "train" else "bins_val_3.2m.json"
        )
        if not split_path.is_file():
            raise FileNotFoundError(f"OmniScene split file not found: {split_path}")
        with split_path.open() as file:
            self.bin_tokens = json.load(file)["bins"]

        if stage == "val":
            self.bin_tokens = self.bin_tokens[:30000:3000][:10]
        elif stage == "test":
            if cfg.test_split == "mini":
                self.bin_tokens = self.bin_tokens[0::14][:2048]
            elif cfg.test_split != "total":
                raise ValueError(f"Unsupported OmniScene test split: {cfg.test_split}")

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def _resolve_image_path(self, image_path: str) -> str:
        return image_path.replace(self.dataset_prefix, str(self.data_root), 1)

    def __getitem__(self, index: int) -> UnbatchedExample:
        bin_token = self.bin_tokens[index]
        bin_path = (
            self.data_root
            / self.data_version
            / "bin_infos_3.2m"
            / f"{bin_token}.pkl"
        )
        if not bin_path.is_file():
            raise FileNotFoundError(f"OmniScene bin info not found: {bin_path}")
        with bin_path.open("rb") as file:
            bin_info = pickle.load(file)

        sensor_info = bin_info["sensor_info"]
        input_paths: list[str] = []
        input_c2ws = []
        output_paths: list[str] = []
        output_c2ws = []

        for camera in self.camera_types:
            camera_frames = sensor_info[camera]
            if len(camera_frames) < 3:
                raise ValueError(
                    f"OmniScene bin {bin_token} camera {camera} has only "
                    f"{len(camera_frames)} frames; expected at least 3"
                )
            image_path, c2w = load_info(camera_frames[0])
            input_paths.append(self._resolve_image_path(image_path))
            input_c2ws.append(c2w)

            for frame_index in (1, 2):
                image_path, c2w = load_info(camera_frames[frame_index])
                output_paths.append(self._resolve_image_path(image_path))
                output_c2ws.append(c2w)

        input_images, input_masks, input_intrinsics, input_rel_depth = load_conditions(
            input_paths,
            self.cfg.image_shape,
            is_input=True,
            load_rel_depth=self.load_rel_depth,
        )
        output_images, output_masks, output_intrinsics, output_rel_depth = (
            load_conditions(
                output_paths,
                self.cfg.image_shape,
                is_input=False,
                load_rel_depth=self.load_rel_depth,
            )
        )

        input_c2ws_tensor = torch.stack(
            [torch.from_numpy(c2w) for c2w in input_c2ws]
        ).float()
        output_c2ws_tensor = torch.stack(
            [torch.from_numpy(c2w) for c2w in output_c2ws]
        ).float()

        target_images = torch.cat((output_images, input_images), dim=0)
        target_masks = torch.cat((output_masks, input_masks), dim=0)
        target_intrinsics = torch.cat(
            (output_intrinsics, input_intrinsics), dim=0
        )
        target_c2ws = torch.cat((output_c2ws_tensor, input_c2ws_tensor), dim=0)

        context = {
            "extrinsics": input_c2ws_tensor,
            "intrinsics": input_intrinsics,
            "image": input_images,
            "near": repeat(
                torch.tensor(self.cfg.near, dtype=torch.float32), "-> v", v=6
            ),
            "far": repeat(
                torch.tensor(self.cfg.far, dtype=torch.float32), "-> v", v=6
            ),
            "index": torch.arange(6, dtype=torch.int64),
        }
        target = {
            "extrinsics": target_c2ws,
            "intrinsics": target_intrinsics,
            "image": target_images,
            "near": repeat(
                torch.tensor(self.cfg.near, dtype=torch.float32), "-> v", v=18
            ),
            "far": repeat(
                torch.tensor(self.cfg.far, dtype=torch.float32), "-> v", v=18
            ),
            "index": torch.arange(18, dtype=torch.int64),
            "masks": target_masks,
        }
        if output_rel_depth is not None and input_rel_depth is not None:
            target["rel_depth"] = torch.cat(
                (output_rel_depth, input_rel_depth), dim=0
            )

        return {"context": context, "target": target, "scene": bin_token}
