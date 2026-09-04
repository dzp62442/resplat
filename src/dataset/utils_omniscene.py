import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_info(info: dict) -> tuple[str, np.ndarray]:
    """Return the image path and OpenCV camera-to-world matrix."""
    image_path = info["data_path"]
    c2w = np.asarray(info["sensor2lidar_transform"], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(
            f"Invalid sensor2lidar_transform shape for {image_path}: {c2w.shape}"
        )
    if not np.isfinite(c2w).all():
        raise ValueError(f"Non-finite sensor2lidar_transform for {image_path}")
    return image_path, c2w


def _replace_image_tree(image_path: str, samples_tree: str, sweeps_tree: str) -> Path:
    for source_tree, target_tree in (
        ("samples_small", samples_tree),
        ("sweeps_small", sweeps_tree),
        ("samples", samples_tree),
        ("sweeps", sweeps_tree),
    ):
        marker = f"/{source_tree}/"
        if marker in image_path:
            return Path(image_path.replace(marker, f"/{target_tree}/", 1))
    raise ValueError(f"Image path is not under a samples/sweeps tree: {image_path}")


def _resize_image_and_intrinsics(
    image: Image.Image,
    image_shape: tuple[int, int],
    intrinsics: np.ndarray,
) -> tuple[Image.Image, np.ndarray, bool]:
    target_height, target_width = image_shape
    if image.height == target_height and image.width == target_width:
        return image, intrinsics, False

    scale_x = target_width / image.width
    scale_y = target_height / image.height
    intrinsics = intrinsics.copy()
    intrinsics[0] *= scale_x
    intrinsics[1] *= scale_y
    image = image.resize((target_width, target_height), Image.BILINEAR)
    return image, intrinsics, True


def _load_relative_depth(
    resized_image_path: Path,
    image_shape: tuple[int, int],
    was_resized: bool,
) -> np.ndarray:
    depth_path = _replace_image_tree(
        str(resized_image_path), "samples_dpt_small", "sweeps_dpt_small"
    ).with_suffix(".npy")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Relative depth file not found: {depth_path}")

    disparity = np.load(depth_path).astype(np.float32)
    if was_resized:
        disparity = np.asarray(
            Image.fromarray(disparity).resize(
                (image_shape[1], image_shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        )
    if disparity.shape != image_shape:
        raise ValueError(
            f"Relative depth shape mismatch for {depth_path}: "
            f"expected {image_shape}, got {disparity.shape}"
        )
    if not np.isfinite(disparity).all():
        raise ValueError(f"Non-finite disparity values in {depth_path}")

    ratio = min(float(disparity.max() / (disparity.min() + 0.001)), 50.0)
    if ratio <= 0:
        raise ValueError(f"Invalid disparity range in {depth_path}")
    min_disparity = float(disparity.max()) / ratio
    relative_depth = 1.0 / np.maximum(disparity, min_disparity)
    depth_range = float(relative_depth.max() - relative_depth.min())
    if depth_range <= 0:
        raise ValueError(f"Constant relative depth in {depth_path}")
    relative_depth = (relative_depth - relative_depth.min()) / depth_range
    return relative_depth.astype(np.float32)


def load_conditions(
    image_paths: list[str],
    image_shape: list[int] | tuple[int, int],
    *,
    is_input: bool,
    load_rel_depth: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Load RGB, static masks, normalized intrinsics, and optional relative depth."""
    target_shape = (int(image_shape[0]), int(image_shape[1]))
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    intrinsics_all: list[np.ndarray] = []
    relative_depths: list[np.ndarray] | None = [] if load_rel_depth else None

    for source_image_path in image_paths:
        parameter_path = _replace_image_tree(
            source_image_path, "samples_param_small", "sweeps_param_small"
        ).with_suffix(".json")
        if not parameter_path.is_file():
            raise FileNotFoundError(
                f"Camera parameter file not found: {parameter_path}"
            )
        with parameter_path.open() as file:
            parameters = json.load(file)
        intrinsics = np.asarray(parameters["camera_intrinsic"], dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                "Invalid camera_intrinsic shape for "
                f"{parameter_path}: {intrinsics.shape}"
            )

        resized_image_path = _replace_image_tree(
            source_image_path, "samples_small", "sweeps_small"
        )
        if not resized_image_path.is_file():
            raise FileNotFoundError(f"RGB image not found: {resized_image_path}")
        with Image.open(resized_image_path) as image_file:
            image = image_file.convert("RGB")
            image, intrinsics, was_resized = _resize_image_and_intrinsics(
                image, target_shape, intrinsics
            )
            image_array = np.asarray(image, dtype=np.uint8)

        intrinsics[0] /= target_shape[1]
        intrinsics[1] /= target_shape[0]
        images.append(image_array)
        intrinsics_all.append(intrinsics)

        if relative_depths is not None:
            relative_depths.append(
                _load_relative_depth(resized_image_path, target_shape, was_resized)
            )

        if is_input:
            mask = np.ones(target_shape, dtype=np.bool_)
        else:
            mask_path = _replace_image_tree(
                str(resized_image_path), "samples_mask_small", "sweeps_mask_small"
            ).with_suffix(".png")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Dynamic-object mask not found: {mask_path}")
            with Image.open(mask_path) as mask_file:
                mask_image = mask_file.convert("L")
                if mask_image.size != (target_shape[1], target_shape[0]):
                    mask_image = mask_image.resize(
                        (target_shape[1], target_shape[0]), Image.BILINEAR
                    )
                mask = np.asarray(mask_image, dtype=np.uint8).astype(np.bool_)
        masks.append(mask)

    image_tensor = (
        torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    )
    mask_tensor = torch.from_numpy(np.stack(masks)).bool()
    intrinsics_tensor = torch.from_numpy(np.stack(intrinsics_all)).float()
    relative_depth_tensor = (
        None
        if relative_depths is None
        else torch.from_numpy(np.stack(relative_depths)).float()
    )
    return image_tensor, mask_tensor, intrinsics_tensor, relative_depth_tensor
