"""Prepare downscaled RGB-D inputs and matching camera intrinsics for smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def intrinsics_for_size(
    height: int,
    width: int,
    *,
    focal: float | None = None,
) -> tuple[float, float, float, float]:
    """Match recgen smoke tests: fx=fy=max(h,w), principal point at image center."""
    fx = focal if focal is not None else float(max(height, width))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def _center_crop_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side]


def _resize_array(arr: np.ndarray, size: int, *, is_mask: bool) -> np.ndarray:
    pil_mode = "F" if arr.dtype in (np.float32, np.float64) else None
    if arr.ndim == 2:
        if pil_mode:
            img = Image.fromarray(arr.astype(np.float32), mode="F")
        else:
            img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.fromarray(arr.astype(np.uint8))

    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    out = img.resize((size, size), resample)
    return np.array(out)


def prepare_smoke_tensors(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float, float]]:
    """Center-crop to square, resize to `size`, return ComfyUI tensors + intrinsics.

    Intrinsics follow the same rule as ``recgen/tests/test_smoke.py`` on the final
  resolution (not the original 1920x1080 calibration file).
    """
    rgb_sq = _center_crop_square(rgb)
    depth_sq = _center_crop_square(depth)
    mask_sq = _center_crop_square(mask)

    rgb_small = _resize_array(rgb_sq, size, is_mask=False)
    depth_small = _resize_array(depth_sq, size, is_mask=True).astype(np.float32)
    mask_small = _resize_array(mask_sq, size, is_mask=True).astype(np.float32)
    if mask_small.max() > 1:
        mask_small = mask_small / 255.0

    image_tensor = torch.from_numpy(rgb_small.astype(np.float32) / 255.0)[None]
    depth_tensor = torch.from_numpy(depth_small)[None, :, :, None]
    mask_tensor = torch.from_numpy(mask_small.astype(np.float32))[None]

    fx, fy, cx, cy = intrinsics_for_size(size, size)
    return image_tensor, depth_tensor, mask_tensor, (fx, fy, cx, cy)


def load_example_smoke_tensors(
    examples_dir: Path,
    example: str = "ex0",
    size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float, float]]:
    """Load ``ex0_rgb/depth/mask`` and prepare smoke tensors."""
    rgb = np.array(Image.open(examples_dir / f"{example}_rgb.png").convert("RGB"))
    depth = np.array(Image.open(examples_dir / f"{example}_depth.png")).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.max() > 100:
        depth = depth / 1000.0

    mask = np.array(Image.open(examples_dir / f"{example}_mask.png"))
    if mask.max() == 1:
        mask = mask * 255

    return prepare_smoke_tensors(rgb, depth, mask, size=size)
