"""Convert ComfyUI depth tensors back to metric depth for RecGen."""

from __future__ import annotations

import numpy as np


def depth_tensor_to_meters(depth_tensor) -> np.ndarray:
    """Convert a ComfyUI depth tensor to float32 depth in metres.

    Accepts:
      - Raw metric depth (values typically 0.1–10).
      - uint16 millimetre depth wrongly passed through an IMAGE node (0–1 float).
    """
    depth_np = depth_tensor.detach().cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[:, :, 0]
    depth_np = depth_np.astype(np.float32)

    finite = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
    if finite.size == 0:
        return depth_np

    # IMAGE widgets normalize uint16 PNG depth to [0, 1].
    if depth_np.max() <= 1.0 + 1e-6 and finite.max() <= 1.0 + 1e-6:
        depth_np = depth_np * 65535.0

    return depth_np
