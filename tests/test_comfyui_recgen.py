"""Lightweight unit tests for ComfyUI-Recgen node metadata."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from comfyui_recgen.depth_utils import depth_tensor_to_meters
from comfyui_recgen.nodes import RecGen2DTo3DNode
from comfyui_recgen.smoke_inputs import intrinsics_for_size, prepare_smoke_tensors


def test_recgen_node_metadata():
    assert RecGen2DTo3DNode.FUNCTION == "generate_3d"
    assert RecGen2DTo3DNode.CATEGORY == "RecGen"
    assert "IMAGE" in RecGen2DTo3DNode.RETURN_TYPES


def test_recgen2d_input_types_without_recgen_runtime():
    """INPUT_TYPES must not import recgen_inference (object_info queries all nodes)."""
    types = RecGen2DTo3DNode.INPUT_TYPES()
    checkpoints = types["required"]["checkpoint"][0]
    assert "recgen_base.multiview_stereo" in checkpoints


def test_intrinsics_for_smoke_resolution():
    fx, fy, cx, cy = intrinsics_for_size(256, 256)
    assert fx == fy == 256.0
    assert cx == cy == 128.0


def test_depth_tensor_to_meters_restores_normalized_uint16():
    import torch

    depth = torch.ones(4, 4) * (1000.0 / 65535.0)
    meters = depth_tensor_to_meters(depth)
    assert meters.max() > 30.0
