#!/usr/bin/env python3
"""Local end-to-end smoke demo for ComfyUI-Recgen (no ComfyUI server required)."""

import os
import sys
from pathlib import Path

# Redirect HuggingFace / torch caches to external drive if available
EXTERNAL_DRIVE = Path("/Volumes/HD-SL2")
if EXTERNAL_DRIVE.exists():
    external_cache = EXTERNAL_DRIVE / ".cache"
    (external_cache / "huggingface").mkdir(parents=True, exist_ok=True)
    (external_cache / "torch").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(external_cache / "huggingface")
    os.environ["TORCH_HOME"] = str(external_cache / "torch")
    print(f"[ComfyUI-Recgen Test] Redirected caches to: {external_cache}")

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "ComfyUI-Recgen" / "src"))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

from comfyui_recgen.nodes import RecGen2DTo3DNode
from comfyui_recgen.smoke_inputs import load_example_smoke_tensors

SMOKE_SIZE = 256


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _print_mps_status() -> None:
    mps_mod = getattr(torch.backends, "mps", None)
    if mps_mod is None:
        print("PyTorch MPS backend: not available in this build")
        return
    print(f"PyTorch MPS built: {mps_mod.is_built()}")
    print(f"PyTorch MPS available: {mps_mod.is_available()}")


def main():
    print("=== ComfyUI-Recgen Local Smoke Demo ===")
    _print_mps_status()

    recgen_dir = WORKSPACE_DIR / "recgen" / "examples"
    if not recgen_dir.is_dir():
        raise SystemExit(f"Missing example data: {recgen_dir}")

    image_tensor, depth_tensor, mask_tensor, (fx, fy, cx, cy) = load_example_smoke_tensors(
        recgen_dir, size=SMOKE_SIZE
    )

    print(f"Smoke RGB tensor:   {tuple(image_tensor.shape)}")
    print(f"Smoke depth tensor: {tuple(depth_tensor.shape)}")
    print(f"Smoke mask tensor:  {tuple(mask_tensor.shape)}")
    print(f"Intrinsics (256²):  fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    device = _pick_device()
    print(f"Device: {device}")

    node = RecGen2DTo3DNode()
    print("Running 3D generation...")
    out_dir, overlay_tensor = node.generate_3d(
        image=image_tensor,
        depth=depth_tensor,
        mask=mask_tensor,
        checkpoint="recgen_base.multiview_stereo",
        device=device,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        seed=42,
        save_splat=False,
        save_glb=False,
    )

    out_path = Path(out_dir)
    mesh_path = out_path / "posed_mesh.obj"
    overlay_path = out_path / "overlay.png"

    print("\n=== Success ===")
    print(f"Output directory: {out_path}")
    print(f"  posed_mesh.obj:       {mesh_path.exists()}")
    print(f"  overlay.png:          {overlay_path.exists()}")
    print(f"  overlay tensor shape: {tuple(overlay_tensor.shape)}")

    if not mesh_path.exists():
        raise SystemExit("Demo failed: posed_mesh.obj was not created")
    if not overlay_path.exists():
        raise SystemExit("Demo failed: overlay.png was not created")

    print("\nLocal smoke demo completed successfully.")


if __name__ == "__main__":
    main()
