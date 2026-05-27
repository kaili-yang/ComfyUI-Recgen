import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .depth_utils import depth_tensor_to_meters

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# Redirect HuggingFace cache to external drive if available to prevent running out of main disk space
EXTERNAL_DRIVE = Path("/Volumes/HD-SL2")
if EXTERNAL_DRIVE.exists():
    external_cache = EXTERNAL_DRIVE / ".cache"
    external_hf_cache = external_cache / "huggingface"
    external_torch_cache = external_cache / "torch"
    external_hf_cache.mkdir(parents=True, exist_ok=True)
    external_torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(external_hf_cache)
    os.environ["TORCH_HOME"] = str(external_torch_cache)
    print(
        "[ComfyUI-Recgen] Redirected HF_HOME and TORCH_HOME to external drive: "
        f"{external_cache}"
    )

# ComfyUI-Recgen package root (…/ComfyUI-Recgen)
NODE_ROOT = Path(__file__).resolve().parents[2]
# Legacy: parent of custom_nodes (used for outputs when folder_paths missing)
PARENT_DIR = Path(__file__).resolve().parents[3]


def _resolve_recgen_root() -> Path | None:
    """Find TRI-ML RecGen checkout (vendor install or workspace sibling)."""
    candidates = [
        NODE_ROOT / "vendor" / "recgen",
        NODE_ROOT.parent.parent.parent / "recgen",
        NODE_ROOT.parent / "recgen",
    ]
    for path in candidates:
        if (path / "recgen_inference").is_dir():
            return path
    return None


recgen_path = _resolve_recgen_root()
if recgen_path is not None and str(recgen_path) not in sys.path:
    sys.path.insert(0, str(recgen_path))

# Set environment variables for spconv, attention, and opengl
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Inject spconv mock so recgen can import spconv.pytorch on macOS / CPU-only hosts
try:
    import spconv.pytorch  # noqa: F401
except ImportError:
    from .spconv_mock import inject_spconv_mock

    inject_spconv_mock()
    print("[ComfyUI-Recgen] Injected pure PyTorch spconv.pytorch mock for macOS compatibility.")

from recgen_inference import build_recgen, generate

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Global cache for loaded pipeline to avoid reloading it on every run
_pipeline_cache = {}

def _activate_pipeline_device(pipeline, device: str) -> str:
    """Move pipeline to the requested runtime device; return the device actually used."""
    runtime = getattr(pipeline, "_recgen_runtime_device", device)
    if runtime == "mps":
        try:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            pipeline.to(runtime)
            print("[ComfyUI-Recgen] Pipeline active on MPS.")
            return runtime
        except Exception as exc:
            print(f"[ComfyUI-Recgen] MPS activation failed ({exc}); using CPU.")
            pipeline.to("cpu")
            return "cpu"
    if runtime == "cuda":
        pipeline.to(runtime)
        return runtime
    pipeline.to("cpu")
    return "cpu"


def get_pipeline(checkpoint, device):
    cache_key = (checkpoint, device)
    if cache_key not in _pipeline_cache:
        load_device = "cpu" if device == "mps" else device
        print(
            f"[ComfyUI-Recgen] Loading pipeline checkpoint: {checkpoint} "
            f"(load on {load_device}, run on {device})..."
        )
        _pipeline_cache[cache_key] = build_recgen.build(checkpoint, device=load_device)
        if device == "mps":
            _pipeline_cache[cache_key]._recgen_runtime_device = "mps"
    return _pipeline_cache[cache_key]

class RecGenLoadExampleInputs:
    """Load the bundled RecGen example RGB, depth, and mask from the sibling recgen repo."""

    @classmethod
    def INPUT_TYPES(cls):
        examples = []
        recgen_examples = (recgen_path / "examples") if recgen_path else None
        if recgen_examples and recgen_examples.is_dir():
            for idx in range(16):
                if (recgen_examples / f"ex{idx}_rgb.png").exists():
                    examples.append(f"ex{idx}")
        if not examples:
            examples = ["ex0"]
        return {
            "required": {
                "example": (examples, {"default": examples[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "image",
        "depth",
        "mask",
        "fx",
        "fy",
        "cx",
        "cy",
    )
    FUNCTION = "load_example"
    CATEGORY = "RecGen"

    def load_example(self, example):
        if recgen_path is None:
            raise FileNotFoundError(
                "RecGen not found. Run install.py or place recgen next to ComfyUI-Recgen."
            )
        recgen_examples = recgen_path / "examples"
        intrinsics_path = recgen_examples / "intrinsics.yaml"
        rgb_path = recgen_examples / f"{example}_rgb.png"
        depth_path = recgen_examples / f"{example}_depth.png"
        mask_path = recgen_examples / f"{example}_mask.png"

        if not rgb_path.exists():
            raise FileNotFoundError(f"Example RGB not found: {rgb_path}")

        rgb_pil = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb_pil).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(rgb_np)[None]

        depth_img = Image.open(depth_path)
        depth_np = np.array(depth_img).astype(np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[:, :, 0]
        if depth_np.max() > 100:
            depth_np = depth_np / 1000.0
        depth_tensor = torch.from_numpy(depth_np)[None, :, :, None]

        mask_pil = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_pil).astype(np.float32)
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        mask_tensor = torch.from_numpy(mask_np)[None]

        fx, fy, cx, cy = 1062.2, 1060.9, -1.0, -1.0
        if intrinsics_path.exists():
            import yaml

            with open(intrinsics_path, "r") as f:
                data = yaml.safe_load(f)
            fx = float(data.get("fu", fx))
            fy = float(data.get("fv", fy))
            cx = float(data.get("pu", cx))
            cy = float(data.get("pv", cy))

        return (image_tensor, depth_tensor, mask_tensor, fx, fy, cx, cy)


class RecGen2DTo3DNode:
    """
    RecGen 2D Image to 3D Object Node
    
    Generates a 3D mesh (.obj) and Gaussian splat (.ply) from a single view RGB, depth, and object mask.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # We try to load checkpoints dynamically, default to "recgen_base.multiview_stereo" if it fails
        try:
            checkpoints = build_recgen.list_checkpoints()
        except Exception:
            checkpoints = ["recgen_base.multiview_stereo"]
            
        # Dynamically determine the best default hardware accelerator for the system
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        else:
            default_device = "cpu"
            
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "mask": ("MASK",),
                "checkpoint": (checkpoints,),
                "device": (["cuda", "cpu", "mps"], {"default": default_device}),
                "fx": ("FLOAT", {"default": 1062.2, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "fy": ("FLOAT", {"default": 1060.9, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "cx": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.1, "tooltip": "Set to -1 for auto (center of image)"}),
                "cy": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.1, "tooltip": "Set to -1 for auto (center of image)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "save_splat": ("BOOLEAN", {"default": True}),
                "save_glb": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_dir", "overlay_image")
    FUNCTION = "generate_3d"
    CATEGORY = "RecGen"

    def generate_3d(self, image, depth, mask, checkpoint, device, fx, fy, cx, cy, seed, save_splat, save_glb):
        # 1. Convert Image tensor to numpy RGB [H, W, 3] in uint8
        img_tensor = image[0]
        rgb_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        H, W, _ = rgb_np.shape

        # 2. Convert Depth tensor to numpy float32 [H, W] (RecGen normalizes units)
        depth_np = depth_tensor_to_meters(depth[0])

        # 3. Convert Mask tensor to numpy uint8 [H, W] (non-zero = object)
        mask_np = mask[0].cpu().numpy() if mask.ndim == 3 else mask.cpu().numpy()
        mask_np = (mask_np * 255.0).clip(0, 255).astype(np.uint8)

        # Ensure shapes match
        if depth_np.shape != (H, W):
            import cv2
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)
        if mask_np.shape != (H, W):
            import cv2
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)

        # 4. Resolve camera intrinsics
        fx_val = fx
        fy_val = fy
        cx_val = W / 2.0 if cx < 0 else cx
        cy_val = H / 2.0 if cy < 0 else cy
        
        K = np.array([
            [fx_val, 0.0, cx_val],
            [0.0, fy_val, cy_val],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 5. Build/fetch the RecGen pipeline and move to MPS/CUDA if requested
        pipeline = get_pipeline(checkpoint, device)
        active_device = _activate_pipeline_device(pipeline, device)

        # 6. Run generation
        print(f"[ComfyUI-Recgen] Running 3D generation on {active_device}...")
        decode_formats = ["mesh", "gaussian", "radiance_field"] if save_splat else ["mesh"]
        result = generate(
            pipeline,
            image=rgb_np,
            depth=depth_np,
            mask=mask_np,
            intrinsics=K,
            seed=seed,
            decode_formats=decode_formats,
        )

        # 7. Save outputs
        if folder_paths is not None:
            try:
                output_base_dir = Path(folder_paths.get_output_directory()) / "recgen_outputs"
            except Exception:
                output_base_dir = PARENT_DIR / "outputs" / "recgen_outputs"
        else:
            output_base_dir = PARENT_DIR / "outputs" / "recgen_outputs"
            
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Unique directory for this run
        import time
        run_name = f"recgen_{int(time.time())}_{seed}"
        out_dir = output_base_dir / run_name
        
        print(f"[ComfyUI-Recgen] Saving outputs to: {out_dir}")
        result.save(out_dir, save_splat=save_splat, save_glb=save_glb)

        # 8. Load overlay image back as ComfyUI image format [1, H, W, 3] float32
        overlay_path = out_dir / "overlay.png"
        if overlay_path.exists():
            overlay_pil = Image.open(overlay_path).convert("RGB")
            overlay_np = np.array(overlay_pil).astype(np.float32) / 255.0
            overlay_tensor = torch.from_numpy(overlay_np)[None] # add batch dim [1, H, W, 3]
        else:
            overlay_tensor = image # fallback to input

        return (str(out_dir), overlay_tensor)

NODE_CLASS_MAPPINGS = {
    "RecGenLoadExampleInputs": RecGenLoadExampleInputs,
    "RecGen2DTo3D": RecGen2DTo3DNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RecGenLoadExampleInputs": "RecGen Load Example Inputs",
    "RecGen2DTo3D": "RecGen 2D to 3D",
}
