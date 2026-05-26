"""Top-level package for ComfyUI-Recgen."""
import sys

# Inject spconv mock before recgen imports sparse conv layers
try:
    import spconv.pytorch  # noqa: F401
except ImportError:
    try:
        from .src.comfyui_recgen.spconv_mock import inject_spconv_mock

        inject_spconv_mock()
        print("[ComfyUI-Recgen] Injected spconv.pytorch mock from root __init__.py")
    except Exception as e:
        print(f"[ComfyUI-Recgen] Failed to inject spconv mock: {e}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = "Kaili Yang"
__email__ = "124ykl@gmail.com"
__version__ = "0.0.1"
__url__ = "https://github.com/kaili-yang/ComfyUI-Recgen"

from .src.comfyui_recgen.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_recgen.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
