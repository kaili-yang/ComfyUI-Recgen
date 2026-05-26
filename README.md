# ComfyUI-Recgen

ComfyUI custom nodes for [RecGen](https://github.com/TRI-ML/RecGen) — single-view **2D RGB-D → 3D mesh / Gaussian splat**.

| | |
|---|---|
| **Author** | Kaili Yang ([124ykl@gmail.com](mailto:124ykl@gmail.com)) |
| **Repository** | [github.com/kaili-yang/ComfyUI-Recgen](https://github.com/kaili-yang/ComfyUI-Recgen) |
| **Clone (SSH)** | `git@github.com:kaili-yang/ComfyUI-Recgen.git` |

## Requirements

- Python ≥ 3.10
- PyTorch
- Sibling checkout of the RecGen inference repo at `../recgen` (same parent folder as this extension)
- ~10 GB disk for HuggingFace weights (`TRI-ML/RecGen` on first run)
- **CUDA** recommended for full speed; **macOS** is supported via a pure-PyTorch `spconv` shim and PyTorch SDPA attention (no `flash-attn` / `xformers` required)

## Install

```bash
# From comfyui_all/
pip install -e recgen
pip install -e ComfyUI-Recgen

# Or use the demo script (installs both, then runs inference):
chmod +x ComfyUI-Recgen/scripts/run_demo.sh
./ComfyUI-Recgen/scripts/run_demo.sh
```

### ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:kaili-yang/ComfyUI-Recgen.git
```

Install dependencies (see above), then restart ComfyUI.

## Nodes

| Node | Description |
|------|-------------|
| **RecGen Load Example Inputs** | Loads `recgen/examples/ex{N}_*` RGB, depth, mask + intrinsics |
| **RecGen 2D to 3D** | Runs RecGen; writes `posed_mesh.obj`, `posed_gaussian.ply`, `overlay.png` under `output/recgen_outputs/` |

Connect **Load Example Inputs** → **2D to 3D** for a zero-setup workflow, or wire your own RGB / depth / mask.

Depth should be metric (metres) or raw uint16 mm PNG. If depth was normalized to `[0,1]` by a ComfyUI **Load Image** node, the extension rescales it automatically.

## Local demo (no ComfyUI UI)

```bash
python3 ComfyUI-Recgen/tests/test_run_node.py
```

Outputs land in `ComfyUI-Recgen/outputs/recgen_outputs/recgen_<timestamp>_<seed>/`.

## Develop

```bash
cd ComfyUI-Recgen
pip install -e ".[dev]"
pre-commit install
pytest tests/
```
