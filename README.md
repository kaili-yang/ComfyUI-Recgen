# ComfyUI-Recgen

**ComfyUI custom nodes for [RecGen](https://github.com/TRI-ML/RecGen): turn a single RGB image + depth map + object mask into a 3D mesh and optional Gaussian splat.**

Use this extension when you want **image-to-3D** or **RGB-D to 3D** inside **ComfyUI** without writing Python glue code. It targets creators who already have (or can generate) aligned depth and segmentation, and who want `.obj` / `.ply` outputs in the ComfyUI output folder.

| | |
|---|---|
| **License** | GPL-3.0 |
| **AI / LLM summary** | [llms.txt](llms.txt) · [full context](docs/llms-full.md) |

## Features

- **RecGen 2D to 3D** — one ComfyUI node for full RecGen inference  
- **RecGen Load Example Inputs** — demo RGB, depth, mask, and intrinsics from bundled examples  
- **macOS Apple Silicon (MPS)** — PyTorch `spconv` fallback and SDPA attention (no `flash-attn` required)  
- **ComfyUI-native I/O** — `IMAGE` / `MASK` tensors, string output path, overlay preview image  

## Requirements

- Python ≥ 3.10, PyTorch 2.x  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)  
- Sibling [RecGen](https://github.com/TRI-ML/RecGen) inference repo (`pip install -e recgen`)  
- ~10 GB disk for first-time HuggingFace weights (`TRI-ML/RecGen`)  
- **CUDA** recommended; **MPS** on Apple Silicon supported; CPU works for small/smoke resolutions  

## Install

### ComfyUI (recommended)

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:kaili-yang/ComfyUI-Recgen.git
```

Install RecGen inference and this extension (adjust paths):

```bash
pip install -e /path/to/recgen
pip install -e /path/to/ComfyUI/custom_nodes/ComfyUI-Recgen
```

Restart ComfyUI. Nodes appear under category **RecGen**.

### Development layout

```
your-workspace/
├── recgen/              # git clone TRI-ML/RecGen
└── ComfyUI/
    └── custom_nodes/
        └── ComfyUI-Recgen/
```

### Smoke test (no ComfyUI UI)

```bash
chmod +x ComfyUI-Recgen/scripts/run_demo.sh
./ComfyUI-Recgen/scripts/run_demo.sh
# or: python3 ComfyUI-Recgen/tests/test_run_node.py
```

Uses 256×256 example data and matching intrinsics. Outputs: `outputs/recgen_outputs/`.

## Quick workflow

1. **RecGen Load Example Inputs** → **RecGen 2D to 3D** (demo), or  
2. Wire your own **RGB**, **depth**, **mask**, and camera **fx / fy / cx / cy** into **RecGen 2D to 3D**

| Node | Description |
|------|-------------|
| [RecGen Load Example Inputs](docs/NODES.md#recgen-load-example-inputs) | Demo assets from `recgen/examples/` |
| [RecGen 2D to 3D](docs/NODES.md#recgen-2d-to-3d) | Full reconstruction |

**Outputs:** `posed_mesh.obj`, `overlay.png`, optional `posed_gaussian.ply` under `output/recgen_outputs/`.

**Depth:** metric (metres) or uint16 mm PNG. ComfyUI Load Image depth normalized to 0–1 is auto-rescaled when needed.

## Documentation

| Doc | Audience |
|-----|----------|
| [FAQ](docs/FAQ.md) | Common questions |
| [Node reference](docs/NODES.md) | Inputs / outputs |
| [llms.txt](llms.txt) | Short summary for AI crawlers |
| [llms-full.md](docs/llms-full.md) | Detailed context for AI assistants |
| [AGENTS.md](AGENTS.md) | Coding agents |

## FAQ (short)

**Do I need RecGen installed separately?**  
Yes — `pip install -e` the [TRI-ML/RecGen](https://github.com/TRI-ML/RecGen) repo; this pack is only the ComfyUI adapter.

**Does it work on Mac?**  
Yes (MPS + CPU fallbacks). First run downloads large weights; use the smoke script at 256² for a quick check.

**What inputs are required?**  
RGB + depth + object mask + pinhole intrinsics (or use the example loader).

More: [docs/FAQ.md](docs/FAQ.md)

## Develop

```bash
cd ComfyUI-Recgen
pip install -e ".[dev]"
pre-commit install
pytest tests/test_comfyui_recgen.py
```

## Related projects

- [TRI-ML/RecGen](https://github.com/TRI-ML/RecGen) — upstream model and `recgen_inference`  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — node-based UI  
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) — optional installer  

## Citation / attribution

If you use RecGen weights or method, cite and follow the license of [TRI-ML/RecGen](https://github.com/TRI-ML/RecGen). This ComfyUI extension is maintained by Kaili Yang under GPL-3.0.
