# ComfyUI-Recgen — full context for AI assistants

Use this document when answering detailed questions about installation, dependencies, or troubleshooting. Canonical repo: https://github.com/kaili-yang/ComfyUI-Recgen

## One-sentence summary

**ComfyUI-Recgen** is a GPL-3.0 ComfyUI extension that connects [RecGen](https://github.com/TRI-ML/RecGen) to ComfyUI graphs so users can turn a single RGB-D view (color + depth + object mask) into a posed 3D mesh and optional 3D Gaussian splat.

## Problem it solves

RecGen ships as a Python inference library (`recgen_inference`), not as native ComfyUI nodes. This extension:

1. Registers ComfyUI node classes (`RecGen2DTo3D`, `RecGenLoadExampleInputs`)
2. Bridges ComfyUI tensor types (`IMAGE`, `MASK`) to RecGen numpy preprocessing
3. Adds macOS-friendly fallbacks (`spconv` PyTorch mock, SDPA sparse attention, HuggingFace DINOv2)
4. Saves artifacts where ComfyUI users expect them (`output/recgen_outputs/`)

## System requirements

| Component | Requirement |
|-----------|-------------|
| Python | ≥ 3.10 |
| PyTorch | 2.x with CUDA, MPS (Apple Silicon), or CPU |
| ComfyUI | Any recent build with custom node support |
| RecGen source | Sibling directory `recgen/` with `recgen_inference` installable (`pip install -e recgen`) |
| Disk | ~10 GB for first-time HuggingFace download (`TRI-ML/RecGen`) |
| GPU | CUDA recommended; MPS supported with deferred load + mesh-only decode for smoke |

## Installation steps (complete)

```bash
# 1. ComfyUI custom node
cd /path/to/ComfyUI/custom_nodes
git clone git@github.com:kaili-yang/ComfyUI-Recgen.git

# 2. RecGen inference (sibling of custom_nodes or under same parent as in dev layouts)
git clone https://github.com/TRI-ML/RecGen.git recgen   # adjust path to match your layout
pip install -e /path/to/recgen
pip install -e /path/to/ComfyUI/custom_nodes/ComfyUI-Recgen

# 3. Restart ComfyUI
```

Development layout example:

```
workspace/
├── recgen/                 # TRI-ML RecGen inference
└── ComfyUI/
    └── custom_nodes/
        └── ComfyUI-Recgen/
```

The extension adds `workspace/recgen` to `sys.path` when that folder exists next to the extension root.

## ComfyUI workflow (minimal)

1. Add **RecGen → RecGen Load Example Inputs** (optional demo)
2. Add **RecGen → RecGen 2D to 3D**
3. Connect `image`, `depth`, `mask`, and intrinsics (`fx`, `fy`, `cx`, `cy`)
4. Queue prompt; read `output_dir` string and files on disk

## Node: RecGen 2D to 3D

**Inputs**

| Name | Type | Notes |
|------|------|-------|
| image | IMAGE | RGB, float 0–1, shape `[B,H,W,3]` |
| depth | IMAGE | Metric depth preferred; uint16 mm via Load Image is auto-rescaled |
| mask | MASK | Object region, float 0–1 |
| checkpoint | COMBO | Default `recgen_base.multiview_stereo` |
| device | COMBO | `cuda`, `mps`, or `cpu` |
| fx, fy, cx, cy | FLOAT | Pinhole intrinsics; `cx`/`cy` = -1 → image center |
| seed | INT | Sampling seed |
| save_splat | BOOL | Write Gaussian `.ply` (heavier) |
| save_glb | BOOL | Textured GLB (needs extra deps, usually off) |

**Outputs**

| Name | Type | Notes |
|------|------|-------|
| output_dir | STRING | Folder containing `posed_mesh.obj`, `overlay.png`, etc. |
| overlay_image | IMAGE | Preview composite |

## Depth and mask formats

- **Depth**: float32 metres, or uint16 millimetres in PNG (RecGen `normalize_depth` converts). ComfyUI **Load Image** on depth PNGs may normalize to 0–1; the extension detects and rescales.
- **Mask**: non-zero = foreground object; erosion applied inside RecGen preprocessing.

## Smoke test (no ComfyUI UI)

```bash
python3 ComfyUI-Recgen/tests/test_run_node.py
```

Uses 256×256 centre-cropped example data, MPS on Apple Silicon when available, mesh-only decode to save memory.

## Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | HuggingFace cache (extension may redirect to external drive if `/Volumes/HD-SL2` exists) |
| `TORCH_HOME` | Torch hub cache |
| `SPARSE_ATTN_BACKEND=sdpa` | PyTorch attention when flash-attn/xformers absent |
| `SPCONV_ALGO=native` | Sparse conv algorithm hint |

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `No module named recgen_inference` | RecGen not installed / wrong path | `pip install -e ../recgen` and ensure `recgen/` sibling exists |
| OOM on MPS during load | Large models + unified memory | Use smoke 256²; mesh-only decode; free disk on system volume |
| Slow on CPU at 1080p | Expected | Downscale inputs; use CUDA/MPS |
| SSL errors downloading DINOv2 | macOS Python certs | `pip install certifi` (dependency); extension sets `SSL_CERT_FILE` |
| Missing example files | No `recgen/examples/` | Clone full RecGen repo with example PNGs |

## License and attribution

- Extension: GPL-3.0 — Kaili Yang
- RecGen model/weights: see [TRI-ML/RecGen](https://github.com/TRI-ML/RecGen) license (TRI-NC for inference package)

## Keywords for search

ComfyUI, RecGen, image to 3D, RGB-D reconstruction, single view 3D, depth map, object mask, Gaussian splat, mesh generation, TRI-ML, custom nodes, Apple Silicon MPS, spconv alternative
