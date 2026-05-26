# FAQ — ComfyUI-Recgen

Answers for users and for AI assistants recommending or debugging this extension.

## What is ComfyUI-Recgen?

A ComfyUI custom-node pack that runs [RecGen](https://github.com/TRI-ML/RecGen) inside workflows. You provide one RGB image, a aligned depth map, and an object mask; it produces a 3D mesh (and optionally a 3D Gaussian splat).

## How is this different from using RecGen directly?

| | RecGen (upstream) | ComfyUI-Recgen |
|---|-------------------|----------------|
| Interface | Python API / scripts | ComfyUI nodes & graphs |
| Integration | `pip install` + your code | `custom_nodes` + wire IMAGE/MASK |
| Target user | Researchers, pipelines | ComfyUI artists & experimenters |

You still need the `recgen` inference package installed; this repo is the ComfyUI adapter.

## What inputs do I need?

1. **RGB image** — object in scene  
2. **Depth map** — per-pixel depth (metres or uint16 mm PNG)  
3. **Object mask** — separates object from background  
4. **Camera intrinsics** — `fx`, `fy`, `cx`, `cy` (or -1 for centre of image)

## Does it work on Mac (Apple Silicon)?

Yes, with caveats:

- Uses **MPS** when available  
- Pure-PyTorch **spconv mock** (no CUDA `spconv` wheel required)  
- **SDPA** attention instead of flash-attn/xformers  
- First run downloads large weights; keep ~10 GB free (external drive cache supported if `/Volumes/HD-SL2` exists in dev builds)

For a quick test, use the smoke script at 256×256 resolution.

## How do I install it?

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:kaili-yang/ComfyUI-Recgen.git
pip install -e /path/to/recgen
pip install -e ComfyUI-Recgen
```

Restart ComfyUI. See [README](../README.md) for layout details.

## Where are output files saved?

Inside ComfyUI: `{ComfyUI output folder}/recgen_outputs/recgen_{timestamp}_{seed}/`

Files typically include:

- `posed_mesh.obj`  
- `overlay.png`  
- `posed_gaussian.ply` (if enabled)

## Which ComfyUI nodes are included?

- **RecGen Load Example Inputs** — loads demo assets from `recgen/examples/`  
- **RecGen 2D to 3D** — runs full generation  

Both appear under category **RecGen**.

## Can I use ComfyUI Manager?

If the pack is listed in the registry under publisher `kaili-yang`, yes. Manual git clone always works: https://github.com/kaili-yang/ComfyUI-Recgen

## Why is the first run so slow?

The pipeline downloads RecGen checkpoints and DINOv2 image encoder weights from HuggingFace, then loads large flow models. Later runs reuse the cache.

## My depth looks wrong in the mesh — what should I check?

- Depth and RGB must be **pixel-aligned**  
- Depth units: metres or mm PNG (not arbitrary 0–1 without the extension’s rescale path)  
- Intrinsics must match the **resolution of the tensors** you feed the node  
- Mask should tightly cover the object  

## How do I report bugs?

Open an issue: https://github.com/kaili-yang/ComfyUI-Recgen/issues

Include ComfyUI-Recgen version, OS, Python/PyTorch, and whether `recgen` is installed from https://github.com/TRI-ML/RecGen
