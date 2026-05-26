# RecGen 2D to 3D

Runs [RecGen](https://github.com/TRI-ML/RecGen) single-view reconstruction inside ComfyUI. Converts aligned **RGB + depth + object mask** into a **3D mesh** (`.obj`) and an on-screen **overlay preview**.

## When to use

- You have a depth map (RGB-D, stereo, or sensor) and a mask for one object  
- You want mesh or Gaussian splat files under ComfyUI’s output directory  

## Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | IMAGE | RGB colour image |
| `depth` | IMAGE | Depth aligned to RGB (metres or rescale-compatible PNG) |
| `mask` | MASK | Foreground object mask |
| `checkpoint` | COMBO | RecGen model checkpoint |
| `device` | COMBO | `cuda`, `mps`, or `cpu` |
| `fx`, `fy` | FLOAT | Focal length in pixels |
| `cx`, `cy` | FLOAT | Principal point (-1 = image centre) |
| `seed` | INT | Random seed |
| `save_splat` | BOOLEAN | Write Gaussian `.ply` |
| `save_glb` | BOOLEAN | Write textured GLB (optional deps) |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `output_dir` | STRING | Folder with `posed_mesh.obj`, `overlay.png`, etc. |
| `overlay_image` | IMAGE | Mesh drawn over the input RGB |

## Tips

- Match intrinsics to the resolution of the tensors you connect  
- First run downloads weights from HuggingFace (~10 GB)  
- On Apple Silicon, prefer moderate resolutions for testing  

See also: [GitHub docs](https://github.com/kaili-yang/ComfyUI-Recgen/blob/main/docs/NODES.md)
