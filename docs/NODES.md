# Node reference — ComfyUI-Recgen

## RecGen Load Example Inputs

Loads bundled demonstration data from the sibling `recgen/examples/` directory (requires the RecGen repo with example PNGs).

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| example | COMBO | Example id, e.g. `ex0`, `ex1`, … |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| image | IMAGE | RGB tensor `[1, H, W, 3]` |
| depth | IMAGE | Depth tensor `[1, H, W, 1]` in metres |
| mask | MASK | Mask tensor `[1, H, W]` |
| fx, fy, cx, cy | FLOAT | Intrinsics from `intrinsics.yaml` when present |

### Typical use

Connect all outputs into **RecGen 2D to 3D** for a zero-config demo workflow.

---

## RecGen 2D to 3D

Runs RecGen single-view reconstruction.

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | IMAGE | — | RGB input |
| depth | IMAGE | — | Depth (metric or rescale-compatible) |
| mask | MASK | — | Object mask |
| checkpoint | COMBO | `recgen_base.multiview_stereo` | RecGen checkpoint name |
| device | COMBO | auto | `cuda`, `mps`, or `cpu` |
| fx | FLOAT | 1062.2 | Focal length x (pixels) |
| fy | FLOAT | 1060.9 | Focal length y (pixels) |
| cx | FLOAT | -1 | Principal point x (-1 = image centre) |
| cy | FLOAT | -1 | Principal point y (-1 = image centre) |
| seed | INT | 42 | Random seed |
| save_splat | BOOLEAN | true | Save `posed_gaussian.ply` |
| save_glb | BOOLEAN | false | Save textured GLB (extra dependencies) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| output_dir | STRING | Directory path with mesh and images |
| overlay_image | IMAGE | Mesh overlay preview |

### Disk outputs (under output_dir)

| File | Always | Description |
|------|--------|-------------|
| posed_mesh.obj | Yes | Textured mesh in camera frame |
| mesh.obj | Yes | Object-centric mesh |
| overlay.png | Yes | Rendered preview |
| posed_gaussian.ply | If save_splat | Gaussian splat |
| metadata.json | Yes | Pose and intrinsics metadata |
