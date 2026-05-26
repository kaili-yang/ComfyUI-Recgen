# RecGen Load Example Inputs

Loads demonstration **RGB**, **depth**, **mask**, and **camera intrinsics** from the sibling `recgen/examples/` folder (requires the [RecGen](https://github.com/TRI-ML/RecGen) repository with example assets).

## When to use

- First-time setup or tutorial workflows  
- Quick test before connecting your own depth/mask  

## Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `example` | COMBO | Example id (`ex0`, `ex1`, …) |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | RGB tensor |
| `depth` | IMAGE | Depth in metres |
| `mask` | MASK | Object mask |
| `fx`, `fy`, `cx`, `cy` | FLOAT | Pinhole intrinsics |

## Typical workflow

Connect all outputs to **RecGen 2D to 3D**.
