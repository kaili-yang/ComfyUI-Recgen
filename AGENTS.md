# AGENTS.md — ComfyUI-Recgen

Guidance for AI coding agents working in this repository.

## Project identity

- **Name**: ComfyUI-Recgen  
- **Purpose**: ComfyUI custom nodes wrapping TRI-ML RecGen (2D RGB-D → 3D mesh / Gaussian splat)  
- **Repo**: https://github.com/kaili-yang/ComfyUI-Recgen  
- **Author**: Kaili Yang (124ykl@gmail.com)  
- **License**: GPL-3.0  

## Layout

```
ComfyUI-Recgen/
├── __init__.py              # ComfyUI entry: NODE_CLASS_MAPPINGS, spconv inject
├── src/comfyui_recgen/
│   ├── nodes.py             # RecGen2DTo3DNode, RecGenLoadExampleInputs
│   ├── spconv_mock.py       # macOS / no-CUDA spconv.pytorch shim
│   ├── depth_utils.py       # ComfyUI depth tensor → metres
│   └── smoke_inputs.py      # 256² smoke test tensors + intrinsics
├── tests/
│   ├── test_run_node.py     # End-to-end smoke (no ComfyUI server)
│   └── test_comfyui_recgen.py
├── docs/                    # FAQ, NODES, llms-full (GEO)
├── llms.txt                 # Short LLM discovery file
└── web/docs/                # ComfyUI node docs UI
```

## Hard dependency

RecGen inference (`recgen_inference`) is resolved from `vendor/recgen` (via `install.py`), a workspace sibling `recgen/`, or `custom_nodes/recgen`. `install.py` clones into `vendor/recgen` so ComfyUI does not treat RecGen as a custom-node pack.

## Conventions

- ComfyUI tensors: IMAGE = float32 `[B,H,W,C]`, MASK = `[B,H,W]`  
- Edit nodes in `src/comfyui_recgen/nodes.py`; register in `NODE_CLASS_MAPPINGS`  
- Category string: `"RecGen"`  
- Prefer minimal diffs; match existing print/log style `[ComfyUI-Recgen] ...`  

## macOS / MPS notes

- Inject `spconv` mock before importing recgen sparse modules  
- Pipeline loads on CPU when `device=mps`, moves to MPS for sampling, may move back to CPU for decode  
- Smoke test uses 256×256 inputs and `intrinsics_for_size()` — do not use full 1080p intrinsics.yaml values at 256² without scaling  

## Tests

```bash
cd ComfyUI-Recgen
pip install -e ".[dev]"
pytest tests/test_comfyui_recgen.py -q
# Full smoke (downloads weights, slow):
python3 tests/test_run_node.py
```

### comfy-test (ComfyUI install + registration + workflows)

Requires [comfy-test](https://github.com/PozzettiAndrea/comfy-test) and path env vars:

```bash
export COMFY_TEST_LOGS_DIR=~/comfy-test-logs
export COMFY_TEST_WORKSPACE_DIR=~/test_workspaces
cd ComfyUI-Recgen
comfy-test run --level registration    # syntax + install + node registration
comfy-test run --level validation      # + workflow validation (load_example_preview.json)
comfy-test run --gpu --level execution # full RecGen smoke (GPU only)
```

Config: `comfy-test.toml`, workflows in `workflows/`, `install.py` clones `TRI-ML/RecGen` into `vendor/recgen`.

## GEO / documentation

When changing user-visible behavior, update:

- `README.md`  
- `docs/FAQ.md`, `docs/NODES.md`, `docs/llms-full.md`  
- `llms.txt`  
- `web/docs/` node markdown if inputs/outputs change  

## Do not

- Commit secrets or HuggingFace tokens  
- Add `Co-authored-by` or AI attribution to git commits (user policy)  
- Modify `ComfyUI/` backend repo from this workspace unless explicitly asked  
