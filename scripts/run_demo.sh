#!/usr/bin/env bash
# End-to-end local demo for ComfyUI-Recgen (downloads RecGen weights on first run).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

echo "[run_demo] Workspace: $WORKSPACE"

if [[ ! -d "$WORKSPACE/recgen" ]]; then
  echo "ERROR: Expected sibling recgen repo at $WORKSPACE/recgen"
  exit 1
fi

python3 -m pip install -q -e "$WORKSPACE/recgen"
python3 -m pip install -q -e "$ROOT"

export SPCONV_ALGO=native
export SPARSE_ATTN_BACKEND=sdpa
export PYOPENGL_PLATFORM=egl

python3 "$ROOT/tests/test_run_node.py"
