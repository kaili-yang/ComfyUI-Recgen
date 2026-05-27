#!/usr/bin/env python3
"""Install TRI-ML RecGen inference under vendor/recgen (not custom_nodes/)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RECGEN_REPO = "https://github.com/TRI-ML/RecGen.git"


def _vendor_recgen() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "recgen"


def ensure_recgen() -> Path:
    recgen = _vendor_recgen()
    marker = recgen / "recgen_inference"
    if not marker.is_dir():
        recgen.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ComfyUI-Recgen] Cloning RecGen into {recgen} ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", RECGEN_REPO, str(recgen)],
        )
    print(f"[ComfyUI-Recgen] Installing recgen_inference from {recgen} ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", str(recgen)],
    )
    return recgen


def main() -> None:
    ensure_recgen()


if __name__ == "__main__":
    main()
