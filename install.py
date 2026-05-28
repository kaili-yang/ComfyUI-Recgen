#!/usr/bin/env python3
"""Install TRI-ML RecGen inference under vendor/recgen (not custom_nodes/)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RECGEN_REPO = "https://github.com/TRI-ML/RecGen.git"


def _node_root() -> Path:
    return Path(__file__).resolve().parent


def _vendor_recgen() -> Path:
    return _node_root() / "vendor" / "recgen"


def _pip_install(*args: str) -> None:
    """Install into the active interpreter (ComfyUI venv in comfy-test CI)."""
    attempts = [
        [sys.executable, "-m", "pip", "install", *args],
        ["uv", "pip", "install", "--python", sys.executable, *args],
    ]
    errors: list[str] = []
    for cmd in attempts:
        try:
            subprocess.check_call(cmd)
            return
        except FileNotFoundError:
            errors.append(f"command not found: {cmd[0]}")
        except subprocess.CalledProcessError as exc:
            errors.append(f"{' '.join(cmd)} failed (exit {exc.returncode})")
    raise RuntimeError(
        "Could not install dependencies via pip or uv pip:\n  "
        + "\n  ".join(errors)
    )


def ensure_recgen() -> Path:
    node_root = _node_root()
    recgen = _vendor_recgen()
    marker = recgen / "recgen_inference"
    if not marker.is_dir():
        recgen.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ComfyUI-Recgen] Cloning RecGen into {recgen} ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", RECGEN_REPO, str(recgen)],
        )
    # Node deps (trimesh, etc.) must be in ComfyUI's venv, not only the host runner.
    print(f"[ComfyUI-Recgen] Installing ComfyUI-Recgen package deps into {sys.executable} ...")
    _pip_install("-e", str(node_root))
    print(f"[ComfyUI-Recgen] Installing recgen_inference from {recgen} ...")
    _pip_install("-e", str(recgen))
    return recgen


def main() -> None:
    ensure_recgen()


if __name__ == "__main__":
    main()
