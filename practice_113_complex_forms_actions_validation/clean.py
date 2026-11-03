#!/usr/bin/env python3
"""Clean generated and temporary files for this practice."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# Practice-specific generated paths (glob patterns relative to ROOT)
EXTRA = [
    "frontend/dist",
    "backend/dist",
    "frontend/.vite",
    ".vite",
]


def _on_rm_error(_func, path, _exc_info):
    os.chmod(path, 0o777)
    try:
        os.unlink(path)
    except OSError:
        pass


def _rm(p: Path) -> None:
    if not p.exists():
        return
    rel = p.relative_to(ROOT)
    print(f"  rm {rel}")
    if p.is_dir():
        shutil.rmtree(p, onerror=_on_rm_error)
    else:
        p.unlink()


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Node.js dependencies and caches
    for pat in ("node_modules", "dist", ".vite", ".turbo"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Python caches (if any helper scripts run)
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Practice-specific
    for pattern in EXTRA:
        for m in ROOT.glob(pattern):
            _rm(m)

    print("  Done.")


if __name__ == "__main__":
    clean()
