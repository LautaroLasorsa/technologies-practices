#!/usr/bin/env python3
"""Clean generated and temporary files for Practice 116."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# Practice-specific generated paths (glob patterns relative to ROOT)
EXTRA = [
    "frontend/node_modules/",
    "frontend/dist/",
    "frontend/.vite/",
]


def _on_rm_error(_func, path, _exc_info):
    """Handle read-only files on Windows during deletion."""
    try:
        os.chmod(path, 0o777)
        os.unlink(path)
    except Exception:
        pass


def _rm(p: Path) -> None:
    if not p.exists():
        return
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        rel = p
    print(f"  rm {rel}")
    if p.is_dir():
        shutil.rmtree(p, onerror=_on_rm_error)
    else:
        try:
            p.unlink()
        except Exception:
            _on_rm_error(None, str(p), None)


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Python caches (recursive)
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Python virtual environments
    for m in ROOT.rglob(".venv"):
        _rm(m)

    # Practice-specific frontend artifacts
    for pattern in EXTRA:
        for m in ROOT.glob(pattern):
            _rm(m)

    print("  Done.")


if __name__ == "__main__":
    clean()
