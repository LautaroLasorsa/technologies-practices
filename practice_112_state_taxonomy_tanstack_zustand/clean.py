#!/usr/bin/env python3
"""Clean generated and temporary files for practice 112."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Paths to remove relative to ROOT
EXTRA = [
    "frontend/dist",
    "frontend/.vite",
    "frontend/node_modules",
]


def _on_rm_error(_func, path, _exc_info):
    os.chmod(path, 0o777)
    os.unlink(path)


def _rm(p: Path) -> None:
    if not p.exists():
        return
    rel = p.relative_to(ROOT)
    print(f"  rm {rel}")
    shutil.rmtree(p, onerror=_on_rm_error) if p.is_dir() else p.unlink()


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Python caches (recursive)
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Python virtual environments
    for m in ROOT.rglob(".venv"):
        _rm(m)

    # Frontend node_modules & build artifacts
    for rel_path in EXTRA:
        target = ROOT / rel_path
        if target.exists():
            _rm(target)

    print("  Done.")


if __name__ == "__main__":
    clean()
