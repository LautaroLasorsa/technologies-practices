#!/usr/bin/env python3
"""Clean generated and temporary files for this practice."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Practice-specific generated paths (glob patterns relative to ROOT)
EXTRA = [
    "data",
    "plots",
    "models",
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

    # Practice-specific
    for pattern in EXTRA:
        for m in ROOT.glob(pattern):
            _rm(m)

    print("  Done.")


if __name__ == "__main__":
    clean()
