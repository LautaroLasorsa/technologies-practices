#!/usr/bin/env python3
"""Clean generated and temporary files for this practice."""
import os
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).parent

EXTRA = [
    "services/python-api/openapi.json",
]


def _on_rm_error(func, path, _exc_info):
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
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
            _on_rm_error(os.unlink, str(p), None)


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Python caches (recursive)
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Python virtual environments
    for m in ROOT.rglob(".venv"):
        _rm(m)

    # Node modules and build outputs
    for pat in ("node_modules", "dist", "build", ".turbo"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # TypeScript build artifacts
    for m in ROOT.rglob("*.tsbuildinfo"):
        _rm(m)

    # Practice-specific
    for pattern in EXTRA:
        for m in ROOT.glob(pattern):
            _rm(m)

    print("  Done.")


if __name__ == "__main__":
    clean()
