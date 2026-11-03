#!/usr/bin/env python3
"""Clean generated and temporary files for practice_117_rendering_architectures_ssr_rsc_bff."""
import os
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).parent

# Practice-specific generated paths
EXTRA = [
    "bff-nextjs/.next",
    "bff-nextjs/out",
    "bff-nextjs/node_modules",
    "python-service/.venv",
    ".venv",
]


def _on_rm_error(func, path, exc_info):
    """Handle read-only or locked files on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"    Warning: could not remove {path}: {e}")


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
            try:
                os.chmod(p, stat.S_IWRITE)
                p.unlink()
            except Exception as e:
                print(f"    Warning: could not delete {p}: {e}")


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Python caches
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Virtual environments & node artifacts
    for item in EXTRA:
        target = ROOT / item
        if target.exists():
            _rm(target)

    # Any remaining .pyc files
    for pyc in ROOT.rglob("*.pyc"):
        _rm(pyc)

    print("  Done.")


if __name__ == "__main__":
    clean()
