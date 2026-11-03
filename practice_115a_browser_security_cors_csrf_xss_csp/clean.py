#!/usr/bin/env python3
"""Clean generated and temporary files for this practice."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Practice-specific generated paths
EXTRA = [
    "frontend/node_modules",
    "frontend/dist",
    "attacker-server/node_modules",
    "backend/.venv",
]


def _on_rm_error(_func, path, _exc_info):
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
        p.unlink()


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Python caches
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Python virtual environments
    for m in ROOT.rglob(".venv"):
        _rm(m)

    # Extra directories (node_modules, dist, etc.)
    for extra_path in EXTRA:
        target = ROOT / extra_path
        if target.exists():
            _rm(target)

    # Log files
    for log_file in ROOT.rglob("*.log"):
        _rm(log_file)

    print("  Done.")


if __name__ == "__main__":
    clean()
