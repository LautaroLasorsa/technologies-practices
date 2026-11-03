#!/usr/bin/env python3
"""Clean generated and temporary files for practice_115b."""
import os
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).parent


def _on_rm_error(func, path, exc_info):
    """Handle read-only files on Windows during deletion."""
    try:
        os.chmod(path, stat.S_IWRITE)
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
            try:
                os.chmod(p, stat.S_IWRITE)
                p.unlink()
            except Exception:
                pass


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Node modules & build artifacts (recursive for frontend and backend)
    for pat in ("node_modules", "dist", "build", ".vite", ".cache", "coverage"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Log files
    for pat in ("*.log", "npm-debug.log*"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Python caches (if any helper scripts run)
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    print("Done cleaning.")


if __name__ == "__main__":
    clean()
