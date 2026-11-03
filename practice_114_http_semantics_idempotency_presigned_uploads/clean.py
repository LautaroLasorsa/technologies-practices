#!/usr/bin/env python3
"""Clean generated and temporary files for Practice 114."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


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
        try:
            p.unlink()
        except Exception:
            pass


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # 1. Stop and remove Docker containers and volumes
    compose_file = ROOT / "docker-compose.yml"
    if compose_file.exists():
        print("  docker compose down -v")
        subprocess.run(["docker", "compose", "down", "-v"], cwd=ROOT, capture_output=True)

    # 2. Python caches and virtual environments
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # 3. Frontend node_modules, build output, and Vite cache
    frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        for pat in ("node_modules", "dist", ".vite"):
            target = frontend_dir / pat
            if target.exists():
                _rm(target)

    # 4. Stray files
    for pat in ("*.pyc", "*.log"):
        for m in ROOT.rglob(pat):
            _rm(m)

    print("  Done.")


if __name__ == "__main__":
    clean()
