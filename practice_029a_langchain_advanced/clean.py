#!/usr/bin/env python3
"""Clean generated and temporary files for this practice."""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# Practice-specific generated paths (glob patterns relative to ROOT)
EXTRA = [
    ".venv/",             # uv virtual environment
    "chroma_data/",       # ChromaDB persistent storage (if used outside in-memory)
]


def _rm(p: Path) -> None:
    if not p.exists():
        return
    rel = p.relative_to(ROOT)
    print(f"  rm {rel}")
    shutil.rmtree(p) if p.is_dir() else p.unlink()


def clean() -> None:
    print(f"Cleaning {ROOT.name} ...")

    # Docker
    if (ROOT / "docker-compose.yml").exists() or (ROOT / "docker-compose.yaml").exists():
        print("  docker compose down -v")
        subprocess.run(["docker", "compose", "down", "-v"], cwd=ROOT, capture_output=True)

    # Python caches (recursive)
    for pat in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        for m in ROOT.rglob(pat):
            _rm(m)

    # Rust target dirs (next to each Cargo.toml)
    for cargo in ROOT.rglob("Cargo.toml"):
        _rm(cargo.parent / "target")

    # C++ / CMake build dirs
    if any(ROOT.rglob("CMakeLists.txt")):
        for d in ("build", "build_clangd"):
            _rm(ROOT / d)
        for f in ROOT.glob("compile_commands.json"):
            _rm(f)

    # Practice-specific
    for pattern in EXTRA:
        for m in ROOT.glob(pattern):
            _rm(m)

    print("  Done.")


if __name__ == "__main__":
    clean()
