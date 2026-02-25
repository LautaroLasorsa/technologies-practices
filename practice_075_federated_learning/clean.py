"""Clean generated/temporary files for practice_075_federated_learning."""

import shutil
from pathlib import Path

PRACTICE_DIR = Path(__file__).resolve().parent

COMMON = [
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]

EXTRA = [
    "data",
    "plots",
    "models",
]


def _on_error(_func, path, _exc_info):
    """Handle read-only files on Windows."""
    import os
    import stat
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)


def clean() -> None:
    targets = COMMON + EXTRA
    for name in targets:
        target = PRACTICE_DIR / name
        if target.is_dir():
            shutil.rmtree(target, onerror=_on_error)
            print(f"  Removed directory: {target.relative_to(PRACTICE_DIR)}")
        elif target.is_file():
            target.unlink()
            print(f"  Removed file: {target.relative_to(PRACTICE_DIR)}")

    # Remove __pycache__ in subdirectories
    for pycache in PRACTICE_DIR.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, onerror=_on_error)
            print(f"  Removed directory: {pycache.relative_to(PRACTICE_DIR)}")

    # Remove .pyc files
    for pyc in PRACTICE_DIR.rglob("*.pyc"):
        pyc.unlink()
        print(f"  Removed file: {pyc.relative_to(PRACTICE_DIR)}")


if __name__ == "__main__":
    print(f"Cleaning {PRACTICE_DIR.name}...")
    clean()
    print("Done.")
