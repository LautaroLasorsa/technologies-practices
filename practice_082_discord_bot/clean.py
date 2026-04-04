"""Clean generated/temporary files for practice_082_discord_bot."""

from __future__ import annotations

import shutil
from pathlib import Path

PRACTICE_DIR = Path(__file__).resolve().parent

REMOVE_DIRS = [
    "__pycache__",
    "cogs/__pycache__",
    ".venv",
]

REMOVE_FILES = [
    ".env",
    "*.log",
]


def _on_rm_error(_func: object, path: str, _exc_info: object) -> None:
    """Handle permission errors on Windows (read-only files)."""
    import os
    import stat
    os.chmod(path, stat.S_IWRITE)
    os.remove(path)


def clean() -> None:
    """Remove all generated/temporary files."""
    for dir_name in REMOVE_DIRS:
        target = PRACTICE_DIR / dir_name
        if target.exists():
            shutil.rmtree(target, onerror=_on_rm_error)
            print(f"  Removed directory: {dir_name}")

    for pattern in REMOVE_FILES:
        for match in PRACTICE_DIR.glob(pattern):
            match.unlink()
            print(f"  Removed file: {match.relative_to(PRACTICE_DIR)}")

    print("Clean complete.")


if __name__ == "__main__":
    clean()
