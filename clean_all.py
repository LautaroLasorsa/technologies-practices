#!/usr/bin/env python3
"""Run clean.py in every practice directory."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> None:
    scripts = sorted(ROOT.glob("practice_*/clean.py"))
    print(f"Found {len(scripts)} practice clean scripts.\n")

    failed: list[str] = []
    for script in scripts:
        practice = script.parent.name
        print(f"{'-' * 60}")
        print(f"  {practice}")
        print(f"{'-' * 60}")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
        )
        if result.returncode != 0:
            failed.append(practice)
        print()

    print(f"{'=' * 60}")
    print(f"Cleaned {len(scripts) - len(failed)}/{len(scripts)} practices.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
