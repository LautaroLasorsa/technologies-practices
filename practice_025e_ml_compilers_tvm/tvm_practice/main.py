"""Main entry point -- runs all 5 phases of the TVM scheduling practice.

Phase 1: TVM Basics & Tensor Expressions
Phase 2: Manual Schedule Optimization
Phase 3: Schedule Analysis & Comparison
Phase 4: Relay Integration
Phase 5: Auto-Tuning with Ansor
"""

from __future__ import annotations

import sys

try:
    import tvm

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False


def check_environment() -> bool:
    """Check that TVM is available and print environment info."""
    print("=" * 60)
    print("  ML Compilers 025e: TVM Scheduling & Auto-Tuning")
    print("=" * 60)

    if not TVM_AVAILABLE:
        print(
            "\n  ERROR: TVM is not installed.\n"
            "\n  This practice requires Docker. Run:\n"
            "    docker compose build\n"
            "    docker compose run --rm tvm python -m tvm_practice.main\n"
        )
        return False

    print(f"\n  TVM version: {tvm.__version__}")
    print(f"  Python: {sys.version}")

    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except ImportError:
        print("  NumPy: NOT FOUND")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
    except ImportError:
        print("  PyTorch: NOT FOUND (Phase 4 will be skipped)")

    # Check available targets
    print(f"\n  Available TVM targets: llvm (CPU)")
    print()
    return True


def main() -> None:
    """Run all phases sequentially."""
    if not check_environment():
        return

    # Phase 1: TVM Basics & Tensor Expressions
    from tvm_practice.te_basics import run_phase as run_phase1
    run_phase1()

    # Phase 2: Manual Schedule Optimization
    from tvm_practice.manual_schedule import run_phase as run_phase2
    run_phase2()

    # Phase 3: Schedule Analysis & Comparison
    from tvm_practice.schedule_analysis import run_phase as run_phase3
    run_phase3()

    # Phase 4: Relay Integration
    from tvm_practice.relay_import import run_phase as run_phase4
    run_phase4()

    # Phase 5: Auto-Tuning with Ansor
    from tvm_practice.auto_tune import run_phase as run_phase5
    run_phase5()

    print("\n" + "=" * 60)
    print("  All phases complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
