"""
Entry point: Run all 5 phases of the ML Compiler Fusion practice sequentially.

Usage:
    uv run python -m fusion.main
"""

from fusion.trace_basics import run_phase as phase1
from fusion.manual_fusion import run_phase as phase2
from fusion.pattern_rewriter import run_phase as phase3
from fusion.benchmark import run_phase as phase4
from fusion.conv_bn_fusion import run_phase as phase5


def main() -> None:
    """Run all phases sequentially."""
    print("=" * 70)
    print("  ML COMPILERS — OPERATOR FUSION & GRAPH REWRITES")
    print("  Running all 5 phases")
    print("=" * 70)

    phase1()
    phase2()
    phase3()
    phase4()
    phase5()

    print("\n" + "=" * 70)
    print("  ALL PHASES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
