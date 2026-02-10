"""Entry point — run all phases of the XLA & HLO Inspection practice.

Executes each phase sequentially:
  Phase 1: JAX Basics (jit, grad, MLP as pure function)
  Phase 2: Jaxpr Inspection (make_jaxpr, primitive analysis)
  Phase 3: HLO Dump & Reading (lowered vs compiled HLO text)
  Phase 4: Optimization Analysis (pre/post optimization comparison)

Usage:
    uv run python -m xla_inspect.main
"""

from xla_inspect.basics import main as run_basics
from xla_inspect.jaxpr_demo import run_phase as run_jaxpr
from xla_inspect.hlo_dump import run_phase as run_hlo
from xla_inspect.optimization_compare import run_phase as run_optimization


def main() -> None:
    """Run all phases sequentially."""
    print("=" * 60)
    print("  XLA & HLO Inspection — All Phases")
    print("=" * 60)
    print()

    # Phase 1: JAX basics — jit, grad, MLP
    print("\n" + "#" * 60)
    print("# PHASE 1: JAX Basics")
    print("#" * 60)
    run_basics()

    # Phase 2: Jaxpr inspection — make_jaxpr, primitive counts
    print("\n\n" + "#" * 60)
    print("# PHASE 2: Jaxpr Inspection")
    print("#" * 60)
    run_jaxpr()

    # Phase 3: HLO dump — lowered vs compiled text
    print("\n\n" + "#" * 60)
    print("# PHASE 3: HLO Dump & Reading")
    print("#" * 60)
    run_hlo()

    # Phase 4: Optimization analysis — pre/post comparison
    print("\n\n" + "#" * 60)
    print("# PHASE 4: Optimization Analysis")
    print("#" * 60)
    run_optimization()

    print("\n" + "=" * 60)
    print("  All phases complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
