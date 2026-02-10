"""Main entry point — runs all 5 phases of the torch.compile practice.

Phase 1: torch.compile Basics — backends and compilation overhead
Phase 2: TorchDynamo Graph Capture — FX graphs and explain()
Phase 3: Graph Break Analysis — detect and fix graph breaks
Phase 4: TorchInductor Code Inspection — generated Triton/C++ code
Phase 5: Benchmarking & Profiling — compilation cost vs runtime speedup
"""

from compile_inspect.basics import run_phase as run_phase1
from compile_inspect.dynamo_capture import run_phase as run_phase2
from compile_inspect.graph_breaks import run_phase as run_phase3
from compile_inspect.inductor_codegen import run_phase as run_phase4
from compile_inspect.benchmark import run_phase as run_phase5


def main() -> None:
    """Run all phases sequentially."""
    print("\n" + "=" * 60)
    print("  ML Compilers 025f: torch.compile Deep Dive")
    print("=" * 60)

    # Phase 1: torch.compile Basics (Windows OK)
    run_phase1()

    # Phase 2: TorchDynamo Graph Capture (Windows OK)
    run_phase2()

    # Phase 3: Graph Break Analysis (Windows OK)
    run_phase3()

    # Phase 4: TorchInductor Code Inspection (Docker recommended)
    run_phase4()

    # Phase 5: Benchmarking & Profiling (Docker recommended)
    run_phase5()

    print("\n" + "=" * 60)
    print("  All phases complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
