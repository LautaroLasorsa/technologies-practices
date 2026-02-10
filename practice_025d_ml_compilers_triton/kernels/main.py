"""Entry point — runs all phases sequentially.

Usage (inside Docker):
    python -m kernels.main

This will run:
    Phase 1: Vector Add
    Phase 2: Softmax
    Phase 3: Fused MatMul + ReLU
    Phase 4: Benchmark
"""

from __future__ import annotations

from kernels import vector_add, softmax, matmul_relu, benchmark


def main() -> None:
    """Run all practice phases in order."""
    vector_add.run_phase()
    softmax.run_phase()
    matmul_relu.run_phase()
    benchmark.run_phase()


if __name__ == "__main__":
    main()
