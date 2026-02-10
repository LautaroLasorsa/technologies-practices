"""Phase 4: Benchmarking — Compare Triton Kernels vs PyTorch.

This module provides a timing framework to measure and compare the performance
of your Triton kernels against their PyTorch equivalents.

Concept overview:
    GPU benchmarking has subtleties that CPU benchmarking doesn't:

    1. **Warm-up runs**: The first kernel launch includes JIT compilation overhead
       (Triton compiles to PTX on first call). We run several warm-up iterations
       before measuring.

    2. **Synchronization**: GPU execution is asynchronous. We must call
       torch.cuda.synchronize() before starting and after stopping the timer,
       otherwise we'd measure only the kernel launch overhead (microseconds)
       instead of actual execution time (milliseconds).

    3. **Multiple iterations**: We run many iterations and report the median
       to reduce noise from OS scheduling and GPU clock boosting.

References:
    - https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#benchmark
"""

from __future__ import annotations

import time
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Import kernel modules
from kernels import vector_add, softmax, matmul_relu


# ---------------------------------------------------------------------------
# Timing utility (fully implemented)
# ---------------------------------------------------------------------------

def benchmark_fn(
    fn: Callable[[], None],
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Benchmark a GPU function, returning median time in milliseconds.

    Args:
        fn: Callable that runs the GPU operation (no return value needed).
        warmup: Number of warm-up iterations (not timed).
        iterations: Number of timed iterations.

    Returns:
        Median execution time in milliseconds.
    """
    # Warm-up: includes JIT compilation on first call
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert to ms

    times.sort()
    return times[len(times) // 2]  # Median


# ---------------------------------------------------------------------------
# Benchmark runner — TODO(human): implement the benchmark comparisons
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run benchmarks comparing Triton kernels vs PyTorch equivalents.

    TODO(human): Implement the benchmark runner.

    The framework is ready — you need to:

    1. DEFINE THE TEST CONFIGURATIONS
       Choose problem sizes that are representative. Suggested:
       - Vector add: n = 1_000_000 (1M elements)
       - Softmax: shape = (1024, 1024) — square matrix
       - MatMul+ReLU: M=K=N=1024 — square matrices

    2. FOR EACH KERNEL, CREATE TWO CALLABLES:
       a) The Triton version (call your kernel with the right grid)
       b) The PyTorch equivalent (e.g., a + b, torch.softmax, torch.relu(a @ b))

    3. BENCHMARK BOTH AND COLLECT RESULTS
       Use benchmark_fn() defined above. It handles warm-up and synchronization.

    4. PRINT A COMPARISON TABLE
       Show: operation name, Triton time (ms), PyTorch time (ms), speedup ratio.

    Example structure for one kernel:

        # --- Vector Add ---
        n = 1_000_000
        a = torch.randn(n, device='cuda')
        b = torch.randn(n, device='cuda')
        c = torch.empty_like(a)
        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        def triton_vadd():
            vector_add.triton_vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)

        def torch_vadd():
            torch.add(a, b, out=c)

        t_triton = benchmark_fn(triton_vadd)
        t_torch = benchmark_fn(torch_vadd)

    Then repeat for softmax and matmul_relu.

    Print format suggestion:
        ┌──────────────────┬────────────┬────────────┬─────────┐
        │ Operation        │ Triton(ms) │ PyTorch(ms)│ Speedup │
        ├──────────────────┼────────────┼────────────┼─────────┤
        │ vector_add (1M)  │     0.042  │     0.038  │   0.91x │
        │ softmax (1Kx1K)  │     0.15   │     0.12   │   0.80x │
        │ matmul+relu (1K) │     0.48   │     0.52   │   1.08x │
        └──────────────────┴────────────┴────────────┴─────────┘

    Key insights to look for:
    - Vector add is memory-bound; PyTorch's CUDA kernel is already optimal, so
      Triton may not be faster (and that's expected).
    - Softmax: Triton's fused kernel avoids an extra pass over the data vs
      a naive implementation, but PyTorch's built-in is also fused.
    - MatMul+ReLU: This is where fusion shines most. torch.relu(A @ B) does
      matmul, writes to memory, reads back, applies ReLU, writes again.
      The Triton fused kernel does it in a single pass.
    """
    if not TRITON_AVAILABLE:
        print("[Phase 4 — Benchmark] Triton not available. Run inside Docker with GPU.")
        return

    print("=" * 60)
    print("Phase 4: Benchmark — Triton vs PyTorch")
    print("=" * 60)

    # ---- STUB: remove this and implement the benchmark runner ----
    print("  [placeholder] Benchmark not yet implemented.")
    print()
    print("  TODO(human): Implement the benchmark runner in this function.")
    print("  See the docstring above for detailed instructions.")
    print()
    print("  Suggested table format:")
    print("  ┌──────────────────┬────────────┬────────────┬─────────┐")
    print("  │ Operation        │ Triton(ms) │ PyTorch(ms)│ Speedup │")
    print("  ├──────────────────┼────────────┼────────────┼─────────┤")
    print("  │ vector_add (1M)  │     ???    │     ???    │   ???   │")
    print("  │ softmax (1Kx1K)  │     ???    │     ???    │   ???   │")
    print("  │ matmul+relu (1K) │     ???    │     ???    │   ???   │")
    print("  └──────────────────┴────────────┴────────────┴─────────┘")
    print()


if __name__ == "__main__":
    run_phase()
