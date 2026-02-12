"""Phase 3: Schedule Analysis & Comparison.

Benchmark different schedule variants to see the real performance impact
of tiling, vectorization, and parallelization. Generate the source code
for each variant and compare side-by-side.

This phase answers the question: "How much faster does each optimization
actually make the matmul?" The answer depends on matrix size, CPU cache
size, and number of cores -- you'll measure it empirically.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import tvm
    from tvm import te

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    print(
        "WARNING: TVM not found. This practice requires Docker.\n"
        "Run: docker compose run --rm tvm python -m tvm_practice.schedule_analysis\n"
    )


# ---------------------------------------------------------------------------
# Matmul declaration (shared)
# ---------------------------------------------------------------------------

def declare_matmul(M: int, K: int, N: int) -> tuple:
    """Declare matmul: C[i,j] = sum_k A[i,k] * B[k,j]."""
    A = te.placeholder((M, K), dtype="float32", name="A")
    B = te.placeholder((K, N), dtype="float32", name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    return A, B, C


# ---------------------------------------------------------------------------
# Schedule builders (fully implemented -- these produce the schedule variants)
# ---------------------------------------------------------------------------

def build_naive(M: int, K: int, N: int) -> tuple:
    """Build a naive (default) schedule for matmul."""
    A, B, C = declare_matmul(M, K, N)
    sch = te.create_schedule(C.op)
    return sch, [A, B, C]


def build_tiled(M: int, K: int, N: int, tile: int = 32) -> tuple:
    """Build a tiled schedule for matmul."""
    A, B, C = declare_matmul(M, K, N)
    sch = te.create_schedule(C.op)
    i, j = sch[C].op.axis
    k = sch[C].op.reduce_axis[0]
    i_o, i_i = sch[C].split(i, factor=tile)
    j_o, j_i = sch[C].split(j, factor=tile)
    sch[C].reorder(i_o, j_o, k, i_i, j_i)
    return sch, [A, B, C]


def build_tiled_parallel(M: int, K: int, N: int, tile: int = 32) -> tuple:
    """Build a tiled + vectorized + parallel schedule for matmul."""
    A, B, C = declare_matmul(M, K, N)
    sch = te.create_schedule(C.op)
    i, j = sch[C].op.axis
    k = sch[C].op.reduce_axis[0]
    i_o, i_i = sch[C].split(i, factor=tile)
    j_o, j_i = sch[C].split(j, factor=tile)
    sch[C].reorder(i_o, j_o, k, i_i, j_i)
    sch[C].vectorize(j_i)
    sch[C].parallel(i_o)
    return sch, [A, B, C]


# ---------------------------------------------------------------------------
# Timing infrastructure (fully implemented)
# ---------------------------------------------------------------------------

def time_schedule(
    func: tvm.runtime.Module,
    M: int,
    K: int,
    N: int,
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> float:
    """Time a compiled TVM function, returning median time in milliseconds.

    Runs n_warmup iterations first (to warm caches and JIT), then
    measures n_repeat iterations and returns the median.
    """
    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    # Warmup
    for _ in range(n_warmup):
        func(a_tvm, b_tvm, c_tvm)

    # Measure
    times: list[float] = []
    for _ in range(n_repeat):
        c_tvm = tvm.nd.array(c_np, dev)  # Reset output
        start = time.perf_counter()
        func(a_tvm, b_tvm, c_tvm)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # Convert to ms

    return float(np.median(times))


def show_generated_source(sch: te.Schedule, tensors: list, label: str) -> None:
    """Print the lowered TIR for a schedule variant."""
    print(f"\n{'─' * 50}")
    print(f"  Generated TIR: {label}")
    print(f"{'─' * 50}")
    lowered = tvm.lower(sch, tensors, simple_mode=True)
    print(lowered)


# ---------------------------------------------------------------------------
# Benchmark & comparison -- TODO(human)
# ---------------------------------------------------------------------------

def run_benchmark(M: int, K: int, N: int) -> None:
    """Build, benchmark, and compare schedule variants.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches empirically measuring schedule impact. Understanding which
    # optimizations help (tiling, parallelization) and when (problem size dependencies)
    # is critical for performance tuning in production ML systems.

    TODO(human): Implement the benchmark comparison.

    This function should:
        1. Build each schedule variant (naive, tiled, tiled+parallel) using the
           build_* functions above.

        2. For each variant:
           a) Show the generated TIR using show_generated_source()
           b) Compile it with tvm.build(sch, tensors, target="llvm", name=label)
           c) Time it with time_schedule(func, M, K, N)
           d) Store the timing result

        3. Print a comparison table like:
           ┌──────────────────────┬──────────────┬─────────┐
           │ Schedule             │ Median (ms)  │ Speedup │
           ├──────────────────────┼──────────────┼─────────┤
           │ naive                │    123.45    │   1.00x │
           │ tiled_32             │     34.56    │   3.57x │
           │ tiled_32_parallel    │      8.91    │  13.86x │
           └──────────────────────┴──────────────┴─────────┘

    Implementation steps:

        1. Define the schedule variants as a list of (label, build_fn) tuples:
               variants = [
                   ("naive", lambda: build_naive(M, K, N)),
                   ("tiled_32", lambda: build_tiled(M, K, N, tile=32)),
                   ("tiled_32_parallel", lambda: build_tiled_parallel(M, K, N, tile=32)),
               ]

        2. For each variant, build and time:
               results = []
               target = tvm.target.Target("llvm")
               for label, build_fn in variants:
                   sch, tensors = build_fn()
                   show_generated_source(sch, tensors, label)
                   func = tvm.build(sch, tensors, target=target, name=label)
                   median_ms = time_schedule(func, M, K, N)
                   results.append((label, median_ms))

        3. Compute speedups relative to the naive baseline:
               naive_ms = results[0][1]
               for label, ms in results:
                   speedup = naive_ms / ms

        4. Print the comparison table. Use f-strings for alignment:
               print(f"  {label:<25s} {ms:>12.3f}    {speedup:>6.2f}x")

    Why this matters:
        Seeing the actual speedup numbers makes the abstract schedule concepts
        concrete. You'll observe:
        - Tiling alone gives 3-5x (cache reuse)
        - Adding vectorize + parallel gives another 2-8x (SIMD + multi-core)
        - The total speedup is often 10-50x over naive

    Args:
        M, K, N: matrix dimensions
    """
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return

    print(f"\nBenchmarking matmul: {M}x{K} @ {K}x{N}")
    print("=" * 60)

    # TODO(human): implement the benchmark comparison
    # Follow the 4 steps above. The build_* and time_schedule functions are ready to use.

    print("\n  (Implement run_benchmark in schedule_analysis.py)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run Phase 3: Schedule Analysis & Comparison."""
    print("\n" + "#" * 60)
    print("  PHASE 3: Schedule Analysis & Comparison")
    print("#" * 60)

    # Test with a medium-sized matrix (large enough to see differences,
    # small enough to benchmark quickly)
    run_benchmark(M=512, K=512, N=512)

    print("\n" + "-" * 60)
    print("KEY TAKEAWAYS:")
    print("-" * 60)
    print("""
  1. Tiling alone gives significant speedup through cache reuse.
  2. Vectorization adds SIMD parallelism (4-16x throughput per core).
  3. Thread-level parallelism scales with number of cores.
  4. The combined effect is multiplicative: tile * vectorize * parallel.
  5. Diminishing returns: at some point, memory bandwidth becomes the bottleneck.
  6. Compare the TIR for each variant to understand what changed.
""")


if __name__ == "__main__":
    run_phase()
