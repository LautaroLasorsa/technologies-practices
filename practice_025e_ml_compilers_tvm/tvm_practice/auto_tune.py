"""Phase 5: Auto-Tuning with Ansor (AutoScheduler).

Ansor is TVM's automatic schedule search engine. Instead of manually writing
split/reorder/vectorize/parallel (Phase 2), Ansor explores the optimization
space automatically using a learned cost model.

How Ansor works:
    1. Task extraction: identify the compute workloads (e.g., matmul 512x512)
    2. Sketch generation: generate schedule templates (tiling patterns, loop orders)
    3. Random annotation: fill in concrete parameters (tile sizes, unroll factors)
    4. Cost model: predict performance of each candidate WITHOUT running it
    5. Evolutionary search: mutate the best candidates to find even better ones
    6. Measure: actually run the top candidates to validate predictions
    7. Update cost model: feed real measurements back to improve predictions

Why Ansor often beats hand-written schedules:
    - The optimization space is HUGE: for a matmul on a modern CPU, there are
      thousands of valid (tile_i, tile_j, tile_k, unroll, vectorize, parallel)
      combinations. Humans can't explore them all.
    - Ansor's cost model learns from measurements: after a few hundred trials,
      it can predict performance of new schedules with ~90% accuracy.
    - Hardware-specific: the optimal schedule depends on cache sizes, SIMD width,
      number of cores, memory bandwidth. Ansor adapts to whatever hardware it runs on.

Tradeoff:
    - Ansor tuning takes TIME (minutes to hours for complex models)
    - But the result is a schedule optimized for YOUR specific hardware
    - You tune once, then deploy the fast schedule forever

This phase has a "quick" mode with few trials for learning purposes.
Production tuning uses 1000+ trials per task.
"""

from __future__ import annotations

import tempfile

import numpy as np

try:
    import tvm
    from tvm import te, auto_scheduler

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    print(
        "WARNING: TVM not found. This practice requires Docker.\n"
        "Run: docker compose run --rm tvm python -m tvm_practice.auto_tune\n"
    )


# ---------------------------------------------------------------------------
# Matmul definition for AutoScheduler (fully implemented)
# ---------------------------------------------------------------------------

@auto_scheduler.register_workload
def matmul_workload(M: int, K: int, N: int) -> list:
    """Define a matmul workload for AutoScheduler.

    AutoScheduler requires workloads to be registered as functions that
    return a list of input/output tensors. This is the same matmul from
    Phase 1-3, but wrapped for the auto_scheduler API.

    The @register_workload decorator makes this function discoverable
    by AutoScheduler's task extraction mechanism.
    """
    A = te.placeholder((M, K), dtype="float32", name="A")
    B = te.placeholder((K, N), dtype="float32", name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    return [A, B, C]


# ---------------------------------------------------------------------------
# Manual schedule baseline (fully implemented)
# ---------------------------------------------------------------------------

def build_manual_schedule(M: int, K: int, N: int, tile: int = 32) -> tvm.runtime.Module:
    """Build the best manual schedule from Phase 2 as a baseline."""
    A = te.placeholder((M, K), dtype="float32", name="A")
    B = te.placeholder((K, N), dtype="float32", name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    sch = te.create_schedule(C.op)
    i, j = sch[C].op.axis
    ko = sch[C].op.reduce_axis[0]
    i_o, i_i = sch[C].split(i, factor=tile)
    j_o, j_i = sch[C].split(j, factor=tile)
    sch[C].reorder(i_o, j_o, ko, i_i, j_i)
    sch[C].vectorize(j_i)
    sch[C].parallel(i_o)

    target = tvm.target.Target("llvm")
    return tvm.build(sch, [A, B, C], target=target, name="manual")


# ---------------------------------------------------------------------------
# Timing helper (fully implemented)
# ---------------------------------------------------------------------------

def benchmark_module(
    func: tvm.runtime.Module,
    M: int,
    K: int,
    N: int,
    n_warmup: int = 5,
    n_repeat: int = 20,
) -> float:
    """Benchmark a compiled module, return median time in ms."""
    import time

    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    for _ in range(n_warmup):
        func(a_tvm, b_tvm, c_tvm)

    times: list[float] = []
    for _ in range(n_repeat):
        c_tvm = tvm.nd.array(c_np, dev)
        start = time.perf_counter()
        func(a_tvm, b_tvm, c_tvm)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return float(np.median(times))


# ---------------------------------------------------------------------------
# Auto-tuning -- TODO(human)
# ---------------------------------------------------------------------------

def auto_tune_matmul(
    M: int,
    K: int,
    N: int,
    n_trials: int = 50,
) -> tvm.runtime.Module | None:
    """Auto-tune a matmul using Ansor and return the compiled module.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches TVM's auto-scheduling (Ansor) — learned cost models for
    # automatic schedule search. Understanding auto-tuning is critical for production
    # ML deployment where manual scheduling doesn't scale.

    TODO(human): Implement the auto-tuning pipeline.

    Ansor's API follows a clear pipeline:
        1. Create a search task
        2. Configure the tuning options
        3. Run the search
        4. Apply the best schedule and compile

    Steps:

        1. Create a search task:
               target = tvm.target.Target("llvm")
               task = auto_scheduler.SearchTask(
                   func=matmul_workload,
                   args=(M, K, N),
                   target=target,
               )
           - func: the registered workload function (matmul_workload above)
           - args: the arguments to pass to the workload (matrix dimensions)
           - target: the compilation target (CPU, GPU, etc.)
           - The task encapsulates: "optimize this specific computation for this target"

        2. Create a temporary file for the tuning log:
               log_file = tempfile.mktemp(suffix=".json")
           - Ansor writes all explored schedules and their measurements to this file.
           - This file can be reused: if you tune again, Ansor reads previous results
             and continues from where it left off (transfer learning).

        3. Configure tuning options:
               tune_option = auto_scheduler.TuningOptions(
                   num_measure_trials=n_trials,
                   measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                   verbose=2,
               )
           - num_measure_trials: how many schedules to actually compile and measure.
             More trials = better result but longer tuning time.
             Quick mode: 50 trials (~1-2 min). Production: 1000+ trials.
           - measure_callbacks: where to save results.
           - verbose=2: print progress during tuning.

        4. Run the auto-tuning search:
               task.tune(tune_option)
           - This is the expensive step: Ansor generates candidates, compiles them,
             runs them, measures wall-clock time, and updates its cost model.
           - For 50 trials on CPU matmul, expect ~1-3 minutes.

        5. Load the best schedule and compile:
               sch, args = task.apply_best(log_file)
               func = tvm.build(sch, args, target=target, name="auto_tuned")
           - apply_best reads the log file and finds the schedule with the
             lowest measured latency.
           - tvm.build compiles that schedule into executable code.

        6. Return the compiled function:
               return func

    Why n_trials matters:
        - 10 trials: Ansor barely explores the space. Result is mediocre.
        - 50 trials: decent exploration. Usually 60-80% of optimal.
        - 200 trials: good results for most workloads.
        - 1000+ trials: near-optimal. Diminishing returns after this.

    Args:
        M, K, N: matrix dimensions
        n_trials: number of measurement trials (default 50 for quick mode)

    Returns:
        Compiled TVM module with the auto-tuned schedule, or None on failure.
    """
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return None

    print(f"\n  Auto-tuning matmul {M}x{K}x{N} with {n_trials} trials...")
    print(f"  (This may take 1-3 minutes for 50 trials)\n")

    # TODO(human): implement the 6 steps above
    # The matmul_workload function is already registered -- just reference it.

    # Stub: return None
    print("  (Implement auto_tune_matmul in auto_tune.py)")
    return None


# ---------------------------------------------------------------------------
# Comparison (fully implemented)
# ---------------------------------------------------------------------------

def compare_manual_vs_auto(
    M: int,
    K: int,
    N: int,
    n_trials: int = 50,
) -> None:
    """Compare manual schedule vs auto-tuned schedule performance."""
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return

    print(f"\n  Matrix dimensions: {M}x{K} @ {K}x{N}")
    print(f"  Auto-tune trials: {n_trials}")

    # Build manual baseline
    print("\n--- Manual Schedule (tiled + vectorize + parallel) ---")
    manual_func = build_manual_schedule(M, K, N)
    manual_ms = benchmark_module(manual_func, M, K, N)
    print(f"  Manual schedule: {manual_ms:.3f} ms")

    # Run auto-tuning
    print("\n--- Auto-Tuned Schedule (Ansor) ---")
    auto_func = auto_tune_matmul(M, K, N, n_trials=n_trials)

    if auto_func is None:
        print("  (Auto-tuning not implemented yet -- skipping comparison)")
        return

    auto_ms = benchmark_module(auto_func, M, K, N)
    print(f"  Auto-tuned schedule: {auto_ms:.3f} ms")

    # Comparison
    speedup = manual_ms / auto_ms if auto_ms > 0 else 0.0
    print("\n" + "=" * 50)
    print("  COMPARISON: Manual vs Auto-Tuned")
    print("=" * 50)
    print(f"  {'Schedule':<25s} {'Median (ms)':>12s}  {'Speedup':>8s}")
    print(f"  {'─' * 25} {'─' * 12}  {'─' * 8}")
    print(f"  {'manual (tile=32)':<25s} {manual_ms:>12.3f}  {'1.00x':>8s}")
    print(f"  {'auto-tuned (Ansor)':<25s} {auto_ms:>12.3f}  {f'{speedup:.2f}x':>8s}")
    print()

    if speedup > 1.0:
        print(f"  Ansor found a schedule {speedup:.1f}x faster than manual!")
    elif speedup > 0.95:
        print(f"  Manual and auto-tuned are comparable (within 5%).")
    else:
        print(f"  Manual schedule is faster -- try more trials for better Ansor results.")

    # Verify correctness
    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    c_manual = np.zeros((M, N), dtype="float32")
    c_auto = np.zeros((M, N), dtype="float32")

    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_m_tvm = tvm.nd.array(c_manual, dev)
    c_a_tvm = tvm.nd.array(c_auto, dev)

    manual_func(a_tvm, b_tvm, c_m_tvm)
    auto_func(a_tvm, b_tvm, c_a_tvm)

    np.testing.assert_allclose(
        c_m_tvm.numpy(), c_a_tvm.numpy(), rtol=1e-3,
    )
    print("  Correctness: PASSED (manual and auto-tuned produce same result)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run Phase 5: Auto-Tuning with Ansor."""
    print("\n" + "#" * 60)
    print("  PHASE 5: Auto-Tuning with Ansor")
    print("#" * 60)

    # Use a moderate size for quick tuning
    compare_manual_vs_auto(M=512, K=512, N=512, n_trials=50)

    print("\n" + "-" * 60)
    print("KEY TAKEAWAYS:")
    print("-" * 60)
    print("""
  1. Ansor automatically explores the schedule space -- no manual tuning needed.
  2. More trials = better schedule, but diminishing returns after ~200-500 trials.
  3. The tuning log can be saved and reused -- tune once, deploy forever.
  4. Ansor adapts to YOUR hardware: cache sizes, SIMD width, core count.
  5. For production models, Ansor+Relay gives end-to-end optimization:
     graph-level fusion (Relay) + operator-level auto-tuning (Ansor).
  6. The tradeoff: tuning time vs execution speed. Worth it for deployed models.
""")


if __name__ == "__main__":
    run_phase()
