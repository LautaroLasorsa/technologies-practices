"""
Phase 4: Benchmarking Fused vs Unfused Models.

GOAL: Measure the actual performance difference between fused and unfused models
using torch.utils.benchmark.Timer — the official PyTorch microbenchmarking tool.

WHY BENCHMARK FUSION?

  Fusion's benefit is reducing memory bandwidth usage. But how much faster does
  it actually make things? The answer depends on:
    - Model size (how much intermediate data is saved)
    - Hardware (GPU memory bandwidth, cache sizes)
    - Batch size (larger batches = more data moving through memory)
    - Operation type (memory-bound vs compute-bound operations)

  Benchmarking lets you quantify the real-world impact rather than guessing.

torch.utils.benchmark.Timer:

  PyTorch provides a benchmarking utility that handles warmup, multiple runs,
  statistical aggregation, and proper GPU synchronization. It's similar to
  Python's timeit but designed for GPU workloads.

  Key features:
    - Automatic warmup (discards first few runs that include JIT compilation,
      CUDA kernel caching, etc.)
    - Reports median, IQR (interquartile range), and number of runs
    - Handles CUDA synchronization (GPU ops are async — you must sync before timing)
    - Clean comparison via Compare class

WHAT WE MEASURE:

  1. LATENCY — How long does a single forward pass take? (milliseconds)
     Lower is better. Directly impacts inference response time.

  2. THROUGHPUT — How many samples per second can the model process?
     Higher is better. Directly impacts batch processing speed.
"""

import torch
import torch.nn as nn
import torch.fx
from torch.utils.benchmark import Timer

from fusion.manual_fusion import (
    StackedLinearReLU,
    apply_linear_relu_fusion,
)


# ---------------------------------------------------------------------------
# Helpers — fully implemented
# ---------------------------------------------------------------------------
def create_fused_model(
    in_features: int = 64, hidden: int = 128, out_features: int = 10,
) -> tuple[nn.Module, torch.fx.GraphModule]:
    """Create an unfused model and its fused counterpart.

    Returns:
        Tuple of (original_model, fused_graph_module).
    """
    model = StackedLinearReLU(in_features, hidden, out_features)
    model.eval()

    traced = torch.fx.symbolic_trace(model)
    fused = apply_linear_relu_fusion(traced)

    return model, fused


def format_time(time_sec: float) -> str:
    """Format a time value to a human-readable string with appropriate units."""
    if time_sec < 1e-6:
        return f"{time_sec * 1e9:.1f} ns"
    elif time_sec < 1e-3:
        return f"{time_sec * 1e6:.1f} us"
    elif time_sec < 1.0:
        return f"{time_sec * 1e3:.2f} ms"
    else:
        return f"{time_sec:.3f} s"


def print_comparison_table(results: list[dict[str, object]]) -> None:
    """Print a formatted comparison table of benchmark results.

    Args:
        results: List of dicts with keys:
            - label (str): description of what was benchmarked
            - batch_size (int): batch size used
            - latency_sec (float): median latency in seconds
            - throughput (float): samples per second
    """
    print(f"\n{'Label':<30} {'Batch':>6} {'Latency':>12} {'Throughput':>15}")
    print("-" * 70)
    for r in results:
        lat_str = format_time(r["latency_sec"])
        tp_str = f"{r['throughput']:.0f} samples/s"
        print(f"{r['label']:<30} {r['batch_size']:>6} {lat_str:>12} {tp_str:>15}")

    # Print speedup if we have exactly 2 results (unfused vs fused)
    if len(results) == 2:
        speedup = results[0]["latency_sec"] / max(results[1]["latency_sec"], 1e-12)
        print(f"\n  Speedup (fused vs unfused): {speedup:.2f}x")
        if speedup < 1.05:
            print("  Note: Speedup may be minimal on CPU. Fusion shines on GPU where")
            print("  memory bandwidth is the bottleneck and kernel launch overhead matters.")


# ---------------------------------------------------------------------------
# TODO(human): Implement the benchmark runner
# ---------------------------------------------------------------------------
def benchmark_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    label: str,
    num_threads: int = 1,
    min_run_time: float = 1.0,
) -> dict[str, object]:
    """Benchmark a single model's forward pass and return timing results.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches using torch.utils.benchmark.Timer for measuring GPU/CPU
    # performance. Understanding microbenchmarking is essential for validating that
    # compiler optimizations actually improve real-world performance.

    TODO(human): Implement this function.

    Use torch.utils.benchmark.Timer to measure the median latency of model(input_tensor).
    Then compute throughput as batch_size / median_latency.

    DETAILED GUIDANCE:

    1. CREATING A TIMER:
       torch.utils.benchmark.Timer takes:
         - stmt: a string of Python code to benchmark (e.g., "model(x)")
         - globals: a dict of variables available to the stmt (e.g., {"model": model, "x": input_tensor})
         - label: a descriptive label for the benchmark
         - num_threads: number of threads for intra-op parallelism

       Example:
           timer = Timer(
               stmt="model(x)",
               globals={"model": model, "x": input_tensor},
               label=label,
               num_threads=num_threads,
           )

    2. RUNNING THE BENCHMARK:
       Call timer.blocked_autorange(min_run_time=min_run_time) to run the benchmark.
       This method:
         - Automatically determines how many iterations to run
         - Runs enough iterations to fill at least min_run_time seconds
         - Handles warmup internally
         - Returns a Measurement object

       Example:
           measurement = timer.blocked_autorange(min_run_time=min_run_time)

    3. EXTRACTING RESULTS:
       The Measurement object has:
         - measurement.median: median time per iteration in SECONDS
         - measurement.mean: mean time per iteration in SECONDS
         - measurement.iqr: interquartile range (measure of variance)
         - measurement.number_per_run: how many iterations per measurement
         - str(measurement): a nicely formatted summary string

       Print str(measurement) to see the full benchmark output, then extract
       the median for our comparison table.

    4. COMPUTING THROUGHPUT:
       Throughput = batch_size / median_latency_in_seconds
       where batch_size = input_tensor.shape[0] (the first dimension).

       Example:
           batch_size = input_tensor.shape[0]
           throughput = batch_size / measurement.median

    5. RETURN VALUE:
       Return a dict with:
           {
               "label": label,
               "batch_size": batch_size,
               "latency_sec": measurement.median,
               "throughput": throughput,
           }

    6. WHY blocked_autorange?
       Alternatives like timeit() require you to specify the number of iterations.
       blocked_autorange figures it out automatically — it keeps running until it
       has collected at least min_run_time seconds of data, giving statistically
       stable results without over-testing fast operations or under-testing slow ones.

    7. GPU CONSIDERATIONS:
       On GPU, you'd need torch.cuda.synchronize() in the stmt or setup.
       Timer handles this if you pass torch.cuda.synchronize into the setup.
       For this practice (CPU only), this isn't needed, but it's good to know.

    Args:
        model: The model to benchmark.
        input_tensor: The input tensor to pass to the model.
        label: Descriptive label for this benchmark run.
        num_threads: Number of CPU threads for parallelism.
        min_run_time: Minimum wall-clock time to run the benchmark (seconds).

    Returns:
        Dict with keys: label, batch_size, latency_sec, throughput.
    """
    # STUB: returns placeholder results so the file runs without errors
    batch_size = input_tensor.shape[0]
    return {
        "label": label,
        "batch_size": batch_size,
        "latency_sec": 0.0,
        "throughput": 0.0,
    }


def run_benchmark_suite(
    batch_sizes: list[int] | None = None,
    in_features: int = 64,
    hidden: int = 128,
    out_features: int = 10,
) -> list[dict[str, object]]:
    """Run the full benchmark suite: unfused vs fused across batch sizes.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches systematic performance comparison — running benchmarks
    # across different workload sizes to quantify optimization impact. Real compiler
    # teams use this methodology to validate every optimization pass.

    TODO(human): Implement this function.

    For each batch size, create the input tensor, benchmark both the unfused and
    fused models, and collect all results.

    DETAILED GUIDANCE:

    1. DEFAULT BATCH SIZES:
       If batch_sizes is None, use a reasonable default like [1, 16, 64, 256].
       These cover single-sample latency (batch=1, typical for online inference)
       through larger batch throughput (batch=256, typical for batch inference).

    2. CREATE MODELS ONCE:
       Call create_fused_model() to get both the original and fused models.
       Create them once outside the batch-size loop — the models themselves don't
       change, only the input tensor size changes.

       IMPORTANT: Both models must be in eval mode (model.eval()) to disable
       dropout, batchnorm training behavior, etc. create_fused_model already
       does this for the original, but you should also ensure the fused model
       is in eval mode.

    3. BENCHMARKING LOOP:
       For each batch_size:
         a) Create an input tensor: torch.randn(batch_size, in_features)
         b) Benchmark the unfused model: benchmark_model(original, x, "Unfused ...")
         c) Benchmark the fused model:   benchmark_model(fused, x, "Fused ...")
         d) Append both results to the results list

    4. USE torch.no_grad():
       Wrap model calls in torch.no_grad() context. Since we're benchmarking
       inference (not training), we don't need gradients. This saves memory and
       can be faster. You can set this globally before the loop:
           torch.set_grad_enabled(False)

    5. RETURN:
       Return the list of all result dicts (one per model per batch size).

    Args:
        batch_sizes: List of batch sizes to test. Defaults to [1, 16, 64, 256].
        in_features: Input feature dimension.
        hidden: Hidden layer dimension.
        out_features: Output dimension.

    Returns:
        List of benchmark result dicts.
    """
    # STUB: returns empty results so the file runs without errors
    return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_phase() -> None:
    """Run Phase 4: benchmark fused vs unfused models."""
    print("\n" + "=" * 70)
    print("  PHASE 4: Benchmarking Fused vs Unfused Models")
    print("=" * 70)

    # Check if the fusion pass is implemented (from Phase 2)
    model = StackedLinearReLU()
    traced = torch.fx.symbolic_trace(model)
    fused = apply_linear_relu_fusion(traced)

    # Quick check: if fusion wasn't implemented, the graph won't have changed
    original_nodes = len(list(torch.fx.symbolic_trace(StackedLinearReLU()).graph.nodes))
    fused_nodes = len(list(fused.graph.nodes))

    if original_nodes == fused_nodes:
        print("\n[HINT] It looks like the fusion pass from Phase 2 is not yet implemented.")
        print("       The benchmark will still run, but unfused and fused models will be identical.")
        print("       Complete Phase 2 first for meaningful benchmark results.\n")

    # Run the benchmark suite
    results = run_benchmark_suite()

    if len(results) == 0:
        print("\n[HINT] run_benchmark_suite() returned no results.")
        print("       Implement the TODO(human) functions:")
        print("         1. benchmark_model()     — use Timer to measure one model")
        print("         2. run_benchmark_suite()  — loop over batch sizes and models")
    else:
        print_comparison_table(results)

    # Key takeaways
    print("\n" + "-" * 70)
    print("KEY TAKEAWAYS:")
    print("-" * 70)
    print("""
  1. torch.utils.benchmark.Timer handles warmup, stats, and GPU sync correctly.
  2. blocked_autorange() automatically picks the right number of iterations.
  3. Fusion speedup is often modest on CPU but significant on GPU (2-3x for
     memory-bound ops) due to eliminated kernel launches and memory traffic.
  4. Batch size affects fusion benefit: larger batches = more data saved from
     unnecessary memory round-trips = bigger absolute speedup.
  5. Always benchmark with realistic batch sizes for your deployment scenario.
""")


if __name__ == "__main__":
    run_phase()
