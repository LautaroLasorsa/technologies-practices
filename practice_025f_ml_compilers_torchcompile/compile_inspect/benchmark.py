"""Phase 5: Benchmarking & Profiling — measure the real impact of torch.compile.

This phase answers the practical question: "Is torch.compile worth it for my model?"

### What to measure

1. **Compilation time.** How long does the first call take (compilation overhead)?
   This is a one-time cost that gets amortized over many inference calls.

2. **Inference latency.** How fast is each subsequent call? This is the steady-state
   performance that matters for serving.

3. **Speedup ratio.** Compiled inference time / eager inference time. Values > 1.0
   mean compilation helped; values < 1.0 mean it hurt.

4. **Break-even point.** How many inference calls until the compilation overhead is
   recovered? If compilation takes 10 seconds and each call saves 1ms, you need
   10,000 calls to break even.

### Factors that affect speedup

- **Model size.** Larger models have more fusion opportunities → bigger speedup.
  Tiny models (like our SimpleMLP) may see minimal or no speedup.

- **Batch size.** Larger batches keep the GPU busier → compilation optimizations
  matter more. Very small batches (1-4) are often memory-bound, not compute-bound.

- **Operation mix.** Models with many pointwise operations (ReLU, sigmoid, dropout,
  normalization) benefit most from fusion. Models that are mostly matmul-bound
  benefit less (matmul is already well-optimized in cuBLAS).

- **Device.** GPU sees much larger speedups than CPU, because:
  - GPU kernel launch overhead is significant → fusion reduces launches
  - GPU memory bandwidth is the bottleneck → fusion reduces memory traffic
  - Triton generates specialized GPU kernels → better than generic ones

- **Dynamic shapes.** If input shapes change between calls, Dynamo must recompile
  (guard failure). This negates the benefit unless you use dynamic=True.

### Warmup

Always do warmup runs before measuring! Reasons:
- First call triggers compilation (huge one-time cost)
- CUDA needs to initialize (first CUDA op is slow)
- CPU caches need to warm up
- PyTorch memory allocator does lazy initialization

Standard practice: 3-5 warmup calls, then measure over 50-200 calls.

### Platform note

On CPU (Windows), speedups will be modest (1.0x-1.3x typically). On GPU (Docker),
expect 1.5x-3x for well-structured models. Transformer-heavy models can see 2x-4x.
"""

import time

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# Models for benchmarking (fully provided)
# ─────────────────────────────────────────────────────────

class SmallMLP(nn.Module):
    """Small MLP — minimal fusion opportunities. Baseline comparison."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MediumTransformerBlock(nn.Module):
    """A single transformer encoder block — good fusion opportunities.

    Contains: LayerNorm, multi-head attention, feedforward, residual connections.
    This is the bread and butter of modern ML — representative of real workloads.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, ff_dim: int = 512) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (used in modern transformers like LLaMA, GPT-NeoX)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out  # Residual connection

        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out  # Residual connection

        return x


class PointwiseChain(nn.Module):
    """Many pointwise operations chained together — ideal for fusion.

    This is the model where torch.compile should shine the most: every operation
    is element-wise and can be fused into a single kernel.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x * torch.tanh(x)
        x = x + 0.5
        x = torch.relu(x)
        x = x * 2.0
        x = torch.log1p(x.abs())
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-6)
        return x


# ─────────────────────────────────────────────────────────
# Timing infrastructure (fully provided)
# ─────────────────────────────────────────────────────────

def measure_compilation_time(model: nn.Module, x: torch.Tensor, backend: str) -> float:
    """Measure the compilation time for a model with a given backend.

    Returns the first-call latency in milliseconds, which includes compilation.
    """
    torch._dynamo.reset()

    compiled = torch.compile(model, backend=backend)

    start = time.perf_counter()
    with torch.no_grad():
        _ = compiled(x)
    compilation_ms = (time.perf_counter() - start) * 1000.0

    torch._dynamo.reset()
    return compilation_ms


def measure_inference_time(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 100,
) -> float:
    """Measure steady-state inference latency in milliseconds.

    Runs warmup iterations first, then measures over num_runs and returns
    the average.

    For GPU: uses torch.cuda.synchronize() to ensure accurate timing.
    For CPU: uses time.perf_counter() directly.
    """
    use_cuda = x.is_cuda

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()

    # Measure
    times: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

    return sum(times) / len(times)


def get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_benchmark_backends() -> list[str]:
    """Return backends available for benchmarking.

    Tests each backend and returns only those that work.
    """
    backends: list[str] = []

    for backend in ["eager", "aot_eager", "inductor"]:
        try:
            model = nn.Linear(8, 8).eval().to(get_device())
            x = torch.randn(1, 8, device=get_device())
            compiled = torch.compile(model, backend=backend)
            with torch.no_grad():
                _ = compiled(x)
            backends.append(backend)
            torch._dynamo.reset()
        except Exception:
            torch._dynamo.reset()

    return backends


# ─────────────────────────────────────────────────────────
# TODO(human): Implement the benchmark suite
# ─────────────────────────────────────────────────────────

def run_benchmark_suite() -> None:
    """Benchmark eager vs compiled across models and batch sizes.

    TODO(human): Implement this function.

    This is the core exercise of Phase 5. You will measure compilation overhead,
    steady-state latency, and compute speedup ratios for different configurations.

    ### Steps to implement:

    1. **Setup.**
       - Get the device with `get_device()` and print it
       - Get available backends with `get_benchmark_backends()` and print them
       - Define batch sizes to test: `[1, 8, 32, 128]` (or adjust for your hardware)

    2. **Define the model configurations.**
       Create a list of (model_class, input_shape_fn, name) tuples:

       ```python
       configs = [
           (SmallMLP, lambda bs: (bs, 256), "SmallMLP"),
           (MediumTransformerBlock, lambda bs: (bs, 16, 256), "TransformerBlock"),
           (PointwiseChain, lambda bs: (bs, 256), "PointwiseChain"),
       ]
       ```

       The lambda takes a batch size and returns the input shape. This lets you
       test each model at different batch sizes.

    3. **For each model, run the benchmark.**
       For each (model_class, shape_fn, name) in configs:

       a) Print a section header for this model.

       b) For each batch_size in batch_sizes:
          - Create input: `x = torch.randn(*shape_fn(batch_size), device=device)`
          - Create model: `model = model_class().eval().to(device)`

          - Measure EAGER baseline:
            `eager_ms = measure_inference_time(model, x)`

          - For each backend in available_backends:
            - Measure compilation time:
              `compile_ms = measure_compilation_time(model, x, backend)`

            - Create compiled model and measure steady-state inference:
              ```python
              torch._dynamo.reset()
              compiled = torch.compile(model, backend=backend)
              # Warmup the compiled model (triggers compilation)
              with torch.no_grad():
                  for _ in range(5):
                      _ = compiled(x)
              compiled_ms = measure_inference_time(compiled, x)
              torch._dynamo.reset()
              ```

            - Compute speedup: `speedup = eager_ms / compiled_ms`
            - Compute break-even: `break_even = compile_ms / max(eager_ms - compiled_ms, 0.001)`

       c) Print a results table for this model. Example format:
          ```
            SmallMLP Results:
            Batch  Backend      Compile(ms)  Eager(ms)  Compiled(ms)  Speedup  Break-even
            ─────  ───────────  ───────────  ─────────  ────────────  ───────  ──────────
                1  eager              12.3       0.45          0.44    1.02x       ∞
                1  aot_eager          45.6       0.45          0.43    1.05x     2,171
                1  inductor          890.1       0.45          0.38    1.18x    12,716
                8  eager              11.8       0.52          0.51    1.02x       ∞
                8  inductor          850.3       0.52          0.35    1.49x     5,002
              ...
          ```

    4. **Print a final comparison summary.**
       After all models, print which model benefited most from compilation and
       at which batch size. Also note which backend gave the best speedup.

       Key observations to print:
       - "PointwiseChain benefits most because fusion eliminates N memory round-trips"
       - "TransformerBlock has moderate speedup — matmul dominates, already optimized"
       - "SmallMLP has minimal speedup — too few operations to amortize overhead"
       - "Larger batch sizes generally show better speedup (more work per kernel)"

    5. **Handle edge cases.**
       - If inductor is not available, only benchmark eager and aot_eager
       - If CUDA is not available, note that speedups will be smaller on CPU
       - If a benchmark fails for any reason, catch the exception, print a warning,
         and continue with the next configuration

    ### Key insights to look for:

    - **Compilation time is O(graph_size)**, not O(batch_size). Compiling once for
      batch_size=1 gives you the same compiled code for batch_size=128.

    - **Speedup increases with batch size** because the GPU is more fully utilized
      and the fixed overhead of kernel launches is amortized.

    - **PointwiseChain should show the biggest speedup** because ALL of its
      element-wise operations can be fused into a single kernel, eliminating
      multiple memory round-trips.

    - **TransformerBlock speedup depends on the attention implementation.** If
      PyTorch uses FlashAttention (2.2+), the attention is already fast; Inductor's
      contribution is mainly fusing the LayerNorm and feedforward pointwise ops.

    - **Break-even analysis matters.** For a serving system doing millions of
      inferences, even a small speedup is worth the compilation cost. For a
      one-off evaluation, the compilation overhead may not be worth it.

    Hint: Use f-strings with format specifiers for clean table formatting:
        f"{name:20s} {compile_ms:10.1f} {eager_ms:9.2f} {compiled_ms:12.2f} {speedup:7.2f}x"
    """
    # TODO(human): implement benchmark suite
    # Stub: print a placeholder message
    print("  [STUB] run_benchmark_suite() not yet implemented.")
    print("  Implement the TODO above to benchmark torch.compile speedup.")


# ─────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────

def run_phase() -> None:
    """Run Phase 5: Benchmarking & Profiling."""
    print("\n" + "#" * 60)
    print("  Phase 5: Benchmarking & Profiling")
    print("#" * 60)
    print()

    device = get_device()
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Running on CPU — speedups will be modest.")
        print(f"  For GPU benchmarks, use Docker.")
    print()

    backends = get_benchmark_backends()
    print(f"  Available backends: {backends}")
    print()

    run_benchmark_suite()


if __name__ == "__main__":
    run_phase()
