"""Phase 1: torch.compile Basics — backends, compilation flow, and first-call overhead.

torch.compile() is PyTorch 2.x's main performance optimization API. It takes a model
(or any callable) and returns a compiled version that can run significantly faster.

The compilation pipeline has three major stages:

    Python code  →  TorchDynamo  →  FX Graph  →  Backend  →  Optimized code
                    (captures)      (IR)         (compiles)   (runs fast)

### Backends

torch.compile accepts a `backend` parameter that controls HOW the FX graph is compiled:

1. **"eager"** — Does nothing. Captures the graph with Dynamo but executes it in
   normal eager mode. Useful for testing that your model CAN be captured by Dynamo
   without actually changing execution. Think of it as a "dry run" for compilation.

2. **"aot_eager"** — "Ahead-of-time eager." Runs the AOTAutograd decomposition step
   (which breaks complex ops into simpler primitives and traces both forward AND
   backward graphs) but then executes the decomposed graph eagerly. Useful for
   debugging AOTAutograd issues without involving code generation.

3. **"inductor"** — The full compilation backend. Takes the FX graph, applies
   optimization passes (fusion, layout optimization, etc.), and generates:
   - **Triton kernels** for GPU (fused, parallel GPU code)
   - **C++/OpenMP code** for CPU (vectorized, threaded loops)
   This is where the real speedup comes from. Requires Triton on Linux/GPU.

### First-call overhead

The FIRST time you call a compiled model, torch.compile must:
1. Trace the Python code with TorchDynamo (bytecode analysis)
2. Decompose ops with AOTAutograd
3. Run the backend compiler (Inductor generates + compiles Triton/C++)
4. Cache the compiled artifact

This first call can take 5-30 seconds (or more for large models). Subsequent calls
with the same input shapes reuse the cached compiled code and are fast.

### Platform note

- **Windows:** `eager` and `aot_eager` work. `inductor` may work for CPU-only
  (generates C++ code) but Triton GPU codegen requires Linux.
- **Docker (Linux + GPU):** All backends work, including full Triton codegen.
"""

import time
from typing import Any

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# Model definition (fully provided)
# ─────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """A simple 3-layer MLP for demonstrating torch.compile.

    Architecture: Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)

    This is intentionally simple so that compilation overhead is visible relative
    to the actual computation time. With larger models, the compilation-to-runtime
    ratio improves significantly.
    """

    def __init__(self, input_dim: int = 784, hidden1: int = 256, hidden2: int = 128, output_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LargerModel(nn.Module):
    """A slightly larger model to make compilation benefits more visible.

    Architecture: 5 linear layers with ReLU activations and layer normalization.
    More operations = more fusion opportunities for the compiler.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ─────────────────────────────────────────────────────────
# Timing helpers (fully provided)
# ─────────────────────────────────────────────────────────

def time_inference(model: nn.Module, x: torch.Tensor, num_runs: int = 50) -> tuple[float, float]:
    """Time model inference, returning (first_call_ms, avg_subsequent_ms).

    The first call is measured separately because it includes compilation overhead
    for compiled models (or JIT warmup for eager mode).

    Returns:
        A tuple of (first_call_ms, average_of_remaining_calls_ms).
    """
    # First call (includes compilation for compiled models)
    torch._dynamo.reset()  # Clear any cached compilations
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    first_call_ms = (time.perf_counter() - start) * 1000.0

    # Subsequent calls (reuse cached compilation)
    times: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times.append((time.perf_counter() - start) * 1000.0)

    avg_ms = sum(times) / len(times) if times else 0.0
    return first_call_ms, avg_ms


def get_available_backends() -> list[str]:
    """Return the list of backends to test based on platform capabilities.

    On Windows, inductor may not work due to Triton dependency.
    We test it and fall back gracefully.
    """
    backends = ["eager", "aot_eager"]

    # Try inductor — it may work on CPU (C++ codegen) even on Windows
    try:
        test_model = SimpleMLP()
        test_input = torch.randn(1, 784)
        compiled = torch.compile(test_model, backend="inductor")
        with torch.no_grad():
            _ = compiled(test_input)
        backends.append("inductor")
        torch._dynamo.reset()
    except Exception as e:
        print(f"  [INFO] 'inductor' backend not available: {e}")
        print(f"  [INFO] This is expected on Windows. Use Docker for full Inductor support.")

    return backends


def verify_outputs_match(
    model: nn.Module,
    compiled_models: dict[str, nn.Module],
    x: torch.Tensor,
) -> None:
    """Verify that all compiled variants produce the same output as eager.

    This is a sanity check: compilation should NEVER change the numerical output
    (within floating-point tolerance). If outputs differ, something is wrong.
    """
    with torch.no_grad():
        reference = model(x)

    for name, compiled in compiled_models.items():
        with torch.no_grad():
            output = compiled(x)
        max_diff = (reference - output).abs().max().item()
        status = "PASS" if max_diff < 1e-5 else f"FAIL (max_diff={max_diff:.6f})"
        print(f"    {name:15s} vs eager: {status}")


# ─────────────────────────────────────────────────────────
# TODO(human): Implement backend comparison
# ─────────────────────────────────────────────────────────

def run_backend_comparison() -> None:
    """Compile a model with different backends, compare outputs and timing.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches torch.compile's compilation pipeline (Dynamo → backend).
    # Understanding backend differences (eager, aot_eager, inductor) and first-call
    # overhead is essential for using torch.compile effectively in production.

    TODO(human): Implement this function.

    This is the core exercise of Phase 1. You will:

    1. **Create the model and input tensor.**
       - Instantiate `SimpleMLP()` and put it in eval mode (`.eval()`)
       - Create a random input: `torch.randn(32, 784)` — batch of 32, MNIST-like input
       - Both on CPU (no `.cuda()` needed for this phase)

    2. **Compile the model with each available backend.**
       Call `get_available_backends()` to get the list of backends that work on your
       platform. Then for each backend:

           compiled = torch.compile(model, backend=backend_name)

       Store the compiled models in a dict: `{"eager": compiled_eager, "aot_eager": compiled_aot, ...}`

       IMPORTANT: `torch.compile` is LAZY — it doesn't actually compile anything until
       the first forward pass. The `torch.compile()` call itself just wraps the model.

    3. **Verify outputs match.**
       Call `verify_outputs_match(model, compiled_models, x)`.
       All backends should produce numerically identical results to the uncompiled model.
       If they don't, there's a bug (in PyTorch or your code).

    4. **Time each backend.**
       For EACH backend AND the original uncompiled model, call:

           first_ms, avg_ms = time_inference(model_variant, x)

       Print a table showing:
       - Backend name
       - First-call latency (includes compilation time)
       - Average subsequent-call latency
       - Speedup vs eager (for subsequent calls)

       Example output format:
       ```
         Backend          First (ms)   Avg (ms)   Speedup
         ─────────────    ──────────   ────────   ───────
         uncompiled            1.23       0.45      1.00x
         eager                15.67       0.44      1.02x
         aot_eager            45.89       0.42      1.07x
         inductor            890.12       0.31      1.45x
       ```

    5. **Print observations.**
       After the table, print a short summary of what you observe:
       - Which backend has the highest first-call overhead? (Should be inductor.)
       - Which backend gives the best subsequent-call performance? (Should be inductor.)
       - Is the speedup worth the compilation cost for this small model?

    ### Key insights to look for:

    - **eager backend** has almost zero first-call overhead because it doesn't really
      compile — it just validates that Dynamo CAN capture the graph.

    - **aot_eager** is slower on first call because AOTAutograd traces both forward
      and backward graphs, but subsequent calls are similar to eager.

    - **inductor** has the highest first-call cost (it generates and compiles Triton/C++
      code) but the fastest subsequent calls due to operator fusion and codegen.

    - For this tiny MLP, the speedup may be small or even negative on CPU. The real
      benefits show on GPU with larger models and batch sizes.

    Hint: Remember to call `torch._dynamo.reset()` between backend tests to clear
    the compilation cache. The `time_inference` helper already does this internally.
    """
    # TODO(human): implement backend comparison
    # Stub: print a placeholder message
    print("  [STUB] run_backend_comparison() not yet implemented.")
    print("  Implement the TODO above to compare torch.compile backends.")


# ─────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────

def run_phase() -> None:
    """Run Phase 1: torch.compile Basics."""
    print("\n" + "#" * 60)
    print("  Phase 1: torch.compile Basics")
    print("#" * 60)
    print()

    print("  Available backends on this platform:")
    backends = get_available_backends()
    for b in backends:
        print(f"    - {b}")
    print()

    torch._dynamo.reset()
    run_backend_comparison()


if __name__ == "__main__":
    run_phase()
