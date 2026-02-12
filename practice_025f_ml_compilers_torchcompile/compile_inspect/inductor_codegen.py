"""Phase 4: TorchInductor Code Inspection — see the generated Triton/C++ code.

TorchInductor is the backend compiler that turns FX graphs into executable code.
This is the "code generation" stage of the pipeline:

    FX Graph  →  TorchInductor  →  Triton kernels (GPU) or C++/OpenMP (CPU)

### What Inductor does

1. **Operator lowering.** Converts high-level PyTorch operations (nn.Linear, ReLU,
   LayerNorm) into lower-level "ATen" operations (matrix multiply, pointwise add,
   element-wise max, etc.).

2. **Fusion planning.** Analyzes the lowered graph to find operations that can be
   fused into a single kernel. Common fusion patterns:
   - **Pointwise fusion:** ReLU + bias_add + dropout → one kernel
   - **Reduction fusion:** softmax (max + subtract + exp + sum + divide) → one kernel
   - **MatMul + epilogue:** linear + ReLU → matmul kernel with fused activation

   Fusion is the #1 source of speedup because it eliminates memory round-trips.
   Without fusion, each operation reads from and writes to GPU global memory.
   With fusion, intermediate values stay in registers or shared memory.

3. **Code generation.** For each fused kernel, Inductor generates:
   - **GPU:** Triton kernel code (Python-like syntax, compiled to PTX by Triton)
   - **CPU:** C++ code with OpenMP pragmas for multi-threading

4. **Compilation.** The generated code is compiled:
   - Triton → PTX → CUDA binary (just-in-time)
   - C++ → shared library via gcc/clang (just-in-time)

### Inspecting generated code

There are several ways to see what Inductor generates:

1. **TORCH_LOGS="output_code"** — Environment variable that prints ALL generated
   code to stderr during compilation. Very verbose but shows everything.

2. **torch._inductor.config.debug = True** — Enables debug mode that writes
   generated code to files in /tmp/torchinductor_<user>/.

3. **torch.compiler.explain()** — Shows the compilation result including graph
   structure (but not the generated code itself).

### Platform note

- **CPU (Windows/Linux):** Inductor generates C++ code. Works on Windows if a C++
  compiler is available. The generated code uses OpenMP for parallelism.
- **GPU (Linux + CUDA):** Inductor generates Triton kernels. Requires Triton package
  (Linux only). The generated Triton code is the same language you learned in 025d.

### What to look for in generated code

- **Kernel boundaries:** Each `@triton.jit` decorated function (GPU) or each C++
  function (CPU) is one fused kernel.
- **Fused operations:** Look for multiple operations inside a single kernel. For
  example, a kernel that does `load → matmul → add_bias → relu → store` has fused
  linear + ReLU.
- **Memory access patterns:** Look for `tl.load` / `tl.store` (Triton) or pointer
  arithmetic (C++). Fewer loads/stores = better fusion.
- **Block size and grid:** The launch configuration tells you about parallelism.
"""

import os
import sys
import io
import logging
from contextlib import redirect_stderr

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# Models for code inspection (fully provided)
# ─────────────────────────────────────────────────────────

class FusionFriendlyModel(nn.Module):
    """A model designed to trigger interesting fusion patterns.

    The sequence Linear → LayerNorm → ReLU → Dropout is a common pattern in
    transformers. Inductor should fuse the pointwise operations (ReLU, LayerNorm
    normalization, dropout) into fewer kernels.

    Expected fusions:
    - LayerNorm decomposition (mean, var, normalize) → one reduction kernel
    - ReLU + dropout → one pointwise kernel (or fused into LayerNorm epilogue)
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc3 = nn.Linear(dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.norm1(self.fc1(x))))
        x = self.dropout(self.relu(self.norm2(self.fc2(x))))
        x = self.fc3(x)
        return x


class PointwiseHeavyModel(nn.Module):
    """A model with many pointwise operations — ideal fusion candidates.

    Each of these element-wise operations (sigmoid, tanh, add, mul) would
    normally require a separate kernel launch and memory round-trip.
    Inductor should fuse ALL of them into a single kernel.

    Without fusion: 6+ kernel launches, 6+ memory round-trips
    With fusion: 1-2 kernels, 1-2 memory accesses
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        # Many pointwise ops — perfect for fusion
        a = torch.sigmoid(x)
        b = torch.tanh(x)
        c = a * b              # Element-wise multiply
        d = c + x              # Residual connection
        e = torch.relu(d)
        f = e * 0.5            # Scale
        return f


class AttentionLikeModel(nn.Module):
    """A simplified attention-like pattern to see how Inductor handles it.

    This mimics Q @ K^T / sqrt(d) → softmax → @ V, the core of transformer
    attention. Inductor should recognize the softmax pattern and generate
    an efficient fused kernel.
    """

    def __init__(self, dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: Q @ K^T / sqrt(d_k)
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Weighted sum: attn @ V
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        return self.out(attn_output)


# ─────────────────────────────────────────────────────────
# Code capture helpers (fully provided)
# ─────────────────────────────────────────────────────────

def capture_inductor_output(
    model: nn.Module,
    x: torch.Tensor,
    model_name: str,
) -> str | None:
    """Compile a model with inductor and capture the generated code logs.

    Uses TORCH_LOGS="output_code" to capture the generated Triton/C++ code.
    Returns the captured log output as a string, or None if inductor is not available.

    The generated code is often hundreds of lines long. This function captures it
    and returns it so you can inspect and analyze it.
    """
    print(f"\n  --- Compiling {model_name} with inductor ---")

    torch._dynamo.reset()

    # Enable output code logging
    os.environ["TORCH_LOGS"] = "output_code"

    # Capture stderr where torch logs go
    captured = io.StringIO()

    try:
        # Configure logging to capture inductor output
        torch_logger = logging.getLogger("torch")
        handler = logging.StreamHandler(captured)
        handler.setLevel(logging.DEBUG)
        torch_logger.addHandler(handler)
        torch_logger.setLevel(logging.DEBUG)

        compiled = torch.compile(model, backend="inductor")

        with torch.no_grad():
            _ = compiled(x)

        torch_logger.removeHandler(handler)

        output = captured.getvalue()

        if output.strip():
            print(f"    Captured {len(output)} characters of generated code")
            return output
        else:
            print(f"    No output captured via logger. Try running with:")
            print(f"    TORCH_LOGS='output_code' python -m compile_inspect.inductor_codegen")
            return None

    except Exception as e:
        print(f"    [ERROR] Inductor compilation failed: {e}")
        print(f"    This is expected on Windows without Triton.")
        print(f"    Use Docker for full Inductor support.")
        return None

    finally:
        os.environ.pop("TORCH_LOGS", None)
        torch._dynamo.reset()


def try_inductor_available() -> bool:
    """Check if the inductor backend is available on this platform."""
    try:
        model = nn.Linear(8, 8).eval()
        x = torch.randn(1, 8)
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            _ = compiled(x)
        torch._dynamo.reset()
        return True
    except Exception:
        torch._dynamo.reset()
        return False


def print_code_summary(code: str, model_name: str) -> None:
    """Print a summary analysis of generated inductor code.

    Looks for patterns in the generated code to identify:
    - Number of kernel functions
    - Types of operations
    - Fusion patterns
    """
    print(f"\n  --- Code Summary for {model_name} ---")

    lines = code.split("\n")
    print(f"    Total lines: {len(lines)}")

    # Look for Triton kernel definitions
    triton_kernels = [l for l in lines if "@triton" in l.lower() or "def triton_" in l.lower()]
    if triton_kernels:
        print(f"    Triton kernels found: {len(triton_kernels)}")
        for k in triton_kernels[:5]:
            print(f"      {k.strip()}")

    # Look for C++ kernel definitions
    cpp_kernels = [l for l in lines if "extern_kernels" in l or "cpp_fused" in l]
    if cpp_kernels:
        print(f"    C++ kernel references: {len(cpp_kernels)}")
        for k in cpp_kernels[:5]:
            print(f"      {k.strip()}")

    # Look for fusion indicators
    fused_ops = [l for l in lines if "fused" in l.lower()]
    if fused_ops:
        print(f"    Fused operation references: {len(fused_ops)}")
        for f in fused_ops[:5]:
            print(f"      {f.strip()}")

    # Look for tl.load / tl.store (Triton memory operations)
    loads = sum(1 for l in lines if "tl.load" in l)
    stores = sum(1 for l in lines if "tl.store" in l)
    if loads or stores:
        print(f"    Memory operations: {loads} loads, {stores} stores")

    if not (triton_kernels or cpp_kernels or fused_ops):
        print(f"    (Could not identify kernel patterns in captured output)")
        print(f"    First 20 lines of captured output:")
        for line in lines[:20]:
            if line.strip():
                print(f"      {line}")


# ─────────────────────────────────────────────────────────
# TODO(human): Implement inductor code inspection
# ─────────────────────────────────────────────────────────

def run_inductor_inspection() -> None:
    """Inspect TorchInductor's generated code for different model patterns.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches inspecting generated Triton/C++ code from TorchInductor.
    # Understanding what code the compiler generates and which operations get fused
    # is essential for optimizing model performance with torch.compile.

    TODO(human): Implement this function.

    This is the core exercise of Phase 4. You will trigger inductor compilation
    on different models and analyze the generated code.

    ### Steps to implement:

    1. **Check inductor availability.**
       Call `try_inductor_available()`. If False, print a message explaining that
       inductor is not available (expected on Windows) and that the user should use
       Docker. Then return early.

       On CPU (without GPU), inductor generates C++ code instead of Triton. This
       is still interesting to inspect — you'll see loop nests, vectorization hints,
       and OpenMP pragmas.

    2. **Compile FusionFriendlyModel and inspect.**
       - Create `FusionFriendlyModel().eval()` and input `torch.randn(8, 256)`
       - Call `capture_inductor_output(model, x, "FusionFriendlyModel")`
       - If code was captured, call `print_code_summary(code, "FusionFriendlyModel")`
       - Print the first 50 lines of the generated code so you can read it
       - Count how many separate kernel functions you see

       Look for:
       - How many kernels were generated? (Fewer = more fusion)
       - Was ReLU fused into the LayerNorm kernel or the Linear kernel?
       - Can you identify the matmul operations vs pointwise operations?

    3. **Compile PointwiseHeavyModel and inspect.**
       - Create `PointwiseHeavyModel().eval()` and input `torch.randn(8, 256)`
       - Same capture + summary as above
       - This model has ~6 pointwise operations. How many kernels did Inductor produce?

       The key insight: Inductor should fuse ALL the pointwise ops (sigmoid, tanh,
       multiply, add, relu, scale) into a SINGLE kernel after the matmul. This means
       instead of 6 kernel launches with 6 memory round-trips, you get ONE kernel
       that reads once and writes once.

       Print how many separate kernels you see and whether the pointwise ops were
       fused.

    4. **Compile AttentionLikeModel and inspect.**
       - Create `AttentionLikeModel().eval()` and input `torch.randn(2, 16, 64)`
       - Same capture + summary
       - This is the most complex model with reshape, matmul, softmax, and transpose

       Look for:
       - How does Inductor handle the softmax? (Should be a reduction kernel)
       - Are the reshape/transpose operations reflected in the generated code, or
         are they folded into the memory access patterns of adjacent kernels?
       - How many total kernels for the attention computation?

    5. **Print a final summary.**
       For each model, print:
       - Model name
       - Number of kernels generated
       - Notable fusion patterns observed

       Example:
       ```
         Model                    Kernels  Notable Fusions
         FusionFriendlyModel          4    LayerNorm+ReLU fused, dropout fused
         PointwiseHeavyModel          2    All pointwise ops fused into 1 kernel
         AttentionLikeModel           6    Softmax fused, QKV reshape folded
       ```

    ### Alternative if capture fails

    If `capture_inductor_output` returns None (logger capture didn't work), you can
    still inspect the code by running from the command line with:

        TORCH_LOGS="output_code" python -m compile_inspect.inductor_codegen

    This prints directly to stderr. Inside Docker:

        docker compose run --rm torchcompile bash -c \
          "TORCH_LOGS='output_code' python -m compile_inspect.inductor_codegen"

    The TODO implementation should handle the None case gracefully (print the
    alternative instructions and continue).

    ### Key insights to look for:

    - **Pointwise fusion is aggressive.** Inductor fuses chains of element-wise
      operations almost always. This is the easiest and most impactful optimization.

    - **Reduction fusion is selective.** Operations like softmax (which require
      reduction across a dimension) are harder to fuse but Inductor handles the
      common patterns well.

    - **MatMul is NOT fused with pointwise.** Matrix multiplication uses specialized
      hardware (tensor cores on GPU) and is typically its own kernel. But the epilogue
      (bias + activation) CAN be fused into the matmul kernel.

    - **Shape operations often disappear.** View, reshape, transpose operations may
      not appear as separate kernels — instead, they change the memory access pattern
      of the adjacent kernel.
    """
    # TODO(human): implement inductor code inspection
    # Stub: print a placeholder message
    print("  [STUB] run_inductor_inspection() not yet implemented.")
    print("  Implement the TODO above to inspect TorchInductor generated code.")


# ─────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────

def run_phase() -> None:
    """Run Phase 4: TorchInductor Code Inspection."""
    print("\n" + "#" * 60)
    print("  Phase 4: TorchInductor Code Inspection")
    print("#" * 60)
    print()

    inductor_ok = try_inductor_available()
    if inductor_ok:
        print("  Inductor backend is available.")
        device = "CUDA (Triton)" if torch.cuda.is_available() else "CPU (C++)"
        print(f"  Code generation target: {device}")
    else:
        print("  Inductor backend is NOT available on this platform.")
        print("  Use Docker for full Inductor support (Triton GPU codegen).")
        print("  On Windows, try: docker compose run --rm torchcompile python -m compile_inspect.inductor_codegen")
    print()

    run_inductor_inspection()


if __name__ == "__main__":
    run_phase()
