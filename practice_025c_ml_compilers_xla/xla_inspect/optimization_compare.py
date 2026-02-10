"""Phase 4: Optimization Analysis — compare un-optimized vs optimized HLO.

This module compares the HLO IR before and after XLA's optimization passes.
By counting instructions, identifying fusions, and diffing the two versions,
you can see exactly what the XLA compiler decided to do.

XLA optimization passes include:
  - **Operator fusion**: Merge element-wise ops, broadcasts, and reductions
    into single fused kernels. This is XLA's most impactful optimization —
    it eliminates memory round-trips between ops.
  - **Algebraic simplification**: Simplify expressions (e.g., x * 1 -> x,
    x + 0 -> x, broadcast of scalar -> constant).
  - **Constant folding**: Evaluate operations on constants at compile time
    instead of runtime (e.g., constant weights * constant inputs).
  - **Layout assignment**: Choose memory layouts (row-major vs column-major)
    for each tensor to optimize cache access patterns for the target hardware.
  - **Dead code elimination**: Remove ops whose results are never used.
  - **Common subexpression elimination (CSE)**: Reuse results of identical
    computations instead of computing them twice.

The comparison workflow:
    lowered = jax.jit(fn).lower(*args)
    pre_opt = lowered.as_text()           # Before XLA optimizations
    compiled = lowered.compile()
    post_opt = compiled.as_text()         # After XLA optimizations

    Compare pre_opt vs post_opt to see what changed.

References:
  - https://openxla.org/xla/operation_semantics
  - https://www.tensorflow.org/xla/developing_new_backend#xla_optimizations
"""

import jax
import jax.numpy as jnp

from xla_inspect.basics import init_mlp_params, mlp_forward, mlp_loss


# ---------------------------------------------------------------------------
# Helpers (fully implemented)
# ---------------------------------------------------------------------------

def extract_both_stages(fn: callable, *args: jax.Array) -> tuple[str, str]:
    """Extract both pre-optimization and post-optimization HLO text.

    Returns both stages from a single lowering, ensuring we compare
    the exact same computation.

    Args:
        fn: The function to compile.
        *args: Example arguments with correct shapes and dtypes.

    Returns:
        Tuple of (pre_optimization_text, post_optimization_text).
    """
    lowered = jax.jit(fn).lower(*args)
    pre_opt = lowered.as_text()
    compiled = lowered.compile()
    post_opt = compiled.as_text()
    return pre_opt, post_opt


def count_lines(text: str) -> int:
    """Count non-empty lines in a text dump.

    Args:
        text: The HLO text to count lines in.

    Returns:
        Number of non-empty, non-whitespace-only lines.
    """
    return sum(1 for line in text.strip().split("\n") if line.strip())


def find_instruction_lines(hlo_text: str) -> list[str]:
    """Extract lines that are HLO instructions (contain '=' and '%').

    Filters out metadata lines, comments, and module/computation headers,
    keeping only actual instruction lines for analysis.

    Args:
        hlo_text: The HLO text dump to parse.

    Returns:
        List of stripped instruction lines.
    """
    instructions = []
    for line in hlo_text.strip().split("\n"):
        stripped = line.strip()
        # HLO instructions have '%name = type opcode(...)' pattern
        if "=" in stripped and "%" in stripped and not stripped.startswith("//"):
            instructions.append(stripped)
    return instructions


def identify_fusions(hlo_text: str) -> list[str]:
    """Find all fusion instructions in an HLO text dump.

    Fusion instructions look like:
        %fusion.1 = f32[4]{0} fusion(...), kind=kOutput, ...
        %fusion.2 = f32[32]{0} fusion(...), kind=kLoop, ...

    Each fusion represents multiple ops merged into a single kernel.
    The 'kind' tells you how XLA implemented it:
      - kOutput: output fusion — writes directly to the output buffer
      - kLoop:   loop fusion — iterates over elements
      - kInput:  input fusion — fused into the consumer of this op

    Args:
        hlo_text: The HLO text dump to search.

    Returns:
        List of fusion instruction lines found.
    """
    fusions = []
    for line in hlo_text.strip().split("\n"):
        stripped = line.strip()
        if "fusion" in stripped.lower() and "=" in stripped and "%" in stripped:
            fusions.append(stripped)
    return fusions


def parse_instruction_types(hlo_text: str) -> dict[str, int]:
    """Parse and count instruction types from HLO text.

    Similar to count_hlo_instructions in hlo_dump.py, but with more robust
    parsing that handles edge cases in optimized HLO (which can have
    more complex instruction formats).

    Args:
        hlo_text: The HLO text dump to parse.

    Returns:
        Dict mapping instruction type -> count, sorted by count descending.
    """
    counts: dict[str, int] = {}
    for line in find_instruction_lines(hlo_text):
        # Extract RHS after '='
        if "=" not in line:
            continue
        rhs = line.split("=", 1)[1].strip()
        parts = rhs.split()
        for part in parts:
            # Skip type annotations
            if part.startswith(("f32", "f16", "bf16", "f64", "s32", "s64",
                                 "u32", "u64", "pred", "(")):
                continue
            opcode = part.split("(")[0].split(".")[0]
            if opcode and opcode.isalpha():
                counts[opcode] = counts.get(opcode, 0) + 1
                break
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def print_comparison_table(
    label: str,
    pre_counts: dict[str, int],
    post_counts: dict[str, int],
) -> None:
    """Print a side-by-side comparison of instruction counts.

    Shows pre-optimization vs post-optimization counts for each instruction
    type, along with the delta (how many were added/removed).

    Args:
        label: Description for the table header.
        pre_counts: Instruction counts before optimization.
        post_counts: Instruction counts after optimization.
    """
    all_ops = sorted(set(pre_counts) | set(post_counts))
    pre_total = sum(pre_counts.values())
    post_total = sum(post_counts.values())

    print(f"\n  Comparison: {label}")
    print(f"  {'Instruction':<25s} {'Pre':>5s} {'Post':>5s} {'Delta':>7s}")
    print(f"  {'─' * 44}")
    for op in all_ops:
        pre = pre_counts.get(op, 0)
        post = post_counts.get(op, 0)
        delta = post - pre
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {op:<25s} {pre:>5d} {post:>5d} {delta_str:>7s}")
    print(f"  {'─' * 44}")
    delta_total = post_total - pre_total
    delta_str = f"+{delta_total}" if delta_total > 0 else str(delta_total)
    print(f"  {'TOTAL':<25s} {pre_total:>5d} {post_total:>5d} {delta_str:>7s}")


# ---------------------------------------------------------------------------
# Part 1: Simple function optimization comparison (fully implemented)
# ---------------------------------------------------------------------------

def demo_simple_comparison() -> None:
    """Compare pre/post optimization for a simple function.

    Fully implemented as a reference. Study the output to see the
    comparison format and what kinds of changes XLA makes.
    """
    print("=" * 60)
    print("PHASE 4A: Optimization comparison — simple function (reference)")
    print("=" * 60)

    def simple_fn(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.dot(x, y) + jnp.sum(jnp.maximum(x, 0.0))

    x = jnp.zeros(8)
    y = jnp.zeros(8)

    pre_text, post_text = extract_both_stages(simple_fn, x, y)

    print(f"\n  Pre-optimization:  {count_lines(pre_text):>4d} lines")
    print(f"  Post-optimization: {count_lines(post_text):>4d} lines")

    pre_counts = parse_instruction_types(pre_text)
    post_counts = parse_instruction_types(post_text)
    print_comparison_table("simple_fn", pre_counts, post_counts)

    # Check for fusions
    fusions = identify_fusions(post_text)
    print(f"\n  Fusions found in optimized HLO: {len(fusions)}")
    for f in fusions:
        # Truncate long lines for readability
        display = f[:100] + "..." if len(f) > 100 else f
        print(f"    {display}")

    print("\n  Observations:")
    print("  - XLA may fuse element-wise ops (maximum + add + reduce) into one kernel")
    print("  - The dot (matrix multiply) typically stays as a separate instruction")
    print("  - Fewer post-opt instructions usually means more fusion happened")


# ---------------------------------------------------------------------------
# Part 2: MLP optimization comparison — TODO(human)
# ---------------------------------------------------------------------------

def demo_mlp_comparison() -> None:
    """Compare pre/post optimization HLO for the MLP forward pass.

    TODO(human): Extract both stages of the MLP compilation and analyze
    what XLA's optimization passes changed.

    This is where you see the compiler's decisions in action. The MLP has
    multiple opportunities for fusion:
      - dot + bias_add can fuse (matmul with integrated bias)
      - bias_add + relu can fuse (element-wise chain)
      - dot + bias_add + relu can all fuse into one kernel (ideal case)

    Whether XLA fuses these depends on the backend (CPU vs GPU), the
    tensor sizes, and XLA's cost model. Part of this exercise is seeing
    what YOUR system's XLA decides.

    Steps:

    1. Create inputs:

           key = jax.random.PRNGKey(0)
           params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
           x = jnp.zeros(8)

    2. Extract both pre- and post-optimization HLO:

           pre_text, post_text = extract_both_stages(mlp_forward, params, x)

    3. Print the line counts to see how much the optimizer compressed things:

           print(f"\\n  Pre-optimization:  {count_lines(pre_text):>4d} lines")
           print(f"  Post-optimization: {count_lines(post_text):>4d} lines")

       A significant reduction in lines indicates aggressive fusion/simplification.

    4. Parse and compare instruction types:

           pre_counts = parse_instruction_types(pre_text)
           post_counts = parse_instruction_types(post_text)
           print_comparison_table("MLP forward", pre_counts, post_counts)

       Look for:
         - Did "add" count decrease? (Fused into other ops)
         - Did "maximum" (ReLU) disappear? (Fused into a fusion kernel)
         - Did "fusion" appear? (XLA created fused kernels)
         - Did "broadcast" count change? (Layout optimization may eliminate broadcasts)

    5. Identify and display fusions:

           fusions = identify_fusions(post_text)
           print(f"\\n  Fusions in optimized MLP HLO: {len(fusions)}")
           for f in fusions:
               display = f[:100] + "..." if len(f) > 100 else f
               print(f"    {display}")

    6. Write a summary of what XLA decided:

           print("\\n  Summary of XLA optimizations on MLP forward:")
           print("  - Fusion: [describe what got fused]")
           print("  - Eliminated ops: [list ops that disappeared]")
           print("  - New ops: [list ops that appeared, e.g., 'fusion']")
           print(f"  - Total instruction reduction: {pre_total} -> {post_total}")

       This summary is the most valuable part — it's what you'd write in a
       performance report or debugging ticket when analyzing XLA behavior.

    What you'll learn:
        - XLA's fusion strategy on CPU (which ops it chooses to fuse)
        - How to read the comparison and identify optimization opportunities
        - The practical workflow: dump HLO -> count ops -> find fusions -> summarize
        - This is exactly the process ML engineers use to debug slow JAX models
    """
    print("\n" + "=" * 60)
    print("PHASE 4B: Optimization comparison — MLP forward pass")
    print("=" * 60)

    # Stub: prints a placeholder message until implemented
    print("\n  [TODO(human): Compare MLP HLO pre/post optimization]")
    print("  See the docstring above for detailed instructions.")


# ---------------------------------------------------------------------------
# Part 3: Gradient optimization comparison — TODO(human)
# ---------------------------------------------------------------------------

def demo_grad_comparison() -> None:
    """Compare pre/post optimization HLO for the MLP gradient computation.

    TODO(human): Repeat the optimization comparison for the gradient of
    mlp_loss. The gradient computation is larger and more complex — XLA
    has even more optimization opportunities here.

    The gradient computation includes both the forward pass AND the backward
    pass. This means:
      - All forward ops appear (dot, add, relu)
      - PLUS backward-specific ops (transpose, reduce for bias grad, etc.)
      - XLA may fuse forward and backward ops together (!)

    Steps:

    1. Create inputs including targets for the loss:

           key = jax.random.PRNGKey(0)
           params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
           x = jnp.zeros(8)
           y = jnp.zeros(4)

    2. Create the grad function:

           grad_loss = jax.grad(mlp_loss, argnums=0)

    3. Extract both stages:

           pre_text, post_text = extract_both_stages(grad_loss, params, x, y)

    4. Print line counts:

           print(f"\\n  Pre-optimization:  {count_lines(pre_text):>4d} lines")
           print(f"  Post-optimization: {count_lines(post_text):>4d} lines")

       Compare with the forward-only line counts from Part 2.
       The gradient should be significantly larger (2-3x more instructions).

    5. Parse, compare, and display:

           pre_counts = parse_instruction_types(pre_text)
           post_counts = parse_instruction_types(post_text)
           print_comparison_table("grad(mlp_loss)", pre_counts, post_counts)

    6. Identify fusions:

           fusions = identify_fusions(post_text)
           print(f"\\n  Fusions in optimized gradient HLO: {len(fusions)}")
           for f in fusions:
               display = f[:100] + "..." if len(f) > 100 else f
               print(f"    {display}")

    7. Write a combined summary comparing forward vs gradient optimization:

           print("\\n  Summary: Forward vs Gradient optimization")
           print("  - Forward: [X] fusions, [Y] total ops after optimization")
           print("  - Gradient: [X] fusions, [Y] total ops after optimization")
           print("  - XLA fused forward+backward ops: [yes/no, describe what you see]")
           print("  - Biggest optimization: [describe the most impactful change]")

    What you'll learn:
        - How XLA handles the combined forward+backward computation
        - Whether XLA can fuse across the forward/backward boundary
        - The relative complexity of gradient vs forward HLO
        - Practical skills for analyzing gradient computation performance
    """
    print("\n" + "=" * 60)
    print("PHASE 4C: Optimization comparison — gradient computation")
    print("=" * 60)

    # Stub: prints a placeholder message until implemented
    print("\n  [TODO(human): Compare gradient HLO pre/post optimization]")
    print("  See the docstring above for detailed instructions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run all Phase 4 demos."""
    print("JAX version:", jax.__version__)
    print()

    demo_simple_comparison()
    demo_mlp_comparison()
    demo_grad_comparison()


if __name__ == "__main__":
    run_phase()
