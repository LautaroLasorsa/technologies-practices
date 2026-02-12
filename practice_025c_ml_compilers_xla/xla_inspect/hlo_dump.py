"""Phase 3: HLO Dump & Reading — trigger XLA compilation and inspect HLO IR.

HLO (High-Level Optimizer) is XLA's intermediate representation. While Jaxpr
is JAX's view of the computation, HLO is what the XLA compiler actually
optimizes and compiles to machine code.

The modern JAX API for HLO inspection:

    lowered = jax.jit(fn).lower(*args)     # Stage 1: JAX -> StableHLO/HLO
    print(lowered.as_text())               # Jaxpr-level / StableHLO text
    compiled = lowered.compile()           # Stage 2: XLA optimizes HLO
    print(compiled.as_text())              # Optimized HLO text

The pipeline is: Python -> Jaxpr -> StableHLO -> HLO -> Optimized HLO -> Machine Code

Key HLO instructions you'll encounter:
  - dot:         Matrix multiplication (the workhorse of neural nets)
  - add:         Element-wise addition
  - maximum:     Element-wise max (used for ReLU)
  - broadcast:   Expand a tensor to match another's shape
  - parameter:   Input to the computation (weights, biases, activations)
  - fusion:      A group of ops fused into a single kernel (XLA's main optimization)
  - constant:    Literal values baked into the computation
  - tuple/get-tuple-element: Packing/unpacking multiple values

References:
  - https://openxla.org/xla/operation_semantics
  - https://jax.readthedocs.io/en/latest/aot.html
  - https://www.tensorflow.org/xla/architecture
"""

import jax
import jax.numpy as jnp

from xla_inspect.basics import init_mlp_params, mlp_forward, mlp_loss


# ---------------------------------------------------------------------------
# Helpers (fully implemented)
# ---------------------------------------------------------------------------

def extract_lowered_text(fn: callable, *args: jax.Array) -> str:
    """Lower a function to StableHLO/Jaxpr-level text without compiling.

    This calls jax.jit(fn).lower(*args).as_text(), which produces the
    intermediate representation BEFORE XLA optimization passes run.

    Args:
        fn: The function to lower (must be JIT-compatible — no side effects).
        *args: Example arguments with the correct shapes and dtypes.

    Returns:
        The StableHLO/Jaxpr-level text representation as a string.
    """
    lowered = jax.jit(fn).lower(*args)
    return lowered.as_text()


def extract_compiled_hlo(fn: callable, *args: jax.Array) -> str:
    """Compile a function with XLA and return the optimized HLO text.

    This calls jax.jit(fn).lower(*args).compile().as_text(), which produces
    the HLO AFTER all XLA optimization passes have run (fusion, layout
    assignment, algebraic simplification, constant folding, etc.).

    This is the "final" IR before machine code generation. Reading this
    tells you exactly what XLA decided to do.

    Args:
        fn: The function to compile.
        *args: Example arguments with the correct shapes and dtypes.

    Returns:
        The optimized HLO text representation as a string.
    """
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    return compiled.as_text()


def print_hlo_section(label: str, hlo_text: str, max_lines: int = 80) -> None:
    """Print an HLO text section with a labeled header, truncating if long.

    HLO dumps can be very long. This helper prints the first `max_lines`
    lines with a note if truncated, so the output stays readable.

    Args:
        label: Description shown in the header banner.
        hlo_text: The HLO text to print.
        max_lines: Maximum number of lines to show (default 80).
    """
    lines = hlo_text.strip().split("\n")
    print(f"\n{'─' * 60}")
    print(f"HLO: {label}")
    print(f"{'─' * 60}")
    for line in lines[:max_lines]:
        print(f"  {line}")
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} more lines, {len(lines)} total)")
    print(f"{'─' * 60}")


def count_hlo_instructions(hlo_text: str) -> dict[str, int]:
    """Count HLO instruction types in an HLO text dump.

    Parses HLO text lines to identify instruction opcodes. HLO instructions
    typically look like:

        %add.5 = f32[32]{0} add(f32[32]{0} %dot.3, f32[32]{0} %parameter.2)
        %fusion.1 = f32[4]{0} fusion(...), kind=kOutput, ...

    This function extracts the opcode (the word after the '=' and type info)
    and counts occurrences.

    Args:
        hlo_text: The HLO text dump to parse.

    Returns:
        Dict mapping instruction type -> count, sorted by count descending.
    """
    counts: dict[str, int] = {}
    for line in hlo_text.strip().split("\n"):
        line = line.strip()
        # HLO instructions have the pattern: %name = type opcode(...)
        # We look for lines containing '=' that are actual instructions
        if "=" in line and "%" in line:
            # Split on '=' and take the right side
            rhs = line.split("=", 1)[1].strip()
            # The opcode comes after the type, e.g., "f32[32]{0} add(...)"
            # Find the first word that looks like an opcode (not a type)
            parts = rhs.split()
            for part in parts:
                # Skip type annotations like f32[32]{0}, pred[], etc.
                if part.startswith(("f32", "f16", "bf16", "f64", "s32", "s64",
                                     "u32", "u64", "pred", "(")):
                    continue
                # The opcode might have parentheses attached
                opcode = part.split("(")[0].split(".")[0]
                if opcode and opcode.isalpha():
                    counts[opcode] = counts.get(opcode, 0) + 1
                    break
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def print_instruction_counts(label: str, counts: dict[str, int]) -> None:
    """Print a formatted table of HLO instruction counts.

    Args:
        label: Description for the table header.
        counts: Dict from count_hlo_instructions().
    """
    total = sum(counts.values())
    print(f"\n  HLO instruction counts ({label}) — {total} total:")
    for instr, count in counts.items():
        print(f"    {instr:30s} {count:>3d}")


# ---------------------------------------------------------------------------
# Part 1: HLO of simple functions (fully implemented reference)
# ---------------------------------------------------------------------------

def demo_simple_hlo() -> None:
    """Demonstrate HLO extraction on a simple function.

    This is fully implemented as a reference. Study the output to understand
    the HLO text format before moving to the MLP in Part 2.
    """
    print("=" * 60)
    print("PHASE 3A: HLO of a simple function (reference)")
    print("=" * 60)

    def simple_fn(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.dot(x, y) + jnp.sum(jnp.maximum(x, 0.0))

    x = jnp.zeros(8)
    y = jnp.zeros(8)

    # --- Pre-optimization (lowered but not compiled) ---
    lowered_text = extract_lowered_text(simple_fn, x, y)
    print_hlo_section("simple_fn — lowered (pre-optimization)", lowered_text)

    # --- Post-optimization (compiled by XLA) ---
    compiled_text = extract_compiled_hlo(simple_fn, x, y)
    print_hlo_section("simple_fn — compiled (post-optimization)", compiled_text)

    # Count and compare
    compiled_counts = count_hlo_instructions(compiled_text)
    print_instruction_counts("simple_fn compiled", compiled_counts)

    print("\n  Key observations:")
    print("  - The lowered text shows StableHLO — close to what JAX traced")
    print("  - The compiled text shows XLA's optimized HLO — may have fusions")
    print("  - 'fusion' instructions mean XLA merged multiple ops into one kernel")
    print("  - 'parameter' entries represent the function's inputs (x, y)")


# ---------------------------------------------------------------------------
# Part 2: HLO of the MLP — TODO(human)
# ---------------------------------------------------------------------------

def demo_mlp_hlo() -> None:
    """Trigger XLA compilation of the MLP and inspect the HLO output.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches reading HLO IR — XLA's internal representation after
    # optimization. HLO shows what the compiler actually compiles to machine code.
    # This is the #1 skill for debugging JAX/TPU performance issues.

    TODO(human): Use the JAX lowering/compilation API to extract HLO
    for the MLP forward pass, and then for the loss+gradient computation.
    Read the HLO text and identify key patterns.

    The MLP is more interesting than simple functions because XLA has more
    opportunities for optimization:
      - Fusing ReLU with the preceding matmul (dot + max -> one kernel)
      - Fusing bias addition with matmul (dot + add -> one kernel)
      - Layout optimization (choosing row-major vs column-major for each tensor)

    Steps:

    1. Set up dummy inputs (same shapes as in jaxpr_demo.py):

           key = jax.random.PRNGKey(0)
           params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
           x = jnp.zeros(8)

    2. Extract and print the LOWERED (pre-optimization) text:

           lowered_text = extract_lowered_text(mlp_forward, params, x)
           print_hlo_section("MLP forward — lowered", lowered_text)

       Study the output:
         - Each JAX operation maps to a StableHLO instruction
         - Look for "stablehlo.dot_general" — these are your matrix multiplications
         - Look for "stablehlo.maximum" — this is your ReLU
         - Look for "stablehlo.add" — these are your bias additions
         - Notice: no fusion yet — each operation is separate

    3. Extract and print the COMPILED (post-optimization) text:

           compiled_text = extract_compiled_hlo(mlp_forward, params, x)
           print_hlo_section("MLP forward — compiled (XLA optimized)", compiled_text)

       Study the output:
         - Look for "fusion" instructions — XLA merged multiple ops
         - Are dot+add+relu fused into a single fusion? (Common on CPU/GPU)
         - Look for "constant" instructions — did XLA fold any computations?
         - Count the total number of instructions — fewer means more fusion

    4. Count and display instruction types:

           compiled_counts = count_hlo_instructions(compiled_text)
           print_instruction_counts("MLP forward compiled", compiled_counts)

    5. BONUS — Extract HLO for the gradient computation:

           y = jnp.zeros(4)  # target for loss
           grad_loss = jax.grad(mlp_loss, argnums=0)
           grad_compiled = extract_compiled_hlo(grad_loss, params, x, y)
           print_hlo_section("grad(mlp_loss) — compiled", grad_compiled, max_lines=100)

           grad_counts = count_hlo_instructions(grad_compiled)
           print_instruction_counts("grad(mlp_loss) compiled", grad_counts)

       Compare with the forward-only HLO:
         - How many more fusions does the gradient have?
         - Are there new instruction types (transpose, reduce)?
         - How much bigger is the gradient HLO compared to forward-only?

    What to look for in the HLO output:
        - "fusion" with "kind=kOutput" or "kind=kLoop": these are XLA fusions.
          kOutput fusions write directly to the output buffer (efficient).
          kLoop fusions iterate over elements (for non-trivially fusible ops).
        - "parameter" entries: these are the inputs to the computation.
          For MLP forward, you'll see parameters for w1, b1, w2, b2, and x.
        - Layout annotations like "f32[32,8]{1,0}": the {1,0} means column-major
          layout (XLA chose this for the matmul). {0,1} would be row-major.
        - "ROOT" instruction: this is the final output of the computation.

    Why this matters:
        When your JAX model is slow, the first debugging step is:
            compiled = jax.jit(fn).lower(*args).compile()
            print(compiled.as_text())
        Then you read the HLO to see if XLA is fusing ops as expected,
        if there are unexpected copies or transposes, and if the layouts
        are efficient for your hardware.
    """
    print("\n" + "=" * 60)
    print("PHASE 3B: HLO of the MLP forward pass")
    print("=" * 60)

    # Stub: prints a placeholder message until implemented
    print("\n  [TODO(human): Extract and analyze MLP HLO]")
    print("  See the docstring above for detailed instructions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run all Phase 3 demos."""
    print("JAX version:", jax.__version__)
    print()

    demo_simple_hlo()
    demo_mlp_hlo()


if __name__ == "__main__":
    run_phase()
