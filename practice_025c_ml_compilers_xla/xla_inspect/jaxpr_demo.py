"""Phase 2: Jaxpr Inspection — capture computation graphs with jax.make_jaxpr.

Jaxpr (JAX Program Representation) is JAX's intermediate representation (IR).
It's a functional, typed, first-order IR that represents computations as a
sequence of primitive operations (like dot_general, add, broadcast, max).

jax.make_jaxpr(fn)(*args) traces `fn` with abstract values (shapes + dtypes,
no real data) and returns a ClosedJaxpr object — a complete description of
the computation graph that JAX would send to XLA.

Key concepts:
  - **Primitives**: The atomic operations JAX knows about (add, mul, dot_general,
    reduce_sum, broadcast_in_dim, etc.). These map ~1:1 to XLA HLO ops.
  - **Variables**: Named intermediate values (a, b, c...) that flow between ops.
  - **Equations**: Each line `var = primitive[params] inputs` is one equation.
  - **Constvars**: Values captured from the enclosing scope (like closure variables).

Reading Jaxpr is the first step to understanding what XLA will compile.
The next step (Phase 3) inspects the actual HLO that XLA produces.

References:
  - https://jax.readthedocs.io/en/latest/jaxpr.html
  - https://jax.readthedocs.io/en/latest/jaxpr.html#understanding-jaxprs
"""

import jax
import jax.numpy as jnp

from xla_inspect.basics import init_mlp_params, mlp_forward, mlp_loss


# ---------------------------------------------------------------------------
# Helpers (fully implemented)
# ---------------------------------------------------------------------------

def print_jaxpr(label: str, jaxpr: jax.core.ClosedJaxpr) -> None:
    """Pretty-print a Jaxpr with a labeled header.

    Args:
        label: Description shown in the header banner.
        jaxpr: The ClosedJaxpr returned by jax.make_jaxpr(fn)(*args).
    """
    print(f"\n{'─' * 60}")
    print(f"Jaxpr: {label}")
    print(f"{'─' * 60}")
    print(jaxpr)
    print(f"{'─' * 60}")


def count_primitives(jaxpr: jax.core.ClosedJaxpr) -> dict[str, int]:
    """Count occurrences of each primitive operation in a Jaxpr.

    Walks the equations in the Jaxpr and tallies how many times each
    primitive appears. This is useful for comparing what operations
    different transformations (grad, vmap) introduce.

    Args:
        jaxpr: The ClosedJaxpr to analyze.

    Returns:
        Dict mapping primitive name -> count, sorted by count descending.

    Example output:
        {"dot_general": 2, "add": 2, "max": 1, "broadcast_in_dim": 1}
    """
    counts: dict[str, int] = {}
    for eqn in jaxpr.jaxpr.eqns:
        name = eqn.primitive.name
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def print_primitive_counts(label: str, counts: dict[str, int]) -> None:
    """Print a formatted table of primitive operation counts.

    Args:
        label: Description for the table header.
        counts: Dict from count_primitives().
    """
    total = sum(counts.values())
    print(f"\n  Primitive counts ({label}) — {total} total ops:")
    for prim, count in counts.items():
        print(f"    {prim:30s} {count:>3d}")


# ---------------------------------------------------------------------------
# Part 1: Jaxpr of simple functions (fully implemented reference)
# ---------------------------------------------------------------------------

def demo_simple_jaxpr() -> None:
    """Demonstrate jax.make_jaxpr on simple mathematical functions.

    This section is fully implemented as a reference. Study the output
    to understand how Jaxpr represents computations before moving to the
    MLP Jaxpr in Part 2.
    """
    print("=" * 60)
    print("PHASE 2A: Jaxpr of simple functions (reference)")
    print("=" * 60)

    # --- Example 1: Element-wise operations ---
    # f(x) = x^2 + 2*x + 1 = (x + 1)^2
    def quadratic(x: jax.Array) -> jax.Array:
        return x ** 2 + 2 * x + 1

    jaxpr_quad = jax.make_jaxpr(quadratic)(jnp.float32(0.0))
    print_jaxpr("quadratic(x) = x^2 + 2*x + 1", jaxpr_quad)
    print_primitive_counts("quadratic", count_primitives(jaxpr_quad))

    # Explanation of expected output:
    # You should see primitives like: integer_pow (for x^2), mul (for 2*x),
    # add (for the two additions). JAX traces the Python operations and records
    # each one as a primitive equation.

    # --- Example 2: Reduction ---
    def sum_of_squares(x: jax.Array) -> jax.Array:
        return jnp.sum(x ** 2)

    jaxpr_sos = jax.make_jaxpr(sum_of_squares)(jnp.zeros(4))
    print_jaxpr("sum_of_squares(x) = sum(x^2)", jaxpr_sos)
    print_primitive_counts("sum_of_squares", count_primitives(jaxpr_sos))

    # --- Example 3: Conditional / max ---
    def relu_manual(x: jax.Array) -> jax.Array:
        return jnp.maximum(x, 0.0)

    jaxpr_relu = jax.make_jaxpr(relu_manual)(jnp.zeros(4))
    print_jaxpr("relu(x) = maximum(x, 0)", jaxpr_relu)
    print_primitive_counts("relu", count_primitives(jaxpr_relu))

    # --- Example 4: dot product ---
    def dot_fn(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.dot(x, y)

    jaxpr_dot = jax.make_jaxpr(dot_fn)(jnp.zeros(4), jnp.zeros(4))
    print_jaxpr("dot(x, y)", jaxpr_dot)
    print_primitive_counts("dot", count_primitives(jaxpr_dot))

    print("\n  Key observations:")
    print("  - Each Python operation becomes one or more Jaxpr equations")
    print("  - Variables (a, b, c...) represent intermediate values")
    print("  - No actual data flows — only shapes and dtypes are tracked")
    print("  - This is how JAX 'records' your computation for XLA")


# ---------------------------------------------------------------------------
# Part 2: Jaxpr of the MLP forward pass — TODO(human)
# ---------------------------------------------------------------------------

def demo_mlp_jaxpr() -> None:
    """Capture and analyze the Jaxpr for the MLP forward pass.

    TODO(human): Use jax.make_jaxpr to capture the MLP's computation graph
    and analyze what primitive operations it contains.

    This is where things get interesting — the MLP forward pass involves:
      - dot_general: the core matrix multiplication primitive in JAX/XLA.
        Unlike np.dot which handles many cases, dot_general has explicit
        "contracting dimensions" and "batch dimensions" parameters.
      - add: bias addition, broadcasted across appropriate dimensions.
      - max: ReLU activation (jax.nn.relu compiles down to a max primitive).

    Steps:

    1. Create dummy inputs for make_jaxpr to trace with.
       jax.make_jaxpr doesn't execute the function — it traces it with
       abstract values that only carry shape and dtype information.
       So we need concrete shapes but the actual values don't matter:

           key = jax.random.PRNGKey(0)
           params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
           x_dummy = jnp.zeros(8)

    2. Capture the Jaxpr of the forward pass:

           jaxpr_fwd = jax.make_jaxpr(mlp_forward)(params, x_dummy)

       Note: make_jaxpr takes the function first, then you call the result
       with the arguments. It returns a ClosedJaxpr object.

    3. Print the Jaxpr using the print_jaxpr helper:

           print_jaxpr("MLP forward pass", jaxpr_fwd)

    4. Count and display the primitive operations:

           counts = count_primitives(jaxpr_fwd)
           print_primitive_counts("MLP forward", counts)

    5. Analyze the output — look for:
       - How many dot_general ops? (Should be 2 — one per linear layer)
       - How many add ops? (Should be 2 — one per bias addition)
       - What does the ReLU compile to? (max primitive)
       - How do the variable names (a, b, c...) show data flow?
       - What are the "constvars"? (The weight/bias arrays from params)

       Print your analysis:
           print("\\n  Analysis:")
           print(f"  - dot_general ops (linear layers): {counts.get('dot_general', 0)}")
           print(f"  - add ops (bias additions):        {counts.get('add', 0)}")
           print(f"  - max ops (ReLU activations):      {counts.get('max', 0)}")

    Why this matters:
        Reading the Jaxpr tells you exactly what JAX will hand off to XLA.
        If you see unexpected ops (e.g., extra transposes, redundant broadcasts),
        it means your code isn't compiling as efficiently as it could.
        This is the first tool you reach for when debugging JAX performance.
    """
    print("\n" + "=" * 60)
    print("PHASE 2B: Jaxpr of the MLP forward pass")
    print("=" * 60)

    # Stub: prints a placeholder message until implemented
    print("\n  [TODO(human): Capture and analyze MLP forward Jaxpr]")
    print("  See the docstring above for detailed instructions.")


# ---------------------------------------------------------------------------
# Part 3: Jaxpr of grad-transformed MLP — TODO(human)
# ---------------------------------------------------------------------------

def demo_grad_jaxpr() -> None:
    """Capture the Jaxpr of the gradient of the MLP loss.

    TODO(human): Use jax.make_jaxpr to see what the backward pass looks like
    as a Jaxpr. This reveals the primitives JAX uses for automatic
    differentiation.

    When you apply jax.grad to a function, JAX doesn't just record the
    forward pass — it also generates the backward pass (reverse-mode AD).
    The resulting Jaxpr contains BOTH the forward and backward computations,
    interleaved or sequenced depending on JAX's internal strategy.

    Steps:

    1. Create the grad-transformed function:

           grad_loss = jax.grad(mlp_loss, argnums=0)

       This creates a new function that, given (params, x, y), returns the
       gradient of mlp_loss w.r.t. params (the first argument, argnums=0).

    2. Create dummy inputs (same shapes as in demo_mlp_jaxpr):

           key = jax.random.PRNGKey(0)
           params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
           x_dummy = jnp.zeros(8)
           y_dummy = jnp.zeros(4)

    3. Capture the Jaxpr:

           jaxpr_grad = jax.make_jaxpr(grad_loss)(params, x_dummy, y_dummy)

    4. Print and count primitives:

           print_jaxpr("grad(mlp_loss) w.r.t. params", jaxpr_grad)
           counts_grad = count_primitives(jaxpr_grad)
           print_primitive_counts("grad(mlp_loss)", counts_grad)

    5. Compare with the forward-only Jaxpr from Part 2:
       - How many MORE dot_general ops? (Backward pass needs transposed matmuls)
       - Do you see "transpose" operations? (Transposing weight matrices for backprop)
       - Are there new primitives like "reduce_sum"? (Gradient accumulation for bias)
       - How many total ops vs the forward pass? (Typically 2-3x more)

       Print your comparison:
           print("\\n  Comparison with forward pass:")
           print(f"  - Forward had ~X total ops, grad has ~Y total ops")
           print(f"  - New primitives in grad: {set(counts_grad) - set(counts_fwd)}")

    What you'll learn:
        - jax.grad doesn't use finite differences or symbolic math — it builds
          a NEW computation graph that computes exact gradients via reverse-mode AD
        - The backward pass reuses intermediate values from the forward pass
          (that's why the Jaxpr may look interleaved)
        - dot_general appears more times because backprop through a linear layer
          requires transposed matrix multiplications
        - Understanding this is essential for reading the HLO of training
          computations in Phase 3
    """
    print("\n" + "=" * 60)
    print("PHASE 2C: Jaxpr of grad(mlp_loss)")
    print("=" * 60)

    # Stub: prints a placeholder message until implemented
    print("\n  [TODO(human): Capture and analyze gradient Jaxpr]")
    print("  See the docstring above for detailed instructions.")


# ---------------------------------------------------------------------------
# Part 4: Jaxpr of vmap-transformed MLP — TODO(human)
# ---------------------------------------------------------------------------

def demo_vmap_jaxpr() -> None:
    """Capture the Jaxpr of the vmap-batched MLP forward pass.

    TODO(human): Use jax.make_jaxpr to see how vmap transforms the
    computation graph to handle batched inputs.

    jax.vmap automatically vectorizes a function that works on single examples
    to work on batches. Instead of writing explicit batch dimensions in your
    code, you write the single-example version and let vmap handle batching.

    The key insight: vmap doesn't add a Python loop — it transforms the Jaxpr
    so that each primitive operates on an extra batch dimension. A dot_general
    on a vector becomes a dot_general on a matrix (batch of vectors).

    Steps:

    1. Create the vmap-transformed function:

           batched_mlp = jax.vmap(mlp_forward, in_axes=(None, 0))

       in_axes=(None, 0) means:
         - params (first arg): DON'T batch — same params for every example
           (None means "broadcast this argument, don't map over it")
         - x (second arg): batch along axis 0 — each row is one example
           (0 means "the first axis of this argument is the batch axis")

    2. Create dummy BATCHED inputs:

           key = jax.random.PRNGKey(0)
           params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
           x_batch_dummy = jnp.zeros((16, 8))  # batch of 16, each with 8 features

       Note: params stays the same (unbatched), but x now has a batch dimension.

    3. Capture the Jaxpr:

           jaxpr_vmap = jax.make_jaxpr(batched_mlp)(params, x_batch_dummy)

    4. Print and count:

           print_jaxpr("vmap(mlp_forward) — batched", jaxpr_vmap)
           counts_vmap = count_primitives(jaxpr_vmap)
           print_primitive_counts("vmap(mlp_forward)", counts_vmap)

    5. Compare with the unbatched forward Jaxpr from Part 2:
       - Same number of dot_general? Or different? (vmap may change the
         contraction dimensions in dot_general rather than adding new ops)
       - Any new broadcast or reshape operations?
       - How did the shapes change? (Look at the type annotations in the Jaxpr)

       Print your observations:
           print("\\n  vmap observations:")
           print("  - Number of primitives: same / more / fewer than unbatched?")
           print("  - How did dot_general parameters change?")
           print("  - Did vmap add any new operation types?")

    What you'll learn:
        - vmap is a *compiler transformation*, not a runtime loop
        - It changes the dimensionality of existing operations rather than
          adding loop-like constructs
        - This is why vmap'd code runs at the same speed as hand-batched code
        - Understanding vmap's Jaxpr helps you verify that batching is happening
          efficiently (no unexpected reshapes or copies)
    """
    print("\n" + "=" * 60)
    print("PHASE 2D: Jaxpr of vmap(mlp_forward) — batched")
    print("=" * 60)

    # Stub: prints a placeholder message until implemented
    print("\n  [TODO(human): Capture and analyze vmap Jaxpr]")
    print("  See the docstring above for detailed instructions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run all Phase 2 demos."""
    print("JAX version:", jax.__version__)
    print()

    demo_simple_jaxpr()
    demo_mlp_jaxpr()
    demo_grad_jaxpr()
    demo_vmap_jaxpr()


if __name__ == "__main__":
    run_phase()
