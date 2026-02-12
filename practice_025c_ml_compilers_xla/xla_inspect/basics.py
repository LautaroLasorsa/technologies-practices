"""Phase 1: JAX Basics — jit, grad, jax.numpy fundamentals.

JAX is a numerical computing library that composes function transformations:
  - jax.jit:  compile a Python function with XLA
  - jax.grad: compute the gradient of a function
  - jax.vmap: automatically vectorize (batch) a function

Unlike PyTorch, JAX is *functional*: no mutable state, no nn.Module.
Parameters are explicit arrays passed as arguments.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Part 1: Simple function with jit and grad (fully implemented)
# ---------------------------------------------------------------------------

def simple_fn(x: jax.Array, y: jax.Array) -> jax.Array:
    """A simple function: dot product + ReLU applied element-wise.

    This is a pure function — no side effects, no mutable state.
    JAX requires functions to be pure for transformations to work.
    """
    return jnp.dot(x, y) + jnp.sum(jnp.maximum(x, 0.0))


def demo_jit_and_grad() -> None:
    """Show jit compilation and gradient computation on simple_fn."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4,))
    y = jax.random.normal(jax.random.PRNGKey(7), (4,))

    print("=" * 60)
    print("PHASE 1A: jit and grad on a simple function")
    print("=" * 60)

    # --- jit compilation ---
    # jax.jit traces the function once and compiles it to XLA.
    # The first call is slow (compilation), subsequent calls reuse the compiled code.
    jitted_fn = jax.jit(simple_fn)

    result_eager = simple_fn(x, y)
    result_jit = jitted_fn(x, y)

    print(f"\nx = {x}")
    print(f"y = {y}")
    print(f"simple_fn(x, y) [eager]  = {result_eager:.6f}")
    print(f"simple_fn(x, y) [jitted] = {result_jit:.6f}")
    print(f"Results match: {jnp.allclose(result_eager, result_jit)}")

    # --- grad ---
    # jax.grad computes the gradient of a *scalar-valued* function.
    # By default, it differentiates w.r.t. the first argument (argnums=0).
    grad_fn = jax.grad(simple_fn, argnums=0)
    grad_x = grad_fn(x, y)

    print(f"\ngrad(simple_fn) w.r.t. x = {grad_x}")
    print("  (This is d/dx [dot(x,y) + sum(relu(x))])")
    print("  = y + d/dx[sum(relu(x))]")
    print("  = y + (1 where x > 0, else 0)")

    # Also compute grad w.r.t. y
    grad_y_fn = jax.grad(simple_fn, argnums=1)
    grad_y = grad_y_fn(x, y)
    print(f"\ngrad(simple_fn) w.r.t. y = {grad_y}")
    print("  (This is d/dy [dot(x,y)] = x)")


# ---------------------------------------------------------------------------
# Part 2: 2-layer MLP as a pure function — TODO(human)
# ---------------------------------------------------------------------------

def init_mlp_params(key: jax.Array, input_dim: int, hidden_dim: int, output_dim: int) -> dict:
    """Initialize parameters for a 2-layer MLP.

    Returns a dict of params (JAX's convention for functional models):
        {
            "w1": jax.Array of shape (input_dim, hidden_dim),
            "b1": jax.Array of shape (hidden_dim,),
            "w2": jax.Array of shape (hidden_dim, output_dim),
            "b2": jax.Array of shape (output_dim,),
        }

    We use Glorot/Xavier initialization: scale = sqrt(2 / (fan_in + fan_out)).
    This is pre-implemented so you can focus on the forward pass.
    """
    k1, k2 = jax.random.split(key)
    scale1 = jnp.sqrt(2.0 / (input_dim + hidden_dim))
    scale2 = jnp.sqrt(2.0 / (hidden_dim + output_dim))
    return {
        "w1": scale1 * jax.random.normal(k1, (input_dim, hidden_dim)),
        "b1": jnp.zeros(hidden_dim),
        "w2": scale2 * jax.random.normal(k2, (hidden_dim, output_dim)),
        "b2": jnp.zeros(output_dim),
    }


def mlp_forward(params: dict, x: jax.Array) -> jax.Array:
    """Forward pass of a 2-layer MLP.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches JAX's functional programming style for neural networks.
    # Unlike PyTorch's nn.Module classes, JAX models are pure functions with explicit
    # parameters. This purity enables jit, grad, and vmap transformations.

    TODO(human): Implement the forward pass as a pure function.

    This is the core JAX pattern: models are just functions that take params
    explicitly. There is no self.weight or nn.Module — everything is a function
    argument. This is what makes jit/grad/vmap composable: they transform
    functions, and functions need all their inputs as arguments.

    Architecture:
        x -> Linear(w1, b1) -> ReLU -> Linear(w2, b2) -> output

    Steps:
        1. Compute the first linear layer: hidden = x @ params["w1"] + params["b1"]
           - jnp.dot(x, params["w1"]) performs the matrix multiply
           - Adding params["b1"] broadcasts the bias across the batch dimension
        2. Apply ReLU activation: hidden = jax.nn.relu(hidden)
           - jax.nn.relu is equivalent to jnp.maximum(hidden, 0)
           - This is the non-linearity that makes the network more than a linear map
        3. Compute the second linear layer: output = hidden @ params["w2"] + params["b2"]
        4. Return the output (no final activation — this is a logits/regression head)

    Why pure functions matter:
        - jax.grad(mlp_forward) computes d(output)/d(params) — only works if
          mlp_forward has no hidden state
        - jax.jit(mlp_forward) traces once, compiles to XLA — only works if the
          function is deterministic given its inputs
        - jax.vmap(mlp_forward, in_axes=(None, 0)) batches over x but shares
          params — only works if params flow through function args

    Args:
        params: dict with keys "w1", "b1", "w2", "b2" (from init_mlp_params)
        x: input array of shape (input_dim,) or (batch, input_dim)

    Returns:
        output array of shape (output_dim,) or (batch, output_dim)
    """
    # Stub: return zeros so the file runs without implementation
    if x.ndim == 1:
        return jnp.zeros(params["b2"].shape[0])
    return jnp.zeros((x.shape[0], params["b2"].shape[0]))


def mlp_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean squared error loss for the MLP.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches composing JAX functions. Loss functions are pure functions
    # that compose with model functions. This composability enables jax.grad to compute
    # gradients through the entire computation automatically.

    TODO(human): Implement the MSE loss using mlp_forward.

    This wraps the forward pass into a scalar-valued function so we can
    use jax.grad on it. jax.grad only works on functions that return a
    scalar (a single float), because gradients are defined as
    d(scalar_output) / d(each_parameter).

    Steps:
        1. Compute predictions: preds = mlp_forward(params, x)
        2. Compute MSE: loss = jnp.mean((preds - y) ** 2)
           - (preds - y) ** 2 gives per-element squared error
           - jnp.mean averages over all elements to get a scalar
        3. Return the scalar loss

    Why scalar output is required:
        - Gradient of a vector-valued function w.r.t. params gives a Jacobian
          (matrix), not a gradient (vector). JAX's grad expects scalar output.
        - For vector outputs, use jax.jacobian or jax.vjp/jax.jvp instead.

    Args:
        params: MLP parameters (from init_mlp_params)
        x: input array
        y: target array (same shape as mlp_forward output)

    Returns:
        Scalar MSE loss
    """
    # Stub: return a scalar zero so the file runs
    return jnp.float32(0.0)


def demo_mlp() -> None:
    """Demonstrate the MLP with jit and grad.

    TODO(human): After implementing mlp_forward and mlp_loss above,
    complete this function to:

    1. JIT-compile mlp_forward:
           jitted_mlp = jax.jit(mlp_forward)
       Then call it and compare with the eager (non-jit) version.

    2. Compute gradients of the loss w.r.t. params:
           grad_fn = jax.grad(mlp_loss, argnums=0)
           grads = grad_fn(params, x, y)
       argnums=0 means "differentiate w.r.t. the first argument (params)".
       The result `grads` has the same tree structure as `params`:
           grads["w1"].shape == params["w1"].shape  (gradient of loss w.r.t. w1)
           grads["b1"].shape == params["b1"].shape   etc.

    3. Print the gradient shapes and norms to verify they make sense:
           for name, g in grads.items():
               print(f"  grad[{name}]: shape={g.shape}, norm={jnp.linalg.norm(g):.6f}")

    Why this matters:
        - This is exactly how JAX training loops work: jit the forward pass
          for speed, grad the loss for parameter updates
        - The grads dict can be directly used for SGD: params[k] -= lr * grads[k]
        - Understanding this pattern is prerequisite for reading Jaxpr/HLO of
          training computations in Phase 2-3
    """
    key = jax.random.PRNGKey(0)
    params = init_mlp_params(key, input_dim=8, hidden_dim=32, output_dim=4)
    x = jax.random.normal(jax.random.PRNGKey(1), (8,))
    y = jax.random.normal(jax.random.PRNGKey(2), (4,))

    print("\n" + "=" * 60)
    print("PHASE 1B: 2-layer MLP (pure functional style)")
    print("=" * 60)

    # Eager forward pass
    output = mlp_forward(params, x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:       {output}")

    # JIT forward pass
    jitted_mlp = jax.jit(mlp_forward)
    output_jit = jitted_mlp(params, x)
    print(f"\nJIT output:   {output_jit}")
    print(f"Match: {jnp.allclose(output, output_jit)}")

    # Gradient computation
    loss_val = mlp_loss(params, x, y)
    print(f"\nMSE loss: {loss_val:.6f}")

    grad_fn = jax.grad(mlp_loss, argnums=0)
    grads = grad_fn(params, x, y)
    print("\nGradient shapes and norms:")
    for name, g in grads.items():
        print(f"  grad[{name}]: shape={g.shape}, norm={jnp.linalg.norm(g):.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 1 demos."""
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
    print()

    demo_jit_and_grad()
    demo_mlp()


if __name__ == "__main__":
    main()
