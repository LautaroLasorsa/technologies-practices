"""Phase 1: Classical Local Optimization with scipy.optimize.

Compares derivative-free (Nelder-Mead) and gradient-based (BFGS, L-BFGS-B)
local minimization methods on the Rosenbrock function. Demonstrates bounded
optimization and nonlinear constraint handling.

Key scipy.optimize functions used:
  - minimize(fun, x0, method=...) — unified interface to local optimizers
  - minimize(..., bounds=...) — box constraints (L-BFGS-B, trust-constr, SLSQP)
  - minimize(..., constraints=...) — nonlinear constraints (trust-constr, SLSQP)
"""

import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
import matplotlib.pyplot as plt


# ============================================================================
# Test functions
# ============================================================================

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2.

    Global minimum at (1, 1) with f = 0.
    The narrow curved valley makes this challenging for local optimizers.
    scipy provides `rosen` but we define it explicitly for clarity.
    """
    return float(rosen(x))


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of the Rosenbrock function.

    scipy provides `rosen_der` — we wrap it for type consistency.
    Gradient-based methods (BFGS, L-BFGS-B) use this for faster convergence.
    """
    return rosen_der(x)


# ============================================================================
# Convergence tracking
# ============================================================================

class ConvergenceTracker:
    """Callback object that records the optimization path and function values.

    Used with scipy.optimize.minimize's `callback` parameter.
    Each method calls the callback with the current iterate x_k after each step.
    """

    def __init__(self, func):
        self.func = func
        self.path = []
        self.values = []

    def __call__(self, xk, *args):
        """Called by minimize after each iteration with the current point xk."""
        self.path.append(xk.copy())
        self.values.append(self.func(xk))


# ============================================================================
# TODO(human): Unconstrained optimization with different methods
# ============================================================================

def optimize_unconstrained(
    x0: np.ndarray,
    method: str,
) -> tuple:
    """Minimize Rosenbrock from x0 using the specified method.

    Args:
        x0: Starting point, e.g. np.array([-1.5, 1.0])
        method: One of "Nelder-Mead", "BFGS", "L-BFGS-B"

    Returns:
        (result, tracker): scipy OptimizeResult and ConvergenceTracker with path/values
    """
    # TODO(human): Unconstrained Minimization with scipy.optimize.minimize
    #
    # Use scipy.optimize.minimize to minimize the Rosenbrock function.
    # The API is: minimize(fun, x0, method=method, jac=jac, callback=callback, options=options)
    #
    # Step 1: Create a ConvergenceTracker(rosenbrock) to record the optimization path.
    #
    # Step 2: Set up the `jac` parameter:
    #   - For gradient-based methods ("BFGS", "L-BFGS-B"), pass jac=rosenbrock_gradient
    #     This provides the analytical gradient, avoiding costly finite-difference approximation.
    #   - For derivative-free methods ("Nelder-Mead"), pass jac=None
    #     Nelder-Mead uses only function values — it doesn't need or use gradients.
    #
    # Step 3: Call minimize() with:
    #   - fun=rosenbrock
    #   - x0=x0
    #   - method=method
    #   - jac=jac (from step 2)
    #   - callback=tracker (from step 1)
    #   - options={"maxiter": 500, "disp": False}
    #
    # Step 4: Return (result, tracker) where result is the OptimizeResult object.
    #
    # The OptimizeResult has key attributes:
    #   result.x — optimal point found
    #   result.fun — function value at optimum
    #   result.nfev — number of function evaluations
    #   result.nit — number of iterations
    #   result.success — whether the optimizer converged
    #
    # Expected: BFGS converges in ~30-50 iterations, Nelder-Mead in ~100-200.
    # Both should find (1, 1) but BFGS is much more efficient due to gradient info.
    raise NotImplementedError("TODO(human): implement unconstrained optimization")


# ============================================================================
# TODO(human): Bounded optimization
# ============================================================================

def optimize_bounded(x0: np.ndarray) -> tuple:
    """Minimize Rosenbrock with box bounds using L-BFGS-B.

    Bounds: -2 <= x[0] <= 0.5, -2 <= x[1] <= 2
    Note: the global minimum (1,1) is OUTSIDE the x[0] bound, so the
    bounded optimum will be at x[0]=0.5 on the boundary.

    Args:
        x0: Starting point, e.g. np.array([-1.5, 1.0])

    Returns:
        (result, tracker): scipy OptimizeResult and ConvergenceTracker
    """
    # TODO(human): Bounded Optimization with L-BFGS-B
    #
    # L-BFGS-B is the standard method for optimization with simple box bounds.
    # Box bounds are specified as a list of (lower, upper) tuples, one per variable.
    #
    # Step 1: Define bounds as a list of (min, max) tuples:
    #   bounds = [(-2.0, 0.5), (-2.0, 2.0)]
    #   This restricts x[0] to [-2, 0.5] and x[1] to [-2, 2].
    #   The global minimum (1, 1) has x[0]=1 > 0.5, so it's infeasible.
    #   The bounded optimum will lie ON the x[0]=0.5 boundary.
    #
    # Step 2: Create a ConvergenceTracker for recording the path.
    #
    # Step 3: Call minimize() with:
    #   - method="L-BFGS-B"
    #   - bounds=bounds
    #   - jac=rosenbrock_gradient (L-BFGS-B uses gradients)
    #   - callback=tracker
    #   - options={"maxiter": 500, "disp": False}
    #
    # Step 4: Return (result, tracker).
    #
    # Key insight: L-BFGS-B projects the gradient onto the feasible set.
    # When a variable hits its bound, the corresponding gradient component
    # is zeroed out if it would push the variable further out of bounds.
    # This is called "gradient projection" — simple but effective for box bounds.
    #
    # The bounded solution should be near (0.5, 0.25) with f ≈ 0.25.
    # Compare: unbounded optimum is (1, 1) with f = 0.
    raise NotImplementedError("TODO(human): implement bounded optimization")


# ============================================================================
# TODO(human): Constrained optimization
# ============================================================================

def optimize_constrained(x0: np.ndarray) -> tuple:
    """Minimize Rosenbrock subject to a nonlinear inequality constraint.

    Constraint: x[0]^2 + x[1]^2 <= 1.5 (must lie inside a circle of radius sqrt(1.5))

    Args:
        x0: Starting point inside the feasible region, e.g. np.array([0.0, 0.0])

    Returns:
        (result, tracker): scipy OptimizeResult and ConvergenceTracker
    """
    # TODO(human): Constrained Optimization with trust-constr or SLSQP
    #
    # Nonlinear constraints in scipy.optimize are specified as dictionaries:
    #   {"type": "ineq", "fun": constraint_func}
    # where "ineq" means constraint_func(x) >= 0 (SLSQP convention).
    #
    # Step 1: Define the constraint function.
    #   We want x[0]^2 + x[1]^2 <= 1.5, which in SLSQP's "ineq" convention is:
    #     constraint_func(x) = 1.5 - (x[0]**2 + x[1]**2) >= 0
    #   So: constraint = {"type": "ineq", "fun": lambda x: 1.5 - x[0]**2 - x[1]**2}
    #
    # Step 2: Create a ConvergenceTracker.
    #
    # Step 3: Call minimize() with:
    #   - method="SLSQP"  (Sequential Least Squares Programming — handles nonlinear constraints)
    #   - constraints=[constraint]  (list of constraint dicts)
    #   - jac=rosenbrock_gradient
    #   - callback=tracker
    #   - options={"maxiter": 500, "disp": False}
    #
    # Step 4: Return (result, tracker).
    #
    # SLSQP works by solving a sequence of QP subproblems (similar to SQP methods).
    # At each step it linearizes constraints and approximates the Lagrangian Hessian.
    #
    # Alternative: method="trust-constr" supports both equality and inequality constraints
    # with a trust-region interior-point approach. For trust-constr, constraints use
    # the newer NonlinearConstraint/LinearConstraint objects instead of dicts.
    #
    # The constrained optimum will be near (1.0, 0.707) on the constraint boundary,
    # since the unconstrained minimum (1,1) has ||x||^2 = 2 > 1.5 (infeasible).
    raise NotImplementedError("TODO(human): implement constrained optimization")


# ============================================================================
# Visualization
# ============================================================================

def plot_rosenbrock_landscape(ax, xlim=(-2, 2), ylim=(-1, 3), levels=50):
    """Plot the Rosenbrock function as a contour map."""
    x = np.linspace(xlim[0], xlim[1], 300)
    y = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

    ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, levels), cmap="viridis", alpha=0.6)
    ax.plot(1, 1, "r*", markersize=15, label="Global min (1,1)")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")


def plot_convergence_paths(trackers: dict[str, ConvergenceTracker]):
    """Plot optimization paths on the Rosenbrock landscape."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: paths on contour plot
    plot_rosenbrock_landscape(ax1)
    colors = ["blue", "red", "green", "orange", "purple"]
    for i, (name, tracker) in enumerate(trackers.items()):
        if tracker.path:
            path = np.array(tracker.path)
            ax1.plot(path[:, 0], path[:, 1], "o-", color=colors[i % len(colors)],
                     markersize=3, linewidth=1, label=f"{name} ({len(path)} iters)")
    ax1.legend(fontsize=8)
    ax1.set_title("Optimization Paths on Rosenbrock")

    # Right: convergence curves (function value vs iteration)
    for i, (name, tracker) in enumerate(trackers.items()):
        if tracker.values:
            ax2.semilogy(tracker.values, color=colors[i % len(colors)], label=name)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x) [log scale]")
    ax2.set_title("Convergence Curves")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase1_convergence.png", dpi=120)
    print("\n  [Saved: phase1_convergence.png]")
    plt.close()


# ============================================================================
# Display helpers
# ============================================================================

def print_result(name: str, result, tracker: ConvergenceTracker) -> None:
    """Print optimization result summary."""
    print(f"\n  {name}:")
    print(f"    Converged: {result.success} ({result.message})")
    print(f"    Optimum:   x = [{result.x[0]:.6f}, {result.x[1]:.6f}]")
    print(f"    f(x):      {result.fun:.8f}")
    print(f"    Func evals: {result.nfev}")
    if hasattr(result, "nit") and result.nit is not None:
        print(f"    Iterations: {result.nit}")
    print(f"    Path points recorded: {len(tracker.path)}")


def print_comparison_table(results: list[tuple[str, object, ConvergenceTracker]]) -> None:
    """Print a comparison table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY: Method Comparison")
    print("=" * 80)
    print(f"  {'Method':<30s} {'f(x*)':>12s} {'x*[0]':>10s} {'x*[1]':>10s} {'nfev':>8s} {'nit':>6s}")
    print(f"  {'-'*76}")
    print(f"  {'True optimum':<30s} {'0.0':>12s} {'1.0':>10s} {'1.0':>10s} {'---':>8s} {'---':>6s}")
    for name, res, tracker in results:
        nit = str(res.nit) if hasattr(res, "nit") and res.nit is not None else "N/A"
        print(f"  {name:<30s} {res.fun:>12.6f} {res.x[0]:>10.6f} {res.x[1]:>10.6f} {res.nfev:>8d} {nit:>6s}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 1: Classical Local Optimization — scipy.optimize.minimize")
    print("=" * 80)

    x0 = np.array([-1.5, 1.0])
    print(f"\nStarting point: x0 = {x0}")
    print(f"Rosenbrock at x0: f(x0) = {rosenbrock(x0):.2f}")
    print(f"True minimum: f(1, 1) = 0.0")

    all_results = []
    all_trackers = {}

    # --- Unconstrained optimization with different methods ---
    print("\n" + "-" * 80)
    print("Unconstrained Optimization")
    print("-" * 80)

    for method in ["Nelder-Mead", "BFGS", "L-BFGS-B"]:
        result, tracker = optimize_unconstrained(x0, method)
        print_result(f"{method} (unconstrained)", result, tracker)
        all_results.append((f"{method} (uncons.)", result, tracker))
        all_trackers[method] = tracker

    # --- Bounded optimization ---
    print("\n" + "-" * 80)
    print("Bounded Optimization: -2 <= x[0] <= 0.5, -2 <= x[1] <= 2")
    print("-" * 80)

    result_bnd, tracker_bnd = optimize_bounded(x0)
    print_result("L-BFGS-B (bounded)", result_bnd, tracker_bnd)
    all_results.append(("L-BFGS-B (bounded)", result_bnd, tracker_bnd))
    all_trackers["L-BFGS-B (bounded)"] = tracker_bnd

    # --- Constrained optimization ---
    print("\n" + "-" * 80)
    print("Constrained Optimization: x[0]^2 + x[1]^2 <= 1.5")
    print("-" * 80)

    x0_constr = np.array([0.0, 0.0])  # start inside feasible region
    result_con, tracker_con = optimize_constrained(x0_constr)
    print_result("SLSQP (constrained)", result_con, tracker_con)
    all_results.append(("SLSQP (constrained)", result_con, tracker_con))
    all_trackers["SLSQP (constrained)"] = tracker_con

    # Verify constraint satisfaction
    x_opt = result_con.x
    print(f"    Constraint check: ||x||^2 = {x_opt[0]**2 + x_opt[1]**2:.4f} <= 1.5")

    # --- Summary ---
    print_comparison_table(all_results)

    # --- Visualization ---
    plot_convergence_paths(all_trackers)


if __name__ == "__main__":
    main()
