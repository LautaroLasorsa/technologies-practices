"""Phase 3: Full Bayesian Optimization Loop.

Implements the complete BO cycle: random initialization → fit GP → maximize
acquisition function → evaluate objective → update data → repeat.
Compares a manual BO implementation against scikit-optimize's gp_minimize
on benchmark functions.

Key components:
  - Latin Hypercube Sampling for initialization
  - scipy.optimize.minimize for acquisition function optimization (inner loop)
  - scikit-optimize gp_minimize for library comparison
"""

import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import matplotlib.pyplot as plt

# scikit-optimize for library comparison
from skopt import gp_minimize
from skopt.space import Real


# ============================================================================
# Test functions (standard BO benchmarks)
# ============================================================================

def branin(x: np.ndarray) -> float:
    """Branin-Hoo function: a standard 2D BO benchmark.

    Three global minima, all with f ≈ 0.397887:
      x* = (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    Domain: x0 in [-5, 10], x1 in [0, 15]
    """
    x0, x1 = x[0], x[1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return float(a * (x1 - b * x0**2 + c * x0 - r)**2 + s * (1 - t) * np.cos(x0) + s)


def six_hump_camel(x: np.ndarray) -> float:
    """Six-Hump Camel function: 2D function with six local minima.

    Two global minima with f ≈ -1.0316:
      x* = (0.0898, -0.7126), (-0.0898, 0.7126)
    Domain: x0 in [-3, 3], x1 in [-2, 2]
    """
    x0, x1 = x[0], x[1]
    return float((4 - 2.1 * x0**2 + x0**4 / 3) * x0**2 + x0 * x1 + (-4 + 4 * x1**2) * x1**2)


def forrester_2d(x: np.ndarray) -> float:
    """2D extension of Forrester: sum of two 1D Forrester functions.

    f(x) = f_1d(x0) + f_1d(x1), domain: [0, 1]^2
    Minimum near (0.757, 0.757).
    """
    def f1d(t):
        return (6 * t - 2) ** 2 * np.sin(12 * t - 4)
    return float(f1d(x[0]) + f1d(x[1]))


BENCHMARKS = {
    "Branin": {
        "func": branin,
        "bounds": [(-5.0, 10.0), (0.0, 15.0)],
        "true_min": 0.397887,
    },
    "Six-Hump Camel": {
        "func": six_hump_camel,
        "bounds": [(-3.0, 3.0), (-2.0, 2.0)],
        "true_min": -1.0316,
    },
    "Forrester 2D": {
        "func": forrester_2d,
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
        "true_min": -12.04,  # approximate
    },
}


# ============================================================================
# Acquisition function (EI, from phase 2 — provided here for completeness)
# ============================================================================

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement (EI) for minimization.

    Provided as a reference implementation for the BO loop.
    You implemented this from scratch in Phase 2.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = f_best - mu - xi
        Z = np.where(sigma > 1e-10, imp / sigma, 0.0)
        ei = np.where(sigma > 1e-10, imp * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)
    return np.maximum(ei, 0.0)


# ============================================================================
# Initialization helpers
# ============================================================================

def latin_hypercube_sample(bounds: list[tuple[float, float]], n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube Samples within the given bounds.

    Latin Hypercube Sampling (LHS) stratifies each dimension into n_samples equal
    intervals and ensures exactly one sample per interval per dimension. This gives
    better space coverage than pure random sampling, especially important for BO
    where the initial points determine the GP's first approximation.

    Returns: array of shape (n_samples, n_dims)
    """
    n_dims = len(bounds)
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)  # in [0, 1]^d

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return qmc.scale(samples, lower, upper)


# ============================================================================
# TODO(human): Complete Bayesian Optimization Loop
# ============================================================================

def bayesian_optimization_loop(
    objective,
    bounds: list[tuple[float, float]],
    n_init: int = 5,
    n_iter: int = 25,
    acquisition: str = "EI",
    seed: int = 42,
) -> dict:
    """Run a complete Bayesian Optimization loop from scratch.

    Args:
        objective: Black-box function f(x) -> float to minimize
        bounds: List of (lower, upper) bounds per dimension
        n_init: Number of initial random (LHS) evaluations
        n_iter: Number of BO iterations (GP fit + acquisition + evaluate)
        acquisition: Acquisition function to use ("EI")
        seed: Random seed

    Returns:
        Dictionary with keys:
          "X": all evaluated points (n_init + n_iter, n_dims)
          "y": all observed values (n_init + n_iter,)
          "best_y": running best (minimum) at each step
          "best_x": location of the running best at each step
    """
    # TODO(human): Implement the Full Bayesian Optimization Loop
    #
    # This is the core algorithm of Bayesian optimization. The loop alternates
    # between fitting a GP surrogate to all observed data and using an acquisition
    # function to select the next point to evaluate. The GP provides a cheap
    # approximation of the expensive objective, and the acquisition function
    # balances exploring uncertain regions vs exploiting promising ones.
    #
    # The complete algorithm:
    #
    # PHASE A — INITIALIZATION (lines of code: ~5)
    #   Step 1: Generate n_init initial points using latin_hypercube_sample(bounds, n_init, seed).
    #     LHS gives better coverage than pure random. For 2D with n_init=5, you get 5
    #     points spread across the domain with no two sharing the same row/column in the grid.
    #
    #   Step 2: Evaluate the objective at each initial point.
    #     X_observed = list of points evaluated so far (start with LHS points)
    #     y_observed = list of function values [objective(x) for x in X_observed]
    #
    #   Step 3: Track convergence. Initialize:
    #     best_y = [min(y_observed)]  — running minimum
    #     best_x = [X_observed[argmin(y_observed)]]  — location of running minimum
    #
    # PHASE B — BO LOOP (repeat n_iter times):
    #   Step 4: Convert X_observed and y_observed to numpy arrays for the GP.
    #     X_array = np.array(X_observed)  — shape (n_observed, n_dims)
    #     y_array = np.array(y_observed)  — shape (n_observed,)
    #
    #   Step 5: Fit a GP to (X_array, y_array).
    #     kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
    #     gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
    #                                    n_restarts_optimizer=5, normalize_y=True)
    #     gp.fit(X_array, y_array)
    #
    #   Step 6: Find x_next by maximizing the acquisition function.
    #     This is the "inner optimization" — it's cheap because it evaluates the GP, not f.
    #     Use scipy_minimize with L-BFGS-B from multiple random starting points:
    #
    #     f_best = min(y_observed)
    #     best_acq_val = infinity
    #     best_x_next = None
    #     for _ in range(20):  # 20 random restarts
    #         x0 = random point in bounds
    #         # Minimize NEGATIVE acquisition (because scipy minimizes, but we want max acq)
    #         def neg_acq(x):
    #             mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    #             return -expected_improvement(mu, sigma, f_best, xi=0.01)[0]
    #         result = scipy_minimize(neg_acq, x0, method="L-BFGS-B", bounds=bounds)
    #         if result.fun < best_acq_val:
    #             best_acq_val = result.fun
    #             best_x_next = result.x
    #
    #   Step 7: Evaluate the TRUE objective at x_next.
    #     y_next = objective(best_x_next)
    #
    #   Step 8: Update observed data.
    #     X_observed.append(best_x_next)
    #     y_observed.append(y_next)
    #
    #   Step 9: Update convergence tracking.
    #     current_best = min(y_observed)
    #     best_y.append(current_best)
    #     best_x.append(X_observed[argmin(y_observed)])
    #
    # Step 10: Return the results dictionary with all data.
    #
    # The beauty of this loop: the ONLY expensive operation is Step 7 (one function eval).
    # Steps 4-6 are cheap (GP fitting is O(n^3) but n is small, acquisition optimization
    # is fast because it queries the GP, not the real function). This is why BO is
    # sample-efficient: it does a LOT of cheap computation to make each expensive
    # evaluation as informative as possible.
    raise NotImplementedError("TODO(human): implement bayesian_optimization_loop")


# ============================================================================
# TODO(human): Compare with scikit-optimize
# ============================================================================

def compare_with_skopt(
    objective,
    bounds: list[tuple[float, float]],
    n_calls: int = 30,
    seed: int = 42,
) -> dict:
    """Run scikit-optimize's gp_minimize and return results for comparison.

    Args:
        objective: Black-box function f(x) -> float to minimize
        bounds: List of (lower, upper) bounds per dimension
        n_calls: Total number of function evaluations
        seed: Random seed

    Returns:
        Dictionary with keys:
          "X": all evaluated points
          "y": all observed values
          "best_y": running best at each step
    """
    # TODO(human): Run scikit-optimize's gp_minimize for Comparison
    #
    # scikit-optimize (skopt) is a production-ready BO library that wraps the
    # same GP + acquisition loop you implemented manually above. Comparing your
    # manual loop against gp_minimize reveals:
    #   - Whether your implementation converges at a similar rate
    #   - What engineering choices the library makes (initialization, kernel, acq function)
    #   - Where library implementations differ from textbook BO
    #
    # Step 1: Convert bounds to skopt format.
    #   skopt uses Real objects: [Real(low, high, name=f"x{i}") for i, (low, high) in enumerate(bounds)]
    #   OR you can pass bounds as a list of tuples directly — gp_minimize accepts both.
    #
    # Step 2: Call gp_minimize with these parameters:
    #   result = gp_minimize(
    #       func=objective,          — the objective function
    #       dimensions=dimensions,   — search space (list of Real or tuples)
    #       n_calls=n_calls,         — total function evaluations (init + BO iterations)
    #       n_initial_points=5,      — how many random points before BO kicks in
    #       acq_func="EI",           — acquisition function: "EI", "PI", "LCB" (=UCB)
    #       random_state=seed,       — for reproducibility
    #       noise=1e-10,             — assumed noise level (near-zero for deterministic functions)
    #   )
    #
    #   gp_minimize internally:
    #   a) Samples n_initial_points random points
    #   b) Fits a GP (Matern 5/2 kernel by default)
    #   c) Maximizes the acquisition function
    #   d) Evaluates the objective
    #   e) Repeats until n_calls total evaluations
    #
    # Step 3: Extract results from the OptimizeResult object.
    #   result.x_iters — list of all evaluated points (list of lists)
    #   result.func_vals — array of all observed values
    #   result.fun — best function value found
    #   result.x — best point found
    #
    # Step 4: Compute running best (cumulative minimum of func_vals).
    #   best_y = [min(result.func_vals[:i+1]) for i in range(len(result.func_vals))]
    #   Or more efficiently: np.minimum.accumulate(result.func_vals)
    #
    # Step 5: Return dict with "X", "y", "best_y" keys.
    #
    # Key differences between gp_minimize and your manual loop:
    #   - gp_minimize uses Cook et al.'s constant liar heuristic when n_initial_points
    #     evaluations are done (not LHS by default — uses Sobol or random)
    #   - It normalizes the input space internally
    #   - It uses a more sophisticated kernel hyperparameter optimization schedule
    #   - It may use a different acquisition function optimizer (not just L-BFGS-B restarts)
    raise NotImplementedError("TODO(human): implement compare_with_skopt")


# ============================================================================
# Visualization
# ============================================================================

def plot_convergence_comparison(
    manual_results: dict,
    skopt_results: dict,
    func_name: str,
    true_min: float,
) -> None:
    """Plot convergence curves: manual BO vs skopt."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Convergence: best value over iterations
    n_manual = len(manual_results["best_y"])
    n_skopt = len(skopt_results["best_y"])

    ax1.plot(range(n_manual), manual_results["best_y"], "b-o", markersize=3, label="Manual BO", linewidth=2)
    ax1.plot(range(n_skopt), skopt_results["best_y"], "r-s", markersize=3, label="skopt gp_minimize", linewidth=2)
    ax1.axhline(true_min, color="green", linestyle=":", alpha=0.7, label=f"True min = {true_min:.4f}")
    ax1.set_xlabel("Function evaluations")
    ax1.set_ylabel("Best f(x) found")
    ax1.set_title(f"{func_name}: Convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Regret: gap from true minimum
    regret_manual = np.array(manual_results["best_y"]) - true_min
    regret_skopt = np.array(skopt_results["best_y"]) - true_min

    ax2.semilogy(range(n_manual), np.maximum(regret_manual, 1e-10), "b-o", markersize=3, label="Manual BO", linewidth=2)
    ax2.semilogy(range(n_skopt), np.maximum(regret_skopt, 1e-10), "r-s", markersize=3, label="skopt gp_minimize", linewidth=2)
    ax2.set_xlabel("Function evaluations")
    ax2.set_ylabel("Simple regret (log scale)")
    ax2.set_title(f"{func_name}: Regret")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = func_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(f"phase3_{safe_name}.png", dpi=120)
    print(f"  [Saved: phase3_{safe_name}.png]")
    plt.close()


def plot_evaluated_points_2d(
    manual_results: dict,
    skopt_results: dict,
    func_name: str,
    bounds: list[tuple[float, float]],
    objective,
) -> None:
    """Plot evaluated points on a 2D contour of the objective."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Contour background
    x0 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x1 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X0, X1 = np.meshgrid(x0, x1)
    Z = np.array([[objective(np.array([xi, yi])) for xi, yi in zip(row0, row1)]
                   for row0, row1 in zip(X0, X1)])

    for ax, results, title in [(ax1, manual_results, "Manual BO"), (ax2, skopt_results, "skopt gp_minimize")]:
        ax.contourf(X0, X1, Z, levels=50, cmap="viridis", alpha=0.7)
        X = np.array(results["X"])
        n_init = 5
        # Initial points
        ax.scatter(X[:n_init, 0], X[:n_init, 1], c="white", s=40, edgecolors="black",
                   marker="o", label="Initial", zorder=5)
        # BO points
        ax.scatter(X[n_init:, 0], X[n_init:, 1], c="red", s=40, edgecolors="black",
                   marker="^", label="BO", zorder=5)
        # Best point
        best_idx = np.argmin(results["y"])
        ax.scatter(X[best_idx, 0], X[best_idx, 1], c="yellow", s=120, edgecolors="black",
                   marker="*", label=f"Best: {results['y'][best_idx]:.4f}", zorder=6)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title(f"{func_name}: {title}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    safe_name = func_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(f"phase3_{safe_name}_points.png", dpi=120)
    print(f"  [Saved: phase3_{safe_name}_points.png]")
    plt.close()


# ============================================================================
# Display helpers
# ============================================================================

def print_results_summary(name: str, results: dict, true_min: float) -> None:
    """Print a summary of BO results."""
    best_y = np.min(results["y"])
    best_idx = np.argmin(results["y"])
    best_x = results["X"][best_idx]
    n_evals = len(results["y"])
    regret = best_y - true_min

    x_str = ", ".join(f"{xi:.4f}" for xi in best_x)
    print(f"    {name}:")
    print(f"      Best f(x) = {best_y:.6f}  (true min: {true_min:.6f}, regret: {regret:.2e})")
    print(f"      Best x = [{x_str}]")
    print(f"      Total evaluations: {n_evals}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 3: Full Bayesian Optimization Loop")
    print("=" * 80)

    n_init = 5
    n_iter = 25
    n_calls = n_init + n_iter  # total evaluations for skopt

    for func_name, info in BENCHMARKS.items():
        print(f"\n{'=' * 80}")
        print(f"Benchmark: {func_name}")
        print(f"  Bounds: {info['bounds']}")
        print(f"  True minimum: {info['true_min']:.6f}")
        print(f"  Budget: {n_init} init + {n_iter} BO = {n_calls} total evaluations")
        print("=" * 80)

        func = info["func"]
        bounds = info["bounds"]
        true_min = info["true_min"]

        # --- Manual BO ---
        print("\n  Running manual BO loop...")
        t0 = time.perf_counter()
        manual_results = bayesian_optimization_loop(
            func, bounds, n_init=n_init, n_iter=n_iter, seed=42
        )
        dt_manual = time.perf_counter() - t0
        print_results_summary("Manual BO", manual_results, true_min)
        print(f"      Time: {dt_manual:.2f}s")

        # --- skopt ---
        print("\n  Running skopt gp_minimize...")
        t0 = time.perf_counter()
        skopt_results = compare_with_skopt(func, bounds, n_calls=n_calls, seed=42)
        dt_skopt = time.perf_counter() - t0
        print_results_summary("skopt gp_minimize", skopt_results, true_min)
        print(f"      Time: {dt_skopt:.2f}s")

        # --- Visualization ---
        plot_convergence_comparison(manual_results, skopt_results, func_name, true_min)
        plot_evaluated_points_2d(manual_results, skopt_results, func_name, bounds, func)

    # --- Overall summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Key takeaways:
  1. The manual BO loop and gp_minimize follow the SAME conceptual algorithm:
     init → fit GP → optimize acquisition → evaluate → repeat.
  2. Differences in convergence come from implementation details:
     - Initialization strategy (LHS vs Sobol vs random)
     - Kernel hyperparameter optimization (restarts, frequency)
     - Acquisition function optimization (restart count, optimizer)
  3. skopt is more polished but your manual implementation reveals the mechanics.
  4. On well-conditioned 2D problems, both should find near-optimal solutions
     within 30 evaluations — far fewer than random search or grid search.
  5. The GP fitting cost (O(n^3)) is negligible for n < 200 but becomes the
     bottleneck for very long optimization runs.
""")


if __name__ == "__main__":
    main()
