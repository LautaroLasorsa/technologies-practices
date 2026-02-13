"""Phase 2: Global Optimization with scipy.optimize.

Applies four global optimization algorithms to multimodal test functions
where local methods fail. Compares differential evolution, dual annealing,
SHGO, and basin-hopping on Rastrigin, Ackley, and Schwefel functions.

Key scipy.optimize functions used:
  - differential_evolution(func, bounds) — population-based (DE/rand/1/bin)
  - dual_annealing(func, bounds) — generalized simulated annealing + local search
  - shgo(func, bounds) — simplicial homology global optimization (finds ALL minima)
  - basinhopping(func, x0) — random perturbation + local minimization
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, shgo, basinhopping
import matplotlib.pyplot as plt


# ============================================================================
# Multimodal test functions
# ============================================================================

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)).

    Global minimum: f(0, 0, ..., 0) = 0.
    Highly multimodal: regular grid of local minima separated by ~1 unit.
    The cosine term creates "bumps" that trap local optimizers.
    Search domain: typically [-5.12, 5.12]^n.
    """
    x = np.asarray(x)
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function: a nearly flat outer region with a large hole at the center.

    Global minimum: f(0, 0, ..., 0) = 0.
    The flat outer region fools gradient-based methods (near-zero gradient far from origin).
    The exponential terms create a deep well near the origin.
    Search domain: typically [-5, 5]^n.

    f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e
    """
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x**2) / n
    sum_cos = np.sum(np.cos(2 * np.pi * x)) / n
    return float(-20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e)


def schwefel(x: np.ndarray) -> float:
    """Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|))).

    Global minimum: f(420.9687, ..., 420.9687) ≈ 0.
    Deceptive: the global minimum is geometrically far from the next-best local minima.
    Local search starting from near the origin will converge to a bad local min.
    Search domain: typically [-500, 500]^n.
    """
    x = np.asarray(x)
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


# Named collection for iteration
TEST_FUNCTIONS = {
    "Rastrigin": {
        "func": rastrigin,
        "bounds": [(-5.12, 5.12)] * 2,
        "true_min": 0.0,
        "true_x": [0.0, 0.0],
    },
    "Ackley": {
        "func": ackley,
        "bounds": [(-5.0, 5.0)] * 2,
        "true_min": 0.0,
        "true_x": [0.0, 0.0],
    },
    "Schwefel": {
        "func": schwefel,
        "bounds": [(-500.0, 500.0)] * 2,
        "true_min": 0.0,
        "true_x": [420.9687, 420.9687],
    },
}


# ============================================================================
# TODO(human): Differential Evolution
# ============================================================================

def optimize_differential_evolution(
    func,
    bounds: list[tuple[float, float]],
    seed: int = 42,
) -> object:
    """Run differential evolution on the given function.

    Args:
        func: Objective function f(x) -> float
        bounds: List of (lower, upper) bounds per dimension
        seed: Random seed for reproducibility

    Returns:
        scipy.optimize.OptimizeResult with .x, .fun, .nfev, etc.
    """
    # TODO(human): Differential Evolution — Population-Based Global Optimizer
    #
    # Differential Evolution (DE) maintains a POPULATION of candidate solutions.
    # For each member, a mutant is created by adding weighted differences of
    # randomly selected members. The mutant is crossed with the original
    # (binomial crossover) to create a trial. If the trial is better, it replaces
    # the original. This is called "DE/rand/1/bin" (the default strategy).
    #
    # Call: differential_evolution(func, bounds, **kwargs)
    #
    # Key parameters to set:
    #   - strategy="best1bin"  — uses the current best vector in mutation.
    #     Alternatives: "rand1bin" (more explorative), "currenttobest1bin" (balanced).
    #     "best1bin" converges faster but may miss the global optimum on very rugged landscapes.
    #
    #   - maxiter=1000  — maximum number of generations (default 1000).
    #     Each generation evaluates popsize candidates.
    #     Total function evaluations ≈ maxiter * popsize * len(bounds).
    #
    #   - tol=1e-8  — convergence tolerance. DE stops when the population's
    #     function value spread is below tol for a number of generations.
    #
    #   - seed=seed  — for reproducibility. DE is stochastic.
    #
    #   - polish=True  — (default True) after DE converges, run L-BFGS-B from
    #     the best point to refine the solution. This "polishing" step uses
    #     gradient info to squeeze out the last few digits of accuracy.
    #
    # The population size is auto-calculated as max(5, 15 * len(bounds)).
    # For 2D problems, popsize ≈ 30. Each generation does popsize evaluations.
    #
    # Return the OptimizeResult directly.
    raise NotImplementedError("TODO(human): implement differential_evolution call")


# ============================================================================
# TODO(human): Dual Annealing
# ============================================================================

def optimize_dual_annealing(
    func,
    bounds: list[tuple[float, float]],
    seed: int = 42,
) -> object:
    """Run dual annealing on the given function.

    Args:
        func: Objective function f(x) -> float
        bounds: List of (lower, upper) bounds per dimension
        seed: Random seed for reproducibility

    Returns:
        scipy.optimize.OptimizeResult
    """
    # TODO(human): Dual Annealing — Generalized Simulated Annealing
    #
    # dual_annealing combines classical simulated annealing with local search.
    # It uses a distorted Cauchy-Lorentz visiting distribution (heavier tails
    # than Gaussian) to make large jumps and escape deep basins.
    #
    # Call: dual_annealing(func, bounds, **kwargs)
    #
    # Key parameters to set:
    #   - maxiter=1000  — maximum number of global iterations.
    #     Each global iteration may include many function evaluations.
    #
    #   - initial_temp=5230.0  — starting temperature (default 5230).
    #     Higher temperatures → more exploration at the start.
    #     The temperature decreases according to the visiting distribution.
    #     Range: (0.01, 5e4].
    #
    #   - visit=2.62  — controls the tail heaviness of the visiting distribution.
    #     Values near 1 → Gaussian-like (short jumps).
    #     Values near 3 → very heavy tails (long jumps, more global exploration).
    #     Default 2.62 is a good balance.
    #
    #   - restart_temp_ratio=2e-5  — when temperature drops to
    #     initial_temp * restart_temp_ratio, reannealing occurs (temperature resets).
    #     This prevents premature convergence by periodically reheating.
    #
    #   - no_local_search=False  — (default) after SA converges, run a local
    #     optimizer to refine the solution. Set True to disable.
    #
    #   - seed=seed  — for reproducibility.
    #
    # The algorithm alternates between:
    #   1. Generalized SA: sample from distorted Cauchy-Lorentz, accept worse
    #      solutions with Boltzmann probability exp(-delta/T).
    #   2. Local search: refine the best point found using L-BFGS-B.
    #
    # Return the OptimizeResult directly.
    raise NotImplementedError("TODO(human): implement dual_annealing call")


# ============================================================================
# TODO(human): SHGO (Simplicial Homology Global Optimization)
# ============================================================================

def optimize_shgo(
    func,
    bounds: list[tuple[float, float]],
) -> object:
    """Run SHGO on the given function.

    Args:
        func: Objective function f(x) -> float
        bounds: List of (lower, upper) bounds per dimension

    Returns:
        scipy.optimize.OptimizeResult (with .xl and .funl listing ALL local minima)
    """
    # TODO(human): SHGO — Simplicial Homology Global Optimization
    #
    # SHGO is fundamentally different from DE and dual_annealing. Instead of
    # stochastic search, it uses TOPOLOGICAL methods to decompose the search
    # space into simplicial complexes and systematically find ALL local minima.
    #
    # Call: shgo(func, bounds, **kwargs)
    #
    # Key parameters to set:
    #   - n=128  — number of sampling points used to construct the simplicial
    #     complex. More points → better coverage of the landscape but more
    #     function evaluations. Default is 100.
    #
    #   - iters=3  — number of iterations of the algorithm. Each iteration
    #     refines the simplicial complex. More iterations → more thorough
    #     search of the landscape.
    #
    #   - sampling_method="sobol"  — how to generate the initial sampling points.
    #     "sobol" uses Sobol' quasi-random sequences (low-discrepancy) for
    #     better space coverage than pure random sampling.
    #     Alternative: "simplicial" uses the Delaunay triangulation directly.
    #
    # SHGO's unique feature: the result object has TWO extra attributes:
    #   result.xl  — list of ALL local minima found (not just the global one)
    #   result.funl — function values at each local minimum
    # No other scipy global optimizer provides this!
    #
    # SHGO works best for:
    #   - Low-dimensional problems (n <= ~10)
    #   - Lipschitz-continuous functions
    #   - When you need ALL local minima, not just the global one
    #
    # Return the OptimizeResult directly.
    raise NotImplementedError("TODO(human): implement shgo call")


# ============================================================================
# TODO(human): Basin-Hopping
# ============================================================================

def optimize_basinhopping(
    func,
    x0: np.ndarray,
    seed: int = 42,
) -> object:
    """Run basin-hopping from a starting point.

    Args:
        func: Objective function f(x) -> float
        x0: Starting point
        seed: Random seed for reproducibility

    Returns:
        scipy.optimize.OptimizeResult
    """
    # TODO(human): Basin-Hopping — Perturbation + Local Minimization
    #
    # Basin-hopping transforms the energy landscape into a collection of
    # "basins" by mapping each point to its local minimum. It then hops
    # between basins via random perturbation + Metropolis acceptance:
    #
    #   1. Start at local minimum x*
    #   2. Perturb: x_trial = x* + random_step
    #   3. Locally minimize from x_trial to find a new local minimum x_trial*
    #   4. Accept x_trial* as new starting point if:
    #      - f(x_trial*) < f(x*), OR
    #      - with probability exp(-(f(x_trial*) - f(x*)) / T)  (Metropolis)
    #   5. Repeat
    #
    # Call: basinhopping(func, x0, **kwargs)
    #
    # Key parameters to set:
    #   - niter=100  — number of basin-hopping iterations (each includes a
    #     local minimization). Total function evals = niter * (local_min_evals).
    #
    #   - T=1.0  — temperature for the Metropolis criterion.
    #     Higher T → more likely to accept uphill moves → more exploration.
    #     Lower T → only accept improvements → more exploitation.
    #     Typical range: 0.5 to 5.0 depending on the function's scale.
    #
    #   - stepsize=0.5  — magnitude of the random perturbation at each hop.
    #     Too small → gets stuck in nearby basins.
    #     Too large → jumps randomly without exploiting landscape structure.
    #     Rule of thumb: ~10-25% of the domain width.
    #
    #   - seed=seed  — for reproducibility.
    #
    #   - minimizer_kwargs={"method": "L-BFGS-B"}  — options passed to the
    #     local minimizer (scipy.optimize.minimize). L-BFGS-B is fast for
    #     smooth functions; Nelder-Mead for non-smooth.
    #
    # Basin-hopping is particularly effective when:
    #   - The energy landscape has a "funnel" structure (many local minima
    #     organized hierarchically around the global minimum)
    #   - Common in molecular geometry optimization, protein folding, clustering
    #
    # Return the OptimizeResult directly.
    raise NotImplementedError("TODO(human): implement basinhopping call")


# ============================================================================
# Visualization
# ============================================================================

def plot_test_function_landscape(name: str, func, bounds: list[tuple[float, float]],
                                  true_x: list[float]):
    """Plot a 2D test function as a filled contour."""
    x = np.linspace(bounds[0][0], bounds[0][1], 300)
    y = np.linspace(bounds[1][0], bounds[1][1], 300)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([xi, yi]) for xi, yi in zip(xrow, yrow)]
                   for xrow, yrow in zip(X, Y)])
    return X, Y, Z


def plot_all_landscapes():
    """Plot all three test functions side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, info) in zip(axes, TEST_FUNCTIONS.items()):
        X, Y, Z = plot_test_function_landscape(name, info["func"], info["bounds"], info["true_x"])
        ax.contourf(X, Y, Z, levels=50, cmap="viridis")
        ax.plot(info["true_x"][0], info["true_x"][1], "r*", markersize=15)
        ax.set_title(f"{name} (min at {info['true_x']})")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
    plt.tight_layout()
    plt.savefig("phase2_landscapes.png", dpi=120)
    print("\n  [Saved: phase2_landscapes.png]")
    plt.close()


def plot_method_comparison(results_table: dict[str, dict[str, dict]]):
    """Plot a bar chart comparing methods across test functions."""
    methods = list(next(iter(results_table.values())).keys())
    functions = list(results_table.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Function value comparison
    x_pos = np.arange(len(functions))
    width = 0.18
    for i, method in enumerate(methods):
        vals = [results_table[f][method]["fun"] for f in functions]
        axes[0].bar(x_pos + i * width, vals, width, label=method)
    axes[0].set_xticks(x_pos + width * (len(methods) - 1) / 2)
    axes[0].set_xticklabels(functions)
    axes[0].set_ylabel("f(x*)")
    axes[0].set_title("Optimal Value Found")
    axes[0].legend(fontsize=7)
    axes[0].set_yscale("symlog", linthresh=1e-6)

    # Function evaluations comparison
    for i, method in enumerate(methods):
        vals = [results_table[f][method]["nfev"] for f in functions]
        axes[1].bar(x_pos + i * width, vals, width, label=method)
    axes[1].set_xticks(x_pos + width * (len(methods) - 1) / 2)
    axes[1].set_xticklabels(functions)
    axes[1].set_ylabel("Function evaluations")
    axes[1].set_title("Computational Cost")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("phase2_comparison.png", dpi=120)
    print("  [Saved: phase2_comparison.png]")
    plt.close()


# ============================================================================
# Display helpers
# ============================================================================

def print_result(method_name: str, func_name: str, result, true_min: float, true_x: list[float]) -> dict:
    """Print and return a single optimization result."""
    error = abs(result.fun - true_min)
    x_str = ", ".join(f"{xi:.6f}" for xi in result.x)
    true_x_str = ", ".join(f"{xi:.4f}" for xi in true_x)
    success_marker = "OK" if error < 1e-2 else "MISS"

    print(f"    [{success_marker}] f(x*) = {result.fun:.8f}  (true: {true_min:.4f}, error: {error:.2e})")
    print(f"         x* = [{x_str}]  (true: [{true_x_str}])")
    print(f"         nfev = {result.nfev}")

    return {"fun": result.fun, "nfev": result.nfev, "error": error, "success": error < 1e-2}


def print_shgo_local_minima(result) -> None:
    """Print all local minima found by SHGO."""
    if hasattr(result, "xl") and result.xl is not None:
        n_minima = len(result.xl)
        print(f"    SHGO found {n_minima} local minim{'um' if n_minima == 1 else 'a'}:")
        for i, (x, f) in enumerate(zip(result.xl[:10], result.funl[:10])):
            x_str = ", ".join(f"{xi:.4f}" for xi in x)
            print(f"      #{i}: f = {f:.6f}  at [{x_str}]")
        if n_minima > 10:
            print(f"      ... and {n_minima - 10} more")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 2: Global Optimization — scipy.optimize")
    print("=" * 80)

    # Visualize test function landscapes
    plot_all_landscapes()

    results_table: dict[str, dict[str, dict]] = {}

    for func_name, info in TEST_FUNCTIONS.items():
        print(f"\n{'=' * 80}")
        print(f"Test Function: {func_name}")
        print(f"  Domain: {info['bounds']}")
        print(f"  True minimum: f({info['true_x']}) = {info['true_min']}")
        print(f"{'=' * 80}")

        func = info["func"]
        bounds = info["bounds"]
        true_min = info["true_min"]
        true_x = info["true_x"]
        results_table[func_name] = {}

        # --- Differential Evolution ---
        print(f"\n  Differential Evolution:")
        t0 = time.perf_counter()
        result_de = optimize_differential_evolution(func, bounds)
        dt = time.perf_counter() - t0
        res_info = print_result("DE", func_name, result_de, true_min, true_x)
        print(f"         time = {dt:.3f}s")
        results_table[func_name]["DE"] = res_info

        # --- Dual Annealing ---
        print(f"\n  Dual Annealing:")
        t0 = time.perf_counter()
        result_da = optimize_dual_annealing(func, bounds)
        dt = time.perf_counter() - t0
        res_info = print_result("DA", func_name, result_da, true_min, true_x)
        print(f"         time = {dt:.3f}s")
        results_table[func_name]["DA"] = res_info

        # --- SHGO ---
        print(f"\n  SHGO:")
        t0 = time.perf_counter()
        result_shgo = optimize_shgo(func, bounds)
        dt = time.perf_counter() - t0
        res_info = print_result("SHGO", func_name, result_shgo, true_min, true_x)
        print(f"         time = {dt:.3f}s")
        print_shgo_local_minima(result_shgo)
        results_table[func_name]["SHGO"] = res_info

        # --- Basin-Hopping ---
        print(f"\n  Basin-Hopping:")
        rng = np.random.default_rng(42)
        x0 = rng.uniform(bounds[0][0], bounds[0][1], size=2)
        t0 = time.perf_counter()
        result_bh = optimize_basinhopping(func, x0)
        dt = time.perf_counter() - t0
        res_info = print_result("BH", func_name, result_bh, true_min, true_x)
        print(f"         x0 = [{x0[0]:.4f}, {x0[1]:.4f}]")
        print(f"         time = {dt:.3f}s")
        results_table[func_name]["BH"] = res_info

    # --- Overall comparison ---
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    print(f"\n  {'Function':<12s} {'Method':<8s} {'f(x*)':<14s} {'nfev':<8s} {'Error':<12s} {'Status'}")
    print(f"  {'-'*65}")
    for func_name, methods in results_table.items():
        for method, info in methods.items():
            status = "OK" if info["success"] else "MISS"
            print(f"  {func_name:<12s} {method:<8s} {info['fun']:<14.8f} {info['nfev']:<8d} {info['error']:<12.2e} {status}")

    # Bar chart comparison
    plot_method_comparison(results_table)


if __name__ == "__main__":
    main()
