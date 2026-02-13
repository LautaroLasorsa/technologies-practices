"""Phase 4: Advanced BO — Constraints & High Dimensions.

Extends Bayesian Optimization to real-world complexities: constrained optimization
(where the feasible region is unknown and expensive to evaluate), batch suggestions
(selecting multiple points to evaluate in parallel), and mixed-variable problems
(continuous, integer, and categorical variables). Compares GP-based BO against
Optuna's TPE on a mixed-variable benchmark.

Key extensions:
  - Feasibility GP: separate GP models for unknown constraints
  - Expected Feasible Improvement (EFI): EI * P(feasible)
  - Kriging Believer: batch BO via fantasized observations
  - Optuna TPE: tree-structured Parzen estimator for mixed-variable problems
"""

import copy
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import matplotlib.pyplot as plt

# Optuna for TPE comparison
import optuna

# scikit-optimize for GP-based BO comparison
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical


# ============================================================================
# Test functions for constrained and mixed-variable BO
# ============================================================================

def constrained_branin(x: np.ndarray) -> float:
    """Branin-Hoo objective function for constrained BO.

    f(x) = (x1 - 5.1/(4*pi^2)*x0^2 + 5/pi*x0 - 6)^2 + 10*(1-1/(8*pi))*cos(x0) + 10
    Domain: x0 in [-5, 10], x1 in [0, 15]
    Three global minima at f ≈ 0.397887.
    """
    x0, x1 = x[0], x[1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return float(a * (x1 - b * x0**2 + c * x0 - r)**2 + s * (1 - t) * np.cos(x0) + s)


def branin_constraint(x: np.ndarray) -> float:
    """Unknown constraint for the Branin problem.

    c(x) = x0 + x1 - 12    (feasible when c(x) <= 0, i.e., x0 + x1 <= 12)

    This constraint is NOT a simple bound — it cuts diagonally through the domain,
    making it impossible to handle via bound constraints alone. The BO must learn
    the feasible region by evaluating the constraint at sampled points, just like
    the objective.
    """
    return float(x[0] + x[1] - 12.0)


def batch_objective(x: np.ndarray) -> float:
    """2D objective for batch BO: Six-Hump Camel function.

    Two global minima at f ≈ -1.0316:
      x* = (0.0898, -0.7126), (-0.0898, 0.7126)
    Domain: x0 in [-2, 2], x1 in [-1, 1]
    """
    x0, x1 = x[0], x[1]
    return float((4 - 2.1 * x0**2 + x0**4 / 3) * x0**2 + x0 * x1 + (-4 + 4 * x1**2) * x1**2)


def mixed_variable_objective(x_continuous: list[float], x_integer: list[int], x_categorical: list[str]) -> float:
    """Objective function with mixed variable types for GP-BO vs TPE comparison.

    A synthetic benchmark that combines:
      - 2 continuous variables: x0 in [-5, 5], x1 in [-5, 5]
      - 1 integer variable: n in {1, 2, ..., 10}
      - 1 categorical variable: kernel_type in {"rbf", "matern", "linear"}

    f = (x0 - n/3)^2 + (x1 + 1)^2 + categorical_penalty + sin(n * x0)

    The categorical variable changes the landscape (simulates choosing a model type).
    The integer variable interacts with continuous vars (simulates a discrete hyperparameter).
    """
    x0, x1 = x_continuous
    n = x_integer[0]
    kernel_type = x_categorical[0]

    # Base quadratic bowl shifted by integer param
    base = (x0 - n / 3.0)**2 + (x1 + 1.0)**2

    # Interaction term: integer * continuous
    interaction = np.sin(n * x0 * 0.5)

    # Categorical penalty (simulates model-type impact)
    cat_penalty = {"rbf": 0.0, "matern": 0.5, "linear": 2.0}[kernel_type]

    return float(base + interaction + cat_penalty)


# ============================================================================
# Shared helpers
# ============================================================================

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement (EI) for minimization. Reference from phase 2/3."""
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = f_best - mu - xi
        Z = np.where(sigma > 1e-10, imp / sigma, 0.0)
        ei = np.where(sigma > 1e-10, imp * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)
    return np.maximum(ei, 0.0)


def latin_hypercube_sample(bounds: list[tuple[float, float]], n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube Samples within the given bounds.

    Returns: array of shape (n_samples, n_dims)
    """
    n_dims = len(bounds)
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return qmc.scale(samples, lower, upper)


def fit_gp_surrogate(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """Fit a GP surrogate with Matern 5/2 kernel. Shared utility for this phase."""
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, normalize_y=True,
    )
    gp.fit(X, y)
    return gp


# ============================================================================
# TODO(human): Constrained Bayesian Optimization
# ============================================================================

def constrained_bo_loop(
    objective,
    constraint,
    bounds: list[tuple[float, float]],
    n_init: int = 8,
    n_iter: int = 25,
    seed: int = 42,
) -> dict:
    """Run constrained Bayesian Optimization with a feasibility GP.

    The constraint c(x) is UNKNOWN and EXPENSIVE — we cannot evaluate it cheaply.
    We model it with a separate GP, compute the probability of feasibility from the
    constraint GP posterior, and multiply it into the acquisition function.

    Args:
        objective: Black-box objective f(x) -> float to minimize
        constraint: Black-box constraint c(x) -> float; feasible when c(x) <= 0
        bounds: List of (lower, upper) bounds per dimension
        n_init: Number of initial random (LHS) evaluations
        n_iter: Number of BO iterations
        seed: Random seed

    Returns:
        Dictionary with keys:
          "X": all evaluated points (n_init + n_iter, n_dims)
          "y_obj": objective values at each point
          "y_con": constraint values at each point (negative = feasible)
          "best_feasible_y": running best feasible objective at each step (NaN if no feasible yet)
          "best_feasible_x": location of the running best feasible point
          "feasible_mask": boolean mask — True if the point is feasible
    """
    # TODO(human): Implement Constrained BO with Feasibility GP
    #
    # Constrained BO handles the case where BOTH the objective f(x) and the
    # constraint c(x) are expensive black-box functions. You cannot simply
    # reject infeasible points a priori — you must LEARN the feasible region
    # by sampling it, just like you learn the objective landscape.
    #
    # The key idea: fit TWO separate GPs:
    #   - GP_obj: models the objective f(x) — gives (mu_obj, sigma_obj)
    #   - GP_con: models the constraint c(x) — gives (mu_con, sigma_con)
    #
    # From GP_con, compute the probability that a point is feasible:
    #   P(feasible | x) = P(c(x) <= 0 | x) = Phi(-mu_con(x) / sigma_con(x))
    #
    # where Phi is the standard normal CDF. This works because the GP posterior
    # at x is approximately Normal(mu_con, sigma_con^2), so the probability of
    # being <= 0 is just the CDF evaluated at the standardized threshold.
    #
    # The acquisition function becomes Expected Feasible Improvement (EFI):
    #   EFI(x) = EI(x) * P(feasible | x)
    #
    # This elegantly handles three scenarios:
    #   1. x is clearly feasible (P ≈ 1): EFI ≈ EI — normal BO behavior
    #   2. x is clearly infeasible (P ≈ 0): EFI ≈ 0 — point is avoided
    #   3. x has uncertain feasibility (P ≈ 0.5): EFI = 0.5 * EI — explores
    #      the constraint boundary, which is valuable information
    #
    # Algorithm:
    #   1. Generate n_init LHS points and evaluate BOTH objective and constraint.
    #   2. For each BO iteration:
    #      a. Fit GP_obj to (X, y_obj) and GP_con to (X, y_con).
    #      b. Find f_best = min objective among FEASIBLE points observed so far.
    #         If no feasible points yet, use the point with smallest constraint
    #         violation as a fallback (f_best = min y_obj overall).
    #      c. Optimize EFI: for each of 20 random restarts, minimize -EFI(x):
    #         - Predict (mu_obj, sigma_obj) from GP_obj and (mu_con, sigma_con) from GP_con
    #         - Compute EI from the objective GP posterior and f_best
    #         - Compute P(feasible) = norm.cdf(-mu_con / sigma_con)
    #         - EFI = EI * P(feasible)
    #         - Return -EFI (because scipy minimizes, we want to maximize EFI)
    #      d. Evaluate objective(x_next) and constraint(x_next), append to data.
    #      e. Update running best feasible objective.
    #   3. Return the full history including feasibility masks.
    raise NotImplementedError("TODO(human): implement constrained_bo_loop")


# ============================================================================
# TODO(human): Batch BO via Kriging Believer
# ============================================================================

def kriging_believer_batch(
    gp: GaussianProcessRegressor,
    bounds: list[tuple[float, float]],
    f_best: float,
    batch_size: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Select a batch of points using the Kriging Believer heuristic.

    After selecting each point, "fantasize" its outcome as the GP mean,
    update the GP with this hallucinated observation, and select the next
    point from the updated posterior.

    Args:
        gp: Fitted GaussianProcessRegressor (will be deep-copied, not modified)
        bounds: List of (lower, upper) bounds per dimension
        f_best: Current best observed value (for EI computation)
        batch_size: Number of points to select in parallel
        seed: Random seed

    Returns:
        Array of shape (batch_size, n_dims) — the batch of suggested points
    """
    # TODO(human): Implement Kriging Believer Batch Suggestion
    #
    # When you can evaluate multiple points IN PARALLEL (e.g., multiple GPUs,
    # multiple lab experiments), you want to select a BATCH of q points at
    # once, not just one. Kriging Believer is the simplest batch BO heuristic.
    #
    # The problem with selecting q points independently from the same GP is
    # that they would all cluster around the same peak of the acquisition
    # function — no diversity. Kriging Believer forces diversity by updating
    # the GP after each selection.
    #
    # The algorithm:
    #   1. Deep-copy the fitted GP (so we don't mutate the caller's GP):
    #      current_gp = copy.deepcopy(gp)
    #      current_f_best = f_best
    #
    #   2. For i = 1 to batch_size:
    #      a. Find x_i by maximizing EI on current_gp (same inner optimization
    #         as the standard BO loop: 20 random restarts of L-BFGS-B,
    #         minimizing -EI(x) where mu, sigma come from current_gp).
    #
    #      b. "Fantasize" the outcome: predict y_i = current_gp.predict(x_i)
    #         (the GP mean — our best guess). This is the "kriging believer"
    #         assumption: we BELIEVE the GP's prediction is the true value.
    #
    #      c. Update the GP with the hallucinated observation:
    #         Append x_i to the training X and y_i to the training y,
    #         then refit: current_gp.fit(X_augmented, y_augmented).
    #
    #      d. Update f_best if y_i < current_f_best.
    #
    #   3. Return the batch of q points as a numpy array (batch_size, n_dims).
    #
    # Why Kriging Believer works:
    #   After fantasizing x_1's outcome, the GP becomes CONFIDENT about x_1
    #   (sigma ≈ 0 there), so EI near x_1 drops. The next point x_2 will
    #   be chosen somewhere else — naturally creating DIVERSITY in the batch.
    #
    # Limitations:
    #   - Greedy: the batch quality depends on the order of selection
    #   - The fantasized y values are just guesses — the real values might differ
    #   - For the same batch size, qEI (joint expected improvement) is theoretically
    #     better but much more expensive to compute (it requires Monte Carlo integration
    #     over all q outcomes jointly, implemented in BoTorch via MC sampling)
    #
    # Implementation detail: to refit the GP with augmented data, extract the
    # current training data from gp.X_train_ and gp.y_train_, concatenate
    # the new point, and call gp.fit() again. The GP will re-optimize kernel
    # hyperparameters, which is slightly expensive but ensures a good fit.
    raise NotImplementedError("TODO(human): implement kriging_believer_batch")


# ============================================================================
# TODO(human): Mixed-Variable BO — GP-BO vs Optuna TPE
# ============================================================================

def mixed_variable_bo(
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Compare GP-based BO (skopt) vs TPE (Optuna) on a mixed-variable problem.

    The mixed_variable_objective has continuous, integer, and categorical parameters.
    GP-based BO (with one-hot encoding for categoricals and rounding for integers)
    is compared against Optuna's TPE, which handles mixed types natively.

    Args:
        n_trials: Total number of function evaluations for each method
        seed: Random seed

    Returns:
        Dictionary with keys:
          "skopt_best_y": running best for skopt, shape (n_trials,)
          "optuna_best_y": running best for Optuna TPE, shape (n_trials,)
          "skopt_best_params": best parameters found by skopt
          "optuna_best_params": best parameters found by Optuna
    """
    # TODO(human): Implement Mixed-Variable BO with GP-BO (skopt) vs TPE (Optuna)
    #
    # Real optimization problems often have MIXED variable types:
    #   - Continuous: learning rate, temperature, weight decay
    #   - Integer: number of layers, batch size, number of trees
    #   - Categorical: optimizer type ("adam", "sgd"), activation ("relu", "tanh")
    #
    # GP-based BO struggles with categorical variables because kernels (Matern, RBF)
    # require a continuous distance metric. Common workarounds:
    #   - One-hot encoding: "rbf" -> [1,0,0], "matern" -> [0,1,0], "linear" -> [0,0,1]
    #   - Hamming kernel: k(x,x') = exp(-sum(x_i != x'_i))
    # Neither is ideal: one-hot inflates dimensionality, Hamming loses GP smoothness.
    #
    # TPE (Tree-structured Parzen Estimator) handles mixed types NATIVELY because
    # it models each parameter independently with kernel density estimators (KDEs),
    # not a joint GP. For categoricals, TPE uses categorical distributions.
    #
    # Implementation plan:
    #
    # Part A — skopt (GP-based BO):
    #   1. Define the search space using skopt dimension types:
    #      dimensions = [
    #          Real(-5.0, 5.0, name="x0"),
    #          Real(-5.0, 5.0, name="x1"),
    #          Integer(1, 10, name="n"),
    #          Categorical(["rbf", "matern", "linear"], name="kernel_type"),
    #      ]
    #
    #   2. Create a wrapper function that unpacks skopt's parameter list:
    #      def skopt_objective(params):
    #          x0, x1, n, kernel_type = params
    #          return mixed_variable_objective([x0, x1], [n], [kernel_type])
    #
    #   3. Run gp_minimize(skopt_objective, dimensions, n_calls=n_trials,
    #                       n_initial_points=10, random_state=seed)
    #
    #   4. Extract running best: np.minimum.accumulate(result.func_vals)
    #
    # Part B — Optuna (TPE):
    #   1. Create an Optuna study: optuna.create_study(direction="minimize",
    #      sampler=optuna.samplers.TPESampler(seed=seed))
    #
    #   2. Define the objective with Optuna's suggest API:
    #      def optuna_objective(trial):
    #          x0 = trial.suggest_float("x0", -5.0, 5.0)
    #          x1 = trial.suggest_float("x1", -5.0, 5.0)
    #          n = trial.suggest_int("n", 1, 10)
    #          kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern", "linear"])
    #          return mixed_variable_objective([x0, x1], [n], [kernel_type])
    #
    #   3. Run study.optimize(optuna_objective, n_trials=n_trials) with
    #      optuna.logging disabled (optuna.logging.set_verbosity(optuna.logging.WARNING))
    #
    #   4. Extract running best from study.trials:
    #      values = [t.value for t in study.trials]
    #      optuna_best_y = np.minimum.accumulate(values)
    #
    # Part C — Return comparison dict with running bests and best params.
    #
    # Expected outcome: TPE should outperform GP-BO on this mixed-variable problem
    # because it handles categorical variables natively without one-hot encoding,
    # and its independent per-parameter modeling avoids the curse of dimensionality
    # that affects GPs in mixed spaces. However, for purely continuous problems in
    # low dimensions, GP-BO typically wins due to its richer correlation modeling.
    raise NotImplementedError("TODO(human): implement mixed_variable_bo")


# ============================================================================
# Visualization
# ============================================================================

def plot_constrained_bo(results: dict, bounds: list[tuple[float, float]], objective, constraint) -> None:
    """Visualize constrained BO results: evaluated points, feasibility, and convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Evaluated points on objective contour with constraint boundary ---
    x0 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x1 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X0, X1 = np.meshgrid(x0, x1)
    Z_obj = np.array([[objective(np.array([xi, yi])) for xi, yi in zip(r0, r1)]
                       for r0, r1 in zip(X0, X1)])
    Z_con = np.array([[constraint(np.array([xi, yi])) for xi, yi in zip(r0, r1)]
                       for r0, r1 in zip(X0, X1)])

    ax1.contourf(X0, X1, Z_obj, levels=50, cmap="viridis", alpha=0.7)
    ax1.contour(X0, X1, Z_con, levels=[0.0], colors="red", linewidths=2, linestyles="--")
    ax1.contourf(X0, X1, Z_con, levels=[0.0, 1000], colors=["red"], alpha=0.1)

    X = np.array(results["X"])
    feasible = np.array(results["feasible_mask"])

    # Infeasible points
    if np.any(~feasible):
        ax1.scatter(X[~feasible, 0], X[~feasible, 1], c="red", s=30, marker="x",
                   label="Infeasible", zorder=5)
    # Feasible points
    if np.any(feasible):
        ax1.scatter(X[feasible, 0], X[feasible, 1], c="white", s=40, edgecolors="black",
                   marker="o", label="Feasible", zorder=5)

    # Best feasible point
    best_feasible = results["best_feasible_y"]
    valid_bests = [v for v in best_feasible if not np.isnan(v)]
    if valid_bests:
        best_idx_feasible = None
        y_obj = np.array(results["y_obj"])
        for i in range(len(X)):
            if feasible[i] and y_obj[i] == valid_bests[-1]:
                best_idx_feasible = i
                break
        if best_idx_feasible is not None:
            ax1.scatter(X[best_idx_feasible, 0], X[best_idx_feasible, 1], c="yellow",
                       s=150, edgecolors="black", marker="*",
                       label=f"Best: {valid_bests[-1]:.3f}", zorder=6)

    ax1.set_xlabel("x[0]")
    ax1.set_ylabel("x[1]")
    ax1.set_title("Constrained BO: Evaluated Points")
    ax1.legend(fontsize=7)

    # --- Right: Convergence of best feasible objective ---
    ax2.plot(range(len(best_feasible)), best_feasible, "b-o", markersize=3, linewidth=2)
    ax2.set_xlabel("Function evaluations")
    ax2.set_ylabel("Best feasible f(x)")
    ax2.set_title("Constrained BO: Convergence")
    ax2.grid(True, alpha=0.3)

    # Mark where first feasible point was found
    for i, v in enumerate(best_feasible):
        if not np.isnan(v):
            ax2.axvline(i, color="green", linestyle=":", alpha=0.5, label=f"First feasible at eval {i}")
            ax2.legend(fontsize=8)
            break

    plt.tight_layout()
    plt.savefig("phase4_constrained_bo.png", dpi=120)
    print("  [Saved: phase4_constrained_bo.png]")
    plt.close()


def plot_batch_bo(X_observed: np.ndarray, y_observed: np.ndarray, batch_points: np.ndarray,
                  bounds: list[tuple[float, float]], objective) -> None:
    """Visualize batch suggestions on the objective landscape."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x0 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x1 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X0, X1 = np.meshgrid(x0, x1)
    Z = np.array([[objective(np.array([xi, yi])) for xi, yi in zip(r0, r1)]
                   for r0, r1 in zip(X0, X1)])

    ax.contourf(X0, X1, Z, levels=50, cmap="viridis", alpha=0.7)
    ax.colorbar = plt.colorbar(ax.contourf(X0, X1, Z, levels=50, cmap="viridis", alpha=0.7), ax=ax)

    # Existing observations
    ax.scatter(X_observed[:, 0], X_observed[:, 1], c="white", s=50, edgecolors="black",
              marker="o", label="Observed", zorder=5)

    # Batch suggestions (numbered)
    colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(batch_points)))
    for i, (pt, color) in enumerate(zip(batch_points, colors)):
        ax.scatter(pt[0], pt[1], c=[color], s=100, edgecolors="black", marker="^", zorder=6)
        ax.annotate(f"B{i+1}", (pt[0], pt[1]), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, fontweight="bold", color="red")

    ax.scatter([], [], c="red", marker="^", s=100, edgecolors="black", label="Batch suggestions")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title(f"Kriging Believer: {len(batch_points)} Batch Suggestions")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("phase4_batch_bo.png", dpi=120)
    print("  [Saved: phase4_batch_bo.png]")
    plt.close()


def plot_mixed_variable_comparison(results: dict) -> None:
    """Plot convergence comparison: GP-BO (skopt) vs TPE (Optuna) on mixed-variable problem."""
    fig, ax = plt.subplots(figsize=(9, 5))

    n_skopt = len(results["skopt_best_y"])
    n_optuna = len(results["optuna_best_y"])

    ax.plot(range(n_skopt), results["skopt_best_y"], "b-o", markersize=3,
            label="GP-BO (skopt)", linewidth=2)
    ax.plot(range(n_optuna), results["optuna_best_y"], "r-s", markersize=3,
            label="TPE (Optuna)", linewidth=2)

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best f(x) found")
    ax.set_title("Mixed-Variable Optimization: GP-BO vs TPE")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase4_mixed_variable.png", dpi=120)
    print("  [Saved: phase4_mixed_variable.png]")
    plt.close()


# ============================================================================
# Display helpers
# ============================================================================

def print_constrained_results(results: dict) -> None:
    """Print summary of constrained BO results."""
    y_obj = np.array(results["y_obj"])
    y_con = np.array(results["y_con"])
    feasible = np.array(results["feasible_mask"])
    n_total = len(y_obj)
    n_feasible = np.sum(feasible)

    print(f"    Total evaluations: {n_total}")
    print(f"    Feasible points: {n_feasible}/{n_total} ({100*n_feasible/n_total:.0f}%)")

    best_feasible = results["best_feasible_y"]
    valid_bests = [v for v in best_feasible if not np.isnan(v)]
    if valid_bests:
        print(f"    Best feasible objective: {valid_bests[-1]:.6f}")
        # Find first feasible
        for i, v in enumerate(best_feasible):
            if not np.isnan(v):
                print(f"    First feasible point found at evaluation: {i}")
                break
    else:
        print("    No feasible points found!")


def print_batch_results(batch_points: np.ndarray, gp: GaussianProcessRegressor) -> None:
    """Print batch point details."""
    mu_batch, sigma_batch = gp.predict(batch_points, return_std=True)
    print(f"    Batch size: {len(batch_points)}")
    for i, (pt, m, s) in enumerate(zip(batch_points, mu_batch, sigma_batch)):
        pt_str = ", ".join(f"{v:.4f}" for v in pt)
        print(f"    Point {i+1}: [{pt_str}]  (GP mean: {m:.4f}, GP std: {s:.4f})")

    # Measure diversity: average pairwise distance
    from scipy.spatial.distance import pdist
    if len(batch_points) > 1:
        dists = pdist(batch_points)
        print(f"    Avg pairwise distance: {np.mean(dists):.4f}")
        print(f"    Min pairwise distance: {np.min(dists):.4f}")


def print_mixed_results(results: dict) -> None:
    """Print mixed-variable comparison results."""
    print(f"    skopt final best: {results['skopt_best_y'][-1]:.6f}")
    print(f"    Optuna final best: {results['optuna_best_y'][-1]:.6f}")
    print(f"    skopt best params: {results['skopt_best_params']}")
    print(f"    Optuna best params: {results['optuna_best_params']}")

    winner = "GP-BO (skopt)" if results["skopt_best_y"][-1] < results["optuna_best_y"][-1] else "TPE (Optuna)"
    print(f"    Winner: {winner}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 4: Advanced BO — Constraints & High Dimensions")
    print("=" * 80)

    # --- Part A: Constrained BO ---
    print("\n" + "-" * 80)
    print("Part A: Constrained Bayesian Optimization")
    print("-" * 80)
    print("  Optimizing Branin with unknown diagonal constraint: x0 + x1 <= 12.")
    print("  The BO must learn the feasible region from expensive constraint evaluations.")
    print("  Uses Expected Feasible Improvement (EFI) = EI * P(feasible).\n")

    constrained_bounds = [(-5.0, 10.0), (0.0, 15.0)]
    t0 = time.perf_counter()
    constrained_results = constrained_bo_loop(
        constrained_branin, branin_constraint, constrained_bounds,
        n_init=8, n_iter=25, seed=42,
    )
    dt = time.perf_counter() - t0
    print_constrained_results(constrained_results)
    print(f"    Time: {dt:.2f}s")
    plot_constrained_bo(constrained_results, constrained_bounds, constrained_branin, branin_constraint)

    # --- Part B: Batch BO via Kriging Believer ---
    print("\n" + "-" * 80)
    print("Part B: Batch BO — Kriging Believer")
    print("-" * 80)
    print("  Selecting 4 points to evaluate in parallel on Six-Hump Camel.")
    print("  Each point is selected sequentially with fantasized GP updates.\n")

    batch_bounds = [(-2.0, 2.0), (-1.0, 1.0)]
    # Generate initial observations
    rng = np.random.default_rng(42)
    X_init = latin_hypercube_sample(batch_bounds, n_samples=10, seed=42)
    y_init = np.array([batch_objective(x) for x in X_init])

    # Fit GP to initial data
    gp_batch = fit_gp_surrogate(X_init, y_init)
    f_best_batch = np.min(y_init)

    print(f"  Initial observations: {len(X_init)}")
    print(f"  Current best: {f_best_batch:.6f}")
    print(f"  Selecting batch of 4 points...\n")

    t0 = time.perf_counter()
    batch_points = kriging_believer_batch(gp_batch, batch_bounds, f_best_batch, batch_size=4, seed=42)
    dt = time.perf_counter() - t0
    print_batch_results(batch_points, gp_batch)
    print(f"    Time: {dt:.2f}s")
    plot_batch_bo(X_init, y_init, batch_points, batch_bounds, batch_objective)

    # --- Part C: GP-BO vs TPE on Mixed Variables ---
    print("\n" + "-" * 80)
    print("Part C: Mixed-Variable BO — GP-BO (skopt) vs TPE (Optuna)")
    print("-" * 80)
    print("  Problem: 2 continuous + 1 integer + 1 categorical variable.")
    print("  GP-BO must encode categoricals; TPE handles them natively.\n")

    t0 = time.perf_counter()
    mixed_results = mixed_variable_bo(n_trials=50, seed=42)
    dt = time.perf_counter() - t0
    print_mixed_results(mixed_results)
    print(f"    Time: {dt:.2f}s")
    plot_mixed_variable_comparison(mixed_results)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Key takeaways:
  1. CONSTRAINED BO: When constraints are unknown and expensive, model them with
     separate GPs and use Expected Feasible Improvement (EFI = EI * P(feasible)).
     The feasibility GP naturally explores the constraint boundary.
  2. BATCH BO: Kriging Believer generates diverse batches by fantasizing outcomes.
     After selecting each point, the GP is updated with the predicted value,
     reducing EI near that point and forcing diversity in subsequent selections.
  3. MIXED VARIABLES: TPE (Optuna) handles categorical/integer variables natively
     because it models each parameter independently. GP-BO requires encoding tricks
     (one-hot, rounding) that inflate dimensionality and break smoothness assumptions.
  4. WHEN TO USE WHAT:
     - Pure continuous, low-d (d < 20): GP-based BO (skopt, BoTorch)
     - Mixed types, medium-d (d < 100): TPE (Optuna, Hyperopt)
     - Very high-d (d > 100): Random search, or BOHB, or TuRBO
     - Unknown constraints: Constrained GP-BO with feasibility GP
     - Parallel resources: Batch BO (Kriging Believer or qEI in BoTorch)
""")


if __name__ == "__main__":
    main()
