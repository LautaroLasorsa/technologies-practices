"""Phase 2: Acquisition Functions for Bayesian Optimization.

Implements the three major acquisition functions — Probability of Improvement (PI),
Expected Improvement (EI), and Upper Confidence Bound (UCB) — from the GP posterior.
Visualizes how each function selects the next evaluation point and how they balance
exploration (sampling where uncertainty is high) vs exploitation (sampling where the
predicted value is good).

Key mathematical functions used:
  - scipy.stats.norm.cdf (Phi) — standard normal CDF for PI and EI
  - scipy.stats.norm.pdf (phi) — standard normal PDF for EI
  - GP posterior: mu(x) and sigma(x) from phase 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


# ============================================================================
# Test function (same Forrester from phase 1)
# ============================================================================

def forrester(x: np.ndarray) -> np.ndarray:
    """Forrester function: f(x) = (6x - 2)^2 * sin(12x - 4). Domain: [0, 1]."""
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


# ============================================================================
# GP helper (reuses logic from phase 1)
# ============================================================================

def fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]:
    """Fit a GP with Matern 5/2 kernel and return predictions.

    Returns:
        (gp, mu, sigma): fitted GP, posterior mean, posterior std at X_test
    """
    kernel = ConstantKernel(1.0) * Matern(length_scale=0.2, nu=2.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, n_restarts_optimizer=10, normalize_y=True
    )
    gp.fit(X_train, y_train)
    mu, sigma = gp.predict(X_test, return_std=True)
    return gp, mu, sigma


# ============================================================================
# TODO(human): Probability of Improvement (PI)
# ============================================================================

def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute the Probability of Improvement acquisition function.

    Args:
        mu: GP posterior mean at candidate points, shape (n,)
        sigma: GP posterior std at candidate points, shape (n,)
        f_best: Current best (minimum) observed function value
        xi: Exploration parameter (>= 0). Higher xi = more exploration.

    Returns:
        PI values at each candidate point, shape (n,)
    """
    # TODO(human): Implement Probability of Improvement (PI)
    #
    # PI measures the PROBABILITY that evaluating at x will improve on the current
    # best observation f_best. It answers: "How likely is it that f(x) < f_best - xi?"
    #
    # The formula (for MINIMIZATION) is:
    #   PI(x) = Phi((f_best - mu(x) - xi) / sigma(x))
    #
    # Where:
    #   - Phi(z) = norm.cdf(z) is the standard normal CDF (probability that Z <= z)
    #   - mu(x) is the GP posterior mean (our best estimate of f(x))
    #   - sigma(x) is the GP posterior std (our uncertainty about f(x))
    #   - f_best is the current best observed value (the target to beat)
    #   - xi >= 0 is the exploration-exploitation tradeoff parameter:
    #     * xi = 0: pure exploitation — maximize probability of ANY improvement
    #     * xi > 0: require improvement of at least xi to count — encourages exploration
    #
    # Step 1: Handle the edge case where sigma(x) = 0 (at or very near observed points).
    #   When sigma = 0, we KNOW the value is mu, so:
    #   - If mu < f_best - xi: PI = 1 (certain improvement)
    #   - Otherwise: PI = 0 (no improvement possible)
    #   Use np.where(sigma > 1e-10, ..., 0.0) or similar to avoid division by zero.
    #
    # Step 2: Compute Z = (f_best - mu - xi) / sigma for points where sigma > 0.
    #   Z is the number of standard deviations below the threshold f_best - xi.
    #   Large positive Z = high probability of improvement.
    #   Large negative Z = very unlikely to improve.
    #
    # Step 3: Return norm.cdf(Z).
    #   norm.cdf(Z) converts the Z-score to a probability in [0, 1].
    #
    # PI's weakness: it only measures PROBABILITY, not MAGNITUDE of improvement.
    # A point with 60% chance of improving by 0.001 scores higher than a point with
    # 40% chance of improving by 100. This causes PI to over-exploit (sample very
    # close to the current best), missing potentially much better regions.
    raise NotImplementedError("TODO(human): implement probability_of_improvement")


# ============================================================================
# TODO(human): Expected Improvement (EI)
# ============================================================================

def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute the Expected Improvement acquisition function.

    Args:
        mu: GP posterior mean at candidate points, shape (n,)
        sigma: GP posterior std at candidate points, shape (n,)
        f_best: Current best (minimum) observed function value
        xi: Exploration parameter (>= 0). Higher xi = more exploration.

    Returns:
        EI values at each candidate point, shape (n,)
    """
    # TODO(human): Implement Expected Improvement (EI)
    #
    # EI measures the EXPECTED AMOUNT of improvement over the current best.
    # Unlike PI (which counts only probability), EI also accounts for HOW MUCH
    # better a point might be. This makes it the most popular acquisition function.
    #
    # The formula (for MINIMIZATION) is:
    #   EI(x) = (f_best - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z)
    #   where Z = (f_best - mu(x) - xi) / sigma(x)
    #
    # Where:
    #   - Phi(z) = norm.cdf(z) — standard normal CDF
    #   - phi(z) = norm.pdf(z) — standard normal PDF
    #   - First term: (f_best - mu - xi) * Phi(Z) — EXPLOITATION term
    #     This is large when mu(x) is much smaller than f_best (predicted improvement)
    #     weighted by the probability of achieving that improvement.
    #   - Second term: sigma(x) * phi(Z) — EXPLORATION term
    #     This is large when sigma(x) is large (high uncertainty).
    #     phi(Z) peaks at Z=0 (where improvement is borderline), so the exploration
    #     bonus is strongest where improvement is uncertain but possible.
    #
    # Step 1: Handle sigma = 0 edge case.
    #   When sigma = 0: EI = max(f_best - mu - xi, 0)
    #   (No uncertainty → improvement is deterministic, either positive or zero.)
    #
    # Step 2: Compute Z = (f_best - mu - xi) / sigma for sigma > 0.
    #
    # Step 3: Compute EI = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z).
    #   Use np.where to handle the sigma=0 case cleanly.
    #
    # Step 4: Return EI values (should be non-negative everywhere).
    #
    # EI naturally balances exploration and exploitation because:
    #   - In well-explored regions (low sigma): the exploitation term dominates,
    #     and EI is high only if mu is significantly better than f_best.
    #   - In unexplored regions (high sigma): the exploration term dominates,
    #     giving EI a nonzero value even if mu is not great — because there's
    #     a chance the true value is much better than the mean predicts.
    raise NotImplementedError("TODO(human): implement expected_improvement")


# ============================================================================
# TODO(human): Upper Confidence Bound (UCB)
# ============================================================================

def upper_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    """Compute the Upper Confidence Bound (GP-UCB) acquisition function.

    For minimization, we use the LOWER confidence bound and negate it
    so that higher values indicate more promising points.

    Args:
        mu: GP posterior mean at candidate points, shape (n,)
        sigma: GP posterior std at candidate points, shape (n,)
        kappa: Exploration parameter (> 0). Higher kappa = more exploration.

    Returns:
        UCB values at each candidate point, shape (n,) — higher is better
    """
    # TODO(human): Implement Upper Confidence Bound (UCB / GP-UCB)
    #
    # UCB is the simplest acquisition function — a linear combination of mean
    # and uncertainty. For MINIMIZATION, we want to sample where the lower
    # confidence bound is lowest, i.e., where mu - kappa * sigma is smallest.
    #
    # To keep the convention "higher acquisition = better candidate" (consistent
    # with PI and EI), we NEGATE the lower confidence bound:
    #
    #   UCB(x) = -(mu(x) - kappa * sigma(x))
    #          = -mu(x) + kappa * sigma(x)
    #
    # Where:
    #   - -mu(x): EXPLOITATION term — prefers points with low predicted value
    #   - kappa * sigma(x): EXPLORATION term — prefers points with high uncertainty
    #   - kappa > 0 controls the tradeoff:
    #     * kappa = 0: pure exploitation (just pick the lowest predicted value)
    #     * kappa → ∞: pure exploration (just pick the most uncertain point)
    #     * kappa = 2.0: standard default, works well in practice
    #
    # Step 1: Compute ucb = -mu + kappa * sigma.
    #   That's it — UCB is a single line of vectorized NumPy.
    #
    # Step 2: Return the ucb array.
    #
    # UCB has theoretical guarantees that EI and PI lack:
    #   Srinivas et al. (2010) proved that GP-UCB achieves SUBLINEAR cumulative
    #   regret (regret grows as O(sqrt(T * log(T))) where T = number of evaluations).
    #   The theoretical kappa schedule is kappa_t = sqrt(2 * log(t * d^2 * pi^2 / (6*delta)))
    #   where t = iteration, d = dimension, delta = failure probability.
    #   In practice, a constant kappa = 2.0 works well.
    #
    # UCB's simplicity is both a strength (easy to implement, interpret, and debug)
    # and a weakness (the exploration-exploitation tradeoff is a linear interpolation,
    # which may not be optimal for all objective function shapes).
    raise NotImplementedError("TODO(human): implement upper_confidence_bound")


# ============================================================================
# Visualization
# ============================================================================

def plot_acquisition_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
) -> None:
    """Plot GP posterior and all three acquisition functions side by side."""
    X_flat = X_test.ravel()

    # Compute acquisition functions
    pi_vals = probability_of_improvement(mu, sigma, f_best, xi=0.01)
    ei_vals = expected_improvement(mu, sigma, f_best, xi=0.01)
    ucb_vals = upper_confidence_bound(mu, sigma, kappa=2.0)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # --- GP Posterior ---
    ax = axes[0]
    ax.plot(X_flat, forrester(X_flat), "k--", alpha=0.4, label="True f(x)")
    ax.fill_between(X_flat, mu - 2 * sigma, mu + 2 * sigma, alpha=0.15, color="blue")
    ax.fill_between(X_flat, mu - sigma, mu + sigma, alpha=0.25, color="blue")
    ax.plot(X_flat, mu, "b-", linewidth=2, label="GP mean")
    ax.scatter(X_train.ravel(), y_train, c="red", s=60, zorder=5, edgecolors="black", label="Observations")
    ax.axhline(f_best, color="green", linestyle=":", alpha=0.7, label=f"f_best = {f_best:.3f}")
    ax.set_ylabel("f(x)")
    ax.set_title("GP Posterior")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- PI ---
    ax = axes[1]
    ax.fill_between(X_flat, 0, pi_vals, alpha=0.3, color="orange")
    ax.plot(X_flat, pi_vals, "orange", linewidth=2)
    x_next_pi = X_flat[np.argmax(pi_vals)]
    ax.axvline(x_next_pi, color="red", linestyle="--", alpha=0.7, label=f"Next: x={x_next_pi:.3f}")
    ax.set_ylabel("PI(x)")
    ax.set_title("Probability of Improvement (PI)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- EI ---
    ax = axes[2]
    ax.fill_between(X_flat, 0, ei_vals, alpha=0.3, color="green")
    ax.plot(X_flat, ei_vals, "green", linewidth=2)
    x_next_ei = X_flat[np.argmax(ei_vals)]
    ax.axvline(x_next_ei, color="red", linestyle="--", alpha=0.7, label=f"Next: x={x_next_ei:.3f}")
    ax.set_ylabel("EI(x)")
    ax.set_title("Expected Improvement (EI)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- UCB ---
    ax = axes[3]
    ax.fill_between(X_flat, np.min(ucb_vals), ucb_vals, alpha=0.3, color="purple")
    ax.plot(X_flat, ucb_vals, "purple", linewidth=2)
    x_next_ucb = X_flat[np.argmax(ucb_vals)]
    ax.axvline(x_next_ucb, color="red", linestyle="--", alpha=0.7, label=f"Next: x={x_next_ucb:.3f}")
    ax.set_ylabel("UCB(x)")
    ax.set_title("Upper Confidence Bound (UCB, κ=2.0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")

    plt.tight_layout()
    plt.savefig("phase2_acquisition_comparison.png", dpi=120)
    print("  [Saved: phase2_acquisition_comparison.png]")
    plt.close()


def plot_ei_xi_sensitivity(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    X_test: np.ndarray,
) -> None:
    """Show how EI changes with different xi (exploration parameter) values."""
    X_flat = X_test.ravel()
    xi_values = [0.0, 0.01, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(len(xi_values), 1, figsize=(10, 2.5 * len(xi_values)), sharex=True)

    for ax, xi in zip(axes, xi_values):
        ei = expected_improvement(mu, sigma, f_best, xi=xi)
        ax.fill_between(X_flat, 0, ei, alpha=0.3, color="green")
        ax.plot(X_flat, ei, "green", linewidth=2)
        x_next = X_flat[np.argmax(ei)]
        ax.axvline(x_next, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(f"EI (ξ={xi})")
        ax.set_title(f"EI with ξ={xi} → next sample at x={x_next:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig("phase2_ei_xi_sensitivity.png", dpi=120)
    print("  [Saved: phase2_ei_xi_sensitivity.png]")
    plt.close()


def plot_ucb_kappa_sensitivity(
    mu: np.ndarray,
    sigma: np.ndarray,
    X_test: np.ndarray,
) -> None:
    """Show how UCB changes with different kappa (exploration parameter) values."""
    X_flat = X_test.ravel()
    kappa_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(len(kappa_values), 1, figsize=(10, 2.5 * len(kappa_values)), sharex=True)

    for ax, kappa in zip(axes, kappa_values):
        ucb = upper_confidence_bound(mu, sigma, kappa=kappa)
        ax.fill_between(X_flat, np.min(ucb), ucb, alpha=0.3, color="purple")
        ax.plot(X_flat, ucb, "purple", linewidth=2)
        x_next = X_flat[np.argmax(ucb)]
        ax.axvline(x_next, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(f"UCB (κ={kappa})")
        ax.set_title(f"UCB with κ={kappa} → next sample at x={x_next:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig("phase2_ucb_kappa_sensitivity.png", dpi=120)
    print("  [Saved: phase2_ucb_kappa_sensitivity.png]")
    plt.close()


# ============================================================================
# Display helpers
# ============================================================================

def print_acquisition_summary(
    X_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
) -> None:
    """Print where each acquisition function would sample next."""
    X_flat = X_test.ravel()

    pi = probability_of_improvement(mu, sigma, f_best, xi=0.01)
    ei = expected_improvement(mu, sigma, f_best, xi=0.01)
    ucb = upper_confidence_bound(mu, sigma, kappa=2.0)

    x_pi = X_flat[np.argmax(pi)]
    x_ei = X_flat[np.argmax(ei)]
    x_ucb = X_flat[np.argmax(ucb)]

    # True function values at proposed points
    f_true_pi = forrester(np.array([x_pi]))[0]
    f_true_ei = forrester(np.array([x_ei]))[0]
    f_true_ucb = forrester(np.array([x_ucb]))[0]

    true_min_x = X_flat[np.argmin(forrester(X_flat))]
    true_min_f = forrester(np.array([true_min_x]))[0]

    print(f"\n  Current best observed: f_best = {f_best:.4f}")
    print(f"  True global minimum: f({true_min_x:.4f}) = {true_min_f:.4f}")
    print()
    print(f"  {'Acquisition':<15s} {'Next x':>10s} {'f_true(x)':>12s} {'Comment'}")
    print(f"  {'-'*60}")
    print(f"  {'PI (ξ=0.01)':<15s} {x_pi:>10.4f} {f_true_pi:>12.4f} {'exploits near best' if abs(x_pi - X_flat[np.argmin(mu)]) < 0.1 else 'explores'}")
    print(f"  {'EI (ξ=0.01)':<15s} {x_ei:>10.4f} {f_true_ei:>12.4f} {'balanced'}")
    print(f"  {'UCB (κ=2.0)':<15s} {x_ucb:>10.4f} {f_true_ucb:>12.4f} {'exploration-heavy' if sigma[np.argmax(ucb)] > np.mean(sigma) else 'balanced'}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 2: Acquisition Functions for Bayesian Optimization")
    print("=" * 80)

    # --- Setup: Fit GP to a sparse set of observations ---
    rng = np.random.default_rng(42)
    X_train = np.array([[0.05], [0.15], [0.4], [0.6], [0.85]])
    y_train = forrester(X_train.ravel()) + rng.normal(0, 0.1, size=len(X_train))
    X_test = np.linspace(0.0, 1.0, 500).reshape(-1, 1)

    print(f"\n  Training points: {len(X_train)}")
    print(f"  Domain: [0, 1]")
    print(f"  True minimum: f(0.757) ≈ -6.02")

    gp, mu, sigma = fit_and_predict(X_train, y_train, X_test)
    f_best = np.min(y_train)

    print(f"  GP fitted. Kernel: {gp.kernel_}")
    print(f"  Current best observation: f_best = {f_best:.4f}")

    # --- Part A: Side-by-side comparison ---
    print("\n" + "-" * 80)
    print("Part A: Acquisition Function Comparison")
    print("-" * 80)
    print("  Comparing PI, EI, and UCB on the same GP posterior.")
    print("  Observe where each function peaks (= where it would sample next).")

    plot_acquisition_comparison(X_train, y_train, X_test, mu, sigma, f_best)
    print_acquisition_summary(X_test, mu, sigma, f_best)

    # --- Part B: EI sensitivity to xi ---
    print("\n" + "-" * 80)
    print("Part B: EI Sensitivity to ξ (exploration parameter)")
    print("-" * 80)
    print("  ξ=0: pure exploitation, ξ large: more exploration.")
    print("  Watch the next sample point shift as ξ increases.")

    plot_ei_xi_sensitivity(mu, sigma, f_best, X_test)

    # --- Part C: UCB sensitivity to kappa ---
    print("\n" + "-" * 80)
    print("Part C: UCB Sensitivity to κ (exploration parameter)")
    print("-" * 80)
    print("  κ=0: pure exploitation (lowest mu), κ large: pure exploration (highest sigma).")

    plot_ucb_kappa_sensitivity(mu, sigma, X_test)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Key takeaways:
  1. PI tends to over-exploit: it peaks near the current best, missing unexplored regions.
  2. EI is the most balanced: it combines probability AND magnitude of improvement.
  3. UCB is the simplest and most tunable: κ directly controls exploration vs exploitation.
  4. All three use the SAME GP posterior (mu, sigma) — they differ only in how they
     combine these two quantities to score candidate points.
  5. In practice, EI is the default choice. UCB is preferred when you want explicit
     control over exploration. PI is rarely used alone.
  6. The xi and kappa parameters should be tuned based on the evaluation budget:
     - Early in optimization (lots of budget left): increase exploration (high xi, high kappa)
     - Late in optimization (budget nearly exhausted): increase exploitation (low xi, low kappa)
""")


if __name__ == "__main__":
    main()
