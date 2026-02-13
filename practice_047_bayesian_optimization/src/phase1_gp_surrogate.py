"""Phase 1: Gaussian Process Surrogate Modeling.

Fits Gaussian Process regression models to noisy observations of 1D test
functions. Visualizes the GP posterior (mean + uncertainty bands), demonstrates
how the posterior evolves as data points are added, and compares kernels
(RBF, Matern 3/2, Matern 5/2) on the same data.

Key scikit-learn classes used:
  - GaussianProcessRegressor — fits GP and predicts mean + std
  - kernels.RBF — squared exponential kernel (infinitely smooth)
  - kernels.Matern — Matern kernel (nu=1.5 or 2.5, controllable smoothness)
  - kernels.ConstantKernel — signal variance multiplier
  - kernels.WhiteKernel — observation noise
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ConstantKernel,
    WhiteKernel,
    Kernel,
)


# ============================================================================
# Test functions
# ============================================================================

def forrester(x: np.ndarray) -> np.ndarray:
    """Forrester function: a standard 1D test function for surrogate modeling.

    f(x) = (6x - 2)^2 * sin(12x - 4)

    Domain: [0, 1]. Has one global minimum near x=0.757 and a local minimum
    near x=0.1. The function is smooth but has both flat and oscillatory regions,
    making it a good test for GP kernel flexibility.
    """
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def gramacy_lee(x: np.ndarray) -> np.ndarray:
    """Gramacy & Lee function: a 1D function with varying frequency.

    f(x) = sin(10*pi*x) / (2*x) + (x - 1)^4

    Domain: [0.5, 2.5]. Has rapidly oscillating behavior near x=0.5 that
    smooths out toward x=2.5. Tests the GP's ability to handle
    non-stationary behavior (different smoothness in different regions).
    """
    return np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4


# ============================================================================
# TODO(human): Fit a Gaussian Process to data
# ============================================================================

def fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: Kernel,
) -> GaussianProcessRegressor:
    """Fit a Gaussian Process regressor to the training data.

    Args:
        X_train: Training inputs, shape (n_samples, 1)
        y_train: Training targets, shape (n_samples,)
        kernel: scikit-learn kernel object (e.g., ConstantKernel() * Matern() + WhiteKernel())

    Returns:
        Fitted GaussianProcessRegressor
    """
    # TODO(human): Create and Fit a GaussianProcessRegressor
    #
    # The GaussianProcessRegressor is scikit-learn's implementation of GP regression.
    # It wraps the Bayesian inference equations (posterior conditioning on data) and
    # kernel hyperparameter optimization (marginal likelihood maximization).
    #
    # Step 1: Create a GaussianProcessRegressor with these parameters:
    #   - kernel=kernel  — the covariance function (passed as argument)
    #   - alpha=1e-6  — small regularization added to the diagonal of K(X,X).
    #     This is NOT the observation noise (that's modeled by WhiteKernel in the kernel).
    #     It's numerical jitter to ensure the kernel matrix K is positive definite.
    #     Without it, K can become singular when two training points are very close.
    #     Think of it as: K_regularized = K + alpha * I.
    #
    #   - n_restarts_optimizer=10  — number of random restarts for the kernel
    #     hyperparameter optimizer. The marginal likelihood is NON-CONVEX in the
    #     hyperparameters (length scale, signal variance, noise), so gradient-based
    #     optimization can get stuck in local optima. Each restart tries a different
    #     random starting point. More restarts = better hyperparameters but slower fitting.
    #
    #   - normalize_y=True  — subtract the mean of y_train before fitting.
    #     This lets the GP prior mean be zero (standard assumption) while still
    #     handling targets that aren't centered. The GP predicts residuals from the
    #     mean, which it can do more easily than predicting absolute values.
    #
    # Step 2: Call gp.fit(X_train, y_train) to condition the GP on the data.
    #   Internally, this:
    #   a) Optimizes kernel hyperparameters by maximizing log marginal likelihood
    #   b) Computes and caches K(X,X)^{-1} @ y for fast prediction
    #   c) Stores the optimized kernel in gp.kernel_ (note the trailing underscore)
    #
    # Step 3: Return the fitted GP object.
    #
    # After fitting, gp.kernel_ contains the OPTIMIZED kernel (with learned
    # hyperparameters), while gp.kernel contains the INITIAL kernel (before optimization).
    # Compare them to see what the data "taught" the GP about the function's smoothness.
    raise NotImplementedError("TODO(human): implement fit_gp")


# ============================================================================
# TODO(human): Predict with uncertainty
# ============================================================================

def predict_with_uncertainty(
    gp: GaussianProcessRegressor,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict GP posterior mean and standard deviation at test points.

    Args:
        gp: Fitted GaussianProcessRegressor
        X_test: Test inputs, shape (n_test, 1)

    Returns:
        (mu, sigma): posterior mean and std, each shape (n_test,)
    """
    # TODO(human): GP Posterior Prediction with Uncertainty
    #
    # The GP posterior at test points x* is a Gaussian distribution:
    #   mu(x*)    = k(x*, X) @ [K(X,X) + sigma_n^2*I]^{-1} @ y
    #   sigma^2(x*) = k(x*, x*) - k(x*, X) @ [K(X,X) + sigma_n^2*I]^{-1} @ k(X, x*)
    #
    # scikit-learn computes this via gp.predict(X_test, return_std=True).
    #
    # Step 1: Call gp.predict(X_test, return_std=True)
    #   This returns (mu, sigma) where:
    #   - mu: shape (n_test,) — posterior mean at each test point
    #   - sigma: shape (n_test,) — posterior standard deviation at each test point
    #
    # The return_std=True flag is what makes this useful for BO: without it,
    # you only get the mean prediction (no uncertainty information). The sigma
    # values are CRITICAL for acquisition functions:
    #   - EI uses both mu and sigma
    #   - UCB uses mu - kappa * sigma
    #   - PI uses (f_best - mu) / sigma
    #
    # Key behaviors to observe:
    #   - sigma ≈ 0 at training points (the GP interpolates exactly if noise-free)
    #   - sigma increases as you move away from training data
    #   - sigma approaches the prior standard deviation far from all data
    #   - The ±2*sigma bands contain ~95% of the GP's probability mass
    #
    # Step 2: Return (mu, sigma) as a tuple.
    raise NotImplementedError("TODO(human): implement predict_with_uncertainty")


# ============================================================================
# TODO(human): Compare kernels
# ============================================================================

def experiment_kernels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]]:
    """Fit GPs with different kernels and compare posteriors.

    Args:
        X_train: Training inputs, shape (n_samples, 1)
        y_train: Training targets, shape (n_samples,)
        X_test: Test inputs for prediction, shape (n_test, 1)

    Returns:
        Dictionary mapping kernel name -> (fitted_gp, mu, sigma)
    """
    # TODO(human): Kernel Comparison — RBF vs Matern 3/2 vs Matern 5/2
    #
    # The kernel is the most important modeling choice in a GP. It determines
    # the smoothness, amplitude, and correlation structure of functions the GP
    # considers plausible. Three common kernels:
    #
    # 1. RBF (Squared Exponential):
    #    k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2*l^2))
    #    - Infinitely differentiable → VERY smooth functions
    #    - In sklearn: ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
    #    - The ConstantKernel multiplier controls sigma_f^2 (signal variance)
    #    - WhiteKernel adds observation noise sigma_n^2
    #
    # 2. Matern 3/2 (nu=1.5):
    #    k(x,x') = sigma_f^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
    #    - Once differentiable → allows "kinks" in the function
    #    - In sklearn: ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-5)
    #
    # 3. Matern 5/2 (nu=2.5):
    #    k(x,x') = sigma_f^2 * (1 + sqrt(5)*r/l + 5r^2/(3l^2)) * exp(-sqrt(5)*r/l)
    #    - Twice differentiable → smoother than 3/2 but rougher than RBF
    #    - The DEFAULT kernel in most BO libraries (good balance)
    #    - In sklearn: ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-5)
    #
    # Step 1: Define three kernels as described above. Use ConstantKernel for signal
    #   variance and WhiteKernel for noise in each. The initial hyperparameter values
    #   (1.0, 1e-5) are just starting points — the GP optimizer will find the best values.
    #
    # Step 2: For each kernel, call fit_gp(X_train, y_train, kernel) to fit the GP.
    #
    # Step 3: For each fitted GP, call predict_with_uncertainty(gp, X_test) to get
    #   posterior mean and std.
    #
    # Step 4: Return a dict mapping kernel name (str) -> (gp, mu, sigma).
    #   e.g., {"RBF": (gp_rbf, mu_rbf, sigma_rbf), "Matern 3/2": (...), ...}
    #
    # After fitting, compare gp.kernel_ across kernels to see learned hyperparameters.
    # Key observations:
    #   - RBF tends to have larger length scales (smoother interpolation)
    #   - Matern 3/2 has tighter uncertainty bands near data but wider between points
    #   - Matern 5/2 is usually the best default for BO applications
    raise NotImplementedError("TODO(human): implement experiment_kernels")


# ============================================================================
# Visualization
# ============================================================================

def plot_gp_posterior(
    ax: plt.Axes,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    true_func=None,
    title: str = "GP Posterior",
) -> None:
    """Plot GP posterior mean, uncertainty bands, and training data."""
    X_test_flat = X_test.ravel()

    # True function (if provided)
    if true_func is not None:
        ax.plot(X_test_flat, true_func(X_test_flat), "k--", linewidth=1, alpha=0.5, label="True f(x)")

    # Uncertainty bands (±1 and ±2 sigma)
    ax.fill_between(
        X_test_flat,
        mu - 2 * sigma,
        mu + 2 * sigma,
        alpha=0.15,
        color="blue",
        label="±2σ (95%)",
    )
    ax.fill_between(
        X_test_flat,
        mu - sigma,
        mu + sigma,
        alpha=0.25,
        color="blue",
        label="±1σ (68%)",
    )

    # Posterior mean
    ax.plot(X_test_flat, mu, "b-", linewidth=2, label="GP mean")

    # Training data
    ax.scatter(
        X_train.ravel(),
        y_train,
        c="red",
        s=50,
        zorder=5,
        edgecolors="black",
        label="Observations",
    )

    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_posterior_evolution(
    func,
    func_name: str,
    domain: tuple[float, float],
    n_points_sequence: list[int],
    seed: int = 42,
) -> None:
    """Show how the GP posterior evolves as more data points are added."""
    rng = np.random.default_rng(seed)
    X_test = np.linspace(domain[0], domain[1], 300).reshape(-1, 1)

    # Generate all training points upfront
    max_n = max(n_points_sequence)
    X_all = rng.uniform(domain[0], domain[1], size=(max_n, 1))
    y_all = func(X_all.ravel()) + rng.normal(0, 0.1, size=max_n)

    n_panels = len(n_points_sequence)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

    for ax, n in zip(axes, n_points_sequence):
        X_train = X_all[:n]
        y_train = y_all[:n]

        gp = fit_gp(X_train, y_train, kernel)
        mu, sigma = predict_with_uncertainty(gp, X_test)

        plot_gp_posterior(
            ax, X_train, y_train, X_test, mu, sigma,
            true_func=func,
            title=f"{func_name}: n={n} points",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

    plt.tight_layout()
    plt.savefig("phase1_posterior_evolution.png", dpi=120)
    print("  [Saved: phase1_posterior_evolution.png]")
    plt.close()


def plot_kernel_comparison(
    results: dict[str, tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    true_func,
    func_name: str,
) -> None:
    """Plot side-by-side comparison of different kernels."""
    n_kernels = len(results)
    fig, axes = plt.subplots(1, n_kernels, figsize=(5 * n_kernels, 4))
    if n_kernels == 1:
        axes = [axes]

    for ax, (name, (gp, mu, sigma)) in zip(axes, results.items()):
        plot_gp_posterior(
            ax, X_train, y_train, X_test, mu, sigma,
            true_func=true_func,
            title=f"{func_name}: {name}",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

        # Print learned hyperparameters
        print(f"\n  {name}:")
        print(f"    Initial kernel: {gp.kernel}")
        print(f"    Optimized kernel: {gp.kernel_}")
        print(f"    Log marginal likelihood: {gp.log_marginal_likelihood_value_:.3f}")

    plt.tight_layout()
    plt.savefig("phase1_kernel_comparison.png", dpi=120)
    print("\n  [Saved: phase1_kernel_comparison.png]")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 1: Gaussian Process Surrogate Modeling")
    print("=" * 80)

    # --- Part A: GP posterior evolution with increasing data ---
    print("\n" + "-" * 80)
    print("Part A: GP Posterior Evolution (Forrester function)")
    print("-" * 80)
    print("  Showing how the GP posterior improves as more observations are added.")
    print("  Watch the uncertainty bands shrink near observed points.\n")

    plot_posterior_evolution(
        func=forrester,
        func_name="Forrester",
        domain=(0.0, 1.0),
        n_points_sequence=[3, 6, 12, 25],
        seed=42,
    )

    # --- Part B: Kernel comparison ---
    print("\n" + "-" * 80)
    print("Part B: Kernel Comparison (Gramacy-Lee function)")
    print("-" * 80)
    print("  Fitting RBF, Matern 3/2, and Matern 5/2 to the same data.")
    print("  Observe differences in smoothness and uncertainty calibration.")

    rng = np.random.default_rng(123)
    X_train = rng.uniform(0.5, 2.5, size=(15, 1))
    y_train = gramacy_lee(X_train.ravel()) + rng.normal(0, 0.05, size=15)
    X_test = np.linspace(0.5, 2.5, 300).reshape(-1, 1)

    results = experiment_kernels(X_train, y_train, X_test)
    plot_kernel_comparison(results, X_train, y_train, X_test, gramacy_lee, "Gramacy-Lee")

    # --- Part C: Prior vs Posterior ---
    print("\n" + "-" * 80)
    print("Part C: GP Prior vs Posterior")
    print("-" * 80)
    print("  Visualizing GP BEFORE conditioning (prior) and AFTER (posterior).")

    X_test_prior = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    kernel_prior = ConstantKernel(1.0) * Matern(length_scale=0.2, nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp_prior = GaussianProcessRegressor(kernel=kernel_prior, alpha=1e-6, optimizer=None)
    # Without fitting, predict gives the prior
    mu_prior, sigma_prior = gp_prior.predict(X_test_prior, return_std=True)

    # Now fit with a few points
    X_few = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
    y_few = forrester(X_few.ravel())
    gp_post = fit_gp(X_few, y_few, kernel_prior)
    mu_post, sigma_post = predict_with_uncertainty(gp_post, X_test_prior)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Prior
    ax1.fill_between(X_test_prior.ravel(), mu_prior - 2 * sigma_prior, mu_prior + 2 * sigma_prior,
                      alpha=0.2, color="blue", label="±2σ prior")
    ax1.plot(X_test_prior.ravel(), mu_prior, "b-", linewidth=2, label="Prior mean")
    ax1.plot(X_test_prior.ravel(), forrester(X_test_prior.ravel()), "k--", alpha=0.5, label="True f(x)")
    ax1.set_title("GP Prior (no data)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")

    # Posterior
    plot_gp_posterior(ax2, X_few, y_few, X_test_prior, mu_post, sigma_post,
                      true_func=forrester, title="GP Posterior (5 points)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")

    plt.tight_layout()
    plt.savefig("phase1_prior_vs_posterior.png", dpi=120)
    print("\n  [Saved: phase1_prior_vs_posterior.png]")
    plt.close()

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Key takeaways:
  1. The GP posterior mean interpolates training data (≈0 error at observed points).
  2. Posterior uncertainty (sigma) is smallest near data and grows in unexplored regions.
  3. Kernel choice affects smoothness: RBF > Matern 5/2 > Matern 3/2.
  4. Kernel hyperparameters are learned automatically via marginal likelihood optimization.
  5. The posterior mean + uncertainty is exactly what acquisition functions need in Phase 2.
""")


if __name__ == "__main__":
    main()
