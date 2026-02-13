"""Phase 1: Least-Squares and Regularized Regression with CVXPY.

Implements ridge (L2) and LASSO (L1) regression as convex optimization problems.
Compares OLS, ridge, and LASSO solutions across different regularization strengths
to demonstrate the bias-variance tradeoff and L1 sparsity.

DCP atoms used:
  - cp.sum_squares(x): convex, positive — ||x||_2^2
  - cp.norm1(x): convex, positive — ||x||_1
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Synthetic data generation
# ============================================================================

def generate_regression_data(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 4,
    noise_std: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data with sparse ground truth.

    Only `n_informative` features have nonzero coefficients — the rest are noise.
    This makes LASSO's sparsity advantage visible.

    Returns:
        A: (n_samples, n_features) design matrix
        b: (n_samples,) response vector
        x_true: (n_features,) true coefficients (sparse)
    """
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((n_samples, n_features))

    # True coefficients: only first n_informative are nonzero
    x_true = np.zeros(n_features)
    x_true[:n_informative] = rng.uniform(1.0, 5.0, size=n_informative)
    # Shuffle to make it less obvious which features matter
    perm = rng.permutation(n_features)
    x_true = x_true[perm]

    b = A @ x_true + noise_std * rng.standard_normal(n_samples)

    return A, b, x_true


def solve_ols(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ordinary Least Squares via numpy (baseline).

    Solves: minimize ||A @ x - b||_2^2
    Closed form: x = (A^T A)^{-1} A^T b
    """
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


# ============================================================================
# TODO(human): Ridge and LASSO regression
# ============================================================================

def solve_ridge(A: np.ndarray, b: np.ndarray, lambd: float) -> np.ndarray:
    """Solve ridge regression using CVXPY.

    minimize ||A @ x - b||_2^2 + lambda * ||x||_2^2

    Args:
        A: (m, n) design matrix
        b: (m,) response vector
        lambd: regularization strength (>= 0)

    Returns:
        x_opt: (n,) optimal coefficient vector
    """
    # TODO(human): Ridge Regression with CVXPY
    # minimize ||A @ x - b||_2^2 + lambda * ||x||_2^2
    #
    # In CVXPY:
    #   x = cp.Variable(n)
    #   objective = cp.Minimize(cp.sum_squares(A @ x - b) + lambd * cp.sum_squares(x))
    #   prob = cp.Problem(objective)
    #   prob.solve()
    #   return x.value
    #
    # DCP check: sum_squares is convex, sum of convex = convex. ✓
    # The lambda parameter controls the bias-variance tradeoff:
    #   lambda = 0 → ordinary least squares (may overfit)
    #   lambda → ∞ → x → 0 (underfits)
    raise NotImplementedError


def solve_lasso(A: np.ndarray, b: np.ndarray, lambd: float) -> np.ndarray:
    """Solve LASSO regression using CVXPY.

    minimize ||A @ x - b||_2^2 + lambda * ||x||_1

    Args:
        A: (m, n) design matrix
        b: (m,) response vector
        lambd: regularization strength (>= 0)

    Returns:
        x_opt: (n,) optimal coefficient vector
    """
    # TODO(human): LASSO Regression with CVXPY
    # minimize ||A @ x - b||_2^2 + lambda * ||x||_1
    #
    # Key difference from ridge: ||x||_1 (L1 norm) produces SPARSE solutions.
    # In CVXPY: cp.norm1(x) or cp.norm(x, 1)
    #
    # DCP: sum_squares (convex) + norm1 (convex) = convex. ✓
    #
    # Compare the sparsity pattern of LASSO vs ridge solutions.
    # LASSO drives some coefficients to EXACTLY zero — feature selection.
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_coefficients(
    name: str,
    x: np.ndarray,
    x_true: np.ndarray,
) -> None:
    """Print coefficient vector with comparison to ground truth."""
    print(f"\n  {name}:")
    print(f"    {'Feature':<10s} {'True':>8s} {'Estimate':>10s} {'Error':>8s}")
    print(f"    {'-'*38}")
    for i in range(len(x)):
        err = abs(x[i] - x_true[i])
        marker = " *" if abs(x_true[i]) > 1e-6 else ""
        print(f"    x[{i}]{marker:<5s} {x_true[i]:8.3f} {x[i]:10.3f} {err:8.3f}")

    mse = np.mean((x - x_true) ** 2)
    n_nonzero = np.sum(np.abs(x) > 1e-4)
    n_true_nonzero = np.sum(np.abs(x_true) > 1e-6)
    print(f"    MSE to true: {mse:.4f}  |  Nonzeros: {n_nonzero}/{len(x)} (true: {n_true_nonzero})")


def print_comparison_table(
    results: list[tuple[str, np.ndarray]],
    x_true: np.ndarray,
) -> None:
    """Print a summary table comparing methods."""
    print("\n" + "=" * 65)
    print("SUMMARY: Method Comparison")
    print("=" * 65)
    print(f"  {'Method':<25s} {'MSE to true':>12s} {'Nonzeros':>10s} {'||x||_1':>10s} {'||x||_2':>10s}")
    print(f"  {'-'*67}")

    n_true_nz = int(np.sum(np.abs(x_true) > 1e-6))
    print(f"  {'Ground truth':<25s} {'---':>12s} {n_true_nz:>10d} {np.linalg.norm(x_true, 1):>10.3f} {np.linalg.norm(x_true, 2):>10.3f}")

    for name, x in results:
        mse = np.mean((x - x_true) ** 2)
        n_nz = int(np.sum(np.abs(x) > 1e-4))
        l1 = np.linalg.norm(x, 1)
        l2 = np.linalg.norm(x, 2)
        print(f"  {name:<25s} {mse:>12.4f} {n_nz:>10d} {l1:>10.3f} {l2:>10.3f}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 1: Regularized Regression — OLS vs Ridge vs LASSO")
    print("=" * 65)

    A, b, x_true = generate_regression_data(
        n_samples=80, n_features=10, n_informative=4, noise_std=1.0
    )
    print(f"\nData: {A.shape[0]} samples, {A.shape[1]} features")
    print(f"True nonzero coefficients: {int(np.sum(np.abs(x_true) > 1e-6))}")

    # OLS baseline
    x_ols = solve_ols(A, b)
    print_coefficients("OLS (no regularization)", x_ols, x_true)

    results = [("OLS", x_ols)]

    # Ridge at multiple lambda values
    print("\n" + "-" * 65)
    print("Ridge Regression (L2 penalty)")
    print("-" * 65)
    for lambd in [0.1, 1.0, 10.0]:
        x_ridge = solve_ridge(A, b, lambd)
        label = f"Ridge (lambda={lambd})"
        print_coefficients(label, x_ridge, x_true)
        results.append((label, x_ridge))

    # LASSO at multiple lambda values
    print("\n" + "-" * 65)
    print("LASSO Regression (L1 penalty)")
    print("-" * 65)
    for lambd in [0.1, 1.0, 10.0]:
        x_lasso = solve_lasso(A, b, lambd)
        label = f"LASSO (lambda={lambd})"
        print_coefficients(label, x_lasso, x_true)
        results.append((label, x_lasso))

    print_comparison_table(results, x_true)


if __name__ == "__main__":
    main()
