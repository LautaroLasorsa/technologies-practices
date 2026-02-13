"""Phase 2: Portfolio Optimization — Markowitz Mean-Variance with CVXPY.

Implements the classic Markowitz portfolio model: minimize portfolio variance
(risk) subject to a target expected return and budget constraints. Traces the
efficient frontier by sweeping over target returns.

DCP atoms used:
  - cp.quad_form(w, Sigma): convex when Sigma is PSD — w^T Sigma w
  - cp.sum(w): affine
  - Dual values: constraint.dual_value for marginal cost of increasing target return
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Data: 5 hypothetical assets with returns and covariance
# ============================================================================

def generate_market_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic market data for 5 assets.

    Returns:
        mu: (5,) expected annual returns
        Sigma: (5, 5) covariance matrix (guaranteed PSD)
        names: asset names
    """
    rng = np.random.default_rng(seed)

    names = ["Tech", "Healthcare", "Energy", "Bonds", "RealEstate"]

    # Expected annual returns (%)
    mu = np.array([0.12, 0.10, 0.08, 0.04, 0.07])

    # Generate a random covariance matrix that's PSD
    # Start with volatilities (annual std dev)
    vols = np.array([0.20, 0.15, 0.18, 0.05, 0.12])

    # Correlation matrix with realistic structure
    corr = np.array([
        [1.00, 0.40, 0.30, -0.10, 0.25],
        [0.40, 1.00, 0.20, 0.05, 0.15],
        [0.30, 0.20, 1.00, -0.05, 0.20],
        [-0.10, 0.05, -0.05, 1.00, 0.10],
        [0.25, 0.15, 0.20, 0.10, 1.00],
    ])

    # Sigma = diag(vols) @ corr @ diag(vols)
    D = np.diag(vols)
    Sigma = D @ corr @ D

    # Ensure PSD by adding small epsilon to diagonal
    Sigma += 1e-8 * np.eye(len(mu))

    return mu, Sigma, names


# ============================================================================
# TODO(human): Markowitz portfolio optimization
# ============================================================================

def solve_markowitz(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
) -> tuple[np.ndarray, float, float, float | None]:
    """Solve the Markowitz mean-variance portfolio optimization.

    minimize    w^T Sigma w          (portfolio variance)
    subject to: mu^T w >= target_return
                sum(w) == 1
                w >= 0               (long-only)

    Args:
        mu: (n,) expected returns for each asset
        Sigma: (n, n) covariance matrix (PSD)
        target_return: minimum required expected return

    Returns:
        Tuple of:
        - weights: (n,) optimal portfolio weights
        - risk: portfolio standard deviation (sqrt of variance)
        - expected_return: mu^T w (actual return, may exceed target)
        - return_dual: dual value of the return constraint (None if unavailable)
    """
    # TODO(human): Markowitz Portfolio with CVXPY
    # minimize w^T Sigma w  (portfolio variance)
    # subject to: mu^T w >= target_return
    #             sum(w) == 1
    #             w >= 0
    #
    # In CVXPY:
    #   w = cp.Variable(n)
    #   risk = cp.quad_form(w, Sigma)  # w^T Sigma w (convex if Sigma PSD)
    #   ret = mu @ w
    #   return_constr = ret >= target_return
    #   prob = cp.Problem(cp.Minimize(risk), [return_constr, cp.sum(w) == 1, w >= 0])
    #
    # DCP: quad_form(w, PSD_matrix) is convex. ✓
    #
    # After solving: access constraint dual values for the "price of risk"
    #   return_constr.dual_value = marginal cost of increasing target return
    #   This is the slope of the efficient frontier at this point.
    #
    # Return (weights, sqrt(risk.value), actual_return, return_dual)
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_portfolio(
    weights: np.ndarray,
    risk: float,
    expected_return: float,
    return_dual: float | None,
    target_return: float,
    names: list[str],
) -> None:
    """Display a single portfolio solution."""
    print(f"\n  Target return: {target_return*100:.1f}%")
    print(f"  Achieved return: {expected_return*100:.2f}%")
    print(f"  Portfolio risk (std): {risk*100:.2f}%")
    if return_dual is not None:
        print(f"  Return constraint dual: {return_dual:.4f}")

    print(f"  Weights:")
    for name, w in zip(names, weights):
        if abs(w) > 1e-4:
            bar = "#" * int(w * 40)
            print(f"    {name:<12s} {w*100:6.1f}%  {bar}")


def print_efficient_frontier(
    frontier: list[tuple[float, float, np.ndarray]],
) -> None:
    """Print a text-based efficient frontier plot.

    Args:
        frontier: list of (risk, return, weights) tuples
    """
    print("\n" + "=" * 65)
    print("EFFICIENT FRONTIER (Risk vs Return)")
    print("=" * 65)

    if not frontier:
        print("  No frontier points computed.")
        return

    risks = [r for r, _, _ in frontier]
    returns = [ret for _, ret, _ in frontier]

    min_risk, max_risk = min(risks), max(risks)
    min_ret, max_ret = min(returns), max(returns)

    # Text-based scatter plot
    rows, cols = 20, 55
    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    for risk, ret, _ in frontier:
        if max_risk > min_risk:
            c = int((risk - min_risk) / (max_risk - min_risk) * (cols - 1))
        else:
            c = cols // 2
        if max_ret > min_ret:
            r = rows - 1 - int((ret - min_ret) / (max_ret - min_ret) * (rows - 1))
        else:
            r = rows // 2
        r = max(0, min(rows - 1, r))
        c = max(0, min(cols - 1, c))
        grid[r][c] = "*"

    # Print with axes
    for r in range(rows):
        ret_val = max_ret - r * (max_ret - min_ret) / (rows - 1) if rows > 1 else max_ret
        label = f"{ret_val*100:5.1f}% |"
        print(f"  {label}{''.join(grid[r])}|")

    print(f"  {'':>7s}+{'-'*cols}+")
    print(f"  {'':>8s}{min_risk*100:<6.1f}%{' '*(cols-14)}{max_risk*100:>6.1f}%")
    print(f"  {'':>8s}{'Risk (std dev)':^{cols}s}")

    # Summary table
    print(f"\n  {'Target':>8s} {'Return':>8s} {'Risk':>8s} {'Sharpe*':>8s}")
    print(f"  {'-'*36}")
    rf = 0.02  # risk-free rate for Sharpe ratio
    for risk, ret, _ in frontier:
        sharpe = (ret - rf) / risk if risk > 1e-8 else float("inf")
        print(f"  {ret*100:7.1f}% {ret*100:7.2f}% {risk*100:7.2f}% {sharpe:8.2f}")
    print(f"  * Sharpe ratio with rf = {rf*100:.0f}%")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 2: Portfolio Optimization — Markowitz Mean-Variance")
    print("=" * 65)

    mu, Sigma, names = generate_market_data()

    print("\nAsset data:")
    print(f"  {'Asset':<12s} {'E[return]':>10s} {'Volatility':>10s}")
    print(f"  {'-'*34}")
    for i, name in enumerate(names):
        print(f"  {name:<12s} {mu[i]*100:9.1f}% {np.sqrt(Sigma[i,i])*100:9.1f}%")

    # Solve for a single target return
    print("\n" + "-" * 65)
    print("Single Portfolio (target = 8%)")
    print("-" * 65)

    weights, risk, ret, dual = solve_markowitz(mu, Sigma, target_return=0.08)
    print_portfolio(weights, risk, ret, dual, 0.08, names)

    # Trace efficient frontier
    print("\n" + "-" * 65)
    print("Tracing Efficient Frontier")
    print("-" * 65)

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, 15)

    frontier = []
    for target in targets:
        try:
            w, r, actual_ret, _ = solve_markowitz(mu, Sigma, target)
            frontier.append((r, actual_ret, w))
        except Exception as e:
            print(f"  target={target*100:.1f}%: infeasible ({e})")

    print_efficient_frontier(frontier)


if __name__ == "__main__":
    main()
