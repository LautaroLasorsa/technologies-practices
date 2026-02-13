"""Phase 3: Markowitz Mean-Variance Portfolio Optimization (NLP/QP).

Minimize portfolio variance subject to achieving a target return.
The objective is quadratic (w^T * Sigma * w) -- this is a QP, solvable
by HiGHS or Ipopt.
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


# ── Stock data ─────────────────────────────────────────────────────────────

STOCK_NAMES = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Expected annual returns (decimal)
EXPECTED_RETURNS = {
    "AAPL": 0.12,
    "GOOGL": 0.10,
    "MSFT": 0.11,
    "AMZN": 0.14,
    "TSLA": 0.18,
}

# Covariance matrix (annual, symmetric positive semi-definite)
# Diagonal = variance, off-diagonal = covariance between pairs
COVARIANCE = {
    ("AAPL", "AAPL"): 0.04,   ("AAPL", "GOOGL"): 0.015,  ("AAPL", "MSFT"): 0.018,
    ("AAPL", "AMZN"): 0.020,  ("AAPL", "TSLA"): 0.025,
    ("GOOGL", "AAPL"): 0.015, ("GOOGL", "GOOGL"): 0.035,  ("GOOGL", "MSFT"): 0.012,
    ("GOOGL", "AMZN"): 0.016, ("GOOGL", "TSLA"): 0.020,
    ("MSFT", "AAPL"): 0.018,  ("MSFT", "GOOGL"): 0.012,   ("MSFT", "MSFT"): 0.032,
    ("MSFT", "AMZN"): 0.014,  ("MSFT", "TSLA"): 0.018,
    ("AMZN", "AAPL"): 0.020,  ("AMZN", "GOOGL"): 0.016,   ("AMZN", "MSFT"): 0.014,
    ("AMZN", "AMZN"): 0.055,  ("AMZN", "TSLA"): 0.030,
    ("TSLA", "AAPL"): 0.025,  ("TSLA", "GOOGL"): 0.020,   ("TSLA", "MSFT"): 0.018,
    ("TSLA", "AMZN"): 0.030,  ("TSLA", "TSLA"): 0.090,
}


# ── Solve portfolio optimization ──────────────────────────────────────────


def solve_portfolio(
    returns: dict[str, float],
    covariance: dict[tuple[str, str], float],
    target_return: float,
) -> pyo.ConcreteModel | None:
    # TODO(human): Markowitz Mean-Variance Portfolio Optimization (NLP)
    #
    # Minimize portfolio variance subject to achieving target return.
    #
    # Variables: w[i] = weight of asset i in portfolio (continuous, [0, 1])
    #
    # Objective (NONLINEAR): minimize w^T * Sigma * w
    #   where Sigma is the covariance matrix
    #   In Pyomo: sum(cov[i,j] * w[i] * w[j] for i in stocks for j in stocks)
    #
    # Constraints:
    #   1. Target return: sum(returns[i] * w[i] for i in stocks) >= target_return
    #   2. Budget: sum(w[i] for i in stocks) == 1
    #   3. No short selling: 0 <= w[i] <= 1 (use bounds=(0, 1) in Var)
    #
    # This is a Quadratic Program (QP) -- convex, solvable efficiently.
    # HiGHS can solve QPs! Or use Ipopt for general NLP.
    #
    # Use SolverFactory('highs') or SolverFactory('ipopt').
    # Check solver availability and fall back if needed.
    #
    # Steps:
    #   1. model = pyo.ConcreteModel()
    #   2. model.stocks = pyo.Set(initialize=list(returns.keys()))
    #   3. model.w = pyo.Var(model.stocks, bounds=(0, 1))
    #   4. Define quadratic objective using sum over i,j
    #   5. Add budget constraint (weights sum to 1)
    #   6. Add return constraint (expected return >= target)
    #   7. Solve and return model (or None if infeasible)
    #
    # After solving, vary target_return to trace the efficient frontier:
    # the set of portfolios with minimum variance for each return level.
    raise NotImplementedError


# ── Display helpers ────────────────────────────────────────────────────────


def display_portfolio(model: pyo.ConcreteModel, target_return: float) -> None:
    """Print portfolio allocation for a single solution."""
    variance = pyo.value(model.obj)
    std_dev = variance ** 0.5
    actual_return = sum(
        EXPECTED_RETURNS[s] * pyo.value(model.w[s]) for s in model.stocks
    )

    print(f"\n  Target return: {target_return:.1%}")
    print(f"  Portfolio variance: {variance:.6f}  (std dev: {std_dev:.4f})")
    print(f"  Achieved return: {actual_return:.4%}")
    print(f"  {'Stock':<8} {'Weight':>8} {'Return':>8} {'Contrib':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for s in model.stocks:
        w = pyo.value(model.w[s])
        if w > 1e-6:
            contrib = EXPECTED_RETURNS[s] * w
            print(f"  {s:<8} {w:>7.1%} {EXPECTED_RETURNS[s]:>7.1%} {contrib:>9.4%}")


def display_efficient_frontier(results: list[tuple[float, float, float]]) -> None:
    """Print the efficient frontier as a table."""
    print(f"\n{'='*50}")
    print(f"  Efficient Frontier")
    print(f"{'='*50}")
    print(f"  {'Target':>8} {'Return':>8} {'StdDev':>8} {'Sharpe*':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    rf = 0.03  # risk-free rate for Sharpe ratio
    for target, ret, std in results:
        sharpe = (ret - rf) / std if std > 1e-8 else 0.0
        print(f"  {target:>7.1%} {ret:>7.2%} {std:>7.4f} {sharpe:>8.3f}")

    print(f"\n  *Sharpe ratio = (return - {rf:.0%} risk-free) / std_dev")
    print(f"  Higher Sharpe = better risk-adjusted return.")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"{'='*50}")
    print(f"  Markowitz Portfolio Optimization")
    print(f"{'='*50}")
    print(f"\n  Stocks: {', '.join(STOCK_NAMES)}")
    print(f"  Expected returns: {', '.join(f'{EXPECTED_RETURNS[s]:.0%}' for s in STOCK_NAMES)}")

    # Sweep target returns to trace the efficient frontier
    min_return = min(EXPECTED_RETURNS.values())
    max_return = max(EXPECTED_RETURNS.values())
    targets = [min_return + i * (max_return - min_return) / 8 for i in range(9)]

    frontier_points: list[tuple[float, float, float]] = []

    for target in targets:
        model = solve_portfolio(EXPECTED_RETURNS, COVARIANCE, target)
        if model is not None:
            variance = pyo.value(model.obj)
            std_dev = variance ** 0.5
            actual_return = sum(
                EXPECTED_RETURNS[s] * pyo.value(model.w[s]) for s in model.stocks
            )
            frontier_points.append((target, actual_return, std_dev))

            # Show detailed allocation for first, middle, and last
            if target in (targets[0], targets[4], targets[-1]):
                display_portfolio(model, target)

    if frontier_points:
        display_efficient_frontier(frontier_points)
    else:
        print("\nERROR: No solutions found. Check solver availability.")

    print(f"\n{'='*50}")
    print("  Key insight: as target return increases, variance increases")
    print("  nonlinearly. Diversification reduces risk -- the portfolio")
    print("  std dev is LESS than the weighted average of individual std devs.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
