"""Phase 2: Robust Portfolio Optimization.

Classical Markowitz portfolio optimization assumes the expected return vector mu
is known exactly. In practice, mu is estimated from historical data and carries
significant estimation error. Robust portfolio optimization protects against
uncertainty in mu by optimizing the WORST-CASE return over an uncertainty set.

This phase compares:
  1. Nominal Markowitz (trust the point estimate of mu)
  2. Box-robust portfolio (mu varies in a box)
  3. Ellipsoidal-robust portfolio (mu in an ellipsoid, becomes SOCP)
  4. Out-of-sample simulation to demonstrate the value of robustness

Key reference: Goldfarb & Iyengar (2003), Tutuncu & Koenig (2004).
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm


# ============================================================================
# Data generation: realistic asset universe
# ============================================================================

def generate_portfolio_data(
    n_assets: int = 8,
    n_samples: int = 252,
    seed: int = 42,
) -> dict:
    """Generate synthetic asset return data for portfolio optimization.

    Simulates daily returns for n_assets from a multivariate normal distribution,
    then estimates mu (expected return) and Sigma (covariance) from the sample.

    The "true" parameters are stored separately to simulate out-of-sample testing.

    Returns:
        dict with keys:
            mu_true:   (n,) true expected returns (annualized)
            Sigma_true: (n,n) true covariance matrix
            mu_hat:    (n,) sample estimate of mu (noisy)
            Sigma_hat: (n,n) sample estimate of Sigma
            returns:   (T, n) matrix of daily return samples
            n_assets:  int
    """
    rng = np.random.default_rng(seed)

    # True parameters (annualized)
    mu_true = rng.uniform(0.04, 0.15, size=n_assets)

    # Generate a random correlation structure
    L = rng.uniform(-0.3, 0.3, size=(n_assets, n_assets))
    Sigma_true = L @ L.T + 0.05 * np.eye(n_assets)
    # Scale to annualized volatility ~15-30%
    vols = rng.uniform(0.15, 0.35, size=n_assets)
    D = np.diag(vols / np.sqrt(np.diag(Sigma_true)))
    Sigma_true = D @ Sigma_true @ D

    # Generate daily returns (annualized -> daily)
    mu_daily = mu_true / 252
    Sigma_daily = Sigma_true / 252
    returns = rng.multivariate_normal(mu_daily, Sigma_daily, size=n_samples)

    # Sample estimates
    mu_hat = returns.mean(axis=0) * 252  # annualize
    Sigma_hat = np.cov(returns.T) * 252  # annualize

    return {
        "mu_true": mu_true,
        "Sigma_true": Sigma_true,
        "mu_hat": mu_hat,
        "Sigma_hat": Sigma_hat,
        "returns": returns,
        "n_assets": n_assets,
    }


def compute_uncertainty_radius(
    Sigma_hat: np.ndarray,
    n_samples: int,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute uncertainty set parameters for the expected return estimate.

    For a sample of size T, the estimation error of mu_hat is approximately:
        mu_hat ~ N(mu_true, Sigma_true / T)

    Box radius:      delta_j = z_{alpha/2} * sigma_j / sqrt(T)
    Ellipsoid matrix: S = Sigma_hat / T   (the covariance of mu_hat)

    Args:
        Sigma_hat: (n, n) estimated covariance
        n_samples: number of return observations T
        confidence: confidence level for the uncertainty set

    Returns:
        (delta_box, S_ellipsoid):
            delta_box: (n,) componentwise half-width of box uncertainty
            S_ellipsoid: (n, n) covariance matrix of the estimation error
    """
    from scipy.stats import norm as normal_dist

    z = normal_dist.ppf(1 - (1 - confidence) / 2)
    sigma = np.sqrt(np.diag(Sigma_hat))
    delta_box = z * sigma / np.sqrt(n_samples)

    S_ellipsoid = Sigma_hat / n_samples

    return delta_box, S_ellipsoid


# ============================================================================
# Nominal Markowitz portfolio
# ============================================================================

def solve_nominal_markowitz(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
) -> tuple[np.ndarray, float, float]:
    """Nominal Markowitz: minimize variance subject to return >= target.

    minimize    w^T Sigma w
    subject to  mu^T w >= target_return
                1^T w = 1
                w >= 0   (long-only)

    Returns:
        (w_opt, portfolio_return, portfolio_risk)
    """
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Sigma)),
        [mu @ w >= target_return, cp.sum(w) == 1, w >= 0],
    )
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Nominal Markowitz: {prob.status}")

    w_val = w.value
    ret = float(mu @ w_val)
    risk = float(np.sqrt(w_val @ Sigma @ w_val))
    return w_val, ret, risk


# ============================================================================
# TODO(human): Box-robust portfolio
# ============================================================================

def solve_robust_box_portfolio(
    mu_hat: np.ndarray,
    Sigma: np.ndarray,
    delta: np.ndarray,
    target_return: float,
) -> tuple[np.ndarray, float, float]:
    """Robust portfolio under BOX uncertainty on expected returns.

    The true expected return mu lies in:
        mu in [mu_hat - delta,  mu_hat + delta]   (componentwise)

    We want the portfolio return to be >= target_return even in the WORST CASE.

    Args:
        mu_hat: (n,) estimated expected returns
        Sigma: (n, n) covariance matrix (treated as known)
        delta: (n,) half-width of the box uncertainty set
        target_return: minimum acceptable return

    Returns:
        (w_opt, worst_case_return, portfolio_risk)
    """
    # TODO(human): Robust Portfolio with Box Uncertainty on Returns
    #
    # MATHEMATICAL DERIVATION:
    # The return constraint must hold for ALL mu in the box:
    #     mu^T w >= target_return   for all mu in [mu_hat - delta, mu_hat + delta]
    #
    # The worst case (minimum return) occurs when each mu_j takes its
    # LOWEST value for assets where w_j > 0, and HIGHEST for w_j < 0:
    #     min_{mu in box} mu^T w = mu_hat^T w - delta^T |w|
    #
    # Since we use long-only constraint (w >= 0), |w| = w, so:
    #     min_{mu in box} mu^T w = mu_hat^T w - delta^T w = (mu_hat - delta)^T w
    #
    # The robust return constraint is simply:
    #     (mu_hat - delta)^T w >= target_return
    #
    # FULL FORMULATION:
    #     minimize    w^T Sigma w                         (portfolio variance)
    #     subject to  (mu_hat - delta) @ w >= target_return  (worst-case return)
    #                 cp.sum(w) == 1                       (budget)
    #                 w >= 0                                (long-only)
    #
    # This is still a QP (same complexity class as nominal Markowitz).
    # The robust solution tilts AWAY from assets with high uncertainty (large delta)
    # — it penalizes assets whose return estimates are unreliable.
    #
    # IMPLEMENTATION STEPS:
    # 1. Compute mu_worst = mu_hat - delta (worst-case return vector)
    # 2. w = cp.Variable(n)
    # 3. Minimize cp.quad_form(w, Sigma) with constraint mu_worst @ w >= target_return
    # 4. Return (w.value, float(mu_worst @ w.value), risk)
    raise NotImplementedError("TODO(human): implement box-robust portfolio")


# ============================================================================
# TODO(human): Ellipsoidal-robust portfolio
# ============================================================================

def solve_robust_ellipsoidal_portfolio(
    mu_hat: np.ndarray,
    Sigma: np.ndarray,
    S: np.ndarray,
    kappa: float,
    target_return: float,
) -> tuple[np.ndarray, float, float]:
    """Robust portfolio under ELLIPSOIDAL uncertainty on expected returns.

    The true expected return mu lies in an ellipsoid:
        { mu : (mu - mu_hat)^T S^{-1} (mu - mu_hat) <= kappa^2 }

    Equivalently: mu = mu_hat + S^{1/2} u,  ||u||_2 <= kappa

    The worst-case return over the ellipsoid becomes a SOCP constraint.

    Args:
        mu_hat: (n,) estimated expected returns
        Sigma: (n, n) covariance matrix
        S: (n, n) covariance of the estimation error (S = Sigma_hat / T)
        kappa: radius of the ellipsoid (controls confidence level,
               e.g., kappa=1.96 for ~95% chi-squared coverage)
        target_return: minimum acceptable return

    Returns:
        (w_opt, worst_case_return, portfolio_risk)
    """
    # TODO(human): Robust Portfolio with Ellipsoidal Uncertainty (SOCP)
    #
    # MATHEMATICAL DERIVATION:
    # The return must hold for ALL mu in the ellipsoid:
    #     mu^T w >= target_return   for all (mu - mu_hat)^T S^{-1} (mu - mu_hat) <= kappa^2
    #
    # The worst case (minimum over the ellipsoid) is:
    #     min_{mu in ellipsoid} mu^T w = mu_hat^T w - kappa * ||S^{1/2} w||_2
    #
    # PROOF (using Cauchy-Schwarz):
    #   Write mu = mu_hat + S^{1/2} u where ||u|| <= kappa.
    #   Then mu^T w = mu_hat^T w + u^T S^{1/2} w.
    #   The minimum over ||u|| <= kappa is: mu_hat^T w - kappa * ||S^{1/2} w||_2
    #   (achieved when u = -kappa * S^{1/2} w / ||S^{1/2} w||)
    #
    # So the robust return constraint is:
    #     mu_hat^T w - kappa * ||S^{1/2} w||_2 >= target_return
    # Rearranged:
    #     mu_hat^T w - target_return >= kappa * ||S^{1/2} w||_2
    #
    # This is a SECOND-ORDER CONE CONSTRAINT!
    #
    # IMPLEMENTATION STEPS:
    # 1. Compute S_half = real part of sqrtm(S)  (matrix square root)
    #    Use: from scipy.linalg import sqrtm; S_half = np.real(sqrtm(S))
    # 2. w = cp.Variable(n)
    # 3. The SOCP constraint is:
    #        mu_hat @ w - target_return >= kappa * cp.norm(S_half @ w, 2)
    # 4. Also add cp.sum(w) == 1, w >= 0
    # 5. Minimize cp.quad_form(w, Sigma)
    # 6. Compute worst_case_return = mu_hat @ w.value - kappa * ||S_half @ w.value||
    # 7. Return (w.value, worst_case_return, risk)
    #
    # KEY INSIGHT: The ellipsoidal model accounts for CORRELATION in estimation
    # errors. If two assets' return estimates are correlated, the ellipsoid
    # captures that — unlike the box which treats each asset independently.
    # This makes ellipsoidal robustness less conservative than box when
    # correlations are present.
    raise NotImplementedError("TODO(human): implement ellipsoidal-robust portfolio")


# ============================================================================
# Efficient frontier comparison
# ============================================================================

def trace_efficient_frontier(
    data: dict,
    n_points: int = 20,
) -> dict:
    """Trace nominal and robust efficient frontiers.

    Sweeps target_return from min to max achievable return and collects
    (risk, return) pairs for each method.

    Returns dict mapping method name -> list of (risk, return) points.
    """
    mu_hat, Sigma_hat = data["mu_hat"], data["Sigma_hat"]
    n_samples = data["returns"].shape[0]
    delta_box, S_ell = compute_uncertainty_radius(Sigma_hat, n_samples)

    # Return range: from min single-asset return to max single-asset return
    ret_min = float(np.min(mu_hat)) * 0.5
    ret_max = float(np.max(mu_hat)) * 0.95
    targets = np.linspace(ret_min, ret_max, n_points)

    frontiers = {"Nominal": [], "Box-Robust": [], "Ellipsoidal-Robust": []}

    for target in targets:
        # Nominal
        try:
            _, ret, risk = solve_nominal_markowitz(mu_hat, Sigma_hat, target)
            frontiers["Nominal"].append((risk, ret))
        except Exception:
            pass

        # Box-robust
        try:
            _, ret, risk = solve_robust_box_portfolio(mu_hat, Sigma_hat, delta_box, target)
            frontiers["Box-Robust"].append((risk, ret))
        except Exception:
            pass

        # Ellipsoidal-robust
        try:
            _, ret, risk = solve_robust_ellipsoidal_portfolio(
                mu_hat, Sigma_hat, S_ell, kappa=1.96, target_return=target,
            )
            frontiers["Ellipsoidal-Robust"].append((risk, ret))
        except Exception:
            pass

    return frontiers


# ============================================================================
# Out-of-sample simulation
# ============================================================================

def out_of_sample_simulation(
    data: dict,
    n_future_samples: int = 1000,
    target_return: float = 0.08,
    seed: int = 999,
) -> dict:
    """Simulate out-of-sample performance using the TRUE distribution.

    For each portfolio method, compute the portfolio weights using the
    ESTIMATED parameters, then evaluate performance using the TRUE
    distribution parameters.

    Returns dict mapping method name -> dict with:
        weights, true_return, true_risk, shortfall_prob
    """
    rng = np.random.default_rng(seed)
    mu_hat, Sigma_hat = data["mu_hat"], data["Sigma_hat"]
    mu_true, Sigma_true = data["mu_true"], data["Sigma_true"]
    n_samples = data["returns"].shape[0]
    delta_box, S_ell = compute_uncertainty_radius(Sigma_hat, n_samples)

    results = {}

    # Solve portfolios using estimated data
    portfolios = {}
    try:
        w, _, _ = solve_nominal_markowitz(mu_hat, Sigma_hat, target_return)
        portfolios["Nominal"] = w
    except Exception:
        pass

    try:
        w, _, _ = solve_robust_box_portfolio(mu_hat, Sigma_hat, delta_box, target_return)
        portfolios["Box-Robust"] = w
    except Exception:
        pass

    try:
        w, _, _ = solve_robust_ellipsoidal_portfolio(
            mu_hat, Sigma_hat, S_ell, kappa=1.96, target_return=target_return,
        )
        portfolios["Ellipsoidal-Robust"] = w
    except Exception:
        pass

    # Evaluate on true distribution
    daily_mu = mu_true / 252
    daily_Sigma = Sigma_true / 252
    future_returns = rng.multivariate_normal(daily_mu, daily_Sigma, size=n_future_samples)

    for name, w in portfolios.items():
        # Portfolio daily returns
        port_daily = future_returns @ w
        port_annual_return = float(np.mean(port_daily) * 252)
        port_annual_risk = float(np.std(port_daily) * np.sqrt(252))

        # Probability of falling below target (shortfall)
        annual_returns = np.sum(
            future_returns.reshape(-1, 252 // 10, 10, data["n_assets"]).mean(axis=2) * 252,
            axis=2,
        ) if n_future_samples >= 252 else port_daily * 252
        # Simplified: use daily returns annualized
        shortfall = float(np.mean(port_daily * 252 < target_return))

        results[name] = {
            "weights": w,
            "true_return": port_annual_return,
            "true_risk": port_annual_risk,
            "shortfall_prob": shortfall,
        }

    return results


# ============================================================================
# Display helpers
# ============================================================================

def print_portfolio(name: str, w: np.ndarray, ret: float, risk: float) -> None:
    """Print portfolio weights and risk-return profile."""
    print(f"\n  {name}:")
    print(f"    Return: {ret * 100:.2f}%   Risk: {risk * 100:.2f}%   Sharpe: {ret / risk:.3f}")
    # Show top 3 holdings
    top_idx = np.argsort(w)[::-1][:3]
    holdings = ", ".join(f"Asset{i}={w[i]*100:.1f}%" for i in top_idx)
    print(f"    Top holdings: {holdings}")
    print(f"    Concentration (||w||_2): {np.linalg.norm(w):.4f}")


def print_oos_results(results: dict) -> None:
    """Print out-of-sample simulation results."""
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE PERFORMANCE (using TRUE distribution)")
    print("=" * 72)
    print(f"  {'Method':<25s} {'True Ret%':>10s} {'True Risk%':>10s} {'Sharpe':>8s} {'P(shortfall)':>12s}")
    print(f"  {'-' * 68}")

    for name, r in results.items():
        ret_pct = r["true_return"] * 100
        risk_pct = r["true_risk"] * 100
        sharpe = r["true_return"] / r["true_risk"] if r["true_risk"] > 0 else 0
        sf_pct = r["shortfall_prob"] * 100
        print(f"  {name:<25s} {ret_pct:>10.2f} {risk_pct:>10.2f} {sharpe:>8.3f} {sf_pct:>11.1f}%")

    print("\n  KEY OBSERVATIONS:")
    print("  - Nominal portfolio may overfit to estimated returns")
    print("  - Robust portfolios are more diversified, less sensitive to estimation error")
    print("  - Robust portfolios often have BETTER out-of-sample Sharpe ratios")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 2: Robust Portfolio Optimization")
    print("=" * 70)

    data = generate_portfolio_data(n_assets=8, n_samples=252)
    mu_hat, Sigma_hat = data["mu_hat"], data["Sigma_hat"]
    n_samples = data["returns"].shape[0]
    delta_box, S_ell = compute_uncertainty_radius(Sigma_hat, n_samples)

    print(f"\nAsset universe: {data['n_assets']} assets, {n_samples} daily observations")
    print(f"Estimated returns: [{', '.join(f'{m*100:.1f}%' for m in mu_hat)}]")
    print(f"Box uncertainty:   [{', '.join(f'{d*100:.2f}%' for d in delta_box)}]")

    target = 0.08  # 8% target return

    # --- Nominal Markowitz ---
    print("\n" + "-" * 70)
    print("1. Nominal Markowitz Portfolio")
    print("-" * 70)
    w_nom, ret_nom, risk_nom = solve_nominal_markowitz(mu_hat, Sigma_hat, target)
    print_portfolio("Nominal", w_nom, ret_nom, risk_nom)

    # --- Box-robust ---
    print("\n" + "-" * 70)
    print("2. Box-Robust Portfolio")
    print("-" * 70)
    w_box, ret_box, risk_box = solve_robust_box_portfolio(
        mu_hat, Sigma_hat, delta_box, target,
    )
    print_portfolio("Box-Robust", w_box, ret_box, risk_box)

    # --- Ellipsoidal-robust ---
    print("\n" + "-" * 70)
    print("3. Ellipsoidal-Robust Portfolio")
    print("-" * 70)
    for kappa in [1.0, 1.96, 3.0]:
        try:
            w_ell, ret_ell, risk_ell = solve_robust_ellipsoidal_portfolio(
                mu_hat, Sigma_hat, S_ell, kappa=kappa, target_return=target,
            )
            print_portfolio(f"Ellipsoidal (kappa={kappa})", w_ell, ret_ell, risk_ell)
        except Exception as e:
            print(f"\n  Ellipsoidal (kappa={kappa}): FAILED — {e}")

    # --- Efficient frontier ---
    print("\n" + "-" * 70)
    print("4. Efficient Frontier Comparison")
    print("-" * 70)
    frontiers = trace_efficient_frontier(data)
    for name, points in frontiers.items():
        if points:
            risks, rets = zip(*points)
            print(f"\n  {name}: {len(points)} points")
            print(f"    Risk range: [{min(risks)*100:.2f}%, {max(risks)*100:.2f}%]")
            print(f"    Return range: [{min(rets)*100:.2f}%, {max(rets)*100:.2f}%]")

    # --- Out-of-sample ---
    print("\n" + "-" * 70)
    print("5. Out-of-Sample Simulation")
    print("-" * 70)
    oos = out_of_sample_simulation(data, target_return=target)
    print_oos_results(oos)


if __name__ == "__main__":
    main()
