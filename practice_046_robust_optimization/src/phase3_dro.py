"""Phase 3: Distributionally Robust Optimization (DRO).

DRO bridges the gap between:
  - Stochastic programming (assumes a KNOWN probability distribution)
  - Robust optimization (assumes NO distribution, just a set of possible outcomes)

DRO works with an AMBIGUITY SET of distributions: a family of distributions
that are "close" to the empirical distribution. It optimizes against the
worst-case distribution in this ambiguity set.

Two main types of ambiguity sets:
  1. Moment-based: all distributions with given mean/covariance bounds (-> SDP)
  2. Wasserstein: all distributions within Wasserstein distance of empirical (-> LP/SOCP)

Key references: Delage & Ye (2010), Mohajerin Esfahani & Kuhn (2018).
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Problem: Newsvendor with uncertain demand
# ============================================================================

def generate_newsvendor_data(
    n_scenarios: int = 50,
    seed: int = 42,
) -> dict:
    """Generate data for a newsvendor problem with uncertain demand.

    The newsvendor must decide how many units to order BEFORE observing demand.
    - Ordering cost: c per unit
    - Selling price: p per unit (p > c)
    - Salvage value: s per unsold unit (s < c)

    Cost function: h(x, d) = c*x - p*min(x, d) - s*max(x-d, 0)
                            = c*x - p*min(x,d) - s*(x-d)^+
    Equivalently:  h(x, d) = (c-s)*x - (p-s)*min(x, d) + s*d  (ignore constant)
    Overage cost:  c_o = c - s   (cost of ordering too much)
    Underage cost: c_u = p - c   (cost of ordering too little)

    Returns dict with:
        c_order:     ordering cost per unit
        p_sell:      selling price per unit
        s_salvage:   salvage value per unit
        demands:     (n_scenarios,) empirical demand samples
        true_mean:   true demand mean (for evaluation)
        true_std:    true demand standard deviation (for evaluation)
    """
    rng = np.random.default_rng(seed)

    c_order = 5.0
    p_sell = 12.0
    s_salvage = 2.0

    # True demand distribution (lognormal — heavy-tailed, realistic)
    true_mean = 100.0
    true_std = 30.0
    # Lognormal parameters
    sigma2 = np.log(1 + (true_std / true_mean) ** 2)
    mu_ln = np.log(true_mean) - sigma2 / 2
    sigma_ln = np.sqrt(sigma2)

    demands = rng.lognormal(mu_ln, sigma_ln, size=n_scenarios)

    return {
        "c_order": c_order,
        "p_sell": p_sell,
        "s_salvage": s_salvage,
        "demands": demands,
        "true_mean": true_mean,
        "true_std": true_std,
    }


# ============================================================================
# Baseline: Sample Average Approximation (SAA)
# ============================================================================

def solve_newsvendor_saa(data: dict) -> tuple[float, float]:
    """Solve newsvendor using Sample Average Approximation.

    Minimize the AVERAGE cost over the empirical scenarios:
        min_x  (1/N) sum_i h(x, d_i)

    This is a standard stochastic programming approach that trusts the
    empirical distribution exactly.

    Returns:
        (x_opt, expected_cost)
    """
    c, p, s = data["c_order"], data["p_sell"], data["s_salvage"]
    demands = data["demands"]
    N = len(demands)

    x = cp.Variable()
    # Cost for each scenario: overage + underage
    # h(x, d_i) = c*x - p*min(x, d_i) - s*max(x - d_i, 0)
    # Using CVXPY: min(x, d_i) = x - max(x - d_i, 0)
    # h(x, d_i) = c*x - p*(x - max(x-d_i, 0)) - s*max(x-d_i, 0)
    #           = (c-p)*x + (p-s)*max(x-d_i, 0)
    # But we also need underage: missed sales when d > x
    # Full cost: (c-s)*max(x-d_i, 0) + (p-c)*max(d_i-x, 0) + (c-s*1)*x ... no
    # Simplest: cost = c*x - p*min(x, d_i) - s*max(x-d_i, 0)
    # In CVXPY: min(x, d_i) is concave, so -min is convex -> use cp.maximum
    # cost_i = c*x - p*(x - cp.maximum(x - d_i, 0)) - s*cp.maximum(x - d_i, 0)
    #        = c*x - p*x + p*cp.maximum(x - d_i, 0) - s*cp.maximum(x - d_i, 0)
    #        = (c-p)*x + (p-s)*cp.maximum(x - d_i, 0)
    # But this misses the revenue from selling min(x, d_i) units...
    # Actually, profit = p*min(x,d) + s*max(x-d,0) - c*x
    # We MINIMIZE negative profit = c*x - p*min(x,d) - s*max(x-d, 0)
    # = c*x - p*x + p*max(x-d,0) - s*max(x-d,0)  [since min(x,d)=x-max(x-d,0)]
    # = (c-p)*x + (p-s)*max(x-d,0)
    # This is convex in x (max is convex, linear combination with positive coeff).

    overage = cp.maximum(x - demands, 0)  # vector of (N,) overages
    cost = (c - p) * x + (p - s) * (1.0 / N) * cp.sum(overage)

    prob = cp.Problem(cp.Minimize(cost), [x >= 0])
    prob.solve()

    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"SAA: {prob.status}")

    return float(x.value), float(prob.value)


# ============================================================================
# Classical Robust (worst-case over support)
# ============================================================================

def solve_newsvendor_robust(
    data: dict,
    d_min: float | None = None,
    d_max: float | None = None,
) -> tuple[float, float]:
    """Solve newsvendor with classical robust optimization.

    Minimize WORST-CASE cost over demand in [d_min, d_max].
    If d_min/d_max not given, use the range of observed samples.

    min_x max_{d in [d_min, d_max]} h(x, d)

    For the newsvendor, the worst case is d=d_min for overage, d=d_max for
    underage. The cost is piecewise linear:
    - If x <= d_min: worst case is underage at d=d_max, cost = (p-c)*(d_max-x) + (c-s)*0
      Actually cost(x,d_max) = (c-p)*x + (p-s)*0 = (c-p)*x  (no overage)
    - If x >= d_max: worst case is overage at d=d_min, cost = (c-p)*x + (p-s)*(x-d_min)
    - In between: worst = max(cost(x,d_min), cost(x,d_max))

    Returns:
        (x_opt, worst_case_cost)
    """
    demands = data["demands"]
    c, p, s = data["c_order"], data["p_sell"], data["s_salvage"]

    if d_min is None:
        d_min = float(np.min(demands)) * 0.8
    if d_max is None:
        d_max = float(np.max(demands)) * 1.2

    x = cp.Variable()
    # Worst-case cost = max over extreme scenarios
    cost_low = (c - p) * x + (p - s) * cp.maximum(x - d_min, 0)
    cost_high = (c - p) * x + (p - s) * cp.maximum(x - d_max, 0)
    worst_cost = cp.maximum(cost_low, cost_high)

    prob = cp.Problem(cp.Minimize(worst_cost), [x >= 0])
    prob.solve()

    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Robust: {prob.status}")

    return float(x.value), float(prob.value)


# ============================================================================
# TODO(human): Moment-based DRO
# ============================================================================

def solve_newsvendor_moment_dro(
    data: dict,
    gamma1: float = 0.2,
    gamma2: float = 0.3,
) -> tuple[float, float]:
    """Distributionally Robust newsvendor with MOMENT-BASED ambiguity set.

    The ambiguity set contains all distributions P such that:
        E_P[d] in [mu_hat - gamma1*mu_hat,  mu_hat + gamma1*mu_hat]
        E_P[d^2] <= (1 + gamma2) * (sigma_hat^2 + mu_hat^2)

    (Mean is within gamma1 fraction of sample mean, second moment bounded.)

    For the newsvendor cost h(x, d) = (c-p)*x + (p-s)*max(x-d, 0), the
    worst-case expected cost over the moment ambiguity set can be reformulated
    as a finite convex program.

    Args:
        data: problem data dict
        gamma1: relative tolerance on the mean (0.2 = 20%)
        gamma2: relative tolerance on the second moment (0.3 = 30%)

    Returns:
        (x_opt, worst_case_expected_cost)
    """
    # TODO(human): Moment-Based Distributionally Robust Optimization
    #
    # MATHEMATICAL BACKGROUND:
    # We want:  min_x  max_{P in ambiguity}  E_P[h(x, d)]
    #
    # The ambiguity set is defined by moment constraints:
    #   P_moment = { P : E[d] in [mu_lo, mu_hi],  E[d^2] <= M2_up,  d >= 0 }
    #
    # where:
    #   mu_hat = np.mean(demands)
    #   sigma_hat = np.std(demands)
    #   mu_lo = mu_hat * (1 - gamma1)
    #   mu_hi = mu_hat * (1 + gamma1)
    #   M2_up = (1 + gamma2) * (sigma_hat**2 + mu_hat**2)
    #
    # For the newsvendor with cost h(x, d) = (c-p)*x + (p-s)*max(x-d, 0):
    # h(x, d) is convex piecewise-linear in d:
    #   h(x, d) = (c-p)*x + (p-s)*(x-d)  if d <= x   (overage)
    #   h(x, d) = (c-p)*x                  if d > x    (no overage, but underage)
    #
    # The worst-case E[h(x,d)] over distributions with known mean and
    # second moment bounds is a classical "generalized Chebyshev" problem.
    #
    # By strong duality, the worst-case expected cost equals:
    #     min_{x, lam, eta}  (c-p)*x + (p-s) * (
    #         max over {mu, sigma2} satisfying moment bounds of:
    #         E[max(x - d, 0)]
    #     )
    #
    # TRACTABLE REFORMULATION (Scarf's bound for single-item newsvendor):
    # For a distribution with mean mu and variance sigma^2, the worst-case
    # E[max(x-d, 0)] over all distributions on [0, inf) is bounded by:
    #
    #   E[max(x-d, 0)] <= (x - mu + sqrt((x-mu)^2 + sigma^2)) / 2
    #                    = ((x-mu) + sqrt((x-mu)^2 + sigma^2)) / 2
    #
    # (This is Scarf's 1958 bound, tight for a two-point distribution.)
    #
    # IMPLEMENTATION APPROACH:
    # Since the exact SDP reformulation of general moment-DRO is complex,
    # we use a practical approach:
    # 1. Compute mu_lo, mu_hi, M2_up from data
    # 2. Search over a grid of (mu, sigma^2) within the ambiguity set
    # 3. For each (mu, sigma^2), compute the Scarf bound
    # 4. Take the worst case (maximum) over the grid
    #
    # CONCRETE STEPS:
    # 1. mu_hat = np.mean(demands), sigma_hat = np.std(demands)
    # 2. mu_lo = mu_hat * (1 - gamma1), mu_hi = mu_hat * (1 + gamma1)
    # 3. M2_up = (1 + gamma2) * (sigma_hat**2 + mu_hat**2)
    # 4. For a given x, the worst-case E[max(x-d,0)] via Scarf's bound:
    #    Over mu in [mu_lo, mu_hi] and sigma^2 = M2_up - mu^2 (tightest bound):
    #      scarf(x, mu, sig) = ((x - mu) + sqrt((x - mu)^2 + sig^2)) / 2
    # 5. Use CVXPY with an auxiliary variable t for the Scarf bound:
    #    x_var = cp.Variable(); t = cp.Variable()
    #    For each (mu_val, sig_val) on a grid of the ambiguity set:
    #      constraints.append(t >= (p-s) * scarf_bound_expr)
    #    Note: scarf_bound with fixed mu, sig is convex in x.
    #    With CVXPY: ((x - mu) + cp.norm(cp.hstack([x - mu, sigma]), 2)) / 2
    #    which is convex (sum of affine and norm).
    # 6. Minimize (c - p) * x_var + t, subject to x_var >= 0 and all scarf constraints
    #
    # KEY INSIGHT: Moment-based DRO is more conservative than SAA but less
    # than worst-case robust. It only requires knowing the mean and variance
    # of demand (or bounds on them), not the full distribution shape.
    raise NotImplementedError("TODO(human): implement moment-based DRO")


# ============================================================================
# TODO(human): Wasserstein DRO
# ============================================================================

def solve_newsvendor_wasserstein_dro(
    data: dict,
    epsilon: float = 5.0,
) -> tuple[float, float]:
    """Distributionally Robust newsvendor with WASSERSTEIN ambiguity set.

    The ambiguity set contains all distributions P whose type-1 Wasserstein
    distance from the empirical distribution P_hat is at most epsilon:
        B_epsilon(P_hat) = { P : W_1(P, P_hat) <= epsilon }

    For piecewise-linear convex cost functions, the worst-case expected cost
    over a Wasserstein ball has a TRACTABLE LP reformulation.

    Args:
        data: problem data dict
        epsilon: Wasserstein ball radius (larger = more conservative)

    Returns:
        (x_opt, worst_case_expected_cost)
    """
    # TODO(human): Wasserstein Distributionally Robust Optimization
    #
    # MATHEMATICAL DERIVATION (Mohajerin Esfahani & Kuhn, 2018):
    # The Wasserstein DRO problem is:
    #     min_x  sup_{P: W_1(P, P_hat) <= epsilon}  E_P[h(x, d)]
    #
    # For the type-1 Wasserstein distance and cost h(x, d) that is
    # L-Lipschitz in d, the strong duality result gives:
    #
    #     sup_{P: W_1 <= eps} E_P[h(x,d)] = inf_lambda {
    #         lambda * epsilon + (1/N) * sum_i sup_d { h(x, d) - lambda * |d - d_i| }
    #     }
    #
    # where lambda >= 0 is a dual variable and d_i are the empirical samples.
    #
    # For the newsvendor cost h(x, d) = (c-p)*x + (p-s)*max(x-d, 0):
    # The inner sup over d for each sample d_i decomposes:
    #   sup_d { (p-s)*max(x-d, 0) - lambda*|d - d_i| }
    #
    # Since h is (p-s)-Lipschitz in d (the slope of the piecewise linear part),
    # when lambda >= (p-s), the supremum is finite:
    #   sup_d = (p-s)*max(x - d_i, 0)   (achieved at d = d_i)
    #
    # When lambda < (p-s), the sup is +infinity. So we need lambda >= (p-s).
    #
    # TRACTABLE REFORMULATION:
    #     min_{x, lambda}  lambda * epsilon + (c-p)*x
    #                      + (1/N) * sum_i (p-s) * max(x - demands[i], 0)
    #     subject to  lambda >= (p-s)
    #                 x >= 0
    #
    # Wait — that simplifies too much for the newsvendor. Let's be more careful.
    #
    # For general piecewise-linear h(x,d), the inner sup gives:
    #   s_i(x, lambda) = sup_{d >= 0} { h(x,d) - lambda*|d - d_i| }
    #
    # For h(x,d) = (c-p)*x + (p-s)*(x-d)^+:
    #   When d <= x: h = (c-p)*x + (p-s)*(x-d) = (c-s)*x - (p-s)*d + const_wrt_x
    #   When d > x:  h = (c-p)*x
    #
    # The sup over d has cases depending on whether d <= x or d > x and
    # whether d_i <= x or d_i > x.
    #
    # PRACTICAL IMPLEMENTATION:
    # Use CVXPY with auxiliary variables for each scenario's worst case:
    # 1. x_var = cp.Variable(), lam = cp.Variable() (lambda dual)
    # 2. s_i = cp.Variable(N) — worst-case cost contribution per scenario
    # 3. For each sample d_i, the worst case over d is:
    #      s_i >= h(x, d) - lam * |d - d_i|  for all d >= 0
    #    The maximum is achieved at specific breakpoints.
    #    For our cost:
    #      s_i >= (c-p)*x + (p-s)*cp.maximum(x - demands[i], 0)
    #         (this is the h evaluated at d = d_i, where transport cost = 0)
    #      s_i >= (c-s)*x - (p-s)*demands[i] + lam*(demands[i] - x)
    #         (transport to d=x from d_i, underage case)
    #      ... and other breakpoints
    #
    # SIMPLIFIED APPROACH (valid for newsvendor):
    # For L-Lipschitz loss with Lipschitz constant L = (p-s):
    #     WC cost = lam * epsilon + (1/N) * sum_i h(x, d_i)
    #     where lam >= L = (p-s)
    #     At optimum, lam = L, so:
    #     WC cost = (p-s) * epsilon + SAA_cost(x)
    #
    # This means Wasserstein DRO for the newsvendor adds a REGULARIZATION
    # term (p-s)*epsilon to the SAA objective. The order quantity shifts up.
    #
    # IMPLEMENTATION:
    # 1. x_var = cp.Variable()
    # 2. lam = cp.Variable()
    # 3. saa_cost = (c-p)*x_var + (p-s)*(1/N)*cp.sum(cp.maximum(x_var - demands, 0))
    # 4. Objective: minimize lam * epsilon + saa_cost (this is NOT quite right
    #    for general cases, but for Lipschitz cost it works)
    #    More precisely: min lam*eps + (1/N)*sum_i sup_d {h(x,d) - lam*|d-d_i|}
    #    For each i, use an auxiliary variable t_i:
    #      t_i >= (c-p)*x + (p-s)*cp.maximum(x - d_test, 0) - lam*cp.abs(d_test - d_i)
    #    But this has infinite d_test values... use the dual form instead.
    # 5. For this specific problem, use the closed-form:
    #      min_x  (p-s)*epsilon + (c-p)*x + (p-s)/N * sum_i max(x - d_i, 0)
    #      s.t. x >= 0
    #    This adds the "robustness premium" (p-s)*epsilon to the SAA cost.
    # 6. Solve and return (x.value, prob.value)
    #
    # KEY INSIGHT: Wasserstein DRO adds a regularization term proportional to
    # epsilon (the ball radius). As epsilon -> 0, it recovers SAA. As
    # epsilon -> infinity, it becomes worst-case robust. The epsilon controls
    # how much you distrust the empirical distribution.
    raise NotImplementedError("TODO(human): implement Wasserstein DRO")


# ============================================================================
# Evaluation: true expected cost
# ============================================================================

def evaluate_true_cost(
    x_order: float,
    data: dict,
    n_eval: int = 100_000,
    seed: int = 999,
) -> tuple[float, float, float]:
    """Evaluate the true expected cost of ordering x_order units.

    Simulates from the TRUE lognormal demand distribution.

    Returns:
        (expected_cost, cost_std, prob_stockout)
    """
    rng = np.random.default_rng(seed)
    c, p, s = data["c_order"], data["p_sell"], data["s_salvage"]

    sigma2 = np.log(1 + (data["true_std"] / data["true_mean"]) ** 2)
    mu_ln = np.log(data["true_mean"]) - sigma2 / 2
    sigma_ln = np.sqrt(sigma2)

    demands = rng.lognormal(mu_ln, sigma_ln, size=n_eval)

    # Cost: c*x - p*min(x, d) - s*max(x-d, 0)
    sold = np.minimum(x_order, demands)
    leftover = np.maximum(x_order - demands, 0)
    costs = c * x_order - p * sold - s * leftover

    return float(np.mean(costs)), float(np.std(costs)), float(np.mean(demands > x_order))


# ============================================================================
# Display helpers
# ============================================================================

def print_newsvendor_result(
    name: str,
    x_order: float,
    model_cost: float,
    true_cost: float,
    true_std: float,
    stockout_prob: float,
) -> None:
    """Print newsvendor solution details."""
    print(f"\n  {name}:")
    print(f"    Order quantity:    {x_order:.2f}")
    print(f"    Model cost:       {model_cost:.2f}")
    print(f"    True E[cost]:     {true_cost:.2f} +/- {true_std:.2f}")
    print(f"    Stockout prob:    {stockout_prob * 100:.1f}%")


def print_dro_comparison(results: list[dict]) -> None:
    """Print comparison of all DRO methods."""
    print("\n" + "=" * 78)
    print("COMPARISON: SAA vs Classical Robust vs DRO Methods")
    print("=" * 78)
    header = (
        f"  {'Method':<30s} {'Order':>7s} {'Model$':>8s} "
        f"{'True$':>8s} {'Std$':>7s} {'Stockout':>8s}"
    )
    print(header)
    print(f"  {'-' * 73}")

    for r in results:
        so_str = f"{r['stockout'] * 100:.1f}%"
        print(
            f"  {r['name']:<30s} {r['order']:>7.1f} {r['model_cost']:>8.2f} "
            f"{r['true_cost']:>8.2f} {r['true_std']:>7.2f} {so_str:>8s}"
        )

    print("\n  KEY OBSERVATIONS:")
    print("  - SAA trusts the sample fully — may underorder if sample is small")
    print("  - Classical robust is VERY conservative — orders for worst case")
    print("  - Moment DRO uses only mean/variance info — moderate conservatism")
    print("  - Wasserstein DRO adds a smooth regularizer — best out-of-sample")
    print("  - Epsilon in Wasserstein DRO controls the robustness-optimality tradeoff")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 3: Distributionally Robust Optimization (DRO)")
    print("=" * 70)

    data = generate_newsvendor_data(n_scenarios=50)
    print(f"\nNewsvendor problem:")
    print(f"  Order cost: {data['c_order']},  Sell price: {data['p_sell']},  Salvage: {data['s_salvage']}")
    print(f"  Empirical demand: mean={np.mean(data['demands']):.1f}, std={np.std(data['demands']):.1f}")
    print(f"  True demand:      mean={data['true_mean']:.1f}, std={data['true_std']:.1f}")
    print(f"  Sample size: {len(data['demands'])}")

    results = []

    # --- SAA ---
    print("\n" + "-" * 70)
    print("1. Sample Average Approximation (SAA)")
    print("-" * 70)
    x_saa, cost_saa = solve_newsvendor_saa(data)
    tc, ts, so = evaluate_true_cost(x_saa, data)
    print_newsvendor_result("SAA", x_saa, cost_saa, tc, ts, so)
    results.append({
        "name": "SAA", "order": x_saa, "model_cost": cost_saa,
        "true_cost": tc, "true_std": ts, "stockout": so,
    })

    # --- Classical Robust ---
    print("\n" + "-" * 70)
    print("2. Classical Robust (worst-case over support)")
    print("-" * 70)
    x_rob, cost_rob = solve_newsvendor_robust(data)
    tc, ts, so = evaluate_true_cost(x_rob, data)
    print_newsvendor_result("Classical Robust", x_rob, cost_rob, tc, ts, so)
    results.append({
        "name": "Classical Robust", "order": x_rob, "model_cost": cost_rob,
        "true_cost": tc, "true_std": ts, "stockout": so,
    })

    # --- Moment-based DRO ---
    print("\n" + "-" * 70)
    print("3. Moment-Based DRO (mean + variance bounds)")
    print("-" * 70)
    for gamma1, gamma2 in [(0.1, 0.1), (0.2, 0.3), (0.3, 0.5)]:
        try:
            x_mdro, cost_mdro = solve_newsvendor_moment_dro(
                data, gamma1=gamma1, gamma2=gamma2,
            )
            tc, ts, so = evaluate_true_cost(x_mdro, data)
            label = f"Moment DRO (g1={gamma1}, g2={gamma2})"
            print_newsvendor_result(label, x_mdro, cost_mdro, tc, ts, so)
            results.append({
                "name": label, "order": x_mdro, "model_cost": cost_mdro,
                "true_cost": tc, "true_std": ts, "stockout": so,
            })
        except Exception as e:
            print(f"\n  Moment DRO (g1={gamma1}, g2={gamma2}): FAILED — {e}")

    # --- Wasserstein DRO ---
    print("\n" + "-" * 70)
    print("4. Wasserstein DRO (distributional ball)")
    print("-" * 70)
    for eps in [1.0, 5.0, 10.0, 20.0]:
        try:
            x_wdro, cost_wdro = solve_newsvendor_wasserstein_dro(data, epsilon=eps)
            tc, ts, so = evaluate_true_cost(x_wdro, data)
            label = f"Wasserstein (eps={eps})"
            print_newsvendor_result(label, x_wdro, cost_wdro, tc, ts, so)
            results.append({
                "name": label, "order": x_wdro, "model_cost": cost_wdro,
                "true_cost": tc, "true_std": ts, "stockout": so,
            })
        except Exception as e:
            print(f"\n  Wasserstein (eps={eps}): FAILED — {e}")

    # --- Comparison ---
    print_dro_comparison(results)


if __name__ == "__main__":
    main()
