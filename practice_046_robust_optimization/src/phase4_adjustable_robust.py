"""Phase 4: Adjustable (Adaptive) Robust Optimization.

In static robust optimization, ALL decisions are made BEFORE uncertainty is
revealed. In adjustable robust optimization (ARO), some decisions can WAIT
until (part of) the uncertainty is observed — these are "wait-and-see" or
"recourse" decisions.

The general ARO is computationally intractable (NP-hard for LPs with
adjustable variables). The standard tractable approximation uses AFFINE
DECISION RULES: the adjustable variables are restricted to be affine
functions of the uncertain parameters.

This phase demonstrates:
  1. Static robust formulation (all decisions made up-front)
  2. Affine decision rule approximation (recourse is affine in uncertainty)
  3. Comparison showing how adaptivity improves the worst-case objective

Key reference: Ben-Tal, Goryashko, Guslitzer & Nemirovski (2004).
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Problem: Two-stage production planning under demand uncertainty
# ============================================================================

def generate_twostage_data(
    n_products: int = 3,
    n_scenarios: int = 4,
    seed: int = 42,
) -> dict:
    """Generate data for a two-stage production planning problem.

    Stage 1 (here-and-now): Decide raw material purchase x (before demand known)
    Stage 2 (wait-and-see): After observing demand d, decide production y and
                             unmet demand z

    minimize   c^T x + max_{d in U} [ q^T y(d) + penalty^T z(d) ]
    subject to A x >= B y(d)            (raw materials suffice for production)
               y(d) + z(d) >= d          (meet demand or pay penalty)
               x >= 0, y(d) >= 0, z(d) >= 0
               d in U (uncertainty set)

    Returns dict with:
        c:        (n,) raw material cost
        q:        (n,) production cost per unit
        penalty:  (n,) unmet demand penalty per unit
        A:        (n, n) material-to-product conversion (diagonal for simplicity)
        d_nom:    (n,) nominal demand
        d_dev:    (n,) maximum demand deviation
        Gamma:    budget of uncertainty (Bertsimas-Sim)
    """
    rng = np.random.default_rng(seed)

    c = rng.uniform(2.0, 5.0, size=n_products)      # raw material cost
    q = rng.uniform(1.0, 3.0, size=n_products)      # production cost
    penalty = rng.uniform(10.0, 20.0, size=n_products)  # unmet demand penalty

    # Conversion: 1 unit raw material -> A[i,i] units of product i
    A = np.diag(rng.uniform(0.8, 1.2, size=n_products))

    d_nom = rng.uniform(50.0, 150.0, size=n_products)  # nominal demand
    d_dev = 0.3 * d_nom                                 # +/- 30% deviation

    Gamma = 1.5  # budget: at most 1.5 products deviate simultaneously

    # Generate scenario matrix for evaluation
    scenarios = []
    for _ in range(n_scenarios):
        # Random demand in the uncertainty set
        u = rng.uniform(-1, 1, size=n_products)
        # Scale to budget
        if np.sum(np.abs(u)) > Gamma:
            u = u * Gamma / np.sum(np.abs(u))
        d = d_nom + d_dev * u
        scenarios.append(d)

    return {
        "c": c,
        "q": q,
        "penalty": penalty,
        "A": A,
        "d_nom": d_nom,
        "d_dev": d_dev,
        "Gamma": Gamma,
        "n_products": n_products,
        "scenarios": np.array(scenarios),
    }


# ============================================================================
# Helper: generate vertices of the uncertainty set
# ============================================================================

def enumerate_uncertainty_vertices(
    d_nom: np.ndarray,
    d_dev: np.ndarray,
    Gamma: float,
) -> list[np.ndarray]:
    """Enumerate vertices of the Bertsimas-Sim budget uncertainty set.

    U = { d : d = d_nom + d_dev * u, |u_j| <= 1, sum |u_j| <= Gamma }

    For small n, we enumerate all 2^n sign combinations and project onto
    the budget constraint. This gives the extreme demands.

    Returns list of demand vectors at the vertices.
    """
    n = len(d_nom)
    vertices = []

    # All sign combinations
    for mask in range(1 << n):
        signs = np.array([(1 if mask & (1 << j) else -1) for j in range(n)], dtype=float)

        # If sum |signs| > Gamma, scale down
        if np.sum(np.abs(signs)) > Gamma:
            # Activate only the top-Gamma components
            # For fractional Gamma: activate floor(Gamma) at full, one at fraction
            Gamma_floor = int(np.floor(Gamma))
            frac = Gamma - Gamma_floor
            # Choose which components are active
            for active_mask in range(1 << n):
                active = [j for j in range(n) if active_mask & (1 << j)]
                if len(active) > Gamma_floor + (1 if frac > 0 else 0):
                    continue
                u = np.zeros(n)
                for k, j in enumerate(active):
                    if k < Gamma_floor:
                        u[j] = signs[j]
                    else:
                        u[j] = signs[j] * frac
                d = d_nom + d_dev * u
                vertices.append(d)

    # Remove duplicates (approximately)
    unique = []
    for v in vertices:
        is_dup = False
        for u in unique:
            if np.allclose(v, u, atol=1e-8):
                is_dup = True
                break
        if not is_dup:
            unique.append(v)

    return unique


def sample_uncertainty_set(
    d_nom: np.ndarray,
    d_dev: np.ndarray,
    Gamma: float,
    n_samples: int = 200,
    seed: int = 77,
) -> np.ndarray:
    """Sample random demand vectors from the budget uncertainty set.

    Returns (n_samples, n) array of demand vectors.
    """
    rng = np.random.default_rng(seed)
    n = len(d_nom)
    samples = []

    for _ in range(n_samples):
        u = rng.uniform(-1, 1, size=n)
        # Project onto budget constraint
        total = np.sum(np.abs(u))
        if total > Gamma:
            u = u * Gamma / total
        samples.append(d_nom + d_dev * u)

    return np.array(samples)


# ============================================================================
# TODO(human): Static Robust Optimization
# ============================================================================

def solve_static_robust(data: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the TWO-STAGE problem with STATIC robust optimization.

    In the static approach, BOTH x (raw materials) AND y (production)
    are decided BEFORE demand is revealed. Only the penalty z adjusts.

    But since y is fixed, we must choose y to be feasible for ALL demands
    in the uncertainty set. This is very conservative.

    minimize   c^T x + q^T y + max_{d in U} penalty^T z
    subject to A x >= B y               (material constraints, B=I here)
               y + z >= d   for ALL d in U  (demand satisfaction)
               x, y, z >= 0

    Since z must satisfy z >= d - y for all d in U, and we want to
    minimize the worst-case penalty, we need:
        z >= d_max_component - y   componentwise
    where d_max[j] = d_nom[j] + d_dev[j] (worst case per component under box).

    But with budget constraint, the worst case is tighter.

    Args:
        data: problem data dict

    Returns:
        (x_opt, y_opt, worst_case_total_cost)
    """
    # TODO(human): Static Robust Two-Stage Problem
    #
    # MATHEMATICAL FORMULATION:
    # In static robust, x and y are fixed. The penalty z must cover the
    # worst-case demand shortfall.
    #
    # For Bertsimas-Sim uncertainty:
    #   d in U = { d_nom + d_dev * u : |u_j| <= 1, sum |u_j| <= Gamma }
    #
    # The worst-case penalty for fixed y is:
    #   max_{d in U}  penalty^T max(d - y, 0)
    #
    # This can be linearized. For each component j:
    #   max(d_j - y_j, 0) is worst when d_j is largest.
    #
    # Using the dual of the inner max (LP duality over the uncertainty set),
    # the static robust counterpart is:
    #
    #   minimize  c^T x + q^T y + Gamma * lam + sum_j pi_j
    #   subject to  lam + pi_j >= penalty_j * s_j    for all j
    #               s_j >= d_nom_j + d_dev_j - y_j    (max demand shortfall)
    #               s_j >= 0                           (no negative shortfall)
    #               A @ x >= y                         (material constraint)
    #               x >= 0, y >= 0
    #               lam >= 0, pi >= 0
    #
    # Here lam and pi are dual variables for the budget constraint,
    # and s_j = max(d_j^worst - y_j, 0) is the worst-case shortfall.
    #
    # IMPLEMENTATION STEPS:
    # 1. Extract c, q, penalty, A, d_nom, d_dev, Gamma from data
    # 2. n = data["n_products"]
    # 3. x = cp.Variable(n), y = cp.Variable(n)
    # 4. lam = cp.Variable(), pi = cp.Variable(n), s = cp.Variable(n)
    # 5. Objective: minimize c @ x + q @ y + Gamma * lam + cp.sum(pi)
    # 6. Constraints:
    #    - lam + pi[j] >= penalty[j] * s[j]   for each j (or vectorized)
    #    - s >= d_nom + d_dev - y   (worst-case shortfall from max demand)
    #    - s >= 0
    #    - A @ x >= y  (material sufficiency, since A is diagonal: x[j]*A[j,j] >= y[j])
    #    - x >= 0, y >= 0, lam >= 0, pi >= 0
    # 7. Solve and return (x.value, y.value, prob.value)
    #
    # KEY INSIGHT: In the static approach, y is committed before seeing demand.
    # This forces y to be conservative (produce less, accept penalties) or
    # aggressive (produce more, waste materials). There's no way to ADAPT
    # production to actual demand — that's why static robust is suboptimal.
    raise NotImplementedError("TODO(human): implement static robust optimization")


# ============================================================================
# TODO(human): Adjustable Robust with Affine Decision Rules
# ============================================================================

def solve_affine_adjustable(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the two-stage problem with AFFINE DECISION RULES.

    The key idea: the wait-and-see variable y is allowed to DEPEND on
    the realized demand d through an AFFINE function:
        y(d) = Y0 + Y1 @ d

    where Y0 (n,) is a constant offset and Y1 (n, n) is a matrix of
    sensitivities. After observing d, we compute y(d) and then z = max(d - y, 0).

    This is an approximation to the fully adjustable problem (which is intractable),
    but it's a GOOD approximation that often captures most of the benefit.

    Args:
        data: problem data dict

    Returns:
        (x_opt, Y0_opt, Y1_opt, worst_case_total_cost)
        where y(d) = Y0_opt + Y1_opt @ d
    """
    # TODO(human): Affine Decision Rules for Adjustable Robust Optimization
    #
    # MATHEMATICAL FORMULATION:
    # Replace y with y(d) = Y0 + Y1 @ d, where Y0 in R^n, Y1 in R^{n x n}.
    #
    # The problem becomes:
    #   minimize  c^T x + max_{d in U} [ q^T (Y0 + Y1 d) + penalty^T z(d) ]
    #   subject to  A x >= Y0 + Y1 d          for ALL d in U
    #               Y0 + Y1 d + z >= d         for ALL d in U
    #               Y0 + Y1 d >= 0             for ALL d in U
    #               z >= 0                      for ALL d in U
    #               x >= 0
    #
    # Each "for ALL d in U" constraint must be robustified.
    #
    # ROBUSTIFICATION (for budget uncertainty with vertices):
    # Since the uncertainty set is a polytope, "for all d in U" is equivalent
    # to "for all vertices of U". We enumerate vertices and add constraints
    # for each one.
    #
    # For a manageable number of products (n=3), the vertices are enumerable.
    # For larger n, you would use the LP dual reformulation instead.
    #
    # IMPLEMENTATION STEPS:
    # 1. Extract c, q, penalty, A, d_nom, d_dev, Gamma, n from data
    # 2. Decision variables:
    #      x = cp.Variable(n)         — raw material purchase
    #      Y0 = cp.Variable(n)        — constant part of decision rule
    #      Y1 = cp.Variable((n, n))   — affine sensitivity to demand
    # 3. Enumerate vertices of the uncertainty set:
    #      vertices = enumerate_uncertainty_vertices(d_nom, d_dev, Gamma)
    #      or use sample_uncertainty_set for an approximation
    # 4. For the worst-case objective, introduce t = cp.Variable():
    #      t >= q @ (Y0 + Y1 @ d_v) + penalty @ cp.maximum(d_v - Y0 - Y1 @ d_v, 0)
    #      for each vertex d_v
    #    Note: cp.maximum is not DCP with variable arguments in both sides.
    #    Instead, introduce auxiliary z_v = cp.Variable(n) for each vertex:
    #      z_v >= d_v - Y0 - Y1 @ d_v
    #      z_v >= 0
    #      t >= q @ (Y0 + Y1 @ d_v) + penalty @ z_v
    # 5. For each vertex d_v, add:
    #      A @ x >= Y0 + Y1 @ d_v     (material sufficiency)
    #      Y0 + Y1 @ d_v >= 0         (production non-negativity)
    # 6. Add x >= 0
    # 7. Objective: minimize c @ x + t
    # 8. Solve and return (x.value, Y0.value, Y1.value, prob.value)
    #
    # USING SAMPLED SCENARIOS (alternative to vertex enumeration):
    # If enumeration produces too many vertices, use sample_uncertainty_set()
    # to get a representative set. The solution is then approximately robust
    # (robust against sampled scenarios, not all of U).
    #
    # KEY INSIGHT: The affine decision rule y(d) = Y0 + Y1 d means:
    # - Y0 is the "base production plan" (what you'd produce at nominal demand)
    # - Y1 captures the "adjustment sensitivity" (how much to increase/decrease
    #   production per unit of demand increase)
    # - Y1[i,i] > 0 means "produce more of product i when its demand is higher"
    # - Y1[i,j] != 0 captures cross-product adjustments
    #
    # The affine rule always gives a BETTER (lower) worst-case cost than the
    # static solution, because static is a special case (Y1 = 0).
    raise NotImplementedError("TODO(human): implement affine adjustable robust optimization")


# ============================================================================
# Evaluation: Monte Carlo worst-case estimation
# ============================================================================

def evaluate_static(
    x: np.ndarray,
    y: np.ndarray,
    data: dict,
    n_scenarios: int = 1000,
    seed: int = 999,
) -> tuple[float, float, float]:
    """Evaluate static solution under random demand scenarios.

    Returns (mean_cost, worst_cost, avg_shortfall_fraction).
    """
    scenarios = sample_uncertainty_set(
        data["d_nom"], data["d_dev"], data["Gamma"],
        n_samples=n_scenarios, seed=seed,
    )
    c, q, penalty = data["c"], data["q"], data["penalty"]

    costs = []
    shortfalls = []
    for d in scenarios:
        z = np.maximum(d - y, 0)
        cost = c @ x + q @ y + penalty @ z
        costs.append(cost)
        shortfalls.append(np.sum(z) / np.sum(d))

    return float(np.mean(costs)), float(np.max(costs)), float(np.mean(shortfalls))


def evaluate_affine(
    x: np.ndarray,
    Y0: np.ndarray,
    Y1: np.ndarray,
    data: dict,
    n_scenarios: int = 1000,
    seed: int = 999,
) -> tuple[float, float, float]:
    """Evaluate affine decision rule under random demand scenarios.

    For each scenario d, computes y(d) = Y0 + Y1 @ d, then z = max(d - y, 0).

    Returns (mean_cost, worst_cost, avg_shortfall_fraction).
    """
    scenarios = sample_uncertainty_set(
        data["d_nom"], data["d_dev"], data["Gamma"],
        n_samples=n_scenarios, seed=seed,
    )
    c, q, penalty = data["c"], data["q"], data["penalty"]

    costs = []
    shortfalls = []
    for d in scenarios:
        y = Y0 + Y1 @ d
        y = np.maximum(y, 0)  # clip to non-negative
        z = np.maximum(d - y, 0)
        cost = c @ x + q @ y + penalty @ z
        costs.append(cost)
        shortfalls.append(np.sum(z) / np.sum(d))

    return float(np.mean(costs)), float(np.max(costs)), float(np.mean(shortfalls))


# ============================================================================
# Display helpers
# ============================================================================

def print_static_result(
    x: np.ndarray,
    y: np.ndarray,
    wc_cost: float,
    eval_result: tuple[float, float, float],
) -> None:
    """Print static robust solution."""
    mean_c, worst_c, avg_sf = eval_result
    print(f"\n  Static Robust Solution:")
    print(f"    Raw materials: [{', '.join(f'{v:.2f}' for v in x)}]")
    print(f"    Fixed production: [{', '.join(f'{v:.2f}' for v in y)}]")
    print(f"    Model worst-case cost: {wc_cost:.2f}")
    print(f"    MC mean cost:          {mean_c:.2f}")
    print(f"    MC worst cost:         {worst_c:.2f}")
    print(f"    Avg shortfall:         {avg_sf * 100:.1f}%")


def print_affine_result(
    x: np.ndarray,
    Y0: np.ndarray,
    Y1: np.ndarray,
    wc_cost: float,
    eval_result: tuple[float, float, float],
) -> None:
    """Print affine adjustable robust solution."""
    mean_c, worst_c, avg_sf = eval_result
    print(f"\n  Affine Decision Rule Solution:")
    print(f"    Raw materials: [{', '.join(f'{v:.2f}' for v in x)}]")
    print(f"    Base production Y0: [{', '.join(f'{v:.2f}' for v in Y0)}]")
    print(f"    Sensitivity Y1 (diagonal): [{', '.join(f'{Y1[i,i]:.4f}' for i in range(len(Y0)))}]")
    print(f"    Model worst-case cost: {wc_cost:.2f}")
    print(f"    MC mean cost:          {mean_c:.2f}")
    print(f"    MC worst cost:         {worst_c:.2f}")
    print(f"    Avg shortfall:         {avg_sf * 100:.1f}%")


def print_improvement(static_cost: float, affine_cost: float) -> None:
    """Print the improvement from adaptivity."""
    improvement = (static_cost - affine_cost) / static_cost * 100
    print(f"\n  IMPROVEMENT FROM ADAPTIVITY:")
    print(f"    Static worst-case:    {static_cost:.2f}")
    print(f"    Affine worst-case:    {affine_cost:.2f}")
    print(f"    Reduction:            {improvement:.1f}%")
    print(f"\n    The affine rule adapts production to observed demand,")
    print(f"    reducing the worst-case cost. Static = special case (Y1=0).")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 4: Adjustable Robust Optimization")
    print("=" * 70)

    data = generate_twostage_data(n_products=3)
    n = data["n_products"]
    print(f"\nTwo-stage production planning: {n} products")
    print(f"  Material costs:  [{', '.join(f'{v:.2f}' for v in data['c'])}]")
    print(f"  Production costs: [{', '.join(f'{v:.2f}' for v in data['q'])}]")
    print(f"  Penalty costs:   [{', '.join(f'{v:.2f}' for v in data['penalty'])}]")
    print(f"  Nominal demand:  [{', '.join(f'{v:.1f}' for v in data['d_nom'])}]")
    print(f"  Demand deviation: [{', '.join(f'{v:.1f}' for v in data['d_dev'])}]")
    print(f"  Budget Gamma:     {data['Gamma']}")

    # Enumerate vertices for reference
    vertices = enumerate_uncertainty_vertices(data["d_nom"], data["d_dev"], data["Gamma"])
    print(f"  Uncertainty set vertices: {len(vertices)}")

    # --- Static Robust ---
    print("\n" + "-" * 70)
    print("1. Static Robust Optimization")
    print("-" * 70)
    print("   All decisions (x, y) made before demand is revealed.")
    x_static, y_static, cost_static = solve_static_robust(data)
    eval_static = evaluate_static(x_static, y_static, data)
    print_static_result(x_static, y_static, cost_static, eval_static)

    # --- Affine Adjustable ---
    print("\n" + "-" * 70)
    print("2. Affine Decision Rule (Adjustable Robust)")
    print("-" * 70)
    print("   Production y(d) = Y0 + Y1 @ d adapts to observed demand.")
    x_affine, Y0, Y1, cost_affine = solve_affine_adjustable(data)
    eval_affine = evaluate_affine(x_affine, Y0, Y1, data)
    print_affine_result(x_affine, Y0, Y1, cost_affine, eval_affine)

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON: Static vs Adjustable")
    print("=" * 70)
    print_improvement(cost_static, cost_affine)

    # --- Demonstrate adaptivity on specific scenarios ---
    print("\n" + "-" * 70)
    print("3. Adaptivity in Action: Production Decisions by Scenario")
    print("-" * 70)
    test_demands = [
        data["d_nom"],                                        # nominal
        data["d_nom"] + data["d_dev"],                        # all high
        data["d_nom"] - data["d_dev"],                        # all low
        data["d_nom"] + data["d_dev"] * np.array([1, -1, 0]), # mixed
    ]
    labels = ["Nominal", "All High", "All Low", "Mixed (+/-/0)"]

    print(f"\n  {'Scenario':<18s} {'Demand':>25s} {'Static y':>25s} {'Affine y(d)':>25s}")
    print(f"  {'-' * 95}")
    for label, d in zip(labels, test_demands):
        # Clip to budget
        u = (d - data["d_nom"]) / data["d_dev"]
        total = np.sum(np.abs(u))
        if total > data["Gamma"]:
            u = u * data["Gamma"] / total
            d = data["d_nom"] + data["d_dev"] * u

        y_adapt = Y0 + Y1 @ d
        y_adapt = np.maximum(y_adapt, 0)

        d_str = f"[{', '.join(f'{v:.1f}' for v in d)}]"
        ys_str = f"[{', '.join(f'{v:.1f}' for v in y_static)}]"
        ya_str = f"[{', '.join(f'{v:.1f}' for v in y_adapt)}]"
        print(f"  {label:<18s} {d_str:>25s} {ys_str:>25s} {ya_str:>25s}")

    print(f"\n  Notice: Static y is IDENTICAL across all scenarios.")
    print(f"  Affine y(d) ADAPTS — producing more when demand is high, less when low.")


if __name__ == "__main__":
    main()
