"""Phase 1: Uncertainty Sets & Robust LP.

Demonstrates how different uncertainty set geometries transform the robust
counterpart of a linear program into different tractable problem classes:
  - Box uncertainty (L-infinity)  -->  LP (linear program)
  - Ellipsoidal uncertainty (L-2) -->  SOCP (second-order cone program)
  - Polyhedral uncertainty         -->  LP (with additional variables)

The "price of robustness" is quantified: how much optimal objective value
you sacrifice in exchange for guaranteed feasibility under perturbation.

Key reference: Ben-Tal & Nemirovski (1998), Bertsimas & Sim (2004).
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Problem data generation
# ============================================================================

def generate_production_lp(
    n_products: int = 4,
    n_resources: int = 6,
    seed: int = 42,
) -> dict:
    """Generate a production planning LP with uncertain resource requirements.

    Nominal problem:
        maximize  p^T x          (profit)
        subject to  A x <= b     (resource constraints)
                    0 <= x <= u  (production bounds)

    Each row A[i, :] of the constraint matrix represents how much of resource i
    is consumed per unit of each product. In reality, these coefficients are
    uncertain (e.g., machine efficiency varies, raw material quality fluctuates).

    Returns a dict with keys:
        p:      (n,) profit per unit of each product
        A_nom:  (m, n) nominal resource consumption matrix
        b:      (m,) resource availability
        u:      (n,) upper bounds on production
        delta:  (m, n) maximum perturbation per entry (for box uncertainty)
    """
    rng = np.random.default_rng(seed)

    p = rng.uniform(5.0, 20.0, size=n_products)
    A_nom = rng.uniform(1.0, 4.0, size=(n_resources, n_products))
    u = rng.uniform(5.0, 15.0, size=n_products)

    # Set b so that producing at 60% capacity is feasible with slack
    x_ref = 0.6 * u
    b = A_nom @ x_ref + rng.uniform(2.0, 8.0, size=n_resources)

    # Perturbation magnitudes: each entry can deviate by up to 20% of nominal
    delta = 0.20 * A_nom

    return {
        "p": p,
        "A_nom": A_nom,
        "b": b,
        "u": u,
        "delta": delta,
    }


# ============================================================================
# Nominal LP (baseline — no robustness)
# ============================================================================

def solve_nominal_lp(data: dict) -> tuple[np.ndarray, float]:
    """Solve the nominal production LP (no uncertainty).

    maximize  p^T x
    subject to  A_nom @ x <= b
                0 <= x <= u
    """
    p, A_nom, b, u = data["p"], data["A_nom"], data["b"], data["u"]
    n = len(p)

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(p @ x),
        [A_nom @ x <= b, x >= 0, x <= u],
    )
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Nominal LP: {prob.status}")
    return x.value, prob.value


# ============================================================================
# TODO(human): Robust LP with BOX uncertainty
# ============================================================================

def solve_robust_box(data: dict) -> tuple[np.ndarray, float]:
    """Robust counterpart of the production LP under BOX (interval) uncertainty.

    Each entry of the constraint matrix A can vary independently within an interval:
        A[i,j] in [A_nom[i,j] - delta[i,j],  A_nom[i,j] + delta[i,j]]

    The box uncertainty set is U_box = { Delta : |Delta[i,j]| <= delta[i,j] }.

    Args:
        data: dict with keys p, A_nom, b, u, delta

    Returns:
        (x_opt, obj_val) — optimal robust solution and objective
    """
    # TODO(human): Robust LP with Box Uncertainty (becomes an LP)
    #
    # MATHEMATICAL DERIVATION:
    # The i-th constraint must hold for ALL Delta in the box:
    #     (A_nom[i,:] + delta_i)^T x <= b[i]   for all |delta_ij| <= delta[i,j]
    #
    # The worst case (maximum LHS) occurs when each delta_ij takes the sign of x_j:
    #     delta_ij = delta[i,j] * sign(x_j)
    #
    # So the worst-case LHS for row i is:
    #     A_nom[i,:] @ x + sum_j delta[i,j] * |x_j|
    #
    # Since x >= 0 in our problem, |x_j| = x_j, so this simplifies to:
    #     A_nom[i,:] @ x + delta[i,:] @ x <= b[i]
    #     equivalently: (A_nom[i,:] + delta[i,:]) @ x <= b[i]
    #
    # This is still LINEAR in x — the robust counterpart is an LP!
    #
    # IMPLEMENTATION STEPS:
    # 1. Create x = cp.Variable(n)
    # 2. Worst-case constraint matrix is A_wc = A_nom + delta (since x >= 0)
    # 3. Constraints: A_wc @ x <= b, 0 <= x <= u
    # 4. Objective: maximize p @ x
    # 5. Solve and return (x.value, prob.value)
    #
    # KEY INSIGHT: Box uncertainty is the cheapest form of robustness —
    # it only changes the constraint coefficients. The problem class stays LP.
    # However, it can be overly conservative because it assumes ALL entries
    # deviate simultaneously in the worst direction.
    raise NotImplementedError("TODO(human): implement robust LP with box uncertainty")


# ============================================================================
# TODO(human): Robust LP with ELLIPSOIDAL uncertainty
# ============================================================================

def solve_robust_ellipsoidal(
    data: dict,
    rho: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Robust counterpart under ELLIPSOIDAL (L2-norm ball) uncertainty.

    Each ROW of the perturbation matrix has bounded Euclidean norm:
        ||Delta[i,:]||_2 <= rho * ||delta[i,:]||_2

    This is an ellipsoidal uncertainty set — less conservative than box
    because it doesn't assume all entries deviate simultaneously at maximum.

    Args:
        data: dict with keys p, A_nom, b, u, delta
        rho: scaling factor for the ellipsoid radius (rho=1 means the L2 ball
             has the same "energy" as the box diagonal)

    Returns:
        (x_opt, obj_val) — optimal robust solution and objective
    """
    # TODO(human): Robust LP with Ellipsoidal Uncertainty (becomes SOCP)
    #
    # MATHEMATICAL DERIVATION (Ben-Tal & Nemirovski):
    # The i-th constraint must hold for all perturbations in the ellipsoid:
    #     (A_nom[i,:] + delta_i)^T x <= b[i]
    #     for all delta_i such that ||diag(1/delta[i,:]) * delta_i||_2 <= rho
    #
    # Equivalently, define D_i = diag(delta[i,:]), then delta_i = D_i * u_i
    # with ||u_i||_2 <= rho. The worst case is:
    #     max_{||u_i|| <= rho}  (A_nom[i,:] + D_i u_i)^T x = A_nom[i,:]^T x + rho * ||D_i x||_2
    #
    # By Cauchy-Schwarz: max_{||u||<=rho} u^T (D_i x) = rho * ||D_i x||_2
    #
    # So the robust constraint for row i is:
    #     A_nom[i,:] @ x + rho * cp.norm(cp.multiply(delta[i,:], x), 2) <= b[i]
    #
    # This is a SECOND-ORDER CONE CONSTRAINT: affine + convex_norm <= constant.
    #
    # IMPLEMENTATION STEPS:
    # 1. Create x = cp.Variable(n)
    # 2. For each row i, add constraint:
    #        A_nom[i,:] @ x + rho * cp.norm(cp.multiply(delta[i,:], x), 2) <= b[i]
    #    Here cp.multiply(delta[i,:], x) does elementwise multiplication (Hadamard product)
    #    and cp.norm(..., 2) computes the Euclidean norm.
    # 3. Also add x >= 0, x <= u
    # 4. Maximize p @ x
    #
    # DCP CHECK: cp.norm(affine, 2) is convex, so affine + convex <= constant is valid.
    #
    # KEY INSIGHT: Ellipsoidal uncertainty is less conservative than box because
    # it constrains the TOTAL perturbation energy (L2 norm) rather than each entry
    # independently. The rho parameter controls the ellipsoid size.
    raise NotImplementedError("TODO(human): implement robust LP with ellipsoidal uncertainty")


# ============================================================================
# TODO(human): Robust LP with POLYHEDRAL (budget) uncertainty
# ============================================================================

def solve_robust_budget(
    data: dict,
    Gamma: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Robust counterpart under BUDGET (Bertsimas-Sim) uncertainty.

    At most Gamma coefficients in each constraint row can deviate from nominal.
    This controls the "budget of uncertainty" — how many parameters can go wrong
    simultaneously.

    The uncertainty set for row i is:
        { delta_i : |delta_ij| <= delta[i,j],  sum_j |delta_ij|/delta[i,j] <= Gamma }

    Args:
        data: dict with keys p, A_nom, b, u, delta
        Gamma: budget of uncertainty (0 = nominal, n = full box). Can be fractional.

    Returns:
        (x_opt, obj_val) — optimal robust solution and objective
    """
    # TODO(human): Robust LP with Budget Uncertainty (Bertsimas & Sim)
    #
    # MATHEMATICAL DERIVATION:
    # For row i, the worst-case perturbation under budget uncertainty is:
    #     max  sum_j delta_ij * x_j
    #     s.t. |delta_ij| <= delta[i,j]  for all j
    #          sum_j |delta_ij| / delta[i,j] <= Gamma
    #
    # This inner maximization is itself a linear program (in delta_ij).
    # By LP duality, the worst-case perturbation equals:
    #     min  Gamma * z_i + sum_j q_ij
    #     s.t. z_i + q_ij >= delta[i,j] * x_j   for all j
    #          z_i >= 0, q_ij >= 0
    #
    # So the robust counterpart is:
    #     maximize  p^T x
    #     subject to A_nom[i,:] @ x + Gamma * z[i] + sum(q[i,:]) <= b[i]  for all i
    #                z[i] + q[i,j] >= delta[i,j] * x[j]                   for all i, j
    #                x >= 0, x <= u
    #                z >= 0, q >= 0
    #
    # IMPLEMENTATION STEPS:
    # 1. Create x = cp.Variable(n), z = cp.Variable(m), q = cp.Variable((m, n))
    # 2. For each row i, add the budget-robust constraint:
    #        A_nom[i,:] @ x + Gamma * z[i] + cp.sum(q[i,:]) <= b[i]
    # 3. For each (i, j), add the dual coupling constraint:
    #        z[i] + q[i,j] >= delta[i,j] * x[j]
    # 4. Add z >= 0, q >= 0, x >= 0, x <= u
    # 5. Maximize p @ x
    #
    # KEY INSIGHT: Bertsimas-Sim budget uncertainty interpolates between
    # nominal (Gamma=0) and full box (Gamma=n). The robust counterpart
    # remains an LP (just with extra variables z, q), so it's computationally
    # as easy as the nominal LP! The Gamma parameter directly controls the
    # "price of robustness" — the tradeoff between protection and optimality.
    raise NotImplementedError("TODO(human): implement robust LP with budget uncertainty")


# ============================================================================
# Monte Carlo feasibility checking
# ============================================================================

def mc_feasibility_check(
    x: np.ndarray,
    data: dict,
    uncertainty_type: str,
    param: float,
    n_trials: int = 5000,
    seed: int = 123,
) -> tuple[float, float]:
    """Monte Carlo: fraction of random perturbations where x stays feasible.

    Generates random perturbations from the specified uncertainty set and
    checks if A_perturbed @ x <= b for each trial.

    Args:
        x: solution vector to test
        data: problem data dict
        uncertainty_type: "box", "ellipsoidal", or "budget"
        param: uncertainty parameter (ignored for box, rho for ellipsoidal,
               Gamma for budget)
        n_trials: number of Monte Carlo samples
        seed: random seed

    Returns:
        (feasibility_rate, max_violation)
    """
    rng = np.random.default_rng(seed)
    A_nom, b, delta = data["A_nom"], data["b"], data["delta"]
    m, n = A_nom.shape

    feasible_count = 0
    max_violation = 0.0

    for _ in range(n_trials):
        if uncertainty_type == "box":
            # Each entry independently in [-delta_ij, +delta_ij]
            Delta = rng.uniform(-1, 1, size=(m, n)) * delta

        elif uncertainty_type == "ellipsoidal":
            # Each row: random direction, norm <= rho * ||delta[i,:]||_2
            raw = rng.standard_normal((m, n))
            # Scale by delta (elementwise) then normalize row to rho * ||delta_i||
            scaled = raw * delta
            row_norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-12)
            radii = rng.uniform(0, param, size=(m, 1))
            Delta = scaled / row_norms * radii * np.linalg.norm(delta, axis=1, keepdims=True)

        elif uncertainty_type == "budget":
            # Budget: at most Gamma deviations active (probabilistic sampling)
            Gamma = param
            Delta = np.zeros((m, n))
            for i in range(m):
                # Randomly choose ~Gamma entries to perturb
                n_active = min(n, max(0, int(rng.poisson(Gamma))))
                active_idx = rng.choice(n, size=min(n_active, n), replace=False)
                Delta[i, active_idx] = rng.uniform(-1, 1, size=len(active_idx)) * delta[i, active_idx]
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")

        Ax = (A_nom + Delta) @ x
        violations = Ax - b
        max_viol = float(np.max(violations))

        if max_viol <= 1e-6:
            feasible_count += 1
        max_violation = max(max_violation, max_viol)

    return feasible_count / n_trials, max_violation


# ============================================================================
# Price of robustness analysis
# ============================================================================

def compute_price_of_robustness(
    nominal_obj: float,
    robust_obj: float,
) -> float:
    """Compute the relative loss in objective due to robustness.

    Price of Robustness = (nominal_obj - robust_obj) / nominal_obj * 100
    For maximization: robust_obj <= nominal_obj, so price >= 0.
    """
    if abs(nominal_obj) < 1e-10:
        return 0.0
    return (nominal_obj - robust_obj) / abs(nominal_obj) * 100.0


# ============================================================================
# Display helpers
# ============================================================================

def print_solution(name: str, x: np.ndarray, obj: float) -> None:
    """Print a solution."""
    print(f"\n  {name}:")
    print(f"    Objective (profit): {obj:.4f}")
    print(f"    Production: [{', '.join(f'{v:.3f}' for v in x)}]")
    print(f"    Total production: {np.sum(x):.3f}")


def print_comparison(results: list[dict]) -> None:
    """Print comparison table of all solutions."""
    print("\n" + "=" * 80)
    print("COMPARISON: Price of Robustness")
    print("=" * 80)
    nom_obj = results[0]["obj"]
    header = f"  {'Method':<35s} {'Profit':>8s} {'Price%':>8s} {'Feas.':>8s} {'MaxViol':>8s}"
    print(header)
    print(f"  {'-' * 73}")

    for r in results:
        price = compute_price_of_robustness(nom_obj, r["obj"])
        feas_str = f"{r['feas'] * 100:.1f}%"
        print(
            f"  {r['name']:<35s} {r['obj']:>8.2f} {price:>7.1f}% "
            f"{feas_str:>8s} {r['viol']:>8.4f}"
        )

    print("\n  KEY OBSERVATIONS:")
    print("  - Box uncertainty is the MOST conservative (worst profit, best feasibility)")
    print("  - Ellipsoidal uncertainty is INTERMEDIATE (less conservative than box)")
    print("  - Budget uncertainty INTERPOLATES between nominal and box via Gamma")
    print("  - The price of robustness is the cost you pay for protection")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 1: Uncertainty Sets & Robust LP")
    print("=" * 70)

    data = generate_production_lp(n_products=4, n_resources=6)
    print(f"\nProduction planning: {len(data['p'])} products, {len(data['b'])} resources")
    print(f"Profit vector: [{', '.join(f'{v:.2f}' for v in data['p'])}]")
    print(f"Upper bounds:  [{', '.join(f'{v:.2f}' for v in data['u'])}]")

    results = []

    # --- Nominal LP ---
    print("\n" + "-" * 70)
    print("1. Nominal LP (no uncertainty)")
    print("-" * 70)
    x_nom, obj_nom = solve_nominal_lp(data)
    print_solution("Nominal", x_nom, obj_nom)
    feas, viol = mc_feasibility_check(x_nom, data, "box", 0.0)
    print(f"    Feasibility under box perturbation: {feas * 100:.1f}%")
    results.append({"name": "Nominal (no protection)", "obj": obj_nom, "feas": feas, "viol": viol})

    # --- Box uncertainty ---
    print("\n" + "-" * 70)
    print("2. Robust LP with BOX uncertainty (worst-case per entry)")
    print("-" * 70)
    x_box, obj_box = solve_robust_box(data)
    print_solution("Robust (Box)", x_box, obj_box)
    feas, viol = mc_feasibility_check(x_box, data, "box", 0.0)
    print(f"    Feasibility under box perturbation: {feas * 100:.1f}%")
    results.append({"name": "Box (full interval)", "obj": obj_box, "feas": feas, "viol": viol})

    # --- Ellipsoidal uncertainty ---
    print("\n" + "-" * 70)
    print("3. Robust LP with ELLIPSOIDAL uncertainty (L2-norm ball)")
    print("-" * 70)
    for rho in [0.5, 1.0, 1.5]:
        x_ell, obj_ell = solve_robust_ellipsoidal(data, rho=rho)
        print_solution(f"Robust (Ellipsoidal, rho={rho})", x_ell, obj_ell)
        feas, viol = mc_feasibility_check(x_ell, data, "ellipsoidal", rho)
        print(f"    Feasibility under ellipsoidal perturbation: {feas * 100:.1f}%")
        results.append({
            "name": f"Ellipsoidal (rho={rho})",
            "obj": obj_ell, "feas": feas, "viol": viol,
        })

    # --- Budget uncertainty ---
    print("\n" + "-" * 70)
    print("4. Robust LP with BUDGET uncertainty (Bertsimas-Sim)")
    print("-" * 70)
    n = len(data["p"])
    for Gamma in [1.0, 2.0, 3.0, float(n)]:
        x_bud, obj_bud = solve_robust_budget(data, Gamma=Gamma)
        gamma_label = f"{Gamma:.0f}" if Gamma <= n else "n"
        print_solution(f"Robust (Budget, Gamma={gamma_label})", x_bud, obj_bud)
        feas, viol = mc_feasibility_check(x_bud, data, "budget", Gamma)
        print(f"    Feasibility under budget perturbation: {feas * 100:.1f}%")
        results.append({
            "name": f"Budget (Gamma={gamma_label})",
            "obj": obj_bud, "feas": feas, "viol": viol,
        })

    # --- Comparison ---
    print_comparison(results)


if __name__ == "__main__":
    main()
