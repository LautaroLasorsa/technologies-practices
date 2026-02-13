"""Phase 4: Robust Linear Programming with CVXPY.

Solves an LP where the constraint matrix has bounded uncertainty.
The robust counterpart guarantees feasibility for ALL perturbations
within an ellipsoidal uncertainty set, at the cost of a more conservative
(worse) objective value.

DCP atoms used:
  - cp.norm(x, 2): convex, positive — Euclidean norm
  - affine + convex <= constant: valid DCP constraint
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Data: LP with uncertain constraints
# ============================================================================

def generate_robust_lp_data(
    n: int = 5,
    m: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a nominal LP: minimize c^T x subject to A_nom x <= b, x >= 0.

    The constraint matrix A_nom represents "best-guess" data that may have
    row-wise perturbations bounded by some epsilon.

    Args:
        n: number of variables
        m: number of constraints

    Returns:
        c: (n,) cost vector
        A_nom: (m, n) nominal constraint matrix
        b: (m,) RHS vector (set so the nominal LP is feasible)
    """
    rng = np.random.default_rng(seed)

    c = rng.uniform(1.0, 5.0, size=n)

    # Generate A with positive entries (resource-type constraints)
    A_nom = rng.uniform(0.5, 3.0, size=(m, n))

    # Choose b so that x = ones(n) is feasible with slack
    x_feas = np.ones(n)
    b = A_nom @ x_feas + rng.uniform(1.0, 5.0, size=m)

    return c, A_nom, b


def solve_nominal_lp(
    c: np.ndarray,
    A_nom: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the nominal LP (no robustness): min c^T x, s.t. A_nom x <= b, x >= 0.

    This is a standard LP — serves as the baseline for comparison.
    """
    n = len(c)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(c @ x),
        [A_nom @ x <= b, x >= 0],
    )
    prob.solve()

    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Nominal LP status: {prob.status}")

    return x.value, prob.value


# ============================================================================
# TODO(human): Robust linear program
# ============================================================================

def solve_robust_lp(
    c: np.ndarray,
    A_nom: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Solve the robust counterpart of an LP with row-wise ellipsoidal uncertainty.

    Standard LP:  minimize c^T x  subject to  A x <= b, x >= 0
    But A is uncertain: A = A_nom + Delta, where each row ||delta_i|| <= epsilon.

    Robust counterpart: for each constraint row i,
        a_i^T x + epsilon * ||x||_2 <= b_i
    (worst-case over all perturbations with ||delta_i|| <= epsilon)

    Args:
        c: (n,) cost vector
        A_nom: (m, n) nominal constraint matrix
        b: (m,) RHS vector
        epsilon: uncertainty radius (>= 0)

    Returns:
        Tuple of:
        - x_opt: (n,) optimal robust solution
        - obj_val: optimal objective value (>= nominal since more constrained)
    """
    # TODO(human): Robust Linear Program with CVXPY
    #
    # Standard LP: minimize c^T x subject to A x <= b
    # But A is uncertain: A = A_nom + Delta, ||Delta_row|| <= epsilon
    #
    # Robust counterpart: for each constraint row i,
    #   a_i^T x + epsilon * ||x||_2 <= b_i
    # (worst-case over all perturbations with ||delta_i|| <= epsilon)
    #
    # In CVXPY:
    #   x = cp.Variable(n)
    #   constraints = [x >= 0]
    #   for i in range(m):
    #       constraints.append(A_nom[i] @ x + epsilon * cp.norm(x, 2) <= b[i])
    #   prob = cp.Problem(cp.Minimize(c @ x), constraints)
    #
    # DCP: norm(x, 2) is convex, affine + convex <= constant is valid. ✓
    #
    # The robust solution is more conservative (worse objective) but guarantees
    # feasibility even when the data is perturbed. The epsilon parameter controls
    # the level of conservatism.
    raise NotImplementedError


# ============================================================================
# Feasibility checking
# ============================================================================

def check_feasibility_under_perturbation(
    x: np.ndarray,
    A_nom: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    n_trials: int = 1000,
    seed: int = 123,
) -> tuple[float, float]:
    """Monte Carlo check: how often is x feasible under random perturbations?

    For each trial, sample Delta where each row has ||delta_i|| <= epsilon,
    then check if (A_nom + Delta) @ x <= b.

    Returns:
        feasibility_rate: fraction of trials where x is feasible
        max_violation: worst constraint violation across all trials
    """
    rng = np.random.default_rng(seed)
    m, n = A_nom.shape

    feasible_count = 0
    max_violation = 0.0

    for _ in range(n_trials):
        # Random perturbation: each row is a random vector with norm <= epsilon
        Delta = rng.standard_normal((m, n))
        row_norms = np.linalg.norm(Delta, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-10)
        # Scale each row to have norm <= epsilon (uniform in ball)
        scales = rng.uniform(0, epsilon, size=(m, 1))
        Delta = Delta / row_norms * scales

        Ax = (A_nom + Delta) @ x
        violations = Ax - b
        max_viol = float(np.max(violations))

        if max_viol <= 1e-6:
            feasible_count += 1
        max_violation = max(max_violation, max_viol)

    return feasible_count / n_trials, max_violation


# ============================================================================
# Display helpers
# ============================================================================

def print_solution(
    name: str,
    x: np.ndarray,
    obj: float,
) -> None:
    """Print a solution vector with its objective."""
    print(f"\n  {name}:")
    print(f"    Objective: {obj:.4f}")
    print(f"    x = [{', '.join(f'{v:.3f}' for v in x)}]")
    print(f"    ||x||_2 = {np.linalg.norm(x, 2):.4f}")


def print_comparison_table(
    results: list[tuple[str, np.ndarray, float, float, float]],
) -> None:
    """Print comparison of nominal vs robust at different epsilon values.

    Args:
        results: list of (name, x, obj, feasibility_rate, max_violation)
    """
    print("\n" + "=" * 72)
    print("COMPARISON: Nominal vs Robust Solutions")
    print("=" * 72)
    print(f"  {'Method':<25s} {'Objective':>10s} {'||x||_2':>8s} {'Feas. rate':>12s} {'Max viol.':>10s}")
    print(f"  {'-'*67}")

    for name, x, obj, feas, viol in results:
        feas_str = f"{feas*100:.1f}%"
        print(f"  {name:<25s} {obj:>10.4f} {np.linalg.norm(x):>8.4f} {feas_str:>12s} {viol:>10.4f}")

    print("\n  Key insight: Robust solutions have WORSE objectives but HIGHER feasibility.")
    print("  The epsilon parameter controls the tradeoff between optimality and robustness.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 4: Robust Linear Programming")
    print("=" * 65)

    c, A_nom, b = generate_robust_lp_data(n=5, m=8)
    print(f"\nProblem: {len(c)} variables, {len(b)} constraints")
    print(f"Cost vector: [{', '.join(f'{v:.2f}' for v in c)}]")

    # Solve nominal LP
    print("\n" + "-" * 65)
    print("Nominal LP (no uncertainty)")
    print("-" * 65)
    x_nom, obj_nom = solve_nominal_lp(c, A_nom, b)
    print_solution("Nominal", x_nom, obj_nom)

    # Test epsilons for robustness
    test_epsilon = 0.5  # perturbation level for Monte Carlo
    results = []

    # Nominal feasibility under perturbation
    feas, viol = check_feasibility_under_perturbation(x_nom, A_nom, b, test_epsilon)
    print(f"  Feasibility under perturbation (epsilon={test_epsilon}): {feas*100:.1f}%")
    results.append(("Nominal (eps=0)", x_nom, obj_nom, feas, viol))

    # Robust solutions at increasing epsilon
    print("\n" + "-" * 65)
    print("Robust LP (increasing uncertainty)")
    print("-" * 65)
    for eps in [0.1, 0.3, 0.5, 1.0]:
        try:
            x_rob, obj_rob = solve_robust_lp(c, A_nom, b, eps)
            print_solution(f"Robust (epsilon={eps})", x_rob, obj_rob)

            feas, viol = check_feasibility_under_perturbation(x_rob, A_nom, b, test_epsilon)
            print(f"  Feasibility under perturbation (test eps={test_epsilon}): {feas*100:.1f}%")
            results.append((f"Robust (eps={eps})", x_rob, obj_rob, feas, viol))

        except Exception as e:
            print(f"\n  Robust (epsilon={eps}): FAILED — {e}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
