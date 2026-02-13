"""Phase 2: Robust Least Squares — SOCP with CVXPY.

Standard least squares minimizes ||Ax - b||_2, but what if A has measurement errors?
If A = A_nom + Delta with ||Delta|| <= epsilon, the worst-case residual is:

    max_{||Delta||<=eps} ||(A_nom + Delta)x - b||  =  ||A_nom x - b|| + eps * ||x||

This is a sum of two norms — an SOCP. The robust solution automatically regularizes:
it trades off data fit vs solution magnitude, identical to Tikhonov/ridge regression.
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Data: Noisy linear system
# ============================================================================

def generate_problem(
    m: int = 30,
    n: int = 5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a noisy least-squares problem.

    Creates A_nom (nominal measurement matrix), b (observations with noise),
    and x_true (ground truth solution).
    """
    rng = np.random.default_rng(seed)

    x_true = rng.standard_normal(n)
    A_true = rng.standard_normal((m, n))
    noise = noise_std * rng.standard_normal(m)
    b = A_true @ x_true + noise

    # Perturb A to simulate measurement uncertainty
    perturbation = 0.1 * rng.standard_normal((m, n))
    A_nom = A_true + perturbation

    return A_nom, b, x_true


# ============================================================================
# Ordinary Least Squares (baseline)
# ============================================================================

def ordinary_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b||_2 via CVXPY (for comparison)."""
    n = A.shape[1]
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.norm(A @ x - b, 2)))
    prob.solve(solver=cp.SCS, verbose=False)
    return x.value


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def robust_least_squares(
    A_nom: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Solve robust least squares: minimize worst-case residual over ||Delta|| <= epsilon.

    Args:
        A_nom: (m, n) nominal measurement matrix.
        b: (m,) observation vector.
        epsilon: uncertainty radius — maximum spectral norm of perturbation Delta.

    Returns:
        Tuple of:
        - x_robust: (n,) optimal robust solution.
        - objective: optimal objective value (worst-case residual).
    """
    # TODO(human): Robust Least Squares via SOCP
    #
    # Standard LS: minimize ||Ax - b||_2
    # But A is uncertain: A = A_nom + Delta, ||Delta|| <= epsilon
    #
    # Robust formulation: minimize max_{||Delta||<=epsilon} ||(A_nom + Delta)x - b||_2
    #
    # This is equivalent to (by triangle inequality):
    #   minimize ||A_nom @ x - b||_2 + epsilon * ||x||_2
    #
    # In CVXPY:
    #   x = cp.Variable(n)
    #   prob = cp.Problem(cp.Minimize(cp.norm(A_nom @ x - b) + epsilon * cp.norm(x)))
    #
    # DCP: norm(affine) + norm(variable) = convex + convex = convex. ✓
    # This is an SOCP (sum of norms).
    #
    # The robust solution trades off fitting the data vs keeping x small
    # (so perturbations in A don't cause large errors).
    # Solve with cp.SCS, verbose=False.
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_comparison(
    x_true: np.ndarray,
    x_ols: np.ndarray,
    x_robust: np.ndarray,
    epsilon: float,
    A_nom: np.ndarray,
    b: np.ndarray,
) -> None:
    """Compare OLS vs robust solutions."""
    print("=" * 65)
    print("ROBUST LEAST SQUARES — SOCP Regularization")
    print("=" * 65)

    print(f"\nUncertainty level (epsilon): {epsilon:.3f}")
    print(f"Problem size: {A_nom.shape[0]} observations, {A_nom.shape[1]} variables\n")

    print(f"{'':>14s}  {'True':>10s}  {'OLS':>10s}  {'Robust':>10s}")
    print("-" * 50)
    for i in range(len(x_true)):
        print(f"  x[{i}]          {x_true[i]:10.4f}  {x_ols[i]:10.4f}  {x_robust[i]:10.4f}")

    err_ols = np.linalg.norm(x_ols - x_true)
    err_robust = np.linalg.norm(x_robust - x_true)

    print(f"\n||x - x_true||_2:")
    print(f"  OLS:    {err_ols:.4f}")
    print(f"  Robust: {err_robust:.4f}")
    winner = "Robust" if err_robust < err_ols else "OLS"
    print(f"  Winner: {winner} (closer to ground truth)")

    norm_ols = np.linalg.norm(x_ols)
    norm_robust = np.linalg.norm(x_robust)
    print(f"\n||x||_2 (solution magnitude):")
    print(f"  OLS:    {norm_ols:.4f}")
    print(f"  Robust: {norm_robust:.4f}")
    print(f"  Robust solution is {'smaller' if norm_robust < norm_ols else 'larger'} "
          f"(regularization effect)")

    res_ols = np.linalg.norm(A_nom @ x_ols - b)
    res_robust = np.linalg.norm(A_nom @ x_robust - b)
    print(f"\n||A_nom @ x - b||_2 (nominal residual):")
    print(f"  OLS:    {res_ols:.4f}")
    print(f"  Robust: {res_robust:.4f}")
    print(f"  OLS fits nominal data better (expected — it doesn't hedge against uncertainty)")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Generating noisy linear system...\n")

    A_nom, b, x_true = generate_problem(m=30, n=5, noise_std=0.5)

    x_ols = ordinary_least_squares(A_nom, b)

    # Try multiple epsilon values to see the regularization effect
    for epsilon in [0.1, 0.5, 1.0]:
        x_robust, obj = robust_least_squares(A_nom, b, epsilon)
        print_comparison(x_true, x_ols, x_robust, epsilon, A_nom, b)
        print(f"\n  Robust objective value: {obj:.4f}")
        print()


if __name__ == "__main__":
    main()
