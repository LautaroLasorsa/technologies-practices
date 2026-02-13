"""Phase 4: Minimum Volume Enclosing Ellipsoid (MVEE / Löwner-John) — SDP.

Find the smallest ellipsoid containing a given set of points. The ellipsoid is
parameterized as {x : ||Ax + b||_2 <= 1}, and we maximize log det(A) (which is
concave on the PSD cone, and larger determinant = smaller volume).

This combines SOCP constraints (point containment) with an SDP-class objective
(log_det). CVXPY handles the mixed cone structure automatically.
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Data: Point clouds in 2D
# ============================================================================

def generate_point_cloud(
    n_points: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2D point cloud (roughly elliptical cluster)."""
    rng = np.random.default_rng(seed)

    # Generate points from a skewed distribution
    raw = rng.standard_normal((n_points, 2))
    # Apply a transformation to make it non-circular
    transform = np.array([[2.0, 0.5],
                          [0.3, 1.5]])
    points = (transform @ raw.T).T
    # Add a shift
    points += np.array([1.0, -0.5])

    return points


def generate_square_points() -> np.ndarray:
    """Generate points at the corners and midpoints of a square (for validation)."""
    return np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1],
    ], dtype=float)


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def min_volume_ellipsoid(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the minimum volume ellipsoid enclosing all points.

    The ellipsoid is {x : ||A @ x + b||_2 <= 1}, where A is a PSD matrix.
    Maximizing log det(A) minimizes the volume (volume ~ 1/det(A)).

    Args:
        points: (m, d) array of points to enclose.

    Returns:
        Tuple of:
        - A_val: (d, d) matrix defining the ellipsoid shape.
        - b_val: (d,) vector defining the ellipsoid center offset.
        - log_det_val: optimal log det(A) value.
    """
    # TODO(human): Minimum Volume Enclosing Ellipsoid (MVEE / Löwner-John)
    #
    # Find the smallest ellipsoid E = {x : ||Ax + b||_2 <= 1} containing all points.
    #
    # Formulation:
    #   maximize log det(A)  (equivalent to minimizing volume)
    #   subject to: ||A @ p_i + b||_2 <= 1  for each point p_i
    #
    # In CVXPY:
    #   A = cp.Variable((d, d), symmetric=True)
    #   b = cp.Variable(d)
    #   constraints = [cp.norm(A @ points[i] + b, 2) <= 1 for i in range(m)]
    #   prob = cp.Problem(cp.Maximize(cp.log_det(A)), constraints)
    #
    # log_det(A) is CONCAVE on PSD matrices, so maximizing it is a convex problem.
    # The SOCP constraints ensure all points are inside the ellipsoid.
    # Combined: this is a mixed SOCP/SDP problem.
    #
    # Solve with cp.SCS, verbose=False, max_iters=5000.
    # Return (A.value, b.value, prob.value).
    #
    # Applications: robust statistics, anomaly detection, collision detection in robotics.
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_ellipsoid_result(
    points: np.ndarray,
    A_val: np.ndarray,
    b_val: np.ndarray,
    log_det_val: float,
    label: str,
) -> None:
    """Display the MVEE result with containment verification."""
    m, d = points.shape

    print("=" * 65)
    print(f"MINIMUM VOLUME ELLIPSOID — {label}")
    print("=" * 65)

    print(f"\nPoints: {m} points in R^{d}")

    # Ellipsoid center: solve A @ center + b = 0  =>  center = -A^{-1} b
    try:
        center = -np.linalg.solve(A_val, b_val)
        print(f"Ellipsoid center: ({center[0]:.4f}, {center[1]:.4f})")
    except np.linalg.LinAlgError:
        center = None
        print("Ellipsoid center: could not compute (A singular)")

    print(f"\nA matrix (shape transform):")
    for i in range(d):
        row_str = "  ".join(f"{A_val[i, j]:8.4f}" for j in range(d))
        print(f"  [{row_str}]")
    print(f"b vector: [{b_val[0]:.4f}, {b_val[1]:.4f}]")
    print(f"log det(A): {log_det_val:.4f}")
    print(f"det(A): {np.exp(log_det_val):.4f}")

    # Semi-axis lengths: eigenvalues of A^{-1}
    try:
        eigvals = np.linalg.eigvalsh(A_val)
        semi_axes = 1.0 / eigvals
        print(f"Semi-axis lengths: {semi_axes[0]:.4f}, {semi_axes[1]:.4f}")
    except np.linalg.LinAlgError:
        pass

    # Verify containment
    print(f"\nContainment check (||A @ p_i + b|| <= 1):")
    max_norm = 0.0
    violations = 0
    for i in range(m):
        norm_val = np.linalg.norm(A_val @ points[i] + b_val)
        max_norm = max(max_norm, norm_val)
        if norm_val > 1.0 + 1e-3:
            violations += 1
    print(f"  Max ||A @ p_i + b||: {max_norm:.6f}")
    print(f"  Violations (> 1 + tol): {violations}")
    print(f"  All points contained: {'YES' if violations == 0 else 'NO'}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # --- Random point cloud ---
    print("Computing MVEE for a random 2D point cloud...\n")
    points = generate_point_cloud(n_points=20)
    A_val, b_val, log_det_val = min_volume_ellipsoid(points)
    print_ellipsoid_result(points, A_val, b_val, log_det_val, "Random Point Cloud (20 pts)")

    # --- Square points (validation) ---
    print("\n")
    pts_sq = generate_square_points()
    A_sq, b_sq, ld_sq = min_volume_ellipsoid(pts_sq)
    print_ellipsoid_result(pts_sq, A_sq, b_sq, ld_sq, "Square Corners + Midpoints")
    print("\n  (For a square [-1,1]^2, the MVEE is the circumscribed circle of radius sqrt(2),")
    print("   so A ~ (1/sqrt(2)) * I and center at origin.)")


if __name__ == "__main__":
    main()
