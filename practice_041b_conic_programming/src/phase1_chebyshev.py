"""Phase 1: Chebyshev Center — SOCP with CVXPY.

Find the largest ball inscribed in a polytope defined by linear inequalities Ax <= b.
The Chebyshev center is the center of this ball — it's the "safest" point in the
feasible region, maximally far from all constraint boundaries.

This is a classic SOCP because the ball-in-halfplane constraint involves a norm:
for the ball {y : ||y - x|| <= r} to satisfy a_i^T y <= b_i, we need
a_i^T x + r * ||a_i||_2 <= b_i (the farthest point of the ball in direction a_i).
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Data: Polytope defined by Ax <= b (irregular pentagon in 2D)
# ============================================================================

# Each row of A is a halfplane normal, b is the RHS.
# These 6 halfplanes define a bounded polygon in R^2.
A_POLYTOPE = np.array([
    [ 1.0,  0.0],   # x <= 4
    [-1.0,  0.0],   # -x <= 1  (i.e., x >= -1)
    [ 0.0,  1.0],   # y <= 3
    [ 0.0, -1.0],   # -y <= 1  (i.e., y >= -1)
    [ 1.0,  1.0],   # x + y <= 5
    [-1.0,  2.0],   # -x + 2y <= 4
])

b_POLYTOPE = np.array([4.0, 1.0, 3.0, 1.0, 5.0, 4.0])


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def chebyshev_center(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find the Chebyshev center of the polytope {x : Ax <= b}.

    Args:
        A: (m, d) matrix of halfplane normals.
        b: (m,) vector of halfplane RHS values.

    Returns:
        Tuple of:
        - center: (d,) array, the Chebyshev center coordinates.
        - radius: float, the radius of the largest inscribed ball.
    """
    # TODO(human): Chebyshev Center via SOCP
    #
    # Find the largest ball (center x, radius r) inscribed in the polytope {x : Ax <= b}.
    #
    # Formulation:
    #   maximize r
    #   subject to: a_i^T x + r * ||a_i||_2 <= b_i  for each row i
    #
    # Why: for the ball {y : ||y - x|| <= r} to be inside the polytope,
    # every halfplane a_i^T y <= b_i must be satisfied at the farthest
    # point of the ball in direction a_i, which is at distance r * ||a_i|| from center.
    #
    # In CVXPY:
    #   x = cp.Variable(d)   # center (d-dimensional)
    #   r = cp.Variable()    # radius (scalar)
    #   constraints = [A[i] @ x + r * np.linalg.norm(A[i]) <= b[i] for i in range(m)]
    #   constraints += [r >= 0]
    #   prob = cp.Problem(cp.Maximize(r), constraints)
    #
    # This is an SOCP because the constraint involves ||a_i|| * r (product of norm and variable).
    # After solving, extract x.value (center) and r.value (radius).
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_chebyshev_result(
    A: np.ndarray,
    b: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> None:
    """Display the Chebyshev center result with constraint analysis."""
    m = A.shape[0]

    print("=" * 65)
    print("CHEBYSHEV CENTER — Largest Inscribed Ball (SOCP)")
    print("=" * 65)

    print(f"\nCenter:  ({center[0]:.4f}, {center[1]:.4f})")
    print(f"Radius:  {radius:.4f}")

    print("\nDistance from center to each halfplane:")
    for i in range(m):
        norm_ai = np.linalg.norm(A[i])
        distance = (b[i] - A[i] @ center) / norm_ai
        binding = "BINDING (touches ball)" if abs(distance - radius) < 1e-4 else ""
        print(f"  Halfplane {i}: dist = {distance:.4f}  {binding}")

    print(f"\nVerification: all distances >= radius ({radius:.4f})? ", end="")
    distances = [(b[i] - A[i] @ center) / np.linalg.norm(A[i]) for i in range(m)]
    all_ok = all(d >= radius - 1e-6 for d in distances)
    print("YES" if all_ok else "NO — check your solution!")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Computing Chebyshev center of a 2D polytope...\n")

    center, radius = chebyshev_center(A_POLYTOPE, b_POLYTOPE)
    print_chebyshev_result(A_POLYTOPE, b_POLYTOPE, center, radius)

    # Bonus: try a simple box [-1, 1]^2 — center should be (0, 0), radius 1
    print("\n" + "=" * 65)
    print("BONUS: Unit box [-1, 1]^2")
    A_box = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    b_box = np.array([1, 1, 1, 1], dtype=float)
    c_box, r_box = chebyshev_center(A_box, b_box)
    print(f"  Center: ({c_box[0]:.4f}, {c_box[1]:.4f})  Radius: {r_box:.4f}")
    print(f"  Expected: (0.0, 0.0), radius 1.0")


if __name__ == "__main__":
    main()
