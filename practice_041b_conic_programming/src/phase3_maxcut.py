"""Phase 3: Max-Cut SDP Relaxation — Goemans-Williamson 1995.

Max-Cut: partition graph vertices into two sets S, S^c to maximize the weight
of edges between S and S^c. NP-hard! But the SDP relaxation replaces binary
variables x_i in {-1, +1} with a PSD correlation matrix X, yielding a
polynomial-time 0.878-approximation.

This is the most celebrated application of semidefinite programming.
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Data: Graphs for Max-Cut
# ============================================================================

def petersen_graph() -> np.ndarray:
    """Return the adjacency matrix of the Petersen graph (10 vertices, 15 edges).

    The Petersen graph is 3-regular (each vertex has degree 3) and is a classic
    example in graph theory. Its max cut is 12 (out of 15 edges).
    """
    n = 10
    W = np.zeros((n, n))
    # Outer cycle: 0-1-2-3-4-0
    outer = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    # Inner star: 5-7-9-6-8-5
    inner = [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    spokes = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

    for i, j in outer + inner + spokes:
        W[i, j] = 1.0
        W[j, i] = 1.0

    return W


def random_graph(n: int = 8, density: float = 0.5, seed: int = 123) -> np.ndarray:
    """Generate a random weighted graph."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.uniform(0.5, 3.0)
                W[i, j] = w
                W[j, i] = w
    return W


def graph_laplacian(W: np.ndarray) -> np.ndarray:
    """Compute the graph Laplacian L = D - W."""
    D = np.diag(W.sum(axis=1))
    return D - W


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def maxcut_sdp(
    W: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Solve the Max-Cut SDP relaxation.

    Args:
        W: (n, n) symmetric adjacency/weight matrix.

    Returns:
        Tuple of:
        - sdp_bound: upper bound on Max-Cut from the SDP relaxation.
        - X_val: (n, n) optimal PSD matrix (correlation matrix).
    """
    # TODO(human): Max-Cut SDP Relaxation (Goemans-Williamson 1995)
    #
    # Max-Cut: partition graph vertices into two sets to maximize edges between sets.
    # NP-hard! But the SDP relaxation gives a 0.878-approximation.
    #
    # Exact formulation: maximize (1/4) sum_{(i,j) in E} w_ij (1 - x_i * x_j)
    #   where x_i in {-1, +1} (which set vertex i belongs to)
    #
    # SDP relaxation: replace x_i * x_j with X_ij where X is PSD:
    #   maximize (1/4) * trace(L @ X)  where L = Laplacian of the graph
    #   subject to: X >> 0 (positive semidefinite)
    #               X_ii = 1 for all i (diagonal = 1)
    #
    # In CVXPY:
    #   n = W.shape[0]
    #   L = graph_laplacian(W)   (use the helper above)
    #   X = cp.Variable((n, n), symmetric=True)
    #   constraints = [X >> 0]  # PSD constraint
    #   constraints += [X[i, i] == 1 for i in range(n)]  # diagonal = 1
    #   prob = cp.Problem(cp.Maximize(0.25 * cp.trace(L @ X)), constraints)
    #   prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
    #
    # Return (prob.value, X.value) — the SDP bound and the optimal matrix.
    raise NotImplementedError


# ============================================================================
# Rounding: random hyperplane
# ============================================================================

def hyperplane_rounding(
    X: np.ndarray,
    W: np.ndarray,
    num_rounds: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Round the SDP solution using Goemans-Williamson random hyperplane rounding.

    1. Cholesky decompose X ~ V^T V (use eigendecomposition for numerical stability).
    2. Generate random vector r ~ N(0, I).
    3. Set x_i = sign(V_i^T r).
    4. Compute cut value. Repeat and take the best.

    Returns:
        Tuple of:
        - best_partition: (n,) array of {-1, +1} labels.
        - best_cut_value: weight of edges between the two sets.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    # Eigendecomposition for numerical stability (X may not be exactly PSD)
    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = np.maximum(eigvals, 0)  # clip negative eigenvalues
    V = eigvecs @ np.diag(np.sqrt(eigvals))  # V such that X ~ V @ V^T

    best_cut = -1.0
    best_partition = None

    for _ in range(num_rounds):
        r = rng.standard_normal(n)
        partition = np.sign(V @ r)
        partition[partition == 0] = 1.0  # break ties

        # Compute cut value: sum of w_ij where i and j are in different sets
        cut_val = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if partition[i] != partition[j]:
                    cut_val += W[i, j]

        if cut_val > best_cut:
            best_cut = cut_val
            best_partition = partition.copy()

    return best_partition, best_cut


# ============================================================================
# Display helpers
# ============================================================================

def print_maxcut_result(
    W: np.ndarray,
    sdp_bound: float,
    partition: np.ndarray,
    cut_value: float,
    graph_name: str,
) -> None:
    """Display Max-Cut results."""
    n = W.shape[0]
    total_weight = W.sum() / 2  # each edge counted once

    print("=" * 65)
    print(f"MAX-CUT SDP RELAXATION — {graph_name}")
    print("=" * 65)

    print(f"\nGraph: {n} vertices, {int((W > 0).sum() / 2)} edges, "
          f"total weight = {total_weight:.1f}")

    print(f"\nSDP relaxation bound: {sdp_bound:.4f}")
    print(f"Rounded cut value:    {cut_value:.4f}")
    print(f"Approximation ratio:  {cut_value / max(sdp_bound, 1e-10):.4f}")
    print(f"  (GW guarantee: >= 0.878)")

    set_s = [i for i in range(n) if partition[i] > 0]
    set_t = [i for i in range(n) if partition[i] < 0]
    print(f"\nPartition:")
    print(f"  Set S: {set_s}")
    print(f"  Set T: {set_t}")

    print(f"\nCut edges:")
    for i in range(n):
        for j in range(i + 1, n):
            if partition[i] != partition[j] and W[i, j] > 0:
                print(f"  ({i}, {j})  weight = {W[i, j]:.1f}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # --- Petersen graph ---
    print("Solving Max-Cut on the Petersen graph...\n")
    W_peter = petersen_graph()
    sdp_bound, X_val = maxcut_sdp(W_peter)
    partition, cut_val = hyperplane_rounding(X_val, W_peter)
    print_maxcut_result(W_peter, sdp_bound, partition, cut_val, "Petersen Graph")
    print(f"\n  (Known optimal Max-Cut for Petersen graph: 12)")

    # --- Random graph ---
    print("\n")
    W_rand = random_graph(n=8, density=0.5)
    sdp_bound2, X_val2 = maxcut_sdp(W_rand)
    partition2, cut_val2 = hyperplane_rounding(X_val2, W_rand)
    print_maxcut_result(W_rand, sdp_bound2, partition2, cut_val2, "Random Graph (n=8)")


if __name__ == "__main__":
    main()
