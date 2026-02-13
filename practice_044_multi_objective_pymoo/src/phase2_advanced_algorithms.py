"""Phase 2: Advanced Algorithms — NSGA-III & MOEA/D.

Applies reference-direction-based algorithms to a 3-objective problem (DTLZ2).
Compares NSGA-III (niche-preserving selection) with MOEA/D (decomposition into
scalar subproblems via Tchebycheff scalarization).

pymoo API patterns used:
  - get_reference_directions("das-dennis", n_obj, n_partitions=p)
  - NSGA3(ref_dirs=ref_dirs)
  - MOEAD(ref_dirs=ref_dirs, n_neighbors=T, prob_neighbor_mating=0.7)
  - get_problem("dtlz2", n_var=N, n_obj=M)
  - 3D Pareto front visualization with mplot3d
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


# ============================================================================
# TODO(human): Run NSGA-III on DTLZ2 (3 objectives)
# ============================================================================

def run_nsga3_on_dtlz2(
    n_partitions: int = 12,
    n_gen: int = 200,
    seed: int = 42,
):
    """Run NSGA-III on the DTLZ2 problem with 3 objectives.

    DTLZ2 with 3 objectives has a spherical Pareto front: the true PF lies
    on the first octant of the unit sphere (f_1^2 + f_2^2 + f_3^2 = 1).

    Args:
        n_partitions: Number of partitions for Das-Dennis reference directions.
            For 3 objectives and p=12, this generates C(3+12-1,12) = 91 points.
        n_gen: Number of generations.
        seed: Random seed.

    Returns:
        result: pymoo Result object.
        pf_true: Analytical Pareto front of DTLZ2.
        ref_dirs: The reference directions used.
    """
    # TODO(human): Set up NSGA-III with Das-Dennis reference directions
    #
    # Step 1: Generate reference directions
    #   ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    #
    #   Das-Dennis generates uniformly distributed points on the unit simplex
    #   in M-dimensional space. For 3 objectives with 12 partitions:
    #     C(3 + 12 - 1, 12) = C(14, 12) = 91 reference points.
    #   Each reference direction is a vector (w_1, w_2, w_3) with w_i >= 0
    #   and w_1 + w_2 + w_3 = 1. These define "target directions" on the
    #   Pareto front where NSGA-III tries to place solutions.
    #
    #   The number of reference directions determines the population size:
    #   pop_size is set automatically to the smallest multiple of 4 >= len(ref_dirs).
    #
    # Step 2: Load DTLZ2 with 3 objectives
    #   problem = get_problem("dtlz2", n_obj=3)
    #
    #   DTLZ2 default: n_var = n_obj + k - 1 = 3 + 10 - 1 = 12 variables.
    #   The first (n_obj-1)=2 variables control position on the PF sphere.
    #   The remaining k=10 variables should converge to 0.5 at the optimum.
    #   The spherical PF makes it easy to visually judge convergence and spread.
    #
    # Step 3: Configure NSGA-III
    #   algorithm = NSGA3(ref_dirs=ref_dirs)
    #
    #   NSGA-III key difference from NSGA-II:
    #   - Still uses non-dominated sorting for ranking.
    #   - Replaces crowding distance with reference-point association:
    #     each solution is associated with its closest reference direction,
    #     and the niche count (how many solutions share that direction)
    #     is used for diversity. Prefer directions with fewer associated solutions.
    #
    # Step 4: Run minimize and get the true PF
    #   result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    #   pf_true = problem.pareto_front(ref_dirs=ref_dirs)
    #
    #   Note: DTLZ2's pareto_front() needs ref_dirs to generate PF points
    #   at matching locations. The true PF is the intersection of the unit
    #   sphere with the positive octant.
    #
    # Return (result, pf_true, ref_dirs).
    raise NotImplementedError("TODO(human): generate ref_dirs, load dtlz2, run NSGA3, return all")


# ============================================================================
# TODO(human): Run MOEA/D on DTLZ2 (3 objectives)
# ============================================================================

def run_moead_on_dtlz2(
    n_partitions: int = 12,
    n_gen: int = 200,
    n_neighbors: int = 15,
    seed: int = 42,
):
    """Run MOEA/D on the DTLZ2 problem with 3 objectives.

    MOEA/D decomposes the multi-objective problem into N single-objective
    subproblems using weight vectors (= reference directions) and Tchebycheff
    scalarization. Each subproblem optimizes in its neighborhood.

    Args:
        n_partitions: Number of partitions for reference direction generation.
        n_gen: Number of generations.
        n_neighbors: Neighborhood size T — how many nearby subproblems
            exchange solutions during mating.
        seed: Random seed.

    Returns:
        result: pymoo Result object.
        pf_true: Analytical Pareto front of DTLZ2.
    """
    # TODO(human): Configure MOEA/D with Tchebycheff decomposition
    #
    # Step 1: Generate reference directions (same as NSGA-III)
    #   ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    #
    #   In MOEA/D, each reference direction defines a scalar subproblem.
    #   With 91 reference directions, MOEA/D maintains 91 solutions, each
    #   optimizing a different Tchebycheff subproblem.
    #
    # Step 2: Load DTLZ2
    #   problem = get_problem("dtlz2", n_obj=3)
    #
    # Step 3: Configure MOEA/D
    #   algorithm = MOEAD(
    #       ref_dirs=ref_dirs,
    #       n_neighbors=n_neighbors,
    #       prob_neighbor_mating=0.7,
    #   )
    #
    #   MOEA/D key concepts:
    #   - Each subproblem i has weight vector λ_i (= ref_dirs[i]).
    #   - Tchebycheff scalarization: g(x|λ) = max_j { λ_j * |f_j(x) - z*_j| }
    #     where z* is the ideal point (best observed value per objective).
    #     This decomposes the MOO into N independent minimization problems.
    #   - Neighborhood: subproblem i's neighbors are the T subproblems
    #     with the closest weight vectors. Mating partners come from neighbors.
    #   - prob_neighbor_mating=0.7: 70% chance to mate within neighborhood,
    #     30% chance to mate with the whole population (for exploration).
    #   - n_neighbors=15: typical for 91 subproblems. Too small → poor
    #     diversity. Too large → slow convergence (acts like a global GA).
    #
    # Step 4: Run minimize
    #   result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    #   pf_true = problem.pareto_front(ref_dirs=ref_dirs)
    #
    # Return (result, pf_true).
    raise NotImplementedError("TODO(human): generate ref_dirs, load dtlz2, run MOEAD, return both")


# ============================================================================
# Visualization helpers
# ============================================================================

def plot_pareto_front_3d(
    F: np.ndarray,
    title: str,
    pf_true: np.ndarray | None = None,
) -> None:
    """Plot a 3D Pareto front with optional true PF comparison."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if pf_true is not None:
        ax.scatter(
            pf_true[:, 0], pf_true[:, 1], pf_true[:, 2],
            c="gray", s=10, alpha=0.3, label="True PF",
        )

    ax.scatter(
        F[:, 0], F[:, 1], F[:, 2],
        c="C0", s=25, alpha=0.7, edgecolors="none", label="Obtained",
    )

    ax.set_xlabel("f₁", fontsize=11)
    ax.set_ylabel("f₂", fontsize=11)
    ax.set_zlabel("f₃", fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_comparison_3d(
    F_nsga3: np.ndarray,
    F_moead: np.ndarray,
    pf_true: np.ndarray | None = None,
) -> None:
    """Side-by-side 3D comparison of NSGA-III and MOEA/D."""
    fig = plt.figure(figsize=(16, 7))

    for idx, (F, name, color) in enumerate([
        (F_nsga3, "NSGA-III", "C0"),
        (F_moead, "MOEA/D", "C1"),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        if pf_true is not None:
            ax.scatter(
                pf_true[:, 0], pf_true[:, 1], pf_true[:, 2],
                c="gray", s=8, alpha=0.2, label="True PF",
            )
        ax.scatter(
            F[:, 0], F[:, 1], F[:, 2],
            c=color, s=25, alpha=0.7, edgecolors="none", label=name,
        )
        ax.set_xlabel("f₁", fontsize=10)
        ax.set_ylabel("f₂", fontsize=10)
        ax.set_zlabel("f₃", fontsize=10)
        ax.set_title(f"DTLZ2 — {name}", fontsize=13)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_reference_directions(ref_dirs: np.ndarray) -> None:
    """Plot Das-Dennis reference directions on the unit simplex."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        ref_dirs[:, 0], ref_dirs[:, 1], ref_dirs[:, 2],
        c="C2", s=40, alpha=0.8, edgecolors="k", linewidths=0.5,
    )
    ax.set_xlabel("w₁", fontsize=11)
    ax.set_ylabel("w₂", fontsize=11)
    ax.set_zlabel("w₃", fontsize=11)
    ax.set_title(f"Das-Dennis Reference Directions ({len(ref_dirs)} points)", fontsize=13)
    plt.tight_layout()
    plt.show()


def print_result_summary(result, name: str) -> None:
    """Print summary statistics for a many-objective result."""
    F = result.F
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Solutions found:   {F.shape[0]}")
    print(f"  Objectives:        {F.shape[1]}")
    for i in range(F.shape[1]):
        print(f"  f_{i+1} range:         [{F[:, i].min():.4f}, {F[:, i].max():.4f}]")

    # For DTLZ2 the PF lies on the unit sphere: sum(f_i^2) = 1
    sphere_err = np.abs(np.sum(F**2, axis=1) - 1.0)
    print(f"  Sphere error:      mean={sphere_err.mean():.6f}, max={sphere_err.max():.6f}")
    print(f"  Execution time:    {result.exec_time:.2f}s")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Phase 2: Advanced Algorithms — NSGA-III & MOEA/D")
    print("=" * 60)

    # --- NSGA-III on DTLZ2 ---
    print("\n--- NSGA-III on DTLZ2 (3 objectives) ---")
    res_nsga3, pf_true_nsga3, ref_dirs = run_nsga3_on_dtlz2(n_partitions=12, n_gen=200)
    print_result_summary(res_nsga3, "DTLZ2 (NSGA-III)")
    print(f"  Reference dirs:    {len(ref_dirs)}")

    # Visualize reference directions
    plot_reference_directions(ref_dirs)

    # 3D Pareto front
    plot_pareto_front_3d(
        res_nsga3.F,
        "DTLZ2 — NSGA-III (3 objectives)",
        pf_true=pf_true_nsga3,
    )

    # --- MOEA/D on DTLZ2 ---
    print("\n--- MOEA/D on DTLZ2 (3 objectives) ---")
    res_moead, pf_true_moead = run_moead_on_dtlz2(
        n_partitions=12, n_gen=200, n_neighbors=15,
    )
    print_result_summary(res_moead, "DTLZ2 (MOEA/D)")

    # Side-by-side comparison
    plot_comparison_3d(
        res_nsga3.F,
        res_moead.F,
        pf_true=pf_true_nsga3,
    )

    # --- Numerical comparison ---
    print("\n" + "=" * 60)
    print("  Algorithm Comparison Summary")
    print("=" * 60)
    for name, F in [("NSGA-III", res_nsga3.F), ("MOEA/D", res_moead.F)]:
        sphere_err = np.abs(np.sum(F**2, axis=1) - 1.0)
        print(f"  {name:<10s}  solutions={F.shape[0]:<4d}  "
              f"sphere_err_mean={sphere_err.mean():.6f}  "
              f"sphere_err_max={sphere_err.max():.6f}")

    print("\n[Phase 2 complete]")


if __name__ == "__main__":
    main()
