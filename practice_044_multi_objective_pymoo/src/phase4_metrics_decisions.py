"""Phase 4: Performance Metrics & Decision Making.

Computes quality indicators (hypervolume, IGD) for Pareto front evaluation,
tracks convergence across generations, and applies decision-making methods
(pseudo-weights, high trade-off points) to select a final solution.

pymoo API patterns used:
  - Hypervolume(ref_point=...).do(F): compute dominated volume
  - IGD(pf_true).do(F): compute inverted generational distance
  - Callback: track metrics per generation
  - PseudoWeights: find solution matching preference weights
  - HighTradeoffPoints: identify knee points on the Pareto front
"""

import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem


# ============================================================================
# TODO(human): Compute quality metrics
# ============================================================================

def compute_quality_metrics(
    F: np.ndarray,
    pf_true: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float]:
    """Compute hypervolume and IGD for a solution set.

    Args:
        F: (n_solutions, n_obj) objective values of the obtained solution set.
        pf_true: (n_pf_points, n_obj) true Pareto front (for IGD).
        ref_point: (n_obj,) reference point for hypervolume (must dominate all F).

    Returns:
        Dictionary with "hv" (hypervolume) and "igd" (inverted generational distance).
    """
    # TODO(human): Compute hypervolume and IGD
    #
    # --- Hypervolume (HV) ---
    # The hypervolume indicator measures the volume of objective space that is
    # dominated by the solution set F and bounded by the reference point.
    #
    # In pymoo:
    #   hv_indicator = Hypervolume(ref_point=ref_point)
    #   hv_value = hv_indicator.do(F)
    #
    # Key properties of hypervolume:
    #   - Higher HV = better (more dominated space)
    #   - Pareto-compliant: if set A dominates set B, HV(A) > HV(B)
    #   - Does NOT require the true PF — only a reference point
    #   - The reference point MUST be dominated by all solutions in F
    #     (i.e., ref_point[i] > max(F[:, i]) for all objectives, when minimizing)
    #   - Sensitive to reference point choice — compare HV values only
    #     when using the same reference point
    #   - Computational cost: O(n * log(n)) for 2 objectives,
    #     exponential for many objectives
    #
    # --- Inverted Generational Distance (IGD) ---
    # IGD measures how well the obtained set covers the true Pareto front.
    # For each point in the true PF, find the nearest obtained solution,
    # then average those distances.
    #
    # In pymoo:
    #   igd_indicator = IGD(pf_true)
    #   igd_value = igd_indicator.do(F)
    #
    # Key properties of IGD:
    #   - Lower IGD = better (closer to true PF + better coverage)
    #   - Measures BOTH convergence (are solutions near the PF?) and
    #     diversity (do solutions cover the entire PF?)
    #   - Requires the true PF — only usable on benchmark problems
    #   - IGD = 0 iff the obtained set contains the entire true PF
    #   - A set with good convergence but poor spread will have high IGD
    #     because distant PF points have no nearby obtained solution
    #
    # Return {"hv": hv_value, "igd": igd_value}
    raise NotImplementedError("TODO(human): compute HV and IGD, return as dict")


# ============================================================================
# TODO(human): Convergence tracking with callback
# ============================================================================

class ConvergenceCallback(Callback):
    """Callback that records hypervolume at each generation.

    pymoo calls notify() after each generation with the current algorithm state.
    We extract the population's objective values and compute HV.
    """

    def __init__(self, ref_point: np.ndarray):
        super().__init__()
        self.ref_point = ref_point
        self.data["hv"] = []

    def notify(self, algorithm):
        # TODO(human): Record hypervolume of the current population
        #
        # The algorithm object provides access to the current population:
        #   F = algorithm.pop.get("F")
        #
        # algorithm.pop is the current population (after selection).
        # .get("F") returns the objective values as a 2D array.
        #
        # Compute hypervolume:
        #   hv_indicator = Hypervolume(ref_point=self.ref_point)
        #   hv_value = hv_indicator.do(F)
        #
        # Append to history:
        #   self.data["hv"].append(hv_value)
        #
        # This callback is invoked EVERY generation, so self.data["hv"]
        # will contain n_gen values when optimization finishes.
        #
        # Note: For large populations or many objectives, computing HV
        # every generation can be expensive. In practice, you might
        # record every 10th generation instead.
        raise NotImplementedError("TODO(human): get F from algorithm.pop, compute HV, append to self.data['hv']")


def track_convergence(
    n_gen: int = 200,
    pop_size: int = 100,
    seed: int = 42,
    ref_point: np.ndarray | None = None,
):
    """Run NSGA-II on ZDT1 with convergence tracking.

    Args:
        n_gen: Number of generations.
        pop_size: Population size.
        seed: Random seed.
        ref_point: Reference point for HV computation.

    Returns:
        result: pymoo Result object.
        hv_history: List of HV values per generation.
        pf_true: True Pareto front.
    """
    # TODO(human): Run NSGA-II with the ConvergenceCallback
    #
    # Step 1: Set up problem and reference point
    #   problem = get_problem("zdt1")
    #   pf_true = problem.pareto_front()
    #   if ref_point is None:
    #       ref_point = np.array([1.1, 1.1])
    #       # Must dominate all PF solutions. ZDT1's PF has f1 in [0,1]
    #       # and f2 in [0,1], so (1.1, 1.1) works.
    #
    # Step 2: Create the callback
    #   callback = ConvergenceCallback(ref_point)
    #
    # Step 3: Configure and run
    #   algorithm = NSGA2(pop_size=pop_size)
    #   result = minimize(
    #       problem, algorithm, ("n_gen", n_gen),
    #       seed=seed, verbose=False,
    #       callback=callback,
    #   )
    #
    #   The callback parameter tells minimize() to call callback.notify()
    #   after each generation. This is the standard way to track
    #   algorithm progress in pymoo.
    #
    # Step 4: Extract history
    #   hv_history = callback.data["hv"]
    #
    # Return (result, hv_history, pf_true).
    raise NotImplementedError("TODO(human): run NSGA2 with ConvergenceCallback, return result + hv_history + pf_true")


# ============================================================================
# TODO(human): Decision making — pseudo-weights
# ============================================================================

def select_solution_pseudo_weights(
    F: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Select a Pareto-optimal solution using pseudo-weights.

    Pseudo-weights map each PF solution to a weight vector that represents
    the relative importance of objectives implied by that solution's position
    on the front. Given desired weights, find the best match.

    Args:
        F: (n_solutions, n_obj) objective values of the Pareto front.
        weights: (n_obj,) desired preference weights (e.g., [0.7, 0.3] means
                 70% importance on f_1, 30% on f_2). Must sum to 1.

    Returns:
        Index of the selected solution in F.
    """
    # TODO(human): Use pymoo's PseudoWeights for decision making
    #
    # Import: from pymoo.mcdm.pseudo_weights import PseudoWeights
    #
    # The PseudoWeights approach:
    # 1. Normalize the PF to [0, 1] on each objective using ideal and nadir.
    # 2. For each solution, compute its "pseudo-weight" vector:
    #    pw_i = (f_i^max - f_i) / (f_i^max - f_i^min)  (higher = better)
    #    then normalize so sum(pw) = 1.
    # 3. Find the solution whose pseudo-weight vector is closest to the
    #    desired weights.
    #
    # Usage:
    #   decomposition = PseudoWeights(weights)
    #   idx = decomposition.do(F, return_pseudo_weights=False)
    #
    #   PseudoWeights.do() returns the INDEX of the best-matching solution.
    #
    # Why pseudo-weights?
    #   - Intuitive: "I care 70% about cost and 30% about deflection"
    #   - Maps continuous preferences to discrete PF solutions
    #   - Works for any number of objectives
    #   - No need to re-run the optimizer — post-hoc selection from existing PF
    #
    # Return the index (int).
    raise NotImplementedError("TODO(human): use PseudoWeights to find the solution matching weights")


# ============================================================================
# TODO(human): Decision making — high trade-off (knee) points
# ============================================================================

def find_knee_points(F: np.ndarray) -> np.ndarray:
    """Identify knee points on the Pareto front.

    Knee points are solutions where the trade-off slope changes most rapidly —
    small improvements in one objective require large sacrifices in another.
    These are often the best compromise solutions.

    Args:
        F: (n_solutions, n_obj) objective values of the Pareto front.

    Returns:
        Array of indices of the knee point(s) in F.
    """
    # TODO(human): Use pymoo's HighTradeoffPoints
    #
    # Import: from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
    #
    # Knee points (high trade-off points) are identified by computing a
    # "trade-off" score for each PF solution. The score measures the
    # local curvature of the PF: how much the slope changes at that point.
    #
    # Usage:
    #   dm = HighTradeoffPoints()
    #   indices = dm.do(F)
    #
    #   Returns an array of indices into F where the trade-off is highest.
    #   For a 2D PF, the "knee" is the point of maximum curvature.
    #
    # Intuition: On a smooth convex PF, the knee is where the front
    # bends most. Moving along the PF from the knee causes the steepest
    # change in trade-off ratio. The knee represents the "best bang for
    # the buck" — the solution where neither objective can be improved
    # cheaply at the expense of the other.
    #
    # For the ZDT1 front f_2 = 1 - sqrt(f_1), the curvature is highest
    # near f_1 ≈ 0.2-0.3 (where the front transitions from steep to flat).
    #
    # Return indices (numpy array).
    raise NotImplementedError("TODO(human): use HighTradeoffPoints to identify knee points")


# ============================================================================
# Visualization helpers
# ============================================================================

def plot_convergence(hv_history: list[float]) -> None:
    """Plot hypervolume convergence across generations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(hv_history) + 1), hv_history, "C0-", linewidth=1.5)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title("NSGA-II Convergence — Hypervolume vs Generation", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark final HV
    final_hv = hv_history[-1]
    ax.axhline(y=final_hv, color="C1", linestyle="--", alpha=0.5, label=f"Final HV = {final_hv:.4f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_pareto_with_decision(
    F: np.ndarray,
    pf_true: np.ndarray | None,
    pw_idx: int | None = None,
    knee_indices: np.ndarray | None = None,
) -> None:
    """Plot Pareto front with decision-making selections highlighted."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if pf_true is not None:
        ax.plot(
            pf_true[:, 0], pf_true[:, 1],
            "k-", linewidth=2, alpha=0.4, label="True PF",
        )

    ax.scatter(
        F[:, 0], F[:, 1],
        c="C0", s=25, alpha=0.5, edgecolors="none", label="Pareto Front",
    )

    if pw_idx is not None:
        ax.scatter(
            F[pw_idx, 0], F[pw_idx, 1],
            c="C1", s=200, marker="*", zorder=5, edgecolors="k",
            linewidths=1, label=f"Pseudo-weight selection (idx={pw_idx})",
        )

    if knee_indices is not None and len(knee_indices) > 0:
        ax.scatter(
            F[knee_indices, 0], F[knee_indices, 1],
            c="C3", s=150, marker="D", zorder=5, edgecolors="k",
            linewidths=1, label=f"Knee point(s) ({len(knee_indices)})",
        )

    ax.set_xlabel("f₁", fontsize=12)
    ax.set_ylabel("f₂", fontsize=12)
    ax.set_title("Decision Making on the Pareto Front", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    metrics_dict: dict[str, dict[str, float]],
) -> None:
    """Bar chart comparing metrics across different configurations."""
    configs = list(metrics_dict.keys())
    hv_values = [metrics_dict[c]["hv"] for c in configs]
    igd_values = [metrics_dict[c]["igd"] for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(configs, hv_values, color="C0", alpha=0.7)
    ax.set_ylabel("Hypervolume (higher = better)", fontsize=11)
    ax.set_title("Hypervolume Comparison", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(configs, igd_values, color="C1", alpha=0.7)
    ax.set_ylabel("IGD (lower = better)", fontsize=11)
    ax.set_title("IGD Comparison", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Phase 4: Performance Metrics & Decision Making")
    print("=" * 60)

    # --- Baseline: run NSGA-II on ZDT1 ---
    problem = get_problem("zdt1")
    pf_true = problem.pareto_front()
    ref_point = np.array([1.1, 1.1])

    # --- Compare different population sizes ---
    print("\n--- Metrics Comparison: Population Size ---")
    metrics_dict = {}
    results_dict = {}

    for pop_size in [50, 100, 200]:
        algorithm = NSGA2(pop_size=pop_size)
        result = minimize(
            problem, algorithm, ("n_gen", 200),
            seed=42, verbose=False,
        )
        metrics = compute_quality_metrics(result.F, pf_true, ref_point)
        label = f"pop={pop_size}"
        metrics_dict[label] = metrics
        results_dict[label] = result
        print(f"  {label}: HV={metrics['hv']:.6f}, IGD={metrics['igd']:.6f}, "
              f"solutions={result.F.shape[0]}")

    plot_metrics_comparison(metrics_dict)

    # --- Convergence tracking ---
    print("\n--- Convergence Tracking ---")
    res_conv, hv_history, pf_conv = track_convergence(n_gen=200, pop_size=100)
    print(f"  Generations: {len(hv_history)}")
    print(f"  Initial HV:  {hv_history[0]:.6f}")
    print(f"  Final HV:    {hv_history[-1]:.6f}")
    print(f"  HV at gen 50: {hv_history[49]:.6f}")

    plot_convergence(hv_history)

    # --- Decision making ---
    print("\n--- Decision Making ---")
    F = res_conv.F

    # Pseudo-weights: prefer f1 (70% weight)
    weights_f1 = np.array([0.7, 0.3])
    idx_f1 = select_solution_pseudo_weights(F, weights_f1)
    print(f"\n  Pseudo-weights [0.7, 0.3] (prefer f₁):")
    print(f"    Selected index: {idx_f1}")
    print(f"    f₁ = {F[idx_f1, 0]:.4f}, f₂ = {F[idx_f1, 1]:.4f}")

    # Pseudo-weights: prefer f2 (30% f1, 70% f2)
    weights_f2 = np.array([0.3, 0.7])
    idx_f2 = select_solution_pseudo_weights(F, weights_f2)
    print(f"\n  Pseudo-weights [0.3, 0.7] (prefer f₂):")
    print(f"    Selected index: {idx_f2}")
    print(f"    f₁ = {F[idx_f2, 0]:.4f}, f₂ = {F[idx_f2, 1]:.4f}")

    # Equal weights
    weights_eq = np.array([0.5, 0.5])
    idx_eq = select_solution_pseudo_weights(F, weights_eq)
    print(f"\n  Pseudo-weights [0.5, 0.5] (balanced):")
    print(f"    Selected index: {idx_eq}")
    print(f"    f₁ = {F[idx_eq, 0]:.4f}, f₂ = {F[idx_eq, 1]:.4f}")

    # Knee points
    knee_indices = find_knee_points(F)
    print(f"\n  Knee points found: {len(knee_indices)}")
    for ki in knee_indices:
        print(f"    Index {ki}: f₁ = {F[ki, 0]:.4f}, f₂ = {F[ki, 1]:.4f}")

    # Plot with all decision selections
    plot_pareto_with_decision(F, pf_conv, pw_idx=idx_eq, knee_indices=knee_indices)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  Summary: Decision Making Methods")
    print(f"{'=' * 60}")
    print(f"  {'Method':<30s} {'f₁':>8s} {'f₂':>8s}")
    print(f"  {'-' * 48}")
    print(f"  {'PW [0.7, 0.3] (prefer f₁)':<30s} {F[idx_f1, 0]:>8.4f} {F[idx_f1, 1]:>8.4f}")
    print(f"  {'PW [0.5, 0.5] (balanced)':<30s} {F[idx_eq, 0]:>8.4f} {F[idx_eq, 1]:>8.4f}")
    print(f"  {'PW [0.3, 0.7] (prefer f₂)':<30s} {F[idx_f2, 0]:>8.4f} {F[idx_f2, 1]:>8.4f}")
    if len(knee_indices) > 0:
        ki = knee_indices[0]
        print(f"  {'Knee point':<30s} {F[ki, 0]:>8.4f} {F[ki, 1]:>8.4f}")

    print("\n[Phase 4 complete]")


if __name__ == "__main__":
    main()
