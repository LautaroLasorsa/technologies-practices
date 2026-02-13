"""Phase 3: Scenario Generation & Reduction.

Generate scenarios from multivariate distributions and reduce them to
a compact representative set using the fast forward selection algorithm
(Heitsch & Romisch, 2003) based on Kantorovich (Wasserstein) distance.

Topics:
  - Sampling from multivariate normal (correlated yields)
  - Kantorovich distance between discrete distributions
  - Greedy forward selection: iteratively add the most representative scenario
  - Probability redistribution after reduction
  - Solution quality comparison: full vs reduced scenario sets

References:
  - Heitsch & Romisch (2003), "Scenario Reduction Algorithms in Stochastic Programming"
  - Dupacova, Growe-Kuska & Romisch (2003), "Scenario Reduction in Stochastic Programming"
"""

import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


# ============================================================================
# Problem data (farmer problem -- same as Phase 1)
# ============================================================================

CROPS = ["wheat", "corn", "sugar_beets"]
TOTAL_LAND = 500.0
PLANTING_COST = {"wheat": 150.0, "corn": 230.0, "sugar_beets": 260.0}
SELLING_PRICE = {"wheat": 170.0, "corn": 150.0, "sugar_beets_favorable": 36.0, "sugar_beets_excess": 10.0}
PURCHASE_PRICE = {"wheat": 238.0, "corn": 210.0}
REQUIREMENTS = {"wheat": 200.0, "corn": 240.0}
SUGAR_BEET_QUOTA = 6000.0

# Multivariate distribution parameters for yield generation
# Yields are correlated: good weather helps all crops, bad weather hurts all
YIELD_MEAN = np.array([2.5, 3.0, 20.0])  # wheat, corn, sugar_beets (tons/acre)
YIELD_COV = np.array([
    [0.16,  0.10,  0.60],   # wheat variance and covariances
    [0.10,  0.25,  0.50],   # corn
    [0.60,  0.50,  9.00],   # sugar_beets
])
# Correlation interpretation: weather affects all crops in the same direction
# Wheat-corn correlation: 0.10 / sqrt(0.16 * 0.25) = 0.50
# Wheat-beets correlation: 0.60 / sqrt(0.16 * 9.0) = 0.50

YIELD_MIN = np.array([1.0, 1.2, 8.0])
YIELD_MAX = np.array([4.0, 4.8, 32.0])


# ============================================================================
# TODO(human): Generate scenarios from multivariate normal
# ============================================================================

def generate_scenarios_from_distribution(
    n_scenarios: int,
    mean: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate scenarios from a multivariate normal distribution.

    Each scenario is a yield vector for all crops. Samples are clipped to
    physical bounds [YIELD_MIN, YIELD_MAX]. Equal probabilities are assigned.

    Args:
        n_scenarios: Number of scenarios to generate.
        mean: Mean yield vector, shape (n_crops,).
        cov: Covariance matrix, shape (n_crops, n_crops).
        rng: NumPy random generator.

    Returns:
        scenarios: Array of shape (n_scenarios, n_crops) -- yield vectors.
        probabilities: Array of shape (n_scenarios,) -- all equal to 1/N.
    """
    # TODO(human): Generate multivariate normal scenarios
    #
    # Use rng.multivariate_normal() to sample correlated yield vectors:
    #   raw = rng.multivariate_normal(mean, cov, size=n_scenarios)
    #   This produces shape (n_scenarios, n_crops) where each row is a
    #   correlated sample from the 3-dimensional normal distribution.
    #
    # Clip to physical bounds to avoid nonsensical yields:
    #   scenarios = np.clip(raw, YIELD_MIN, YIELD_MAX)
    #
    # Assign equal probabilities:
    #   probabilities = np.full(n_scenarios, 1.0 / n_scenarios)
    #
    # WHY multivariate normal instead of independent normals?
    #   Crop yields are correlated through weather. If we sample independently,
    #   we might generate a scenario with great wheat but terrible corn yields
    #   that is physically implausible (both would be affected by the same
    #   drought). The covariance matrix captures these dependencies.
    #
    # WHY equal probabilities?
    #   With Monte Carlo sampling, each sample has equal weight 1/N.
    #   After scenario reduction, the probabilities will be redistributed
    #   to account for deleted scenarios, breaking the uniformity.
    #
    # Return (scenarios, probabilities).
    raise NotImplementedError("TODO(human): sample from multivariate normal, clip, assign equal probs")


# ============================================================================
# TODO(human): Kantorovich distance computation
# ============================================================================

def kantorovich_distance(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    selected_indices: list[int],
) -> float:
    """Compute the Kantorovich distance between full and selected scenario sets.

    For each non-selected scenario i, the distance contribution is:
      p_i * min_{j in selected} ||ξ_i - ξ_j||_2

    The total Kantorovich distance is the sum over all non-selected scenarios.
    Lower distance means the selected subset better represents the original.

    Args:
        scenarios: All scenarios, shape (S, n_crops).
        probabilities: Probabilities for all scenarios, shape (S,).
        selected_indices: Indices of scenarios in the reduced set.

    Returns:
        Kantorovich distance (float, >= 0).
    """
    # TODO(human): Compute Kantorovich distance
    #
    # The Kantorovich (Wasserstein-1) distance for discrete distributions
    # measures the minimum "transportation cost" to move probability mass
    # from the full set to the selected subset.
    #
    # Algorithm:
    #   total_dist = 0.0
    #   selected_set = set(selected_indices)
    #   selected_scenarios = scenarios[selected_indices]  # shape (|J|, n_crops)
    #
    #   for i in range(len(scenarios)):
    #       if i not in selected_set:
    #           # Distance from scenario i to its nearest selected scenario
    #           diffs = selected_scenarios - scenarios[i]     # shape (|J|, n_crops)
    #           dists = np.linalg.norm(diffs, axis=1)         # shape (|J|,)
    #           min_dist = np.min(dists)
    #           total_dist += probabilities[i] * min_dist
    #
    # Vectorized alternative (faster):
    #   non_selected = [i for i in range(len(scenarios)) if i not in selected_set]
    #   For each non-selected i, compute distance to ALL selected, take min.
    #   Use broadcasting: diffs has shape (|non_selected|, |J|, n_crops)
    #
    # MATHEMATICAL CONTEXT:
    #   The Kantorovich distance D_K satisfies:
    #     D_K(P, Q) = inf { E[c(X,Y)] : (X,Y) has marginals P and Q }
    #   For our case with discrete measures and 1-Wasserstein distance:
    #     D_K = Σ_{i not in J} p_i * min_{j in J} ||ξ_i - ξ_j||
    #   This is the cost of "assigning" each deleted scenario to its
    #   nearest retained scenario, weighted by probability.
    #
    # Return total_dist (float).
    raise NotImplementedError("TODO(human): compute Kantorovich distance between full and selected sets")


# ============================================================================
# TODO(human): Fast forward selection scenario reduction
# ============================================================================

def fast_forward_selection(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Reduce scenarios using the fast forward selection algorithm.

    Greedily add scenarios one at a time, always picking the one that
    reduces the Kantorovich distance the most. After selection, redistribute
    deleted scenarios' probabilities to their nearest retained scenario.

    Algorithm (Heitsch & Romisch, 2003):
      1. Start with J = {} (empty selected set)
      2. For each step until |J| = n_target:
         a. For each candidate j not in J:
            - Compute D_j = kantorovich_distance(scenarios, probs, J ∪ {j})
         b. Add the j that minimizes D_j
      3. Redistribute probabilities: for each deleted scenario i,
         add p_i to its nearest scenario in J

    Args:
        scenarios: All scenarios, shape (S, n_crops).
        probabilities: Original probabilities, shape (S,).
        n_target: Number of scenarios to retain.

    Returns:
        reduced_scenarios: shape (n_target, n_crops).
        reduced_probabilities: shape (n_target,) -- redistributed, sum to 1.
        selected_indices: which original indices were retained.
    """
    # TODO(human): Implement fast forward selection
    #
    # This is a greedy algorithm. At each iteration, try every candidate
    # and pick the one that minimizes the Kantorovich distance.
    #
    # PSEUDOCODE:
    #   S = len(scenarios)
    #   selected = []
    #
    #   for step in range(n_target):
    #       best_candidate = -1
    #       best_distance = float('inf')
    #
    #       for j in range(S):
    #           if j in selected:
    #               continue
    #           trial = selected + [j]
    #           d = kantorovich_distance(scenarios, probabilities, trial)
    #           if d < best_distance:
    #               best_distance = d
    #               best_candidate = j
    #
    #       selected.append(best_candidate)
    #       print(f"    Step {step+1}: added scenario {best_candidate}, "
    #             f"Kantorovich distance = {best_distance:.6f}")
    #
    # OPTIMIZATION HINT:
    #   The naive approach recomputes kantorovich_distance from scratch
    #   at each trial. For large S, this is O(S^2 * n_target). A faster
    #   approach caches distances and updates incrementally. For this
    #   practice, the naive approach is fine (S <= 500).
    #
    # PROBABILITY REDISTRIBUTION:
    #   After selecting the n_target scenarios, redistribute probabilities:
    #   reduced_probs = np.zeros(n_target)
    #   For each scenario i (selected or not):
    #     Find j_nearest = argmin_{j in selected} ||ξ_i - ξ_j||
    #     reduced_probs[index_of_j_nearest_in_selected_list] += probabilities[i]
    #
    #   This ensures sum(reduced_probs) == 1.0 and each retained scenario
    #   "absorbs" the probability mass of its deleted neighbors.
    #
    # RETURN:
    #   reduced_scenarios = scenarios[selected]
    #   reduced_probabilities = the redistributed probabilities
    #   selected_indices = selected (list of original indices)
    raise NotImplementedError("TODO(human): implement greedy forward selection with probability redistribution")


# ============================================================================
# TODO(human): Compare solutions with full vs reduced scenarios
# ============================================================================

def compare_solutions(
    scenarios_full: np.ndarray,
    probs_full: np.ndarray,
    scenarios_reduced: np.ndarray,
    probs_reduced: np.ndarray,
) -> dict:
    """Solve the farmer problem with full and reduced scenario sets.

    Compare first-stage decisions and objective values to assess
    how well the reduced set approximates the full problem.

    Args:
        scenarios_full: Full scenario set, shape (S, n_crops).
        probs_full: Full probabilities, shape (S,).
        scenarios_reduced: Reduced scenarios, shape (n_target, n_crops).
        probs_reduced: Redistributed probabilities, shape (n_target,).

    Returns:
        Dictionary with objective values and solutions for both.
    """
    # TODO(human): Solve both problems and compare
    #
    # For each scenario set (full and reduced), build and solve the farmer's
    # extensive form. You need to convert the numpy arrays into the format
    # expected by a Pyomo model.
    #
    # Build an extensive form model for a given (scenarios, probabilities):
    #   model = pyo.ConcreteModel()
    #   model.x = pyo.Var(CROPS, within=pyo.NonNegativeReals)
    #   model.land = pyo.Constraint(expr=sum(model.x[c] for c in CROPS) <= TOTAL_LAND)
    #
    #   For each scenario index s with scenarios[s] and probabilities[s]:
    #     Create a Block with second-stage variables and linking constraints
    #     (same pattern as build_and_solve_saa_model from Phase 2).
    #
    #   You can factor out a helper function:
    #     def solve_with_scenarios(scens, probs) -> (obj_val, x_sol):
    #       Build model, solve, return objective and first-stage decisions.
    #
    # Then:
    #   obj_full, x_full = solve_with_scenarios(scenarios_full, probs_full)
    #   obj_reduced, x_reduced = solve_with_scenarios(scenarios_reduced, probs_reduced)
    #
    # Return {
    #   "obj_full": obj_full, "x_full": x_full,
    #   "obj_reduced": obj_reduced, "x_reduced": x_reduced,
    #   "obj_gap_pct": 100 * abs(obj_full - obj_reduced) / abs(obj_full),
    # }
    #
    # INTERPRETATION:
    #   If the reduced set is a good approximation, first-stage decisions
    #   and objectives should be close. A small objective gap (<1%) means
    #   the reduction preserved the essential information.
    raise NotImplementedError("TODO(human): solve farmer problem with full and reduced scenarios, compare")


# ============================================================================
# Visualization helpers (provided)
# ============================================================================

def plot_scenarios_2d(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    title: str,
    selected_indices: list[int] | None = None,
    crop_x: int = 0,
    crop_y: int = 1,
) -> None:
    """Plot scenarios in 2D (two crop yields), optionally highlighting selected."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all scenarios
    sizes = probabilities * 5000  # scale for visibility
    ax.scatter(
        scenarios[:, crop_x], scenarios[:, crop_y],
        s=sizes, c="lightblue", edgecolors="steelblue", alpha=0.6,
        label=f"All ({len(scenarios)})",
    )

    if selected_indices is not None:
        ax.scatter(
            scenarios[selected_indices, crop_x],
            scenarios[selected_indices, crop_y],
            s=100, c="red", marker="*", zorder=5,
            label=f"Selected ({len(selected_indices)})",
        )

    crop_names = CROPS
    ax.set_xlabel(f"{crop_names[crop_x]} yield (tons/acre)", fontsize=12)
    ax.set_ylabel(f"{crop_names[crop_y]} yield (tons/acre)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_reduction_convergence(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    max_selected: int,
) -> None:
    """Plot Kantorovich distance as scenarios are added during forward selection."""
    # Track distance at each step
    distances = []
    selected = []
    S = len(scenarios)

    for step in range(min(max_selected, S)):
        best_j = -1
        best_d = float("inf")
        for j in range(S):
            if j in selected:
                continue
            trial = selected + [j]
            d = kantorovich_distance(scenarios, probabilities, trial)
            if d < best_d:
                best_d = d
                best_j = j
        selected.append(best_j)
        distances.append(best_d)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(distances) + 1), distances, "o-", color="C0")
    ax.set_xlabel("Number of selected scenarios", fontsize=12)
    ax.set_ylabel("Kantorovich distance", fontsize=12)
    ax.set_title("Scenario Reduction: Kantorovich Distance vs Selected Count", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_probability_comparison(
    probs_original: np.ndarray,
    probs_reduced: np.ndarray,
    selected_indices: list[int],
) -> None:
    """Compare original and redistributed probabilities for selected scenarios."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(selected_indices))
    width = 0.35

    ax.bar(x_pos - width / 2, probs_original[selected_indices], width,
           label="Original probability", color="C0", alpha=0.7)
    ax.bar(x_pos + width / 2, probs_reduced, width,
           label="Redistributed probability", color="C1", alpha=0.7)

    ax.set_xlabel("Selected scenario index", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Probability Redistribution After Scenario Reduction", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in selected_indices], fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def print_comparison(result: dict) -> None:
    """Print comparison of full vs reduced solutions."""
    print(f"\n{'=' * 65}")
    print(f"  Full vs Reduced Scenario Set Comparison")
    print(f"{'=' * 65}")
    print(f"  {'':>12} | {'Full Set':>14} | {'Reduced Set':>14}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*14}")
    print(f"  {'Objective':>12} | {result['obj_full']:>14,.2f} | {result['obj_reduced']:>14,.2f}")
    for c in CROPS:
        print(f"  {c:>12} | {result['x_full'][c]:>14.1f} | {result['x_reduced'][c]:>14.1f}")
    print(f"  {'Gap (%)':>12} | {'':>14} | {result['obj_gap_pct']:>13.2f}%")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 3: Scenario Generation & Reduction")
    print("=" * 65)

    rng = np.random.default_rng(42)

    # --- Generate scenarios ---
    print("\n--- Generating 200 Scenarios from Multivariate Normal ---")
    scenarios, probs = generate_scenarios_from_distribution(200, YIELD_MEAN, YIELD_COV, rng)
    print(f"  Generated {len(scenarios)} scenarios")
    print(f"  Yield ranges:")
    for i, c in enumerate(CROPS):
        print(f"    {c:>12}: [{scenarios[:, i].min():.2f}, {scenarios[:, i].max():.2f}]"
              f"  (mean={scenarios[:, i].mean():.2f})")

    plot_scenarios_2d(scenarios, probs, "Generated Scenarios (Wheat vs Corn)")
    plot_scenarios_2d(scenarios, probs, "Generated Scenarios (Wheat vs Sugar Beets)",
                      crop_x=0, crop_y=2)

    # --- Scenario reduction: 200 -> 20 ---
    print("\n--- Fast Forward Selection: 200 -> 20 scenarios ---")
    reduced_scen, reduced_probs, selected_idx = fast_forward_selection(scenarios, probs, 20)
    print(f"  Selected {len(selected_idx)} scenarios")
    print(f"  Kantorovich distance: {kantorovich_distance(scenarios, probs, selected_idx):.6f}")
    print(f"  Probability range: [{reduced_probs.min():.4f}, {reduced_probs.max():.4f}]")
    print(f"  Probability sum: {reduced_probs.sum():.6f}")

    plot_scenarios_2d(scenarios, probs, "Scenario Reduction: 200 -> 20",
                      selected_indices=selected_idx)
    plot_probability_comparison(probs, reduced_probs, selected_idx)

    # --- Reduction convergence ---
    print("\n--- Kantorovich Distance Convergence ---")
    plot_reduction_convergence(scenarios, probs, max_selected=30)

    # --- Solution comparison ---
    print("\n--- Solution Quality: Full vs Reduced ---")
    comparison = compare_solutions(scenarios, probs, reduced_scen, reduced_probs)
    print_comparison(comparison)

    # --- Different reduction levels ---
    print("\n--- Reduction Level Sensitivity ---")
    for n_keep in [5, 10, 20, 50]:
        red_s, red_p, red_idx = fast_forward_selection(scenarios, probs, n_keep)
        comp = compare_solutions(scenarios, probs, red_s, red_p)
        dk = kantorovich_distance(scenarios, probs, red_idx)
        print(f"  n={n_keep:>3}: obj_gap={comp['obj_gap_pct']:.2f}%, "
              f"Kantorovich={dk:.4f}")

    print("\n[Phase 3 complete]")


if __name__ == "__main__":
    main()
