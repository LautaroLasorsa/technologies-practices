"""Phase 2: Sample Average Approximation (SAA).

When the number of scenarios is large or continuous, the extensive form
becomes intractable. SAA samples N scenarios, solves the resulting
extensive form, and repeats M times to get statistical bounds.

This phase applies SAA to a variant of the farmer's problem with
continuously distributed yields (normal around expected values).

SAA procedure:
  1. Draw M independent batches of N scenarios each
  2. Solve the extensive form for each batch -> M candidate solutions
  3. Evaluate the best candidate on a large independent sample
  4. Compute confidence intervals for the optimality gap

Pyomo patterns used:
  - Same Block-based extensive form as Phase 1
  - Repeated model construction and solving in a loop
  - numpy for sampling and statistics
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


# ============================================================================
# Problem data (same as Phase 1, but yields will be sampled)
# ============================================================================

CROPS = ["wheat", "corn", "sugar_beets"]
TOTAL_LAND = 500.0
PLANTING_COST = {"wheat": 150.0, "corn": 230.0, "sugar_beets": 260.0}
SELLING_PRICE = {"wheat": 170.0, "corn": 150.0, "sugar_beets_favorable": 36.0, "sugar_beets_excess": 10.0}
PURCHASE_PRICE = {"wheat": 238.0, "corn": 210.0}
REQUIREMENTS = {"wheat": 200.0, "corn": 240.0}
SUGAR_BEET_QUOTA = 6000.0

# Expected yields and standard deviations for continuous sampling
EXPECTED_YIELDS = {"wheat": 2.5, "corn": 3.0, "sugar_beets": 20.0}
YIELD_STDDEV = {"wheat": 0.4, "corn": 0.5, "sugar_beets": 3.0}

# Minimum yields (physical lower bound -- can't be negative)
MIN_YIELDS = {"wheat": 1.0, "corn": 1.2, "sugar_beets": 8.0}
MAX_YIELDS = {"wheat": 4.0, "corn": 4.8, "sugar_beets": 32.0}


# ============================================================================
# TODO(human): Sample scenarios from continuous distribution
# ============================================================================

def sample_scenarios(n_scenarios: int, rng: np.random.Generator) -> list[dict]:
    """Generate N random yield scenarios by sampling from normal distributions.

    Each scenario is a dictionary mapping crop names to yield values (tons/acre).
    Yields are sampled from Normal(EXPECTED_YIELDS[c], YIELD_STDDEV[c]) and
    clipped to [MIN_YIELDS[c], MAX_YIELDS[c]] to stay physically plausible.

    Args:
        n_scenarios: Number of scenarios to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of N dictionaries, each mapping crop -> yield (tons/acre).
    """
    # TODO(human): Sample yield scenarios
    #
    # For each of the n_scenarios, sample yields for all 3 crops:
    #   For each crop c in CROPS:
    #     raw_yield = rng.normal(loc=EXPECTED_YIELDS[c], scale=YIELD_STDDEV[c])
    #     clipped_yield = np.clip(raw_yield, MIN_YIELDS[c], MAX_YIELDS[c])
    #
    # You can do this efficiently with rng.normal() for arrays:
    #   means = np.array([EXPECTED_YIELDS[c] for c in CROPS])       # shape (3,)
    #   stds  = np.array([YIELD_STDDEV[c] for c in CROPS])          # shape (3,)
    #   mins  = np.array([MIN_YIELDS[c] for c in CROPS])            # shape (3,)
    #   maxs  = np.array([MAX_YIELDS[c] for c in CROPS])            # shape (3,)
    #   raw = rng.normal(loc=means, scale=stds, size=(n_scenarios, 3))
    #   clipped = np.clip(raw, mins, maxs)                          # shape (N, 3)
    #
    # Then convert each row into a dict:
    #   scenarios = [
    #       {CROPS[j]: clipped[i, j] for j in range(len(CROPS))}
    #       for i in range(n_scenarios)
    #   ]
    #
    # WHY clip? Negative yields are physically impossible. Clipping to
    # [MIN, MAX] keeps scenarios realistic. This is a simple truncated-normal
    # approximation -- more sophisticated approaches use moment matching.
    #
    # Return the list of scenario dicts.
    raise NotImplementedError("TODO(human): sample N yield scenarios from normal distributions")


# ============================================================================
# Provided: Build extensive form for SAA (same as Phase 1 but generalized)
# ============================================================================

def build_and_solve_saa_model(scenarios: list[dict]) -> tuple[float, dict]:
    """Build and solve the extensive form for a set of equi-probable scenarios.

    Each scenario has probability 1/N. The model structure is identical to
    Phase 1's extensive form.

    Args:
        scenarios: List of yield dictionaries (one per scenario).

    Returns:
        Tuple of (optimal_objective, first_stage_solution).
        first_stage_solution is a dict: crop -> acres allocated.
    """
    n = len(scenarios)
    prob = 1.0 / n

    model = pyo.ConcreteModel()
    model.x = pyo.Var(CROPS, within=pyo.NonNegativeReals)

    # Land constraint
    model.land = pyo.Constraint(
        expr=sum(model.x[c] for c in CROPS) <= TOTAL_LAND
    )

    # Scenario blocks
    scenario_names = list(range(n))

    def scenario_block_rule(block, s):
        yields = scenarios[s]
        block.y_sell_wheat = pyo.Var(within=pyo.NonNegativeReals)
        block.y_sell_corn = pyo.Var(within=pyo.NonNegativeReals)
        block.y_sell_beets_fav = pyo.Var(within=pyo.NonNegativeReals)
        block.y_sell_beets_exc = pyo.Var(within=pyo.NonNegativeReals)
        block.y_buy_wheat = pyo.Var(within=pyo.NonNegativeReals)
        block.y_buy_corn = pyo.Var(within=pyo.NonNegativeReals)

        block.wheat_balance = pyo.Constraint(
            expr=yields["wheat"] * model.x["wheat"] + block.y_buy_wheat - block.y_sell_wheat >= REQUIREMENTS["wheat"]
        )
        block.corn_balance = pyo.Constraint(
            expr=yields["corn"] * model.x["corn"] + block.y_buy_corn - block.y_sell_corn >= REQUIREMENTS["corn"]
        )
        block.beet_sell = pyo.Constraint(
            expr=block.y_sell_beets_fav + block.y_sell_beets_exc <= yields["sugar_beets"] * model.x["sugar_beets"]
        )
        block.beet_quota = pyo.Constraint(
            expr=block.y_sell_beets_fav <= SUGAR_BEET_QUOTA
        )

    model.scenarios = pyo.Block(scenario_names, rule=scenario_block_rule)

    # Probability-weighted objective
    first_stage = sum(PLANTING_COST[c] * model.x[c] for c in CROPS)

    second_stage = sum(
        prob * (
            - SELLING_PRICE["wheat"] * model.scenarios[s].y_sell_wheat
            - SELLING_PRICE["corn"] * model.scenarios[s].y_sell_corn
            - SELLING_PRICE["sugar_beets_favorable"] * model.scenarios[s].y_sell_beets_fav
            - SELLING_PRICE["sugar_beets_excess"] * model.scenarios[s].y_sell_beets_exc
            + PURCHASE_PRICE["wheat"] * model.scenarios[s].y_buy_wheat
            + PURCHASE_PRICE["corn"] * model.scenarios[s].y_buy_corn
        )
        for s in scenario_names
    )

    model.obj = pyo.Objective(expr=first_stage + second_stage, sense=pyo.minimize)

    solver = pyo.SolverFactory("highs")
    result = solver.solve(model)
    assert result.solver.termination_condition == TerminationCondition.optimal

    obj_val = pyo.value(model.obj)
    x_sol = {c: pyo.value(model.x[c]) for c in CROPS}
    return obj_val, x_sol


# ============================================================================
# TODO(human): Single SAA replication
# ============================================================================

def saa_single_replication(n_scenarios: int, rng: np.random.Generator) -> tuple[float, dict]:
    """Run a single SAA replication: sample N scenarios and solve.

    Args:
        n_scenarios: Number of scenarios to sample for this replication.
        rng: NumPy random generator.

    Returns:
        Tuple of (optimal_objective, first_stage_solution).
    """
    # TODO(human): One SAA replication
    #
    # This is straightforward -- chain together sample_scenarios and
    # build_and_solve_saa_model:
    #
    #   1. scenarios = sample_scenarios(n_scenarios, rng)
    #   2. obj_val, x_sol = build_and_solve_saa_model(scenarios)
    #   3. return (obj_val, x_sol)
    #
    # WHY is this a separate function? In the full SAA loop (run_saa),
    # we call this M times with different random seeds. Each replication
    # gives a different scenario sample and hence a different optimal
    # objective. The variation across replications measures the sampling
    # error and determines the confidence interval width.
    #
    # The optimal objective from each replication is a BIASED ESTIMATOR
    # of the true optimal value -- it's biased LOW because we optimize
    # over the sampled scenarios (optimization bias). The average across
    # M replications provides a statistical lower bound.
    raise NotImplementedError("TODO(human): sample scenarios and solve the extensive form")


# ============================================================================
# TODO(human): Full SAA procedure with confidence intervals
# ============================================================================

def run_saa(
    n_scenarios: int,
    n_replications: int,
    n_eval_scenarios: int = 5000,
    seed: int = 42,
) -> dict:
    """Run the full SAA procedure.

    Steps:
      1. Run M replications of SAA, each with N scenarios.
      2. Collect objective values -> compute lower bound + CI.
      3. Pick the best candidate solution (lowest objective).
      4. Evaluate it on a large independent sample -> upper bound + CI.
      5. Optimality gap = upper bound - lower bound.

    Args:
        n_scenarios: Scenarios per replication (N).
        n_replications: Number of replications (M).
        n_eval_scenarios: Scenarios for evaluating the best candidate.
        seed: Master random seed.

    Returns:
        Dictionary with SAA statistics.
    """
    # TODO(human): Implement the full SAA loop
    #
    # STEP 1: Run M replications
    #   rng = np.random.default_rng(seed)
    #   objectives = []
    #   solutions = []
    #   for m in range(n_replications):
    #       obj, x_sol = saa_single_replication(n_scenarios, rng)
    #       objectives.append(obj)
    #       solutions.append(x_sol)
    #
    # STEP 2: Lower bound statistics
    #   The average SAA objective is a statistical lower bound (biased low).
    #   lower_bound = np.mean(objectives)
    #   lower_std = np.std(objectives, ddof=1)
    #   lower_ci = 1.96 * lower_std / np.sqrt(n_replications)  # 95% CI half-width
    #
    # STEP 3: Select the best candidate
    #   best_idx = np.argmin(objectives)
    #   best_x = solutions[best_idx]
    #
    # STEP 4: Evaluate the best candidate on a LARGE independent sample
    #   Generate n_eval_scenarios scenarios with a FRESH rng (independent of training).
    #   For each evaluation scenario, fix x = best_x and solve only the second stage.
    #   The average second-stage cost + first-stage cost = unbiased upper bound.
    #
    #   eval_rng = np.random.default_rng(seed + 999)  # independent seed
    #   eval_scenarios = sample_scenarios(n_eval_scenarios, eval_rng)
    #
    #   To evaluate x_fixed on a set of scenarios:
    #     - Build the extensive form with eval_scenarios
    #     - Fix model.x[c] to best_x[c] for each crop
    #     - Solve and record objective
    #   OR more efficiently: for each scenario, compute the second-stage cost
    #   analytically given x_fixed (since it's a simple LP per scenario).
    #
    #   For simplicity, you can build one model with all eval scenarios but
    #   fix the first-stage variables:
    #     eval_model = build extensive form with eval_scenarios
    #     for c in CROPS: eval_model.x[c].fix(best_x[c])
    #     solve eval_model
    #     upper_bound = pyo.value(eval_model.obj)
    #
    #   To get a CI on the upper bound, you can split eval_scenarios into
    #   batches and compute per-batch objectives, then take mean/std.
    #
    # STEP 5: Compute gap
    #   gap = upper_bound - lower_bound (should be >= 0)
    #
    # Return a dict with:
    #   "lower_bound", "lower_ci", "upper_bound", "gap",
    #   "best_x" (the best first-stage solution),
    #   "all_objectives" (list of per-replication objectives for plotting)
    raise NotImplementedError("TODO(human): implement SAA loop with confidence intervals")


# ============================================================================
# Provided: Convergence study
# ============================================================================

def saa_convergence_study(
    scenario_counts: list[int],
    n_replications: int = 20,
    seed: int = 42,
) -> dict:
    """Study how SAA converges as N increases.

    Runs SAA for each N in scenario_counts and collects statistics.

    Args:
        scenario_counts: List of N values to test.
        n_replications: Replications per N.
        seed: Random seed.

    Returns:
        Dictionary with per-N statistics for plotting.
    """
    results = {"N": [], "mean_obj": [], "std_obj": [], "ci_width": []}

    for n_scen in scenario_counts:
        print(f"  SAA with N={n_scen:>5} scenarios, M={n_replications} replications...", end=" ")
        t0 = time.time()
        saa_result = run_saa(n_scen, n_replications, n_eval_scenarios=2000, seed=seed)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")

        results["N"].append(n_scen)
        results["mean_obj"].append(saa_result["lower_bound"])
        results["std_obj"].append(np.std(saa_result["all_objectives"], ddof=1))
        results["ci_width"].append(saa_result["lower_ci"])

    return results


def plot_saa_convergence(conv_results: dict) -> None:
    """Plot SAA convergence: mean objective and CI width vs N."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ns = conv_results["N"]
    means = conv_results["mean_obj"]
    cis = conv_results["ci_width"]
    stds = conv_results["std_obj"]

    # Left: Mean objective with error bars
    ax = axes[0]
    ax.errorbar(ns, means, yerr=cis, fmt="o-", capsize=5, color="C0")
    ax.set_xlabel("Number of scenarios (N)", fontsize=12)
    ax.set_ylabel("Mean SAA objective", fontsize=12)
    ax.set_title("SAA Lower Bound vs Sample Size", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Right: Standard deviation / CI width
    ax = axes[1]
    ax.plot(ns, stds, "s-", color="C1", label="Std dev across replications")
    ax.plot(ns, cis, "^-", color="C2", label="95% CI half-width")
    ax.set_xlabel("Number of scenarios (N)", fontsize=12)
    ax.set_ylabel("Variability", fontsize=12)
    ax.set_title("SAA Variability vs Sample Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_saa_summary(saa_result: dict, n_scenarios: int, n_replications: int) -> None:
    """Print SAA results summary."""
    print(f"\n{'=' * 65}")
    print(f"  SAA Results (N={n_scenarios}, M={n_replications})")
    print(f"{'=' * 65}")
    print(f"  Lower bound (mean SAA obj):  {saa_result['lower_bound']:>12,.2f}")
    print(f"  Lower 95% CI half-width:     {saa_result['lower_ci']:>12,.2f}")
    print(f"  Upper bound (eval on large):  {saa_result['upper_bound']:>12,.2f}")
    print(f"  Optimality gap:              {saa_result['gap']:>12,.2f}")
    print(f"\n  Best first-stage solution:")
    for c in CROPS:
        print(f"    {c:>12}: {saa_result['best_x'][c]:>8.1f} acres")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 2: Sample Average Approximation (SAA)")
    print("=" * 65)

    # --- Single SAA run with moderate sample size ---
    print("\n--- SAA with N=50 scenarios, M=20 replications ---")
    saa_result = run_saa(n_scenarios=50, n_replications=20, seed=42)
    print_saa_summary(saa_result, 50, 20)

    # --- Convergence study ---
    print("\n--- SAA Convergence Study ---")
    print("  Testing N = [10, 25, 50, 100, 200, 500]")
    conv = saa_convergence_study(
        scenario_counts=[10, 25, 50, 100, 200, 500],
        n_replications=15,
        seed=42,
    )
    plot_saa_convergence(conv)

    # --- Effect of replications ---
    print("\n--- SAA with larger N=200, M=30 replications ---")
    saa_large = run_saa(n_scenarios=200, n_replications=30, seed=123)
    print_saa_summary(saa_large, 200, 30)

    print("\n[Phase 2 complete]")


if __name__ == "__main__":
    main()
