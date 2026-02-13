"""Phase 4: Multi-Stage Stochastic Program -- Inventory Management.

Extend from two-stage to three-stage: an inventory management problem where
demand is uncertain at each period. The decision-maker orders inventory at
the start of each period, observes demand, and carries excess to the next
period (or incurs backlog cost).

Key concepts:
  - Scenario tree (branching at each stage)
  - Non-anticipativity constraints (same decisions for indistinguishable scenarios)
  - Extensive form with explicit non-anticipativity
  - Rolling horizon comparison (myopic two-stage approach)

Pyomo patterns:
  - Nested scenario indexing
  - Non-anticipativity constraints linking scenario variables
  - Multi-period inventory balance
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


# ============================================================================
# Problem data -- 3-period inventory management
# ============================================================================

N_PERIODS = 3

# Cost parameters
ORDER_COST = 5.0        # $/unit ordered
HOLDING_COST = 1.0      # $/unit held per period
BACKLOG_COST = 10.0     # $/unit backordered per period (penalty for unmet demand)
SALVAGE_VALUE = 2.0     # $/unit remaining at end (negative cost)

# Order constraints
MAX_ORDER = 200         # max units orderable per period
INITIAL_INVENTORY = 50  # starting inventory

# Demand uncertainty: at each stage, demand can be Low, Medium, or High
DEMAND_OUTCOMES = {
    1: {"Low": 50, "Medium": 100, "High": 150},   # Period 1 demand
    2: {"Low": 60, "Medium": 110, "High": 160},   # Period 2 demand
    3: {"Low": 40, "Medium": 90, "High": 140},    # Period 3 demand
}

DEMAND_PROBABILITIES = {
    1: {"Low": 0.25, "Medium": 0.50, "High": 0.25},
    2: {"Low": 0.30, "Medium": 0.40, "High": 0.30},
    3: {"Low": 0.20, "Medium": 0.50, "High": 0.30},
}


# ============================================================================
# TODO(human): Build scenario tree
# ============================================================================

def build_scenario_tree(
    demand_outcomes: dict,
    demand_probabilities: dict,
) -> dict:
    """Build a scenario tree from per-period demand outcomes.

    The tree has branching at each stage. For 3 periods with 3 outcomes each,
    there are 3^3 = 27 leaf scenarios. Each scenario is a path from root to leaf.

    The tree structure encodes:
      - Which scenarios share the same history at each stage
      - Path probabilities (product of per-stage probabilities)

    Args:
        demand_outcomes: {period: {outcome_name: demand_value}}.
        demand_probabilities: {period: {outcome_name: probability}}.

    Returns:
        Dictionary with:
          "scenarios": list of scenario dicts, each with:
              "name": tuple of outcome names, e.g. ("Low", "Medium", "High")
              "demands": list of demand values per period
              "probability": path probability (product of per-stage probs)
          "n_scenarios": total number of leaf scenarios
          "periods": list of period indices [1, 2, 3]
          "nodes": dict mapping (period, history_prefix) -> list of scenario indices
                   that share that history. Used for non-anticipativity.
    """
    # TODO(human): Construct the scenario tree
    #
    # STEP 1: Generate all leaf scenarios using itertools.product
    #   The scenarios are the Cartesian product of outcomes at each stage.
    #   For 3 periods with outcomes ["Low", "Medium", "High"] each:
    #     outcomes_per_period = [
    #       list(demand_outcomes[1].keys()),  # ["Low", "Medium", "High"]
    #       list(demand_outcomes[2].keys()),
    #       list(demand_outcomes[3].keys()),
    #     ]
    #     all_paths = list(itertools.product(*outcomes_per_period))
    #     => [("Low","Low","Low"), ("Low","Low","Medium"), ..., ("High","High","High")]
    #     This gives 3 * 3 * 3 = 27 scenarios.
    #
    # STEP 2: For each path, compute demands and probability
    #   For path = ("Low", "Medium", "High"):
    #     demands = [demand_outcomes[1]["Low"], demand_outcomes[2]["Medium"], demand_outcomes[3]["High"]]
    #     probability = demand_probabilities[1]["Low"] * demand_probabilities[2]["Medium"] * demand_probabilities[3]["High"]
    #
    # STEP 3: Build non-anticipativity node mapping
    #   At stage t, scenarios that share the same history (outcomes for
    #   periods 1..t-1) must make the SAME ordering decision.
    #
    #   nodes = {}
    #   For each period t in [1, 2, 3]:
    #     For each scenario s with path (o_1, o_2, o_3):
    #       history = path[:t-1]  (everything before period t)
    #       => At period 1: history = () (empty -- all share same info)
    #       => At period 2: history = (o_1,) (only period 1 outcome known)
    #       => At period 3: history = (o_1, o_2) (periods 1-2 known)
    #
    #       key = (t, history)
    #       nodes[key] = list of scenario indices with this history
    #
    #   Example: nodes[(1, ())] = [0, 1, ..., 26]  (ALL scenarios -- period 1 has no info)
    #            nodes[(2, ("Low",))] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  (scenarios starting with Low)
    #
    # This node mapping is used to enforce non-anticipativity: for each
    # node (t, history), all scenarios in that node must have the same
    # order quantity at period t.
    #
    # Return the dict with "scenarios", "n_scenarios", "periods", "nodes".
    raise NotImplementedError("TODO(human): build scenario tree with paths, probabilities, and node mapping")


# ============================================================================
# TODO(human): Build multi-stage extensive form with non-anticipativity
# ============================================================================

def build_multi_stage_model(
    scenario_tree: dict,
    initial_inventory: float = INITIAL_INVENTORY,
) -> pyo.ConcreteModel:
    """Build the extensive form of the 3-stage inventory problem.

    Variables (per scenario per period):
      - order[s, t]: units ordered at start of period t in scenario s
      - inventory[s, t]: units on hand at END of period t (can be negative = backlog)
      - holding[s, t]: max(0, inventory) at end of period t
      - backlog[s, t]: max(0, -inventory) at end of period t

    Constraints:
      - Inventory balance: inv[s,t] = inv[s,t-1] + order[s,t] - demand[s,t]
      - Holding/backlog decomposition: holding[s,t] - backlog[s,t] = inv[s,t]
      - Order bounds: 0 <= order[s,t] <= MAX_ORDER
      - Non-anticipativity: for all (s1, s2) sharing the same node at stage t,
                            order[s1, t] = order[s2, t]

    Objective: minimize expected total cost (ordering + holding + backlog - salvage).

    Args:
        scenario_tree: Output of build_scenario_tree().
        initial_inventory: Starting inventory at period 0.

    Returns:
        Solved Pyomo ConcreteModel.
    """
    # TODO(human): Build and solve the multi-stage extensive form
    #
    # This is more complex than the two-stage model because:
    #   1. Variables are indexed by BOTH scenario AND period
    #   2. Non-anticipativity constraints explicitly link scenarios
    #   3. Inventory balance chains across periods
    #
    # STEP 1: Create model and index sets
    #   model = pyo.ConcreteModel("Inventory_MultiStage")
    #   S = scenario_tree["n_scenarios"]
    #   T = len(scenario_tree["periods"])
    #   model.S = pyo.RangeSet(0, S - 1)   # scenario indices
    #   model.T = pyo.RangeSet(1, T)        # period indices 1..T
    #
    # STEP 2: Variables
    #   model.order = pyo.Var(model.S, model.T, bounds=(0, MAX_ORDER))
    #   model.inv = pyo.Var(model.S, model.T, within=pyo.Reals)  # can be negative (backlog)
    #   model.holding = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    #   model.backlog = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    #
    # STEP 3: Inventory balance constraints
    #   For each scenario s and period t:
    #     if t == 1:
    #       inv[s,1] = initial_inventory + order[s,1] - demand[s,1]
    #     else:
    #       inv[s,t] = inv[s,t-1] + order[s,t] - demand[s,t]
    #
    #   Where demand[s,t] comes from scenario_tree["scenarios"][s]["demands"][t-1]
    #
    # STEP 4: Holding/backlog decomposition
    #   For each s, t:
    #     holding[s,t] - backlog[s,t] == inv[s,t]
    #   Combined with holding >= 0 and backlog >= 0, this implements:
    #     holding = max(0, inv), backlog = max(0, -inv)
    #   This is a standard LP trick to linearize max(0, x).
    #
    # STEP 5: NON-ANTICIPATIVITY CONSTRAINTS (the key concept)
    #   For each node (t, history) in scenario_tree["nodes"]:
    #     Let scenario_indices = list of scenarios in this node
    #     Pick the first scenario as reference: s_ref = scenario_indices[0]
    #     For each other scenario s in scenario_indices[1:]:
    #       model.add_component(
    #           f"na_{t}_{s_ref}_{s}",
    #           pyo.Constraint(expr=model.order[s, t] == model.order[s_ref, t])
    #       )
    #
    #   This ensures that scenarios sharing the same information at stage t
    #   make the same ordering decision. Without these constraints, the model
    #   would "cheat" by using future information (clairvoyance).
    #
    #   Non-anticipativity is THE defining concept of multi-stage stochastic
    #   programming. It encodes: "you can only use information available at
    #   the time of the decision."
    #
    # STEP 6: Objective
    #   For each scenario s with probability p_s:
    #     cost_s = Σ_t (ORDER_COST * order[s,t]
    #                    + HOLDING_COST * holding[s,t]
    #                    + BACKLOG_COST * backlog[s,t])
    #              - SALVAGE_VALUE * holding[s, T]  (salvage leftover at end)
    #
    #   model.obj = Objective(expr=Σ_s p_s * cost_s, sense=minimize)
    #
    # STEP 7: Solve
    #   solver = pyo.SolverFactory('highs')
    #   result = solver.solve(model)
    #   assert result.solver.termination_condition == TerminationCondition.optimal
    #
    # Return the solved model.
    raise NotImplementedError("TODO(human): build multi-stage model with non-anticipativity constraints")


# ============================================================================
# TODO(human): Rolling horizon (myopic) approach
# ============================================================================

def rolling_horizon(
    scenario_tree: dict,
    initial_inventory: float = INITIAL_INVENTORY,
    seed: int = 42,
) -> dict:
    """Simulate the rolling horizon (myopic two-stage) approach.

    At each period:
      1. Given current inventory, solve a two-stage problem:
         - First stage: order decision for THIS period
         - Second stage: recourse for the REMAINING periods (simplified)
      2. Implement the first-stage order decision
      3. Observe the realized demand (simulate one realization)
      4. Update inventory and move to next period

    This is repeated for each demand realization to get the expected cost.
    We average over all 27 leaf scenarios.

    Args:
        scenario_tree: Output of build_scenario_tree().
        initial_inventory: Starting inventory.
        seed: Random seed for selecting scenarios during simulation.

    Returns:
        Dictionary with rolling horizon results:
          "expected_cost": average cost across all scenarios
          "per_scenario_costs": list of costs for each scenario
          "decisions": dict mapping scenario index -> list of order quantities
    """
    # TODO(human): Implement rolling horizon simulation
    #
    # The rolling horizon approach is a MYOPIC alternative to the full
    # multi-stage model. Instead of optimizing all periods jointly with
    # a scenario tree, it solves a sequence of simpler problems:
    #
    # For each scenario s (we evaluate on ALL 27 scenarios to get expected cost):
    #   current_inv = initial_inventory
    #   total_cost = 0.0
    #   orders = []
    #
    #   For each period t = 1, 2, 3:
    #     # Solve a single-period (or two-stage) problem:
    #     # Given current_inv, decide how much to order for period t.
    #     #
    #     # Simple heuristic: solve a newsvendor-like problem.
    #     # Use expected demand for remaining periods.
    #     # Order to bring expected inventory to a target level.
    #     #
    #     # A reasonable approach for this practice:
    #     #   expected_demand_t = sum(prob * demand for each outcome at period t)
    #     #   target_inv = expected_demand_t  (order enough to meet expected demand)
    #     #   order = max(0, min(MAX_ORDER, target_inv - current_inv))
    #     #
    #     # Or more sophisticated: build a Pyomo model for the remaining
    #     # periods with expected demands and solve it. The first-stage
    #     # decision from this model is the order for period t.
    #     #
    #     # For simplicity, use the expected-demand heuristic:
    #     expected_demand = sum(
    #         DEMAND_PROBABILITIES[t][outcome] * DEMAND_OUTCOMES[t][outcome]
    #         for outcome in DEMAND_OUTCOMES[t]
    #     )
    #     order = max(0.0, min(MAX_ORDER, expected_demand - current_inv))
    #     orders.append(order)
    #
    #     # Compute costs for this period with ACTUAL demand from scenario s
    #     actual_demand = scenario_tree["scenarios"][s]["demands"][t - 1]
    #     current_inv = current_inv + order - actual_demand
    #     holding = max(0.0, current_inv)
    #     backlog = max(0.0, -current_inv)
    #     total_cost += ORDER_COST * order + HOLDING_COST * holding + BACKLOG_COST * backlog
    #
    #   # Salvage at the end
    #   total_cost -= SALVAGE_VALUE * max(0.0, current_inv)
    #   Store per-scenario cost and decisions
    #
    # expected_cost = sum(scen["probability"] * cost_s for each scenario)
    #
    # WHY ROLLING HORIZON IS SUBOPTIMAL:
    #   The rolling horizon cannot hedge against future uncertainty because
    #   it uses expected values instead of modeling the full scenario tree.
    #   It cannot "build buffer stock" in anticipation of high-demand scenarios.
    #   The gap between rolling horizon and multi-stage optimal quantifies
    #   the value of look-ahead optimization.
    #
    # Return {"expected_cost": ..., "per_scenario_costs": [...], "decisions": {...}}
    raise NotImplementedError("TODO(human): implement rolling horizon simulation")


# ============================================================================
# Helpers (provided)
# ============================================================================

def print_scenario_tree(tree: dict) -> None:
    """Print scenario tree structure summary."""
    print(f"\n{'=' * 65}")
    print(f"  Scenario Tree Summary")
    print(f"{'=' * 65}")
    print(f"  Periods: {tree['periods']}")
    print(f"  Total leaf scenarios: {tree['n_scenarios']}")
    print(f"  Probability sum: {sum(s['probability'] for s in tree['scenarios']):.6f}")

    print(f"\n  Non-anticipativity nodes:")
    for key, indices in sorted(tree["nodes"].items()):
        period, history = key
        hist_str = "root" if len(history) == 0 else " -> ".join(history)
        print(f"    Period {period}, history=({hist_str}): {len(indices)} scenarios")

    print(f"\n  Sample scenarios:")
    for i in [0, len(tree["scenarios"]) // 2, -1]:
        s = tree["scenarios"][i]
        print(f"    {s['name']}: demands={s['demands']}, p={s['probability']:.4f}")


def print_multi_stage_solution(model: pyo.ConcreteModel, scenario_tree: dict) -> None:
    """Print the multi-stage model solution."""
    print(f"\n{'=' * 65}")
    print(f"  Multi-Stage Model Solution")
    print(f"{'=' * 65}")
    print(f"  Objective (expected cost): {pyo.value(model.obj):>12,.2f}")

    # Print decisions at each non-anticipativity node
    print(f"\n  Optimal ordering decisions by node:")
    for key in sorted(scenario_tree["nodes"].keys()):
        period, history = key
        indices = scenario_tree["nodes"][key]
        s_ref = indices[0]
        order_val = pyo.value(model.order[s_ref, period])
        hist_str = "root" if len(history) == 0 else " -> ".join(history)
        print(f"    Period {period} ({hist_str}): order = {order_val:>8.1f} units")


def print_comparison(multi_stage_obj: float, rolling_obj: float) -> None:
    """Print comparison of multi-stage vs rolling horizon."""
    print(f"\n{'=' * 65}")
    print(f"  Multi-Stage vs Rolling Horizon Comparison")
    print(f"{'=' * 65}")
    print(f"  Multi-stage optimal (expected cost): {multi_stage_obj:>12,.2f}")
    print(f"  Rolling horizon (expected cost):     {rolling_obj:>12,.2f}")
    gap = rolling_obj - multi_stage_obj
    gap_pct = 100 * gap / abs(multi_stage_obj) if multi_stage_obj != 0 else 0
    print(f"  Gap (rolling - optimal):             {gap:>12,.2f} ({gap_pct:.1f}%)")
    print(f"\n  Interpretation:")
    if gap > 0:
        print(f"    The multi-stage model saves ${gap:,.2f} ({gap_pct:.1f}%) over the myopic approach.")
        print(f"    This is the value of look-ahead planning under uncertainty.")
    else:
        print(f"    The rolling horizon performs comparably -- look-ahead adds little value here.")


def plot_decision_tree(scenario_tree: dict, model: pyo.ConcreteModel) -> None:
    """Visualize the decision tree with order quantities at each node."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect unique nodes and their positions
    nodes_by_period = {}
    for key in sorted(scenario_tree["nodes"].keys()):
        period, history = key
        if period not in nodes_by_period:
            nodes_by_period[period] = []
        indices = scenario_tree["nodes"][key]
        s_ref = indices[0]
        order_val = pyo.value(model.order[s_ref, period])
        nodes_by_period[period].append((history, order_val, len(indices)))

    # Plot nodes
    x_positions = {1: 0.15, 2: 0.5, 3: 0.85}
    for period, nodes in nodes_by_period.items():
        x = x_positions[period]
        n_nodes = len(nodes)
        for idx, (history, order_val, n_scen) in enumerate(nodes):
            y = (idx + 0.5) / n_nodes
            hist_str = "Start" if len(history) == 0 else "\n".join(history)
            ax.annotate(
                f"Order: {order_val:.0f}\n({hist_str})",
                (x, y),
                fontsize=8,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=f"C{period-1}", alpha=0.3),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_xticks([0.15, 0.5, 0.85])
    ax.set_xticklabels(["Period 1\n(decide before\nany demand)", "Period 2\n(after observing\nperiod 1 demand)", "Period 3\n(after observing\nperiods 1-2 demand)"])
    ax.set_title("Multi-Stage Decision Tree: Order Quantities by Node", fontsize=13)
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 4: Multi-Stage Stochastic Program -- Inventory Management")
    print("=" * 65)

    # --- Build scenario tree ---
    print("\n--- Building Scenario Tree ---")
    tree = build_scenario_tree(DEMAND_OUTCOMES, DEMAND_PROBABILITIES)
    print_scenario_tree(tree)

    # --- Solve multi-stage model ---
    print("\n--- Solving Multi-Stage Extensive Form ---")
    model = build_multi_stage_model(tree)
    print_multi_stage_solution(model, tree)
    multi_stage_obj = pyo.value(model.obj)

    # --- Visualize decision tree ---
    plot_decision_tree(tree, model)

    # --- Rolling horizon ---
    print("\n--- Rolling Horizon (Myopic) Approach ---")
    rh_result = rolling_horizon(tree)
    rolling_obj = rh_result["expected_cost"]
    print(f"  Rolling horizon expected cost: {rolling_obj:>12,.2f}")

    # --- Comparison ---
    print_comparison(multi_stage_obj, rolling_obj)

    # --- Sensitivity to demand variability ---
    print(f"\n{'=' * 65}")
    print(f"  Sensitivity: Effect of Demand Spread")
    print(f"{'=' * 65}")
    for spread_factor in [0.5, 1.0, 1.5, 2.0]:
        # Scale demand outcomes around their mean
        scaled_outcomes = {}
        for t in DEMAND_OUTCOMES:
            mean_d = sum(
                DEMAND_PROBABILITIES[t][o] * DEMAND_OUTCOMES[t][o]
                for o in DEMAND_OUTCOMES[t]
            )
            scaled_outcomes[t] = {
                o: max(0, mean_d + spread_factor * (DEMAND_OUTCOMES[t][o] - mean_d))
                for o in DEMAND_OUTCOMES[t]
            }

        tree_scaled = build_scenario_tree(scaled_outcomes, DEMAND_PROBABILITIES)
        model_scaled = build_multi_stage_model(tree_scaled)
        rh_scaled = rolling_horizon(tree_scaled)

        ms_obj = pyo.value(model_scaled.obj)
        rh_obj = rh_scaled["expected_cost"]
        gap_pct = 100 * (rh_obj - ms_obj) / abs(ms_obj) if ms_obj != 0 else 0

        print(f"  Spread x{spread_factor:.1f}: multi-stage={ms_obj:>10,.1f}, "
              f"rolling={rh_obj:>10,.1f}, gap={gap_pct:.1f}%")

    print("\n[Phase 4 complete]")


if __name__ == "__main__":
    main()
