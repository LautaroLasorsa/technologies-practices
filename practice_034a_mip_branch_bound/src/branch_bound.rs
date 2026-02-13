// Phase 3: Full Branch & Bound with DFS
//
// This binary assembles the complete Branch & Bound solver. LP relaxation
// and branching are provided (implemented). You implement the search loop:
// node selection, solve, bound check, integrality check, prune or branch.

#[path = "common.rs"]
mod common;

use common::*;
use good_lp::{constraint, variable, Expression, ProblemVariables, Solution, SolverModel};

// ============================================================================
// Provided: LP relaxation solver (same as Phase 1).
// Copy your working implementation from Phase 1 here.
// ============================================================================

/// Solve the LP relaxation of a MIP, respecting the node's tightened bounds.
///
/// This version takes a BBNode so that branching-imposed bounds are used
/// instead of the original problem bounds.
fn solve_lp_relaxation_node(
    problem: &MIPProblem,
    node: &BBNode,
) -> Result<(Vec<f64>, f64), String> {
    // Build a modified problem with the node's bounds, then solve.
    let modified = MIPProblem {
        lower_bounds: node.lower_bounds.clone(),
        upper_bounds: node.upper_bounds.clone(),
        ..problem.clone()
    };
    solve_lp_relaxation(&modified)
}

/// Solve the LP relaxation of a MIP problem.
/// (Copy your implementation from Phase 1.)
fn solve_lp_relaxation(problem: &MIPProblem) -> Result<(Vec<f64>, f64), String> {
    // Re-use your Phase 1 implementation here.
    todo!("TODO(human): Copy your solve_lp_relaxation from Phase 1")
}

// ============================================================================
// Provided: Branching (same as Phase 2, adapted for BBNode).
// Copy your working implementation from Phase 2 here.
// ============================================================================

/// Create two child BBNodes by branching on variable `var_idx`.
/// Uses the parent node's bounds as the starting point.
fn branch_on_variable_node(
    parent: &BBNode,
    var_idx: usize,
    fractional_value: f64,
    next_id: &mut usize,
) -> (BBNode, BBNode) {
    let mut left = BBNode {
        lower_bounds: parent.lower_bounds.clone(),
        upper_bounds: parent.upper_bounds.clone(),
        lp_bound: f64::INFINITY,
        depth: parent.depth + 1,
        id: *next_id,
    };
    *next_id += 1;

    let mut right = BBNode {
        lower_bounds: parent.lower_bounds.clone(),
        upper_bounds: parent.upper_bounds.clone(),
        lp_bound: f64::INFINITY,
        depth: parent.depth + 1,
        id: *next_id,
    };
    *next_id += 1;

    // Left child: x_i <= floor(f)
    left.upper_bounds[var_idx] = fractional_value.floor();
    // Right child: x_i >= ceil(f)
    right.lower_bounds[var_idx] = fractional_value.ceil();

    (left, right)
}

// ============================================================================
// TODO(human): Variable selection
// ============================================================================

/// Select the branching variable: first fractional integer/binary variable.
///
/// # TODO(human): Implement this function
///
/// Scan through the solution values. For each variable:
///   - Skip if var_types[i] is Continuous
///   - Skip if the value is integer (within INT_TOLERANCE)
///   - Return Some(i) for the FIRST fractional integer/binary variable
///
/// Return None if all integer variables are integer-valued (solution is
/// integer-feasible — this shouldn't happen if the caller checks first,
/// but handle it gracefully).
///
/// "First fractional" is the simplest branching strategy. It picks the
/// variable with the lowest index that has a fractional value. This is
/// a baseline — not necessarily good, but easy to implement and debug.
/// You'll compare it with "most fractional" in Phase 4.
///
/// Implementation is straightforward:
///   for i in 0..solution.len() {
///       if var_types[i] is Integer or Binary, and !is_integer(solution[i], tolerance):
///           return Some(i)
///   }
///   return None
pub fn select_branching_variable_first_fractional(
    solution: &[f64],
    var_types: &[VarType],
    tolerance: f64,
) -> Option<usize> {
    todo!("TODO(human): Implement first-fractional variable selection")
}

// ============================================================================
// TODO(human): Full Branch & Bound loop
// ============================================================================

/// Solve a MIP using Branch & Bound with DFS node selection.
///
/// # TODO(human): Implement this function
///
/// Branch and Bound (DFS):
///
/// Initialize:
///   - Create a Vec<BBNode> as a stack (DFS uses push/pop = LIFO)
///   - Push the root node: make_root_node(problem)
///   - incumbent_solution: Option<Vec<f64>> = None
///   - incumbent_obj: f64 = -INFINITY for Maximize, +INFINITY for Minimize
///   - next_node_id: usize = 1 (root is 0)
///   - Counters: nodes_explored, pruned_by_bound, pruned_by_infeasibility, pruned_by_integrality
///
/// Determine comparison direction from problem.sense:
///   - Maximize: a new solution is better if obj > incumbent_obj
///               a node is pruned if lp_bound <= incumbent_obj
///   - Minimize: a new solution is better if obj < incumbent_obj
///               a node is pruned if lp_bound >= incumbent_obj
///
/// Loop while stack is not empty:
///   node = stack.pop()  (last element — DFS)
///   nodes_explored += 1
///
///   1. SOLVE LP relaxation of node:
///      Call solve_lp_relaxation_node(problem, &node)
///      If Err (infeasible):
///        pruned_by_infeasibility += 1
///        continue  (skip to next node)
///      Else: get (solution, lp_obj)
///
///   2. BOUND CHECK:
///      For Maximize: if lp_obj <= incumbent_obj → prune by bound, continue
///      For Minimize: if lp_obj >= incumbent_obj → prune by bound, continue
///      (The LP bound tells us the BEST possible integer solution in this
///       subtree. If that best can't beat the incumbent, skip the whole subtree.)
///
///   3. INTEGER FEASIBILITY CHECK:
///      If is_integer_feasible(&solution, &problem.var_types, INT_TOLERANCE):
///        This LP solution is also a valid MIP solution!
///        For Maximize: if lp_obj > incumbent_obj → update incumbent
///        For Minimize: if lp_obj < incumbent_obj → update incumbent
///        pruned_by_integrality += 1
///        continue  (no need to branch further — this node is a leaf)
///
///   4. BRANCH:
///      Select a fractional variable using select_branching_variable_first_fractional()
///      If None → skip (shouldn't happen — we just checked integrality)
///      Create two children via branch_on_variable_node()
///      Push both onto the stack (left first, then right — DFS explores right first)
///
/// After loop: build and return BBResult with incumbent and counters.
///
/// Key insight: pruning by bound is what makes B&B fast. Without it,
/// DFS would explore every possible integer assignment (2^n for n binary
/// variables). With a good incumbent found early, vast subtrees get pruned.
///
/// Tip: Print progress every N nodes (e.g., every 10 or 100) to see the
/// solver working. Print: node ID, depth, LP bound, incumbent objective.
pub fn branch_and_bound(problem: &MIPProblem) -> BBResult {
    todo!("TODO(human): Implement the complete Branch & Bound loop with DFS")
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Phase 3: Full Branch & Bound (DFS)                 ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // -----------------------------------------------------------------------
    // Problem 1: Small binary knapsack (3 variables)
    // -----------------------------------------------------------------------
    println!("=== Problem 1: Binary Knapsack (3 vars) ===\n");
    println!("  maximize  8*x1 + 5*x2 + 4*x3");
    println!("  s.t.      6*x1 + 4*x2 + 3*x3 <= 12");
    println!("            x_i in {{0, 1}}\n");

    let knapsack = knapsack_problem();

    // First show LP relaxation bound for reference
    let root = make_root_node(&knapsack);
    if let Ok((_, lp_obj)) = solve_lp_relaxation_node(&knapsack, &root) {
        println!("LP relaxation bound: {:.6}", lp_obj);
    }

    let result = branch_and_bound(&knapsack);
    print_bb_result(&result, &knapsack);

    // Verify: optimal should be x1=1, x2=1, x3=0 → obj=13
    if let Some(obj) = result.objective {
        println!("Expected optimal: 13.0 (x1=1, x2=1, x3=0)");
        println!("Got: {:.1}\n", obj);
    }

    // -----------------------------------------------------------------------
    // Problem 2: Set cover (4 variables, minimization)
    // -----------------------------------------------------------------------
    println!("=== Problem 2: Set Cover (4 vars, minimization) ===\n");
    println!("  minimize  3*x1 + 2*x2 + 4*x3 + 3*x4");
    println!("  s.t.      x1 + x3         >= 1  (cover element 1)");
    println!("            x1 + x2 + x4    >= 1  (cover element 2)");
    println!("            x2 + x3 + x4    >= 1  (cover element 3)");
    println!("            x_i in {{0, 1}}\n");

    let setcover = set_cover_problem();

    let root = make_root_node(&setcover);
    if let Ok((_, lp_obj)) = solve_lp_relaxation_node(&setcover, &root) {
        println!("LP relaxation bound: {:.6}", lp_obj);
    }

    let result = branch_and_bound(&setcover);
    print_bb_result(&result, &setcover);

    // -----------------------------------------------------------------------
    // Problem 3: Larger knapsack (6 variables)
    // -----------------------------------------------------------------------
    println!("=== Problem 3: Larger Knapsack (6 vars) ===\n");
    println!("  maximize  10*x1 + 9*x2 + 8*x3 + 7*x4 + 6*x5 + 5*x6");
    println!("  s.t.      6*x1 + 5*x2 + 5*x3 + 4*x4 + 3*x5 + 3*x6 <= 15");
    println!("            x_i in {{0, 1}}\n");

    let large = large_knapsack_problem();

    let root = make_root_node(&large);
    if let Ok((_, lp_obj)) = solve_lp_relaxation_node(&large, &root) {
        println!("LP relaxation bound: {:.6}", lp_obj);
    }

    let result = branch_and_bound(&large);
    print_bb_result(&result, &large);
}
