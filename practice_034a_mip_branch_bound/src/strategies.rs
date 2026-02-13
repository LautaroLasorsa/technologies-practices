// Phase 4: Branching Strategies
//
// This binary compares different B&B strategies on the same problems:
//   - Node selection: DFS (stack) vs Best-First (BinaryHeap by LP bound)
//   - Variable selection: First-Fractional vs Most-Fractional
//
// The goal is to observe how strategy choices affect: nodes explored,
// nodes pruned, and (implicitly) memory usage.

#[path = "common.rs"]
mod common;

use common::*;
use good_lp::{constraint, variable, Expression, ProblemVariables, Solution, SolverModel};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ============================================================================
// Provided: LP relaxation solvers (same pattern as branch_bound.rs).
// Copy your Phase 1 implementation into solve_lp_relaxation().
// ============================================================================

/// Solve the LP relaxation of a MIP problem.
/// (Copy your implementation from Phase 1.)
fn solve_lp_relaxation(problem: &MIPProblem) -> Result<(Vec<f64>, f64), String> {
    // Re-use your Phase 1 implementation here.
    todo!("TODO(human): Copy your solve_lp_relaxation from Phase 1")
}

/// Solve the LP relaxation of a MIP, respecting the node's tightened bounds.
fn solve_lp_relaxation_node(
    problem: &MIPProblem,
    node: &BBNode,
) -> Result<(Vec<f64>, f64), String> {
    let modified = MIPProblem {
        lower_bounds: node.lower_bounds.clone(),
        upper_bounds: node.upper_bounds.clone(),
        ..problem.clone()
    };
    solve_lp_relaxation(&modified)
}

// ============================================================================
// Provided: Branching mechanics (same as branch_bound.rs).
// ============================================================================

/// Create two child BBNodes by branching on variable `var_idx`.
fn branch_on_variable(
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
// Provided: First-Fractional variable selection (baseline from Phase 3).
// ============================================================================

/// Select the branching variable: first fractional integer/binary variable.
/// (Same logic as Phase 3 — baseline for comparison.)
fn select_branching_variable_first_fractional(
    solution: &[f64],
    var_types: &[VarType],
) -> Option<usize> {
    for (i, (&val, &vtype)) in solution.iter().zip(var_types.iter()).enumerate() {
        match vtype {
            VarType::Continuous => continue,
            VarType::Integer | VarType::Binary => {
                if !is_integer(val, INT_TOLERANCE) {
                    return Some(i);
                }
            }
        }
    }
    None
}

// ============================================================================
// TODO(human) #1: Most-Fractional variable selection
// ============================================================================

/// Select the branching variable: most fractional integer/binary variable.
///
/// # TODO(human): Implement this function
///
/// Most-Fractional Variable Selection
///
/// Among all integer variables with fractional LP values, pick the one
/// whose fractional part is closest to 0.5 (i.e., "most fractional").
///
/// For each integer variable x_i with LP value v:
///   fractionality = 0.5 - |v - v.round()|
///     (equivalently: min(v - v.floor(), v.ceil() - v))
///
/// Select the variable with MAXIMUM fractionality (closest to 0.5).
///
/// Intuition: branching on x_i = 0.99 barely changes the LP (floor=0, ceil=1,
/// but the LP "wants" x_i ≈ 1 anyway). Branching on x_i = 0.5 creates the
/// most balanced split — both children's LPs change significantly.
///
/// Skip variables that are already integer (|v - v.round()| < 1e-6).
/// Return None if all integer variables are integral.
///
/// Implementation:
///   let mut best_idx = None;
///   let mut best_frac = -1.0;
///   for (i, (&val, &vtype)) in solution.iter().zip(var_types).enumerate() {
///       if vtype is not Integer/Binary: continue
///       let frac_part = val - val.floor();
///       let fractionality = 0.5 - (frac_part - 0.5).abs();  // max at 0.5
///       if fractionality > best_frac + 1e-9 {
///           best_frac = fractionality;
///           best_idx = Some(i);
///       }
///   }
///   best_idx  (None if all integer)
pub fn select_branching_variable_most_fractional(
    solution: &[f64],
    var_types: &[VarType],
) -> Option<usize> {
    todo!("TODO(human): Implement most-fractional variable selection")
}

// ============================================================================
// Provided: DFS Branch & Bound (from Phase 3, for comparison).
// ============================================================================

/// Solve a MIP using Branch & Bound with DFS and configurable variable selection.
///
/// `var_selector` chooses which fractional variable to branch on.
fn branch_and_bound_dfs(
    problem: &MIPProblem,
    var_selector: fn(&[f64], &[VarType]) -> Option<usize>,
    label: &str,
) -> BBResult {
    println!("  Running DFS B&B ({})...", label);

    let is_max = problem.sense == Sense::Maximize;
    let mut incumbent_solution: Option<Vec<f64>> = None;
    let mut incumbent_obj = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
    let mut next_id: usize = 1;

    let mut nodes_explored: usize = 0;
    let mut pruned_by_bound: usize = 0;
    let mut pruned_by_infeasibility: usize = 0;
    let mut pruned_by_integrality: usize = 0;

    let root = make_root_node(problem);
    let mut stack: Vec<BBNode> = vec![root];

    while let Some(node) = stack.pop() {
        nodes_explored += 1;

        // Solve LP relaxation
        let (solution, lp_obj) = match solve_lp_relaxation_node(problem, &node) {
            Ok(result) => result,
            Err(_) => {
                pruned_by_infeasibility += 1;
                continue;
            }
        };

        // Bound check
        let prune_by_bound = if is_max {
            lp_obj <= incumbent_obj + 1e-9
        } else {
            lp_obj >= incumbent_obj - 1e-9
        };
        if prune_by_bound {
            pruned_by_bound += 1;
            continue;
        }

        // Integer feasibility check
        if is_integer_feasible(&solution, &problem.var_types, INT_TOLERANCE) {
            let is_better = if is_max {
                lp_obj > incumbent_obj + 1e-9
            } else {
                lp_obj < incumbent_obj - 1e-9
            };
            if is_better {
                incumbent_obj = lp_obj;
                incumbent_solution = Some(solution);
            }
            pruned_by_integrality += 1;
            continue;
        }

        // Branch
        if let Some(var_idx) = var_selector(&solution, &problem.var_types) {
            let (left, right) = branch_on_variable(&node, var_idx, solution[var_idx], &mut next_id);
            stack.push(left);
            stack.push(right);
        }
    }

    BBResult {
        solution: incumbent_solution,
        objective: if incumbent_obj.is_finite() {
            Some(incumbent_obj)
        } else {
            None
        },
        nodes_explored,
        pruned_by_bound,
        pruned_by_infeasibility,
        pruned_by_integrality,
    }
}

// ============================================================================
// NodeWrapper for BinaryHeap (best-first search).
// ============================================================================

/// Wrapper around BBNode for use in BinaryHeap.
///
/// BinaryHeap is a max-heap. For maximization, we want the node with the
/// highest LP bound on top. For minimization, we want the lowest LP bound
/// on top (negate or reverse the ordering).
///
/// This wrapper stores the LP bound alongside the node and implements
/// Ord by LP bound (descending for max, ascending for min is handled
/// by the caller negating the bound for minimization problems).
struct NodeWrapper {
    node: BBNode,
    lp_bound: f64,
}

impl PartialEq for NodeWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.lp_bound == other.lp_bound
    }
}

impl Eq for NodeWrapper {}

impl PartialOrd for NodeWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by lp_bound. For maximization, highest bound first.
        // Use ordered comparison; NaN sorts last.
        self.lp_bound
            .partial_cmp(&other.lp_bound)
            .unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// TODO(human) #2: Best-First Branch & Bound
// ============================================================================

/// Solve a MIP using Best-First (Best-Bound) Branch & Bound.
///
/// `var_selector` chooses which fractional variable to branch on.
///
/// # TODO(human): Implement this function
///
/// Best-First (Best-Bound) Branch & Bound
///
/// Instead of DFS (stack, LIFO), use a priority queue (BinaryHeap)
/// sorted by LP bound (best bound explored first).
///
/// For MAXIMIZATION: explore the node with the HIGHEST LP bound next.
/// This proves optimality fastest because it always works on the most
/// promising part of the tree.
///
/// Algorithm (same as DFS B&B from Phase 3, but with BinaryHeap):
///   let mut heap = BinaryHeap::new();
///   heap.push(NodeWrapper { node: root, lp_bound: f64::INFINITY });
///
///   while let Some(NodeWrapper { node, lp_bound }) = heap.pop() {
///       // Bound check: if this node's bound <= incumbent, skip
///       // (the heap may contain stale nodes from before incumbent improved)
///       if lp_bound <= best_obj + 1e-9 { nodes_pruned += 1; continue; }
///
///       // Solve LP relaxation
///       // ... same logic as DFS: check feasibility, bound, integrality
///
///       // Branch: create children, solve their LPs to get bounds,
///       // push NodeWrapper { node: child, lp_bound: child_lp_obj } onto heap
///       // Only push if child's bound > best_obj (pre-prune)
///   }
///
/// Key differences from DFS:
///   - Uses BinaryHeap (max-heap) instead of Vec as stack
///   - NodeWrapper implements Ord by lp_bound
///   - May use more memory (entire frontier stored)
///   - Finds optimal proof faster but feasible solutions slower
///
/// For MINIMIZATION: we need a min-heap. Since BinaryHeap is a max-heap,
/// negate the LP bound when pushing (push -lp_obj) and negate when comparing.
/// Alternatively, reverse the NodeWrapper ordering. The simplest approach:
/// for minimization, push NodeWrapper with lp_bound = -lp_obj, and do the
/// bound check with the negated value. Or handle maximization/minimization
/// separately in the comparison.
///
/// Suggested approach (simpler): always store the actual lp_bound, and for
/// minimization, reverse the Ord in NodeWrapper. But since NodeWrapper uses
/// max-heap ordering, for minimization you can negate: push with
/// lp_bound = -lp_obj, and when checking bounds, compare -lp_bound against
/// -incumbent_obj. OR just keep the same code as DFS but swap the stack
/// for the heap, adjusting comparisons for sense.
///
/// Cleanest approach: keep the same pruning logic as DFS (it already handles
/// both senses). For the heap ordering, store the "priority" as:
///   - Maximize: priority = lp_obj (higher is better, max-heap works)
///   - Minimize: priority = -lp_obj (lower obj → higher priority in max-heap)
///
/// Compare: nodes explored, nodes pruned, memory (heap size) vs DFS
fn branch_and_bound_best_first(
    problem: &MIPProblem,
    var_selector: fn(&[f64], &[VarType]) -> Option<usize>,
    label: &str,
) -> BBResult {
    println!("  Running Best-First B&B ({})...", label);
    todo!("TODO(human): Implement best-first (best-bound) branch and bound")
}

// ============================================================================
// Test problems
// ============================================================================

/// Larger knapsack with 8 binary variables — enough to show strategy differences.
///   maximize  12*x1 + 11*x2 + 10*x3 + 9*x4 + 8*x5 + 7*x6 + 6*x7 + 5*x8
///   s.t.      7*x1 + 6*x2 + 6*x3 + 5*x4 + 5*x5 + 4*x6 + 3*x7 + 3*x8 <= 20
///             x_i in {0, 1}
fn large_strategy_problem() -> MIPProblem {
    MIPProblem {
        objective: vec![12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        sense: Sense::Maximize,
        constraints: vec![vec![7.0, 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 3.0]],
        rhs: vec![20.0],
        constraint_types: vec![ConstraintType::Leq],
        var_types: vec![VarType::Binary; 8],
        lower_bounds: vec![0.0; 8],
        upper_bounds: vec![1.0; 8],
        var_names: (1..=8).map(|i| format!("x{}", i)).collect(),
    }
}

/// Multi-constraint knapsack (10 binary variables, 3 constraints).
/// Designed to produce many fractional variables and differentiate strategies.
///   maximize  15*x1 + 14*x2 + 13*x3 + 12*x4 + 11*x5 + 10*x6 + 9*x7 + 8*x8 + 7*x9 + 6*x10
///   s.t.      8*x1 + 7*x2 + 6*x3 + 5*x4 + 5*x5 + 4*x6 + 4*x7 + 3*x8 + 2*x9 + 2*x10 <= 25
///             3*x1 + 4*x2 + 5*x3 + 3*x4 + 2*x5 + 4*x6 + 3*x7 + 5*x8 + 4*x9 + 3*x10 <= 18
///             2*x1 + 3*x2 + 2*x3 + 4*x4 + 5*x5 + 2*x6 + 3*x7 + 2*x8 + 4*x9 + 5*x10 <= 16
///             x_i in {0, 1}
fn multi_constraint_problem() -> MIPProblem {
    MIPProblem {
        objective: vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0],
        sense: Sense::Maximize,
        constraints: vec![
            vec![8.0, 7.0, 6.0, 5.0, 5.0, 4.0, 4.0, 3.0, 2.0, 2.0],
            vec![3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 3.0],
            vec![2.0, 3.0, 2.0, 4.0, 5.0, 2.0, 3.0, 2.0, 4.0, 5.0],
        ],
        rhs: vec![25.0, 18.0, 16.0],
        constraint_types: vec![ConstraintType::Leq, ConstraintType::Leq, ConstraintType::Leq],
        var_types: vec![VarType::Binary; 10],
        lower_bounds: vec![0.0; 10],
        upper_bounds: vec![1.0; 10],
        var_names: (1..=10).map(|i| format!("x{}", i)).collect(),
    }
}

// ============================================================================
// Main: run all strategy combinations and compare
// ============================================================================

/// Print a comparison table row.
fn print_result_row(label: &str, result: &BBResult) {
    let obj_str = match result.objective {
        Some(obj) => format!("{:.1}", obj),
        None => "INFEASIBLE".to_string(),
    };
    let total_pruned =
        result.pruned_by_bound + result.pruned_by_infeasibility + result.pruned_by_integrality;
    println!(
        "  {:40} | {:>6} | {:>6} | {:>6} | {:>6} | {:>10}",
        label,
        result.nodes_explored,
        result.pruned_by_bound,
        result.pruned_by_infeasibility,
        total_pruned,
        obj_str,
    );
}

fn run_comparison(problem: &MIPProblem, problem_name: &str) {
    println!("=== {} ===\n", problem_name);

    // Show LP relaxation bound
    let root = make_root_node(problem);
    if let Ok((_, lp_obj)) = solve_lp_relaxation_node(problem, &root) {
        println!("  LP relaxation bound: {:.6}\n", lp_obj);
    }

    // Run all four combinations
    let dfs_first = branch_and_bound_dfs(
        problem,
        select_branching_variable_first_fractional,
        "first-fractional",
    );
    let dfs_most = branch_and_bound_dfs(
        problem,
        select_branching_variable_most_fractional,
        "most-fractional",
    );
    let bf_first = branch_and_bound_best_first(
        problem,
        select_branching_variable_first_fractional,
        "first-fractional",
    );
    let bf_most = branch_and_bound_best_first(
        problem,
        select_branching_variable_most_fractional,
        "most-fractional",
    );

    // Print comparison table
    println!();
    println!(
        "  {:40} | {:>6} | {:>6} | {:>6} | {:>6} | {:>10}",
        "Strategy", "Nodes", "Bound", "Infeas", "Total", "Objective"
    );
    println!("  {:-<40}-+-{:-<6}-+-{:-<6}-+-{:-<6}-+-{:-<6}-+-{:-<10}", "", "", "", "", "", "");
    print_result_row("DFS + First-Fractional", &dfs_first);
    print_result_row("DFS + Most-Fractional", &dfs_most);
    print_result_row("Best-First + First-Fractional", &bf_first);
    print_result_row("Best-First + Most-Fractional", &bf_most);
    println!();

    // Print solutions for verification
    if let (Some(sol), Some(obj)) = (&dfs_first.solution, dfs_first.objective) {
        println!("  Optimal objective: {:.1}", obj);
        let active: Vec<String> = sol
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.5)
            .map(|(i, _)| problem.var_names[i].clone())
            .collect();
        println!("  Active variables: {}", active.join(", "));
    }
    println!();
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Phase 4: Branching Strategy Comparison              ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Problem 1: Small knapsack from earlier phases (baseline, quick)
    println!("────────────────────────────────────────────────────────");
    run_comparison(&knapsack_problem(), "Problem 1: Binary Knapsack (3 vars)");

    // Problem 2: Medium knapsack (6 vars)
    println!("────────────────────────────────────────────────────────");
    run_comparison(&large_knapsack_problem(), "Problem 2: Knapsack (6 vars)");

    // Problem 3: Larger knapsack (8 vars) — enough to see strategy differences
    println!("────────────────────────────────────────────────────────");
    run_comparison(&large_strategy_problem(), "Problem 3: Knapsack (8 vars)");

    // Problem 4: Multi-constraint problem (10 vars, 3 constraints)
    println!("────────────────────────────────────────────────────────");
    run_comparison(
        &multi_constraint_problem(),
        "Problem 4: Multi-Constraint Knapsack (10 vars, 3 constraints)",
    );

    // Problem 5: Set cover (minimization — tests min-sense handling)
    println!("────────────────────────────────────────────────────────");
    run_comparison(&set_cover_problem(), "Problem 5: Set Cover (4 vars, minimization)");

    // Summary
    println!("════════════════════════════════════════════════════════");
    println!("Observations to look for:");
    println!("  - DFS tends to find good incumbents EARLY (fewer nodes to first feasible)");
    println!("  - Best-First explores fewer nodes to PROVE optimality (tighter gap)");
    println!("  - Most-Fractional creates more balanced splits (often fewer total nodes)");
    println!("  - First-Fractional is simpler but may create lopsided subtrees");
    println!("  - On small problems, differences are minor; they diverge on larger ones");
    println!("  - Best-First may use more memory (stores entire frontier in heap)");
    println!("════════════════════════════════════════════════════════");
}
