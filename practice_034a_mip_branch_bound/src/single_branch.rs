// Phase 2: Single Branch
//
// This binary demonstrates one branching step in isolation. Given the
// fractional LP relaxation solution from Phase 1, we pick the most
// fractional variable and create two child subproblems:
//   Left child:  x_i <= floor(fractional_value)
//   Right child: x_i >= ceil(fractional_value)
// We solve both children's LP relaxations and compare their bounds
// to the parent's bound, observing the monotonicity property.

#[path = "common.rs"]
mod common;

use common::*;
use good_lp::{constraint, variable, Expression, ProblemVariables, Solution, SolverModel};

// ============================================================================
// LP relaxation solver — same as Phase 1 (provided here for self-containment).
// If you already implemented Phase 1, you can copy your implementation here.
// Otherwise, this is a second chance to implement it.
// ============================================================================

/// Solve the LP relaxation of a MIP problem.
/// (Same function as Phase 1 — copy your implementation or re-implement.)
fn solve_lp_relaxation(problem: &MIPProblem) -> Result<(Vec<f64>, f64), String> {
    // Re-use your Phase 1 implementation here.
    // If you haven't done Phase 1 yet, implement it now — it's the same function.
    todo!("TODO(human): Copy your solve_lp_relaxation from Phase 1, or re-implement it")
}

/// Create two child MIP problems by branching on variable `var_idx`.
///
/// Given x_i = fractional_value (from LP relaxation), create:
///   Left child:  add constraint x_i <= floor(fractional_value)
///   Right child: add constraint x_i >= ceil(fractional_value)
///
/// # TODO(human): Implement this function
///
/// Branching: given x_i = f (fractional), create two subproblems:
///   Left child:  add constraint x_i <= floor(f)
///   Right child: add constraint x_i >= ceil(f)
///
/// For binary variables (0-1), this simplifies to:
///   Left child:  fix x_i = 0 (upper bound becomes 0)
///   Right child: fix x_i = 1 (lower bound becomes 1)
///
/// For general integer variables with x_i = 3.7:
///   Left child:  x_i <= 3 (upper bound tightened to 3)
///   Right child: x_i >= 4 (lower bound tightened to 4)
///
/// Implementation steps:
///   1. Clone the problem twice (left and right).
///   2. For the LEFT child: set upper_bounds[var_idx] = floor(fractional_value)
///   3. For the RIGHT child: set lower_bounds[var_idx] = ceil(fractional_value)
///
/// The key insight: branching TIGHTENS the feasible region.
/// Each child's LP relaxation will have a bound that is no better than
/// the parent's bound:
///   - For maximization: child_bound <= parent_bound
///   - For minimization: child_bound >= parent_bound
///
/// This monotonicity is fundamental to B&B correctness — we are
/// partitioning the feasible region into two parts, each at least as
/// constrained as the original. The child LP can't do better because
/// it has strictly fewer feasible points.
///
/// Edge case: if floor(f) < lower_bounds[var_idx], the left child is
/// infeasible. Similarly, if ceil(f) > upper_bounds[var_idx], the right
/// child is infeasible. B&B handles this naturally — the LP solve will
/// return infeasible and the node gets pruned.
pub fn branch_on_variable(
    problem: &MIPProblem,
    var_idx: usize,
    fractional_value: f64,
) -> (MIPProblem, MIPProblem) {
    todo!("TODO(human): Implement branching — create left (floor) and right (ceil) child problems")
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Phase 2: Single Branching Step                     ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let problem = knapsack_problem();

    println!("Problem: Binary Knapsack");
    println!("  maximize  8*x1 + 5*x2 + 4*x3");
    println!("  s.t.      6*x1 + 4*x2 + 3*x3 <= 12");
    println!("            x1, x2, x3 in {{0, 1}}\n");

    // Step 1: Solve root LP relaxation
    println!("--- Step 1: Solve Root LP Relaxation ---\n");
    let (root_solution, root_obj) = solve_lp_relaxation(&problem)
        .expect("Root LP should be feasible");

    println!("Root LP objective: {:.6}", root_obj);
    print_solution(&root_solution, &problem.var_names);

    // Step 2: Find the most fractional variable
    let fractional = find_fractional_variables(&root_solution, &problem.var_types, INT_TOLERANCE);

    if fractional.is_empty() {
        println!("\nNo fractional variables — LP solution is already integer-feasible!");
        println!("MIP optimum = LP optimum = {:.6}", root_obj);
        return;
    }

    // Pick the most fractional variable (closest to 0.5)
    let &(branch_idx, branch_val) = fractional
        .iter()
        .max_by(|(_, a), (_, b)| {
            let frac_a = 0.5 - (a - a.floor() - 0.5).abs();
            let frac_b = 0.5 - (b - b.floor() - 0.5).abs();
            frac_a.partial_cmp(&frac_b).unwrap()
        })
        .unwrap();

    println!(
        "\nMost fractional variable: {} = {:.6} (frac part = {:.6})",
        problem.var_names[branch_idx],
        branch_val,
        branch_val - branch_val.floor()
    );

    // Step 3: Branch
    println!("\n--- Step 2: Branch on {} ---\n", problem.var_names[branch_idx]);
    let (left_problem, right_problem) = branch_on_variable(&problem, branch_idx, branch_val);

    println!(
        "Left child:  {} <= {:.0} (fix to 0 for binary)",
        problem.var_names[branch_idx],
        branch_val.floor()
    );
    println!(
        "Right child: {} >= {:.0} (fix to 1 for binary)",
        problem.var_names[branch_idx],
        branch_val.ceil()
    );

    // Step 4: Solve both children
    println!("\n--- Step 3: Solve Child LP Relaxations ---\n");

    println!("Left child ({}=0):", problem.var_names[branch_idx]);
    match solve_lp_relaxation(&left_problem) {
        Ok((left_sol, left_obj)) => {
            println!("  LP objective: {:.6}", left_obj);
            print_solution(&left_sol, &problem.var_names);
            let left_frac = find_fractional_variables(&left_sol, &problem.var_types, INT_TOLERANCE);
            if left_frac.is_empty() {
                println!("  Integer-feasible! This is a candidate MIP solution.");
            } else {
                println!("  Still fractional — would need further branching.");
            }
        }
        Err(e) => println!("  INFEASIBLE: {}", e),
    }

    println!("\nRight child ({}=1):", problem.var_names[branch_idx]);
    match solve_lp_relaxation(&right_problem) {
        Ok((right_sol, right_obj)) => {
            println!("  LP objective: {:.6}", right_obj);
            print_solution(&right_sol, &problem.var_names);
            let right_frac = find_fractional_variables(&right_sol, &problem.var_types, INT_TOLERANCE);
            if right_frac.is_empty() {
                println!("  Integer-feasible! This is a candidate MIP solution.");
            } else {
                println!("  Still fractional — would need further branching.");
            }
        }
        Err(e) => println!("  INFEASIBLE: {}", e),
    }

    // Step 5: Compare bounds
    println!("\n--- Step 4: Compare Bounds ---\n");
    println!("Root LP bound:       {:.6}", root_obj);

    let left_bound = solve_lp_relaxation(&left_problem)
        .map(|(_, obj)| obj)
        .unwrap_or(f64::NEG_INFINITY);
    let right_bound = solve_lp_relaxation(&right_problem)
        .map(|(_, obj)| obj)
        .unwrap_or(f64::NEG_INFINITY);

    println!("Left child bound:    {:.6}", left_bound);
    println!("Right child bound:   {:.6}", right_bound);

    println!("\nObservations:");
    println!("  - Both child bounds <= root bound (monotonicity)");
    println!("  - The gap between root and children shows branching progress");
    println!("  - To solve the MIP, we'd continue branching on fractional children");
    println!("  - With pruning (skipping nodes whose bound <= incumbent), we avoid");
    println!("    exploring the entire tree → that's Branch & Bound (Phase 3).");
}
