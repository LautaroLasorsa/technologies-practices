// ============================================================================
// Phase 4: Branch-and-Cut with Rounding Heuristic
// ============================================================================
//
// The algorithm that powers Gurobi, CPLEX, SCIP, HiGHS:
// Branch-and-Bound + Cutting Planes + Primal Heuristics.
//
// This phase integrates:
//   1. Branch-and-Bound (from 034a) for systematic enumeration
//   2. Gomory cutting planes (Phase 1) for LP bound tightening
//   3. A rounding heuristic for finding integer incumbents quickly
//
// You compare pure B&B (no cuts) vs branch-and-cut (B&B + cuts + heuristic)
// on the same problem to see the node reduction from cuts.
//
// Problem: medium-size knapsack (10 binary variables)
//   max  sum_j v_j * x_j
//   s.t. sum_j w_j * x_j <= capacity
//        x_j in {0, 1}
//
// ============================================================================

#[path = "common.rs"]
mod common;

use common::*;

use std::time::Instant;

// ─── B&B Node ───────────────────────────────────────────────────────────────

/// A node in the branch-and-cut tree.
///
/// Stores variable bounds imposed by branching, plus any cuts
/// specific to this node (in a simple implementation, cuts are
/// inherited from the parent).
#[derive(Debug, Clone)]
struct BCNode {
    /// Lower bounds on each variable (branching may tighten these)
    lower_bounds: Vec<f64>,
    /// Upper bounds on each variable (branching may tighten these)
    upper_bounds: Vec<f64>,
    /// Additional cut constraints accumulated at this node
    /// (in a full solver, these would come from a global cut pool)
    extra_constraints: Vec<LinearConstraint>,
    /// Depth in the B&B tree
    depth: usize,
    /// Node ID for tracking
    id: usize,
}

// ─── Provided: LP solving ───────────────────────────────────────────────────

/// Solve the LP relaxation of a MIP problem using HiGHS,
/// respecting the bounds from a BCNode.
fn solve_lp_relaxation_node(
    problem: &MIPProblem,
    node: &BCNode,
) -> Result<(Vec<f64>, f64), String> {
    use good_lp::*;

    let n = problem.n_vars;

    let mut vars = ProblemVariables::new();
    let x: Vec<Variable> = (0..n)
        .map(|i| {
            vars.add(variable().min(node.lower_bounds[i]).max(node.upper_bounds[i]))
        })
        .collect();

    // Objective
    let obj_expr: Expression = problem.obj.iter().zip(x.iter())
        .map(|(&c, &v)| c * v)
        .sum();

    let mut model = vars.minimise(obj_expr).using(highs);

    // Original constraints
    for c in &problem.constraints {
        let lhs: Expression = c.coeffs.iter().zip(x.iter())
            .map(|(&a, &v)| a * v)
            .sum();
        match c.sense {
            ConstraintSense::Le => { model = model.with(lhs.leq(c.rhs)); }
            ConstraintSense::Ge => { model = model.with(lhs.geq(c.rhs)); }
            ConstraintSense::Eq => {
                model = model.with(lhs.clone().leq(c.rhs));
                model = model.with(lhs.geq(c.rhs));
            }
        }
    }

    // Extra constraints (cuts) at this node
    for c in &node.extra_constraints {
        let lhs: Expression = c.coeffs.iter().zip(x.iter())
            .map(|(&a, &v)| a * v)
            .sum();
        match c.sense {
            ConstraintSense::Le => { model = model.with(lhs.leq(c.rhs)); }
            ConstraintSense::Ge => { model = model.with(lhs.geq(c.rhs)); }
            ConstraintSense::Eq => {
                model = model.with(lhs.clone().leq(c.rhs));
                model = model.with(lhs.geq(c.rhs));
            }
        }
    }

    match model.solve() {
        Ok(solution) => {
            let values: Vec<f64> = x.iter().map(|&v| solution.value(v)).collect();
            let objective: f64 = problem.obj.iter().zip(values.iter())
                .map(|(c, v)| c * v)
                .sum();
            Ok((values, objective))
        }
        Err(_) => Err("LP infeasible".to_string()),
    }
}

/// Solve the LP relaxation without any B&B node (root solve).
fn solve_lp_relaxation(problem: &MIPProblem) -> Result<(Vec<f64>, f64), String> {
    let root = BCNode {
        lower_bounds: vec![0.0; problem.n_vars],
        upper_bounds: problem.var_types.iter()
            .map(|vt| if *vt == VarType::Binary { 1.0 } else { f64::INFINITY })
            .collect(),
        extra_constraints: Vec::new(),
        depth: 0,
        id: 0,
    };
    solve_lp_relaxation_node(problem, &root)
}

// ─── Provided: Branching ────────────────────────────────────────────────────

/// Select the most fractional integer/binary variable for branching.
fn select_branching_variable(
    solution: &[f64],
    var_types: &[VarType],
) -> Option<usize> {
    most_fractional_variable(solution, var_types)
}

/// Create two child nodes by branching on variable `var_idx`.
fn branch_on_variable(
    parent: &BCNode,
    var_idx: usize,
    fractional_value: f64,
    next_id: &mut usize,
) -> (BCNode, BCNode) {
    let mut left = BCNode {
        lower_bounds: parent.lower_bounds.clone(),
        upper_bounds: parent.upper_bounds.clone(),
        extra_constraints: parent.extra_constraints.clone(),
        depth: parent.depth + 1,
        id: *next_id,
    };
    *next_id += 1;

    let mut right = BCNode {
        lower_bounds: parent.lower_bounds.clone(),
        upper_bounds: parent.upper_bounds.clone(),
        extra_constraints: parent.extra_constraints.clone(),
        depth: parent.depth + 1,
        id: *next_id,
    };
    *next_id += 1;

    // Left: x_i <= floor(f)
    left.upper_bounds[var_idx] = fractional_value.floor();
    // Right: x_i >= ceil(f)
    right.lower_bounds[var_idx] = fractional_value.ceil();

    (left, right)
}

// ─── Provided: Gomory cut generation via manual simplex ─────────────────────

/// Generate a Gomory cut from a MIP problem at the current LP optimum.
///
/// Uses the manual simplex tableau to extract fractional row data,
/// then computes the Gomory cut. Returns the cut as a >= constraint
/// or None if no fractional variable found.
///
/// The cut is returned as a LinearConstraint in the ORIGINAL variable
/// space (n_vars coefficients).
fn generate_gomory_cut_at_node(
    problem: &MIPProblem,
    node: &BCNode,
) -> Option<LinearConstraint> {
    // Build a local MIPProblem with node bounds as explicit constraints
    let mut local = problem.clone();

    // Add bound constraints from node
    for i in 0..problem.n_vars {
        if node.lower_bounds[i] > ZERO_TOL {
            local.constraints.push(LinearConstraint {
                coeffs: {
                    let mut c = vec![0.0; problem.n_vars];
                    c[i] = -1.0; // -x_i <= -lb  ⟹  x_i >= lb
                    c
                },
                sense: ConstraintSense::Le,
                rhs: -node.lower_bounds[i],
            });
        }
        let ub = node.upper_bounds[i];
        if ub < 1e10 {
            local.constraints.push(LinearConstraint {
                coeffs: {
                    let mut c = vec![0.0; problem.n_vars];
                    c[i] = 1.0;
                    c
                },
                sense: ConstraintSense::Le,
                rhs: ub,
            });
        }
    }

    // Add node's extra constraints
    for ec in &node.extra_constraints {
        local.constraints.push(ec.clone());
    }

    // Solve via manual simplex
    let tableau = match solve_lp_manual(&local) {
        Ok(t) => t,
        Err(_) => return None,
    };

    // Find fractional row
    let n_orig = problem.n_vars;
    let mut best_row = None;
    let mut best_frac = 0.0;
    for i in 0..tableau.m {
        let var_idx = tableau.basis[i];
        if var_idx >= n_orig {
            continue;
        }
        if problem.var_types[var_idx] == VarType::Continuous {
            continue;
        }
        let rhs_val = tableau.rhs(i);
        let f = frac(rhs_val);
        let dist = f.min(1.0 - f);
        if dist > INT_TOL && dist > best_frac {
            best_frac = dist;
            best_row = Some(i);
        }
    }

    let row = best_row?;
    let full_row = tableau.full_row(row);
    let rhs_val = full_row[tableau.n];

    // Compute Gomory cut in the full variable space (original + slacks)
    let f0 = frac(rhs_val);
    let cut_coeffs_full: Vec<f64> = full_row[..tableau.n]
        .iter()
        .map(|&a| frac(a))
        .collect();

    // Project back to original variables only
    // The cut sum_j f_j x_j >= f0 in the original space uses only
    // the first n_orig coefficients (slacks correspond to the constraints
    // and are non-negative, so their fractional parts only strengthen the cut).
    let cut_coeffs: Vec<f64> = cut_coeffs_full[..n_orig].to_vec();

    Some(LinearConstraint {
        coeffs: cut_coeffs,
        sense: ConstraintSense::Ge,
        rhs: f0,
    })
}

// ─── Provided: Pure B&B for comparison ──────────────────────────────────────

/// Pure branch-and-bound WITHOUT cuts (for comparison).
/// DFS node selection, most-fractional branching.
fn branch_and_bound_no_cuts(problem: &MIPProblem) -> BBResult {
    let n = problem.n_vars;
    let root = BCNode {
        lower_bounds: vec![0.0; n],
        upper_bounds: problem.var_types.iter()
            .map(|vt| if *vt == VarType::Binary { 1.0 } else { f64::INFINITY })
            .collect(),
        extra_constraints: Vec::new(),
        depth: 0,
        id: 0,
    };

    let mut stack = vec![root];
    let mut incumbent: Option<Vec<f64>> = None;
    let mut best_obj = f64::INFINITY; // minimization
    let mut nodes_explored = 0usize;
    let mut next_id = 1usize;

    while let Some(node) = stack.pop() {
        nodes_explored += 1;

        // Solve LP relaxation
        let (sol, lp_obj) = match solve_lp_relaxation_node(problem, &node) {
            Ok(r) => r,
            Err(_) => continue, // infeasible
        };

        // Bound check (minimization: prune if lp_obj >= best_obj)
        if lp_obj >= best_obj - EQ_TOL {
            continue;
        }

        // Integer check
        if is_integer_feasible(&sol, &problem.var_types) {
            if lp_obj < best_obj - EQ_TOL {
                best_obj = lp_obj;
                incumbent = Some(sol);
            }
            continue;
        }

        // Branch
        if let Some(var_idx) = select_branching_variable(&sol, &problem.var_types) {
            let (left, right) = branch_on_variable(&node, var_idx, sol[var_idx], &mut next_id);
            stack.push(right);
            stack.push(left);
        }
    }

    BBResult {
        solution: incumbent,
        objective: if best_obj < f64::INFINITY { Some(best_obj) } else { None },
        nodes_explored,
        cuts_generated: 0,
        incumbent_from_heuristic: false,
    }
}

// ─── TODO(human): Rounding Heuristic ────────────────────────────────────────

/// Attempt to find a feasible integer solution by rounding.
///
/// # Arguments
/// - `solution`: current LP relaxation solution (may have fractional values)
/// - `var_types`: variable types (Binary, Integer, Continuous)
/// - `problem`: the MIP problem (for constraint checking)
///
/// # Returns
/// `Some((rounded_solution, objective))` if the rounded solution satisfies
/// all constraints, `None` otherwise.
fn rounding_heuristic(
    solution: &[f64],
    var_types: &[VarType],
    problem: &MIPProblem,
) -> Option<(Vec<f64>, f64)> {
    // TODO(human): Rounding Heuristic
    //
    // Given a fractional LP solution, try to find a feasible integer solution
    // by rounding. This is the simplest primal heuristic.
    //
    // Algorithm:
    //   1. Clone the solution vector
    //   2. For each integer/binary variable:
    //      Round to nearest integer (0.7 → 1, 0.3 → 0)
    //   3. Check feasibility: does the rounded solution satisfy ALL constraints?
    //      For each constraint: sum(A[row][j] * x_rounded[j]) <= b[row]?
    //   4. If feasible: compute objective = sum(c[j] * x_rounded[j]), return Some(...)
    //      If infeasible: return None
    //
    // Use the provided `is_feasible()` function from common.rs to check
    // constraint satisfaction, and `objective_value()` to compute the objective.
    //
    // This often fails because rounding can violate constraints.
    // More sophisticated heuristics (feasibility pump, diving) handle this
    // by iterating between rounding and re-solving. For learning purposes,
    // simple rounding is sufficient.
    //
    // Even when it fails often, trying at every B&B node is cheap and
    // occasionally finds good incumbents that enable aggressive pruning.
    todo!("TODO(human): not implemented")
}

// ─── TODO(human): Branch-and-Cut ────────────────────────────────────────────

/// Solve a MIP using branch-and-cut: B&B with Gomory cuts and a rounding
/// heuristic at each node.
///
/// # Arguments
/// - `problem`: the MIP problem to solve
/// - `max_cuts_per_node`: maximum Gomory cuts to generate at each B&B node
///
/// # Returns
/// A `BBResult` with solution, objective, node count, cut count, and whether
/// the incumbent was found via the rounding heuristic.
fn branch_and_cut(problem: &MIPProblem, max_cuts_per_node: usize) -> BBResult {
    // TODO(human): Branch-and-Cut
    //
    // This is B&B with two enhancements at each node:
    //   1. Try rounding heuristic (cheap, may find incumbent)
    //   2. Generate cutting planes (tighten LP bound)
    //
    // Algorithm (DFS for simplicity):
    //   Initialize stack with root node, incumbent = None, best_obj = +inf (min)
    //
    //   While stack not empty:
    //     node = stack.pop()
    //
    //     // Solve LP relaxation
    //     let (solution, lp_obj) = solve_lp_relaxation_node(problem, &node)?;
    //
    //     // Bound check (minimization: prune if lp_obj >= best_obj)
    //     if lp_obj >= best_obj - EQ_TOL { prune; continue }
    //
    //     // Try rounding heuristic
    //     if let Some((rounded, rounded_obj)) = rounding_heuristic(&solution, ...) {
    //         if rounded_obj < best_obj { update incumbent }
    //     }
    //
    //     // Integer check
    //     if is_integer_feasible(&solution, ...) { update incumbent; continue }
    //
    //     // === CUTTING PLANE PHASE ===
    //     // Generate up to max_cuts_per_node Gomory cuts
    //     // For each cut: add as constraint to a local copy of the problem
    //     // Re-solve LP after adding cuts
    //     // This tightens the bound at this node
    //     //
    //     // Use generate_gomory_cut_at_node() to get a cut, then add it to
    //     // the node's extra_constraints. After adding cuts, re-solve the LP
    //     // at this node to get the tighter bound.
    //     //
    //     // Note: cuts generated at a node are valid globally (for all nodes),
    //     // but for simplicity we only use them at this node and its children
    //     // (since children inherit extra_constraints from the parent).
    //     // Production solvers maintain a global "cut pool."
    //
    //     // After cuts: re-check bound and integrality
    //     // If still fractional: branch
    //     let var_idx = select_branching_variable(&solution, ...);
    //     let (left, right) = branch_on_variable(&node, var_idx, solution[var_idx], ...);
    //     stack.push(right);
    //     stack.push(left);
    //
    // Return results with comparison metrics
    todo!("TODO(human): not implemented")
}

// ─── Test problem: larger knapsack ──────────────────────────────────────────

/// A knapsack problem with 10 binary variables for branch-and-cut testing.
///
/// max  45*x0 + 48*x1 + 35*x2 + 38*x3 + 28*x4 + 33*x5 + 22*x6 + 25*x7 + 18*x8 + 15*x9
/// s.t. 12*x0 + 13*x1 + 10*x2 + 11*x3 +  8*x4 +  9*x5 +  7*x6 +  8*x7 +  6*x8 +  5*x9 <= 40
///      x_j in {0, 1}
///
/// Minimization form: negate objective coefficients.
fn knapsack_10_items() -> MIPProblem {
    let values = vec![45.0, 48.0, 35.0, 38.0, 28.0, 33.0, 22.0, 25.0, 18.0, 15.0];
    let weights = vec![12.0, 13.0, 10.0, 11.0, 8.0, 9.0, 7.0, 8.0, 6.0, 5.0];
    let capacity = 40.0;
    let n = values.len();

    MIPProblem::new(
        values.iter().map(|v| -v).collect(), // minimize -values
        vec![LinearConstraint {
            coeffs: weights,
            sense: ConstraintSense::Le,
            rhs: capacity,
        }],
        vec![VarType::Binary; n],
        (0..n).map(|i| format!("x{}", i)).collect(),
    )
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 4: Branch-and-Cut vs Pure Branch-and-Bound          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let problem = knapsack_10_items();
    let n = problem.n_vars;
    let weights: Vec<f64> = problem.constraints[0].coeffs.clone();
    let values: Vec<f64> = problem.obj.iter().map(|c| -c).collect();
    let capacity = problem.constraints[0].rhs;

    println!("=== Problem: 10-Item Binary Knapsack ===\n");
    println!("  max {}", values.iter().enumerate()
        .map(|(i, v)| format!("{:.0}*x{}", v, i))
        .collect::<Vec<_>>().join(" + "));
    println!("  s.t. {} <= {:.0}", weights.iter().enumerate()
        .map(|(i, w)| format!("{:.0}*x{}", w, i))
        .collect::<Vec<_>>().join(" + "), capacity);
    println!("       x_j in {{0, 1}}\n");

    // LP relaxation bound
    match solve_lp_relaxation(&problem) {
        Ok((lp_sol, lp_obj)) => {
            println!("LP relaxation bound: {:.6} (max value: {:.2})\n", lp_obj, -lp_obj);
            println!("LP solution:");
            for (i, &v) in lp_sol.iter().enumerate() {
                if v > ZERO_TOL {
                    let frac = if !is_integer(v) { " *" } else { "" };
                    println!("  x{} = {:.4}{}", i, v, frac);
                }
            }
            println!();
        }
        Err(e) => {
            println!("LP relaxation failed: {}\n", e);
        }
    }

    // ─── Run 1: Pure B&B ────────────────────────────────────────────────────

    println!("{'=':<60}");
    println!("=== Run 1: Pure Branch-and-Bound (no cuts) ===\n");

    let t1 = Instant::now();
    let result_bb = branch_and_bound_no_cuts(&problem);
    let time_bb = t1.elapsed();

    println!("{}", result_bb);
    println!("  Time: {:.3}ms\n", time_bb.as_secs_f64() * 1000.0);

    // ─── Run 2: Branch-and-Cut ──────────────────────────────────────────────

    println!("{'=':<60}");
    println!("=== Run 2: Branch-and-Cut (max 3 cuts/node) ===\n");

    let t2 = Instant::now();
    let result_bc = branch_and_cut(&problem, 3);
    let time_bc = t2.elapsed();

    println!("{}", result_bc);
    println!("  Time: {:.3}ms\n", time_bc.as_secs_f64() * 1000.0);

    // ─── Comparison ─────────────────────────────────────────────────────────

    println!("{'=':<60}");
    println!("=== Comparison ===\n");
    println!(
        "  {:>30}  {:>12}  {:>12}",
        "", "Pure B&B", "Branch-Cut"
    );
    println!(
        "  {:>30}  {:>12}  {:>12}",
        "Nodes explored",
        result_bb.nodes_explored,
        result_bc.nodes_explored
    );
    println!(
        "  {:>30}  {:>12}  {:>12}",
        "Cuts generated",
        result_bb.cuts_generated,
        result_bc.cuts_generated
    );
    println!(
        "  {:>30}  {:>12}  {:>12}",
        "Incumbent from heuristic",
        result_bb.incumbent_from_heuristic,
        result_bc.incumbent_from_heuristic
    );
    if let (Some(obj1), Some(obj2)) = (result_bb.objective, result_bc.objective) {
        println!(
            "  {:>30}  {:>12.4}  {:>12.4}",
            "Objective (min form)", obj1, obj2
        );
        println!(
            "  {:>30}  {:>12.4}  {:>12.4}",
            "Objective (max value)", -obj1, -obj2
        );
    }
    println!(
        "  {:>30}  {:>9.3}ms  {:>9.3}ms",
        "Time",
        time_bb.as_secs_f64() * 1000.0,
        time_bc.as_secs_f64() * 1000.0
    );

    if result_bc.nodes_explored < result_bb.nodes_explored {
        let reduction = 100.0 * (1.0 - result_bc.nodes_explored as f64 / result_bb.nodes_explored as f64);
        println!(
            "\n  Branch-and-cut reduced nodes by {:.1}% ({} → {})",
            reduction, result_bb.nodes_explored, result_bc.nodes_explored
        );
    }

    println!("\n  Key insight: cuts tighten LP bounds at each node, enabling");
    println!("  more aggressive pruning and fewer nodes to prove optimality.");
    println!("  The rounding heuristic provides incumbents cheaply, further");
    println!("  enabling bound-based pruning.");

    println!("\n=== Phase 4 Complete ===");
}
