// ============================================================================
// Phase 1: Gomory Fractional Cuts
// ============================================================================
//
// This phase teaches the most fundamental cutting plane: the Gomory fractional
// cut. You will:
//
//   1. Solve the LP relaxation of a small MIP
//   2. Identify a fractional basic variable in the simplex tableau
//   3. Derive a Gomory cut from that tableau row
//   4. Add the cut and re-solve to observe LP bound tightening
//
// The key insight: Gomory cuts are purely mechanical --- derived from the
// simplex tableau, requiring NO problem-specific knowledge. This generality
// is their power (works on any MIP) and their weakness (may not exploit
// structure as well as specialized cuts).
//
// Example problem (small MIP):
//   min  -5*x0 - 4*x1
//   s.t.  6*x0 + 4*x1 <= 24
//           x0 + 2*x1 <=  6
//         x0, x1 >= 0, integer
//
// LP relaxation optimal: x0 = 3.0, x1 = 1.5, obj = -21.0
// MIP optimal:           x0 = 4, x1 = 0, obj = -20
//                     or x0 = 3, x1 = 1, obj = -19
//                     Best: x0 = 4, x1 = 0, obj = -20
//
// ============================================================================

#[path = "common.rs"]
mod common;

use common::*;

// ─── Provided: LP relaxation and tableau display ─────────────────────────────

/// Solve the LP relaxation and display the optimal tableau.
/// Returns the tableau for cut generation.
fn solve_and_display(problem: &MIPProblem) -> SimplexTableau {
    println!("=== Solving LP Relaxation ===\n");
    println!("Problem:");
    println!(
        "  min  {}",
        problem
            .obj
            .iter()
            .zip(problem.var_names.iter())
            .map(|(c, n)| format!("{:+.0}*{}", c, n))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for (i, c) in problem.constraints.iter().enumerate() {
        let sense_str = match c.sense {
            ConstraintSense::Le => "<=",
            ConstraintSense::Ge => ">=",
            ConstraintSense::Eq => "==",
        };
        println!(
            "  s.t. {} {} {:.0}   (constraint {})",
            c.coeffs
                .iter()
                .zip(problem.var_names.iter())
                .map(|(a, n)| format!("{:+.0}*{}", a, n))
                .collect::<Vec<_>>()
                .join(" "),
            sense_str,
            c.rhs,
            i
        );
    }
    println!();

    let tableau = solve_lp_manual(problem).expect("LP should be feasible");
    println!("{}", tableau);

    let sol = tableau.solution();
    println!("LP optimal solution:");
    print_solution(&problem.var_names, &sol[..problem.n_vars], &problem.var_types);
    println!("LP optimal objective: {:.6}\n", tableau.objective_value());

    // Check if LP is already integer
    if is_integer_feasible(&sol[..problem.n_vars], &problem.var_types) {
        println!("LP solution is already integer-feasible! No cuts needed.");
    } else {
        println!("LP solution is FRACTIONAL. Gomory cuts can tighten the relaxation.");
    }
    println!();

    tableau
}

/// Find a tableau row with a fractional RHS (corresponding to a fractional
/// basic variable that should be integer).
///
/// Returns (row_index, basic_variable_index, fractional_value) or None.
fn find_fractional_row(
    tableau: &SimplexTableau,
    var_types: &[VarType],
) -> Option<(usize, usize, f64)> {
    let mut best_row = None;
    let mut best_frac = 0.0;

    for i in 0..tableau.m {
        let var_idx = tableau.basis[i];
        // Only consider original variables that should be integer
        if var_idx >= var_types.len() {
            continue; // slack variable, skip
        }
        if var_types[var_idx] == VarType::Continuous {
            continue;
        }
        let rhs = tableau.rhs(i);
        let f = frac(rhs);
        let dist = f.min(1.0 - f);
        if dist > INT_TOL && dist > best_frac {
            best_frac = dist;
            best_row = Some((i, var_idx, rhs));
        }
    }

    best_row
}

// ─── TODO(human): Gomory cut derivation ──────────────────────────────────────

/// Derive a Gomory fractional cut from a simplex tableau row.
///
/// # Arguments
/// - `tableau_row`: The FULL tableau row coefficients (length = n, all variables).
///   For the basic variable in this row, the coefficient is 1.0.
///   For other basic variables, the coefficient is 0.0.
///   For non-basic variables, the coefficient is the tableau entry ā_{ij}.
/// - `rhs`: The RHS value b̄_i of this tableau row (fractional).
///
/// # Returns
/// `(cut_coefficients, cut_rhs)` representing:
///   sum_j cut_coefficients[j] * x_j >= cut_rhs
///
/// where cut_coefficients has length n (one per variable in the tableau,
/// including slacks).
///
/// # Gomory Fractional Cut Derivation
///
/// From a simplex tableau row for basic variable x_i:
///   x_i + Σ_j ā_{ij} * x_j = b̄_i   (where x_j are non-basic)
///
/// If b̄_i is fractional (not integer), define:
///   f₀ = frac(b̄_i)         -- fractional part of the RHS
///   f_j = frac(ā_{ij})      -- fractional part of each coefficient
///
/// The Gomory fractional cut is:
///   Σ_j f_j * x_j ≥ f₀
///
/// where the sum is over ALL variables (basic variables have f_j = 0
/// since their tableau column is a unit vector).
///
/// # Why this is valid
///
/// From the tableau row: x_i = b̄_i - Σ_j ā_{ij} * x_j
/// Decompose each ā_{ij} = floor(ā_{ij}) + f_j and b̄_i = floor(b̄_i) + f₀.
/// Then: x_i - floor(b̄_i) + Σ_j floor(ā_{ij}) * x_j = f₀ - Σ_j f_j * x_j
///
/// The LHS is an integer (since x_i and x_j are integers). Therefore:
///   f₀ - Σ_j f_j * x_j must be an integer.
///
/// Also: Σ_j f_j * x_j ≥ 0 (since f_j ≥ 0 and x_j ≥ 0).
/// So f₀ - Σ_j f_j * x_j ≤ f₀ < 1.
/// Combined: f₀ - Σ_j f_j * x_j is an integer ≤ f₀ < 1, so it's ≤ 0.
/// Therefore: Σ_j f_j * x_j ≥ f₀.
///
/// # Key subtlety: the frac() function
///
/// frac(a) = a - floor(a), always in [0, 1).
/// For NEGATIVE numbers: frac(-0.3) = -0.3 - (-1) = 0.7, NOT -0.3.
/// This is critical for correctness.
/// Use the provided `frac()` function from common.rs.
///
/// # Current LP violation
///
/// At the current LP solution, all non-basic variables are 0, so:
///   LHS = Σ f_j * 0 = 0 < f₀ = RHS
/// The cut is violated! It will tighten the LP relaxation.
fn compute_gomory_cut(tableau_row: &[f64], rhs: f64) -> (Vec<f64>, f64) {
    // Hint: the implementation is ~5 lines.
    // 1. Compute f₀ = frac(rhs)
    // 2. For each coefficient in tableau_row, compute f_j = frac(coefficient)
    // 3. Return (f_j vector, f₀)
    //
    // Note: basic variables have coefficient 0.0 or 1.0 in their column,
    // so frac(0.0) = 0.0 and frac(1.0) = 0.0 — they contribute nothing.
    // This is correct: the cut only involves non-basic variables effectively.

    let _ = (tableau_row, rhs);
    todo!("TODO(human): Implement Gomory fractional cut derivation")
}

// ─── TODO(human): Add cut and re-solve ───────────────────────────────────────

/// Add a Gomory cut to the LP tableau and re-solve.
///
/// # Arguments
/// - `tableau`: The current simplex tableau (will be modified in place).
/// - `cut_coeffs`: Cut coefficients for all variables (length = tableau.n).
///   Represents: Σ cut_coeffs[j] * x_j >= cut_rhs.
/// - `cut_rhs`: The RHS of the cut (f₀ > 0).
///
/// # Returns
/// `(solution, objective)` after adding the cut and re-solving.
///
/// # How to add the cut
///
/// The Gomory cut is: Σ f_j * x_j ≥ f₀
///
/// To add this to the tableau:
/// 1. Call `tableau.add_gomory_cut_row(cut_coeffs, cut_rhs)` which:
///    - Adds a surplus variable s_new >= 0
///    - Creates the row: Σ cut_coeffs[j] * x_j - s_new = cut_rhs
///    - The new surplus variable is initially basic with value = cut_rhs
///      but may be negative after expressing in the current basis
/// 2. Call `tableau.dual_simplex_restore(1000)` to restore primal feasibility
///    (the new row may have negative RHS, which dual simplex handles)
/// 3. Extract the new solution and objective from the re-solved tableau
///
/// # Why dual simplex?
///
/// When we add a Gomory cut as a new constraint, the current solution becomes
/// infeasible (the cut was specifically chosen to be violated). The dual
/// simplex method restores primal feasibility while maintaining dual feasibility
/// (optimality conditions). This is more efficient than re-solving from scratch.
///
/// In production solvers, this is exactly how cuts are added at each B&B node:
/// warm-start dual simplex from the previous optimal basis with the new constraint.
fn add_cut_and_resolve(
    tableau: &mut SimplexTableau,
    cut_coeffs: &[f64],
    cut_rhs: f64,
) -> (Vec<f64>, f64) {
    // Hint:
    // 1. Add the cut row to the tableau
    // 2. Run dual simplex to restore feasibility
    // 3. Return (tableau.solution(), tableau.objective_value())
    //
    // Error handling: if dual simplex fails, print an error and return
    // the current (possibly infeasible) solution.

    let _ = (tableau, cut_coeffs, cut_rhs);
    todo!("TODO(human): Add Gomory cut to tableau and re-solve with dual simplex")
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 1: Gomory Fractional Cuts                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let problem = example_mip_small();

    // Step 1: Solve LP relaxation, display tableau
    let mut tableau = solve_and_display(&problem);

    // Step 2: Find a fractional row
    println!("=== Finding Fractional Basic Variable ===\n");
    let (row, var_idx, rhs_val) = find_fractional_row(&tableau, &problem.var_types)
        .expect("Expected a fractional basic variable");

    println!(
        "Selected row {} (basic variable x{} = {:.6}, fractional part = {:.6})\n",
        row,
        var_idx,
        rhs_val,
        frac(rhs_val)
    );

    // Display the tableau row
    let full_row = tableau.full_row(row);
    println!("Tableau row coefficients:");
    for (j, &val) in full_row.iter().enumerate() {
        if j < tableau.n {
            print!("  x{}: {:+.6}", j, val);
        }
    }
    println!("  | RHS: {:.6}\n", full_row[tableau.n]);

    // Step 3: Compute Gomory cut
    println!("=== Computing Gomory Cut ===\n");
    let (cut_coeffs, cut_rhs) = compute_gomory_cut(&full_row[..tableau.n], rhs_val);

    let cut = Cut {
        coeffs: cut_coeffs.clone(),
        rhs: cut_rhs,
        description: format!("Gomory cut from row {} (x{})", row, var_idx),
    };
    println!("Generated cut: {}\n", cut);

    // Verify the cut is violated by the current LP solution
    let current_sol = tableau.solution();
    let lhs: f64 = cut_coeffs
        .iter()
        .zip(current_sol.iter())
        .map(|(c, x)| c * x)
        .sum();
    println!(
        "Current LP: LHS = {:.6}, RHS = {:.6} → {} (should be violated)",
        lhs,
        cut_rhs,
        if lhs >= cut_rhs - EQ_TOL {
            "satisfied"
        } else {
            "VIOLATED"
        }
    );
    println!();

    // Step 4: Add cut and re-solve
    println!("=== Adding Cut and Re-solving ===\n");
    let (new_sol, new_obj) = add_cut_and_resolve(&mut tableau, &cut_coeffs, cut_rhs);

    println!("New LP solution after cut:");
    print_solution(
        &problem.var_names,
        &new_sol[..problem.n_vars],
        &problem.var_types,
    );
    println!("New LP objective: {:.6}", new_obj);
    println!(
        "Bound improved from {:.6} to {:.6}",
        -21.0, // original LP bound
        new_obj
    );

    // Check if now integer
    if is_integer_feasible(&new_sol[..problem.n_vars], &problem.var_types) {
        println!("\nSolution is now integer-feasible! Cut closed the gap.");
    } else {
        println!("\nSolution is still fractional. More cuts may be needed.");
        println!("(This is expected — one cut rarely suffices.)");
    }

    println!("\n=== Phase 1 Complete ===");
}
