// ============================================================================
// Phase 3: Pure Cutting Plane Algorithm (Gomory)
// ============================================================================
//
// This phase assembles the Gomory cut generator into a complete iterative
// algorithm. You will see how pure cutting planes converge (slowly) to the
// integer optimum, understand why cut accumulation causes numerical issues,
// and appreciate why branch-and-cut is preferred.
//
// Iterate: solve LP → generate Gomory cut → add cut → re-solve → repeat
// until integer-feasible or iteration limit reached.
//
// Example problem (same small MIP as Phase 1):
//   min  -5*x0 - 4*x1
//   s.t.  6*x0 + 4*x1 <= 24
//           x0 + 2*x1 <=  6
//         x0, x1 >= 0, integer
//
// LP optimal: x0 = 3.0, x1 = 1.5, obj = -21.0
// MIP optimal: x0 = 4, x1 = 0, obj = -20
//
// The cutting plane algorithm should tighten the LP bound from -21.0
// toward -20.0 with each Gomory cut, ideally reaching the integer optimum.
//
// ============================================================================

#[path = "common.rs"]
mod common;

use common::*;

// ─── Provided: Solve LP and extract tableau ─────────────────────────────────

/// Solve the LP relaxation using the manual simplex and return the
/// tableau for cut generation.
///
/// Returns (solution, objective, tableau) or an error string.
fn solve_and_get_tableau(problem: &MIPProblem) -> Result<(Vec<f64>, f64, SimplexTableau), String> {
    let tableau = solve_lp_manual(problem)?;
    let sol = tableau.solution();
    let obj = tableau.objective_value();
    Ok((sol, obj, tableau))
}

/// Compute a Gomory cut from a full tableau row with fractional RHS.
/// (Same derivation as Phase 1.)
///
/// Given the full tableau row: x_i + sum_j a_bar_ij * x_j = b_bar_i
/// The Gomory cut is: sum_j frac(a_bar_ij) * x_j >= frac(b_bar_i)
///
/// Returns (cut_coefficients, cut_rhs).
fn compute_gomory_cut(tableau_row: &[f64], rhs: f64) -> (Vec<f64>, f64) {
    // This is the same as Phase 1's compute_gomory_cut.
    // Copy your Phase 1 implementation here after completing it.
    //
    // For reference, the Gomory cut formula:
    //   f_0 = frac(rhs)
    //   f_j = frac(tableau_row[j]) for each j
    //   Cut: sum_j f_j * x_j >= f_0
    let _ = (tableau_row, rhs);
    todo!(
        "TODO(human): Copy your Gomory cut implementation from Phase 1 \
         (compute_gomory_cut in gomory.rs)"
    )
}

/// Find the best fractional basic variable for cut generation.
/// Returns (row_index, var_index, rhs_value) or None.
fn find_fractional_row(
    tableau: &SimplexTableau,
    var_types: &[VarType],
) -> Option<(usize, usize, f64)> {
    let mut best_row = None;
    let mut best_frac = 0.0;

    for i in 0..tableau.m {
        let var_idx = tableau.basis[i];
        if var_idx >= var_types.len() {
            continue; // slack variable
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

/// Print a summary of one round of the cutting plane algorithm.
fn print_round_summary(round: usize, obj: f64, n_fractional: usize, cut_info: &str) {
    println!(
        "  Round {:>2}: obj = {:>10.6}  |  fractional vars = {}  |  {}",
        round, obj, n_fractional, cut_info
    );
}

// ─── TODO(human): Pure Cutting Plane Algorithm ──────────────────────────────

/// Run the pure cutting plane algorithm using Gomory cuts.
///
/// # Arguments
/// - `problem`: the MIP problem to solve
/// - `max_rounds`: maximum number of cutting plane iterations
///
/// # Returns
/// A `CuttingPlaneResult` with the final solution, objective, round count,
/// whether the solution is integer-feasible, and the LP bound history.
fn cutting_plane_algorithm(problem: &MIPProblem, max_rounds: usize) -> CuttingPlaneResult {
    // TODO(human): Pure Cutting Plane Algorithm
    //
    // 1. Solve LP relaxation of the initial problem
    // 2. Check if solution is integer-feasible (within tolerance 1e-6)
    //    → If yes, return as optimal
    // 3. Find a fractional integer variable in the solution
    // 4. Generate a Gomory cut from the tableau row of that variable
    //    (Use compute_gomory_cut from Phase 1)
    // 5. Add the cut as a new constraint to the problem
    // 6. Re-solve the LP
    // 7. Record: current LP bound, number of fractional variables
    // 8. Repeat from step 2 until:
    //    - Solution is integer-feasible (optimal!)
    //    - max_rounds reached (return best bound)
    //    - LP becomes infeasible (problem is infeasible)
    //
    // Implementation notes:
    //   - Keep a mutable copy of the problem, adding cuts as new constraints
    //   - Each round adds ONE cut (simplest approach)
    //   - Track bound_history to show convergence
    //   - Print progress each round: LP bound, # fractional vars, cut details
    //
    // Suggested structure:
    //   let (mut sol, mut obj, mut tableau) = solve_and_get_tableau(problem)?;
    //   let mut bound_history = vec![obj];
    //   let mut cuts_added = 0;
    //
    //   for round in 1..=max_rounds {
    //       // Check integer feasibility
    //       if is_integer_feasible(&sol[..problem.n_vars], &problem.var_types) { break; }
    //
    //       // Find fractional row
    //       let (row, var_idx, rhs_val) = find_fractional_row(&tableau, &problem.var_types)?;
    //
    //       // Generate Gomory cut from that row
    //       let full_row = tableau.full_row(row);
    //       let (cut_coeffs, cut_rhs) = compute_gomory_cut(&full_row[..tableau.n], rhs_val);
    //
    //       // Add cut to tableau and re-solve with dual simplex
    //       tableau.add_gomory_cut_row(&cut_coeffs, cut_rhs);
    //       tableau.dual_simplex_restore(1000)?;
    //
    //       // Extract new solution
    //       sol = tableau.solution();
    //       obj = tableau.objective_value();
    //       cuts_added += 1;
    //       bound_history.push(obj);
    //
    //       // Count fractional variables and print progress
    //       let n_frac = count fractional vars in sol[..problem.n_vars]
    //       print_round_summary(round, obj, n_frac, &format!("cut from x{}", var_idx));
    //   }
    //
    // What you'll observe:
    //   - LP bound tightens each round (monotonically for maximization: decreases)
    //   - Number of fractional variables may oscillate
    //   - After several rounds, the LP may become integer-feasible
    //   - Or numerical issues may prevent convergence (this is expected!)
    //
    // This is why pure cutting planes aren't used alone — branch-and-cut
    // combines cutting with branching for robust convergence.
    todo!("TODO(human): not implemented")
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 3: Pure Cutting Plane Algorithm (Gomory)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Problem 1: Small MIP
    println!("=== Problem 1: Small MIP ===\n");
    let problem1 = example_mip_small();
    println!("  min  -5*x0 - 4*x1");
    println!("  s.t.  6*x0 + 4*x1 <= 24");
    println!("          x0 + 2*x1 <=  6");
    println!("        x0, x1 >= 0, integer\n");
    println!("  LP optimum:  -21.0 (x0=3.0, x1=1.5)");
    println!("  MIP optimum: -20.0 (x0=4, x1=0)\n");

    println!("Running cutting plane algorithm (max 20 rounds):\n");
    let result1 = cutting_plane_algorithm(&problem1, 20);
    println!("\n{}", result1);

    if result1.is_integer {
        println!("SUCCESS: Pure cutting planes found the integer optimum!");
    } else {
        println!("Cutting planes did not converge to integer optimum in 20 rounds.");
        println!("This is typical — pure Gomory cuts can be slow to converge.");
    }

    // Problem 2: Medium MIP
    println!("\n{'=':<60}\n");
    println!("=== Problem 2: Medium MIP (3 variables) ===\n");
    let problem2 = example_mip_medium();
    println!("  min  -3*x0 - 2*x1 - 4*x2");
    println!("  s.t.  2*x0 +   x1 + 3*x2 <= 10");
    println!("          x0 + 2*x1 +   x2 <=  8");
    println!("        x0, x1, x2 >= 0, integer\n");

    println!("Running cutting plane algorithm (max 30 rounds):\n");
    let result2 = cutting_plane_algorithm(&problem2, 30);
    println!("\n{}", result2);

    if result2.is_integer {
        println!("SUCCESS: Found integer optimum via pure cutting planes!");
    } else {
        println!("Did not converge. This motivates branch-and-cut (Phase 4).");
    }

    // Summary
    println!("\n=== Summary ===\n");
    println!("Problem 1: {} rounds, {} cuts, obj = {:.6}, integer = {}",
        result1.rounds, result1.cuts_added, result1.objective, result1.is_integer);
    println!("Problem 2: {} rounds, {} cuts, obj = {:.6}, integer = {}",
        result2.rounds, result2.cuts_added, result2.objective, result2.is_integer);
    println!();
    println!("Key observations:");
    println!("  - LP bound tightens monotonically with each cut");
    println!("  - Convergence can be slow (many rounds for small problems)");
    println!("  - Numerical issues may arise as cuts accumulate");
    println!("  - This motivates branch-and-cut: combine cuts with branching");

    println!("\n=== Phase 3 Complete ===");
}
