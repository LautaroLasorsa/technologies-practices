// Phase 1: LP Relaxation
//
// This binary demonstrates the foundational concept of LP relaxation.
// We take a binary knapsack MIP, drop the integrality constraints (treat
// binary variables as continuous in [0, 1]), and solve the resulting LP.
// The LP optimum provides an upper bound on the MIP optimum (for maximization).
// The solution will typically be fractional, motivating Branch & Bound.

#[path = "common.rs"]
mod common;

use common::*;

/// Solve the LP relaxation of a MIP problem using good_lp with the HiGHS backend.
///
/// Returns (solution_vector, objective_value) or an error string.
///
/// # TODO(human): Implement this function
///
/// LP Relaxation: drop integrality constraints, solve as continuous LP.
///
/// Steps using good_lp:
/// 1. Create a ProblemVariables container:
///      let mut vars = good_lp::ProblemVariables::new();
///
/// 2. Add one continuous variable per MIP variable. For each variable j:
///      let x_j = vars.add(good_lp::variable().min(lb[j]).max(ub[j]));
///    Store all Variable handles in a Vec<good_lp::Variable>.
///    IMPORTANT: do NOT use .integer() or .binary() — this is a RELAXATION.
///    Binary variables become continuous in [0, 1], integer variables
///    become continuous in [lb, ub].
///
/// 3. Build the objective expression:
///      let objective: good_lp::Expression = problem.objective.iter()
///          .zip(x_vars.iter())
///          .map(|(&c, &x)| c * x)
///          .sum();
///
/// 4. Create the optimization model. Use problem.sense to decide:
///      Sense::Maximize => vars.maximise(&objective).using(good_lp::solvers::highs::highs)
///      Sense::Minimize => vars.minimise(&objective).using(good_lp::solvers::highs::highs)
///
/// 5. Add constraints. For each constraint i:
///      Build the LHS expression: sum of constraints[i][j] * x_vars[j]
///      Match on constraint_types[i]:
///        ConstraintType::Leq => model.with(good_lp::constraint!(lhs <= rhs[i]))
///        ConstraintType::Geq => model.with(good_lp::constraint!(lhs >= rhs[i]))
///        ConstraintType::Eq  => model.with(good_lp::constraint!(lhs == rhs[i]))
///
///    Note: good_lp's .with() takes ownership and returns the model, so chain
///    calls or use a mutable binding with reassignment:
///      let mut model = vars.maximise(...).using(highs);
///      model = model.with(constraint);  // .with() returns the model
///
///    Actually, with() takes &mut self on some backends. Check the API —
///    you may need:
///      model.with(constraint);  // returns &mut Self
///    The key pattern: build all constraints and add them to the model,
///    then call model.solve().
///
/// 6. Solve: let solution = model.solve().map_err(|e| format!("{:?}", e))?;
///
/// 7. Extract values: solution.value(x_vars[j]) for each variable j.
///    Compute objective: solution.eval(objective_expr) or sum manually.
///
/// Refer to good_lp docs: https://docs.rs/good_lp/latest/good_lp/
/// Key API: ProblemVariables::new(), variable().min().max(), Expression,
///          constraint!(), .maximise()/.minimise(), .using(highs), .solve()
///
/// The LP relaxation gives a BOUND on the MIP optimum:
///   - For maximization: LP_opt >= MIP_opt (LP is less constrained)
///   - For minimization: LP_opt <= MIP_opt
///
/// If the LP solution happens to be integer-feasible, we got lucky —
/// it's also optimal for the MIP! This rarely happens in practice.
pub fn solve_lp_relaxation(problem: &MIPProblem) -> Result<(Vec<f64>, f64), String> {
    todo!("TODO(human): Implement LP relaxation solver using good_lp with HiGHS backend")
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Phase 1: LP Relaxation of Binary Knapsack          ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let problem = knapsack_problem();

    println!("Problem: Binary Knapsack");
    println!("  maximize  8*x1 + 5*x2 + 4*x3");
    println!("  s.t.      6*x1 + 4*x2 + 3*x3 <= 12");
    println!("            x1, x2, x3 in {{0, 1}}\n");

    println!("--- Solving LP Relaxation (x_i in [0, 1] instead of {{0, 1}}) ---\n");

    match solve_lp_relaxation(&problem) {
        Ok((solution, obj_value)) => {
            println!("LP Relaxation Objective: {:.6}", obj_value);
            print_solution(&solution, &problem.var_names);

            // Identify fractional variables
            let fractional = find_fractional_variables(&solution, &problem.var_types, INT_TOLERANCE);

            if fractional.is_empty() {
                println!("\n  All variables are integer! LP relaxation = MIP optimum.");
                println!("  (This is rare — we got lucky.)");
            } else {
                println!("\n  Fractional variables (integrality violated):");
                for &(idx, val) in &fractional {
                    let frac_part = val - val.floor();
                    println!(
                        "    {} = {:.6}  (fractional part = {:.6})",
                        problem.var_names[idx], val, frac_part
                    );
                }
                println!("\n  The LP relaxation bound ({:.6}) is an UPPER bound on the MIP optimum.", obj_value);
                println!("  To find the true integer optimum, we need Branch & Bound.");
            }

            // Show what rounding would give (naive approach)
            println!("\n--- Naive Rounding (for comparison) ---\n");
            let rounded: Vec<f64> = solution
                .iter()
                .zip(problem.var_types.iter())
                .map(|(&v, &vt)| match vt {
                    VarType::Continuous => v,
                    VarType::Integer | VarType::Binary => v.round(),
                })
                .collect();

            let rounded_obj: f64 = rounded
                .iter()
                .zip(problem.objective.iter())
                .map(|(x, c)| x * c)
                .sum();

            // Check feasibility of rounded solution
            let feasible = problem
                .constraints
                .iter()
                .zip(problem.rhs.iter())
                .zip(problem.constraint_types.iter())
                .all(|((row, &rhs), &ct)| {
                    let lhs: f64 = row.iter().zip(rounded.iter()).map(|(a, x)| a * x).sum();
                    match ct {
                        ConstraintType::Leq => lhs <= rhs + 1e-9,
                        ConstraintType::Geq => lhs >= rhs - 1e-9,
                        ConstraintType::Eq => (lhs - rhs).abs() < 1e-9,
                    }
                });

            print_solution(&rounded, &problem.var_names);
            println!("  Rounded objective: {:.6}", rounded_obj);
            println!(
                "  Feasible: {}",
                if feasible { "YES" } else { "NO — rounding can violate constraints!" }
            );
            if feasible {
                println!("  Gap from LP bound: {:.6} ({:.2}%)",
                    obj_value - rounded_obj,
                    100.0 * (obj_value - rounded_obj) / obj_value,
                );
            }
        }
        Err(e) => {
            eprintln!("ERROR solving LP relaxation: {}", e);
        }
    }
}
