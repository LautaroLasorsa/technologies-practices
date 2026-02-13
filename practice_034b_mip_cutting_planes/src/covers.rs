// ============================================================================
// Phase 2: Cover Inequalities for Knapsack
// ============================================================================
//
// Cover cuts exploit the structure of binary knapsack constraints.
// Unlike Gomory cuts (general-purpose), cover cuts are specific to
// 0-1 knapsack substructures — and they're among the most effective.
//
// A knapsack constraint has the form:
//   sum_j a_j * x_j <= b,   x_j in {0, 1}
//
// A "cover" C is a subset of items whose total weight exceeds capacity:
//   sum_{j in C} a_j > b
//
// The cover inequality says: you can't select all items in C:
//   sum_{j in C} x_j <= |C| - 1
//
// This is valid because if all items in C were selected, the knapsack
// constraint would be violated. At least one must be excluded.
//
// Example problem (binary knapsack):
//   max 16*x0 + 22*x1 + 12*x2 + 8*x3 + 11*x4 + 19*x5
//   s.t. 5*x0 + 7*x1 + 4*x2 + 3*x3 + 4*x4 + 6*x5 <= 15
//        x_j in {0, 1}
//
// LP relaxation will have some items at fractional values.
// We find a cover whose inequality is violated by the LP solution
// and add it to tighten the relaxation.
//
// ============================================================================

#[path = "common.rs"]
mod common;

use common::*;

// ─── Provided: LP solving and display ───────────────────────────────────────

/// Solve the LP relaxation of a knapsack problem using HiGHS.
fn solve_lp_relaxation(problem: &MIPProblem) -> Result<(Vec<f64>, f64), String> {
    let lp = solve_lp_highs(problem)?;
    Ok((lp.values, lp.objective))
}

/// Pretty-print a knapsack solution with item details.
fn print_knapsack_solution(
    solution: &[f64],
    item_names: &[String],
    weights: &[f64],
    values: &[f64],
) {
    println!("  {:>6}  {:>8}  {:>8}  {:>8}", "Item", "x_j", "Weight", "Value");
    println!("  {:-<6}  {:-<8}  {:-<8}  {:-<8}", "", "", "", "");
    let mut total_weight = 0.0;
    let mut total_value = 0.0;
    for (i, name) in item_names.iter().enumerate() {
        let w = weights[i] * solution[i];
        let v = values[i] * solution[i];
        total_weight += w;
        total_value += v;
        let frac_marker = if !is_integer(solution[i]) && solution[i] > ZERO_TOL {
            " *"
        } else {
            ""
        };
        println!(
            "  {:>6}  {:>8.4}  {:>8.2}  {:>8.2}{}",
            name, solution[i], w, v, frac_marker
        );
    }
    println!("  {:-<6}  {:-<8}  {:-<8}  {:-<8}", "", "", "", "");
    println!(
        "  {:>6}  {:>8}  {:>8.2}  {:>8.2}",
        "Total", "", total_weight, total_value
    );
    println!("  (* = fractional)");
}

// ─── TODO(human): Find Minimal Cover ────────────────────────────────────────

/// Find a minimal cover for a knapsack constraint that is violated by
/// the current LP solution.
///
/// # Arguments
/// - `weights`: coefficient a_j for each item in the knapsack constraint
/// - `capacity`: the RHS b of the knapsack constraint (sum a_j x_j <= b)
/// - `lp_solution`: current LP relaxation values for each variable
///
/// # Returns
/// `Some(cover)` where `cover` is a Vec of item indices forming a minimal
/// cover whose cover inequality is violated by `lp_solution`, or `None`
/// if no violated cover is found by the greedy heuristic.
fn find_minimal_cover(
    weights: &[f64],
    capacity: f64,
    lp_solution: &[f64],
) -> Option<Vec<usize>> {
    // TODO(human): Find a Minimal Cover
    //
    // A cover C for knapsack sum_j(a_j * x_j) <= b is a subset where
    // sum_{j in C} a_j > b (the items in C exceed capacity).
    //
    // We want a cover whose inequality is VIOLATED by the current LP solution:
    //   sum_{j in C} x*_j > |C| - 1
    //
    // Greedy heuristic:
    //   1. Create a list of (index, lp_value, weight) for all items
    //   2. Sort by LP value DESCENDING (fractional items with high LP value first)
    //   3. Greedily add items to cover set C:
    //      - total_weight += weight[j]
    //      - Add j to C
    //      - If total_weight > capacity → we have a cover, stop adding
    //   4. Minimize the cover: try removing each item (in reverse order of addition)
    //      - If removing item j still leaves total_weight > capacity → remove it
    //      - This ensures minimality (no proper subset is a cover)
    //   5. Check violation: sum_{j in C} x*_j > |C| - 1?
    //      If violated → return Some(cover)
    //      If not → return None (no violated cover found by this heuristic)
    //
    // Note: this greedy approach may miss some violated covers. An exact
    // approach would solve a knapsack problem to find the most violated cover.
    // The greedy heuristic is sufficient for learning.
    todo!("TODO(human): not implemented")
}

// ─── TODO(human): Generate Cover Inequality ─────────────────────────────────

/// Generate the cover inequality from a cover set.
///
/// # Arguments
/// - `cover`: indices of items in the cover
/// - `n_vars`: total number of variables in the problem
///
/// # Returns
/// `(coefficients, rhs)` representing the inequality:
///   sum_{j in C} x_j <= |C| - 1
///
/// Returned in the form suitable for adding as a Le constraint:
///   coefficients[j] = 1.0 if j in cover, 0.0 otherwise
///   rhs = (|C| - 1) as f64
fn generate_cover_inequality(cover: &[usize], n_vars: usize) -> (Vec<f64>, f64) {
    // TODO(human): Generate Cover Inequality
    //
    // Given a cover set C = {j1, j2, ..., jk}, the cover inequality is:
    //   sum_{j in C} x_j <= |C| - 1
    //
    // In our constraint format (Ax <= b for the cut):
    //   coefficients: 1.0 for each j in C, 0.0 for others
    //   rhs: (|C| - 1) as f64
    //
    // Return: (coefficients vec of length n_vars, rhs)
    //
    // Example: if n_vars = 6 and cover = [1, 3, 5], then:
    //   coefficients = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    //   rhs = 2.0  (|C| - 1 = 3 - 1 = 2)
    //
    // This says: you can pick at most 2 out of items {1, 3, 5}.
    todo!("TODO(human): not implemented")
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2: Cover Inequalities for Knapsack                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let problem = example_knapsack();
    let n = problem.n_vars;

    // Extract knapsack data for display
    let weights: Vec<f64> = problem.constraints[0].coeffs.clone();
    let values: Vec<f64> = problem.obj.iter().map(|c| -c).collect(); // negate (obj is min -values)
    let capacity = problem.constraints[0].rhs;

    println!("=== Knapsack Problem ===\n");
    println!("  max  {}", values.iter().enumerate()
        .map(|(i, v)| format!("{:.0}*x{}", v, i))
        .collect::<Vec<_>>().join(" + "));
    println!("  s.t. {} <= {:.0}", weights.iter().enumerate()
        .map(|(i, w)| format!("{:.0}*x{}", w, i))
        .collect::<Vec<_>>().join(" + "), capacity);
    println!("       x_j in {{0, 1}}\n");

    // Step 1: Solve LP relaxation
    println!("=== Step 1: LP Relaxation ===\n");
    let (lp_sol, lp_obj) = solve_lp_relaxation(&problem).expect("LP should be feasible");
    println!("LP optimal objective: {:.6} (maximization value: {:.6})\n", lp_obj, -lp_obj);
    print_knapsack_solution(&lp_sol, &problem.var_names, &weights, &values);
    println!();

    if is_integer_feasible(&lp_sol, &problem.var_types) {
        println!("LP solution is already integer-feasible. No cuts needed.");
        return;
    }
    println!("LP solution is FRACTIONAL — cover cuts can tighten it.\n");

    // Step 2: Find a minimal cover
    println!("=== Step 2: Find Minimal Cover ===\n");
    let cover = find_minimal_cover(&weights, capacity, &lp_sol);

    match cover {
        None => {
            println!("No violated cover found by greedy heuristic.");
            println!("(This can happen — the heuristic is not exact.)");
        }
        Some(ref cover_items) => {
            let cover_weight: f64 = cover_items.iter().map(|&j| weights[j]).sum();
            let cover_lp_sum: f64 = cover_items.iter().map(|&j| lp_sol[j]).sum();

            println!("Found cover: {:?}", cover_items);
            println!(
                "  Items:       {}",
                cover_items.iter().map(|j| format!("x{}", j)).collect::<Vec<_>>().join(", ")
            );
            println!("  Total weight: {:.1} > {:.1} (capacity)  ✓ (exceeds capacity)", cover_weight, capacity);
            println!("  |C| = {}", cover_items.len());
            println!(
                "  LP sum:       {:.4} > {:.1} (|C| - 1)  → {}",
                cover_lp_sum,
                (cover_items.len() - 1) as f64,
                if cover_lp_sum > (cover_items.len() - 1) as f64 + EQ_TOL {
                    "VIOLATED ✓"
                } else {
                    "satisfied (not violated)"
                }
            );
            println!();

            // Step 3: Generate cover inequality
            println!("=== Step 3: Generate Cover Inequality ===\n");
            let (cut_coeffs, cut_rhs) = generate_cover_inequality(cover_items, n);

            let cut_terms: Vec<String> = cut_coeffs.iter().enumerate()
                .filter(|(_, &c)| c > ZERO_TOL)
                .map(|(j, _)| format!("x{}", j))
                .collect();
            println!("Cover inequality: {} <= {:.0}", cut_terms.join(" + "), cut_rhs);
            println!("  (at most {} of {} items can be selected)\n",
                cut_rhs as usize, cover_items.len());

            // Step 4: Add the cover cut and re-solve
            println!("=== Step 4: Add Cut and Re-solve ===\n");
            let mut problem_with_cut = problem.clone();
            problem_with_cut.constraints.push(LinearConstraint {
                coeffs: cut_coeffs,
                sense: ConstraintSense::Le,
                rhs: cut_rhs,
            });

            match solve_lp_relaxation(&problem_with_cut) {
                Ok((new_sol, new_obj)) => {
                    println!("New LP objective: {:.6} (maximization: {:.6})", new_obj, -new_obj);
                    println!("Old LP objective: {:.6} (maximization: {:.6})\n", lp_obj, -lp_obj);
                    print_knapsack_solution(&new_sol, &problem.var_names, &weights, &values);
                    println!();

                    // Since we minimize -values, a higher (less negative) objective = tighter bound
                    let improvement = new_obj - lp_obj;
                    if improvement.abs() > EQ_TOL {
                        println!("Bound tightened by {:.6} (LP bound moved toward MIP optimum)", improvement.abs());
                    } else {
                        println!("No bound improvement (cut was not strong enough at this LP vertex).");
                    }

                    if is_integer_feasible(&new_sol, &problem.var_types) {
                        println!("New solution is integer-feasible! Cover cut closed the gap.");
                    } else {
                        println!("Solution still fractional. More cuts or branching needed.");
                    }
                }
                Err(e) => {
                    println!("Error re-solving LP with cover cut: {}", e);
                }
            }
        }
    }

    println!("\n=== Phase 2 Complete ===");
}
