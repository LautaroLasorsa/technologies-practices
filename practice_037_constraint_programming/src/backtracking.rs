// Phase 3: Backtracking Search with AC-3 + MRV
//
// This binary combines AC-3 propagation with backtracking search to create
// a complete CSP solver. The algorithm is called MAC (Maintaining Arc
// Consistency): at each search node, after assigning a value to a variable,
// run AC-3 to propagate consequences. MRV (Minimum Remaining Values) picks
// the most constrained variable to branch on.
//
// MAC + MRV is the standard algorithm in CP solvers. It transforms brute-force
// backtracking into a practical solver for problems like Sudoku and N-Queens.

#[path = "common.rs"]
mod common;

use common::*;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// AC-3 (provided — same as Phase 2 solution)
// ---------------------------------------------------------------------------

/// Revise the domain of var_i w.r.t. var_j. Remove unsupported values.
fn revise(
    domains: &mut [Domain],
    var_i: usize,
    var_j: usize,
    check: &dyn Fn(i32, i32) -> bool,
) -> bool {
    let values_i: Vec<i32> = domains[var_i].iter().copied().collect();
    let mut revised = false;
    for v in values_i {
        let has_support = domains[var_j].iter().any(|&w| check(v, w));
        if !has_support {
            domains[var_i].remove(&v);
            revised = true;
        }
    }
    revised
}

/// AC-3: enforce arc consistency. Returns false if any domain becomes empty.
fn ac3(csp: &mut CSP, stats: &mut CSPResult) -> bool {
    let neighbor_map = csp.build_neighbor_map();
    let mut domains: Vec<Domain> = csp.variables.iter().map(|v| v.domain.clone()).collect();

    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
    for (ci, c) in csp.constraints.iter().enumerate() {
        queue.push_back((c.var_i, c.var_j, ci));
        queue.push_back((c.var_j, c.var_i, ci));
    }

    while let Some((i, j, ci)) = queue.pop_front() {
        stats.propagations += 1;
        let check = &csp.constraints[ci].check;
        // Determine which direction to check: if i is var_i of constraint, use check(v, w)
        // If i is var_j of constraint, we need check(w, v) — i.e., swap arguments.
        let is_forward = csp.constraints[ci].var_i == i;

        let revised = if is_forward {
            revise(&mut domains, i, j, &|a, b| check(a, b))
        } else {
            revise(&mut domains, i, j, &|a, b| check(b, a))
        };

        if revised {
            if domains[i].is_empty() {
                // Write back and signal failure
                for (vi, var) in csp.variables.iter_mut().enumerate() {
                    var.domain = domains[vi].clone();
                }
                return false;
            }
            // Re-enqueue arcs (k, i) for all neighbors k != j
            if let Some(neighbors) = neighbor_map.get(&i) {
                for &(nci, k) in neighbors {
                    if k != j {
                        queue.push_back((k, i, nci));
                    }
                }
            }
        }
    }

    // Write back reduced domains
    for (vi, var) in csp.variables.iter_mut().enumerate() {
        var.domain = domains[vi].clone();
    }
    true
}

// ---------------------------------------------------------------------------
// Variable selection: MRV (Fail-First)
// ---------------------------------------------------------------------------

/// Select the unassigned variable with the smallest domain (MRV heuristic).
///
/// An "unassigned" variable is one with domain size > 1 (not yet fixed to
/// a single value). Among all such variables, return the index of the one
/// with the fewest remaining values.
///
/// # TODO(human): Implement MRV variable selection
///
/// Minimum Remaining Values (MRV) / Fail-First:
///   Pick the unassigned variable with the SMALLEST domain.
///   Intuition: if a variable has only 2 values left, it's the most
///   constrained — try it now so we fail fast if there's no solution.
///   Variables with domain size 1 are already assigned (skip them).
///
///   Break ties by variable index (smallest index first — arbitrary
///   but deterministic).
///
/// Implementation:
///   1. Iterate over csp.variables with enumerate().
///   2. Filter: only variables where domain.len() > 1.
///   3. Find the one with minimum domain.len().
///   4. Return its index.
///   5. If no unassigned variable exists, return None.
///
/// Use .filter().min_by_key() or a simple loop with tracking.
pub fn select_unassigned_variable(csp: &CSP) -> Option<usize> {
    todo!("TODO(human): Implement MRV — return index of variable with smallest domain > 1")
}

// ---------------------------------------------------------------------------
// Backtracking search with MAC
// ---------------------------------------------------------------------------

/// Recursive backtracking search with AC-3 propagation (MAC).
///
/// Returns true if a solution is found (all domains are singletons).
///
/// # TODO(human): Implement backtracking search with AC-3
///
/// Backtracking search with constraint propagation (MAC):
///
///   1. If all domains are singletons (csp.is_solved()): return true.
///
///   2. var = select_unassigned_variable(csp)  // MRV
///      If None: return csp.is_solved() (shouldn't happen if logic is correct).
///
///   3. Copy the domain of var (the values to try): values = csp.variables[var].domain.clone()
///
///   4. For each value v in values:
///      a. Save current domains: saved = csp.save_domains()
///      b. Reduce domain of var to {v}: csp.variables[var].domain = single {v}
///      c. stats.nodes_explored += 1
///      d. Run AC-3 to propagate: let consistent = ac3(csp, stats)
///      e. If consistent AND backtrack(csp, stats) returns true:
///           return true  // Solution found down this path
///      f. Restore domains: csp.restore_domains(&saved)  // Undo everything
///
///   5. Return false (no value for this variable leads to a solution — backtrack)
///
/// The domain save/restore is crucial: AC-3 modifies domains throughout
/// the CSP, not just the assigned variable. When we backtrack, we must
/// undo ALL propagated domain reductions, not just the assignment.
///
/// The AC-3 call after each assignment is the key optimization:
/// without it, failures are detected only when we try to assign a variable
/// and no value works. With AC-3, failures are detected immediately when
/// propagation empties a domain, pruning entire subtrees.
pub fn backtrack(csp: &mut CSP, stats: &mut CSPResult) -> bool {
    todo!("TODO(human): Implement MAC backtracking — assign, propagate, recurse or undo")
}

/// Solve a CSP using backtracking search with AC-3 propagation and MRV.
pub fn solve(csp: &mut CSP) -> CSPResult {
    let mut stats = CSPResult::new();

    // Initial AC-3 propagation (before any search)
    println!("  Running initial AC-3 propagation...");
    let consistent = ac3(csp, &mut stats);
    if !consistent {
        println!("  AC-3 detected inconsistency before search!");
        return stats;
    }

    if csp.is_solved() {
        println!("  AC-3 alone solved the CSP!");
        stats.solution = csp.extract_solution();
        return stats;
    }

    println!("  AC-3 reduced domains. Starting backtracking search...");

    // Run backtracking
    if backtrack(csp, &mut stats) {
        stats.solution = csp.extract_solution();
    }

    stats
}

// ---------------------------------------------------------------------------
// Example problems
// ---------------------------------------------------------------------------

/// 4-Queens: 4 variables (rows), domain {0..3} (columns).
fn four_queens() -> CSP {
    let n = 4;
    let mut csp = CSP::new();
    for i in 0..n {
        csp.add_variable(Variable::new_range(&format!("Q{}", i), 0, (n - 1) as i32));
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (j - i) as i32;
            csp.add_constraint(BinaryConstraint {
                var_i: i,
                var_j: j,
                name: format!("Q{} vs Q{}", i, j),
                check: Box::new(move |qi, qj| qi != qj && (qi - qj).abs() != diff),
            });
        }
    }
    csp
}

/// Small graph coloring: triangle (K3), 3 colors.
fn triangle_coloring() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("V0", 0, 2));
    csp.add_variable(Variable::new_range("V1", 0, 2));
    csp.add_variable(Variable::new_range("V2", 0, 2));
    csp.add_not_equal(0, 1, "V0 != V1");
    csp.add_not_equal(1, 2, "V1 != V2");
    csp.add_not_equal(0, 2, "V0 != V2");
    csp
}

/// Graph coloring: K4 with 3 colors (infeasible — K4 needs 4 colors).
fn k4_three_colors() -> CSP {
    let mut csp = CSP::new();
    for i in 0..4 {
        csp.add_variable(Variable::new_range(&format!("V{}", i), 0, 2));
    }
    for i in 0..4 {
        for j in (i + 1)..4 {
            csp.add_not_equal(i, j, &format!("V{} != V{}", i, j));
        }
    }
    csp
}

/// Chain constraints: A < B < C < D < E, all in {1..8}.
fn chain_example() -> CSP {
    let mut csp = CSP::new();
    for name in ["A", "B", "C", "D", "E"] {
        csp.add_variable(Variable::new_range(name, 1, 8));
    }
    for i in 0..4 {
        let ni = csp.variables[i].name.clone();
        let nj = csp.variables[i + 1].name.clone();
        csp.add_constraint(BinaryConstraint {
            var_i: i,
            var_j: i + 1,
            name: format!("{} < {}", ni, nj),
            check: Box::new(|a, b| a < b),
        });
    }
    csp
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Phase 3: Backtracking Search with AC-3 + MRV          ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // --- 4-Queens ---
    println!("━━━ Problem 1: 4-Queens ━━━\n");
    let mut csp = four_queens();
    println!("  4 queens on a 4x4 board, no two attacking.");
    println!("  Variables: Q0..Q3 (row), domains: {{0..3}} (column)\n");
    print_domains(&csp);

    let result = solve(&mut csp);
    println!("\n  Nodes explored: {}", result.nodes_explored);
    println!("  Propagations: {}", result.propagations);
    match &result.solution {
        Some(sol) => {
            print_solution(&csp, sol);
            println!("\n  Board:");
            print_queens_board(4, sol);
        }
        None => println!("  No solution found!"),
    }

    // --- Triangle coloring (3 vertices, 3 colors) ---
    println!("\n━━━ Problem 2: Triangle Coloring (K3, 3 colors) ━━━\n");
    let mut csp = triangle_coloring();
    println!("  3-vertex complete graph, 3 colors.");
    print_domains(&csp);

    let result = solve(&mut csp);
    println!("\n  Nodes explored: {}", result.nodes_explored);
    println!("  Propagations: {}", result.propagations);
    match &result.solution {
        Some(sol) => print_solution(&csp, sol),
        None => println!("  No solution found!"),
    }

    // --- K4 with 3 colors (infeasible) ---
    println!("\n━━━ Problem 3: K4 with 3 Colors (infeasible) ━━━\n");
    let mut csp = k4_three_colors();
    println!("  4-vertex complete graph, only 3 colors (needs 4 — should fail).");
    print_domains(&csp);

    let result = solve(&mut csp);
    println!("\n  Nodes explored: {}", result.nodes_explored);
    println!("  Propagations: {}", result.propagations);
    match &result.solution {
        Some(sol) => {
            println!("  Unexpected solution found!");
            print_solution(&csp, sol);
        }
        None => println!("  Correctly detected: NO SOLUTION (K4 needs >= 4 colors)."),
    }

    // --- Chain A < B < C < D < E ---
    println!("\n━━━ Problem 4: Chain A < B < C < D < E, all in {{1..8}} ━━━\n");
    let mut csp = chain_example();
    print_domains(&csp);

    let result = solve(&mut csp);
    println!("\n  Nodes explored: {}", result.nodes_explored);
    println!("  Propagations: {}", result.propagations);
    match &result.solution {
        Some(sol) => {
            print_solution(&csp, sol);
            println!(
                "  Verify: {} < {} < {} < {} < {}",
                sol[0], sol[1], sol[2], sol[3], sol[4]
            );
        }
        None => println!("  No solution found!"),
    }

    println!("\n━━━ Phase 3 Complete ━━━");
    println!("  You implemented MAC (Maintaining Arc Consistency) with MRV.");
    println!("  This is the standard algorithm in CP solvers.");
    println!("  Next: Phase 4 applies the solver to Sudoku, 8-Queens, and graph coloring.");
}
