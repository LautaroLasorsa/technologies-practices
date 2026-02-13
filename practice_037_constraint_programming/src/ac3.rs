// Phase 2: AC-3 Arc Consistency Algorithm
//
// This binary implements the AC-3 algorithm (Mackworth, 1977) — the standard
// propagation algorithm for binary CSPs. AC-3 maintains a queue of constraint
// arcs, revises each one, and re-enqueues affected arcs when domains change.
//
// AC-3 is the workhorse of CP: it runs in O(e * d^3) and is called at every
// node of backtracking search (Phase 3) to propagate the consequences of
// each variable assignment.

#[path = "common.rs"]
mod common;

use common::*;
use std::collections::{HashMap, VecDeque};

/// Revise the domain of variable `var_i` w.r.t. variable `var_j`.
/// (Same as Phase 1 — copied here for self-contained binary.)
///
/// For each value v in D_i, check if any w in D_j satisfies the constraint.
/// Remove v if no support exists. Return true if D_i changed.
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

/// AC-3: enforce arc consistency on the entire CSP.
///
/// Returns true if all arcs are consistent (no domain became empty).
/// Returns false if some domain became empty (inconsistency — no solution).
///
/// After AC-3, every value in every domain has at least one support in
/// every constraining neighbor's domain. This may fully solve the CSP
/// (all domains size 1) or just reduce domains (search needed).
///
/// # TODO(human): Implement the AC-3 algorithm
///
/// AC-3 (Arc Consistency Algorithm #3) by Mackworth (1977):
///
///   1. Initialize a queue with ALL constraint arcs (both directions).
///      For each constraint c with (var_i, var_j):
///        - Enqueue (var_i, var_j, constraint_index)
///        - Enqueue (var_j, var_i, constraint_index)
///      This ensures we check support in both directions.
///
///   2. While queue is not empty:
///      Dequeue (i, j, ci).
///      if revise(domains, i, j, constraint[ci].check):
///        if domains[i] is empty:
///          return false  // Inconsistency detected — no solution
///        // Domain of i changed → neighbors of i might lose support.
///        // Re-enqueue all arcs (k, i) where k != j:
///        for each (constraint_idx, k) in neighbor_map[i] where k != j:
///          enqueue (k, i, constraint_idx)
///
///   3. Return true (all arcs consistent, no empty domains)
///
/// Important details:
///   - Use a VecDeque<(usize, usize, usize)> as the arc queue.
///     Each entry is (var_to_revise, other_var, constraint_index).
///   - Pre-build the neighbor map using csp.build_neighbor_map() to
///     efficiently find which arcs to re-enqueue.
///   - The `revise` function operates on a `&mut [Domain]` slice.
///     Extract domains from csp.variables before the loop:
///       let mut domains: Vec<Domain> = csp.variables.iter().map(|v| v.domain.clone()).collect();
///     Then write them back after AC-3 completes.
///   - The constraint's check function is accessed via csp.constraints[ci].check.
///     IMPORTANT: When re-enqueuing, you need to determine the correct
///     constraint index for the arc (k, i). The neighbor_map gives you this.
///   - Count the total number of revise() calls in `stats.propagations`.
///
/// Complexity: O(e * d^3) where e = number of arcs, d = max domain size.
/// Each arc is enqueued at most O(d) times, and each revise costs O(d^2).
pub fn ac3(csp: &mut CSP, stats: &mut CSPResult) -> bool {
    todo!("TODO(human): Implement AC-3 — arc consistency propagation algorithm")
}

// ---------------------------------------------------------------------------
// Example problems for Phase 2
// ---------------------------------------------------------------------------

/// Map coloring: 3 regions, 3 colors, each pair must differ.
/// A, B, C in {0, 1, 2}, A != B, B != C, A != C.
/// AC-3 won't fully solve this (multiple solutions), but will maintain domains.
fn map_coloring_3() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("A", 0, 2));
    csp.add_variable(Variable::new_range("B", 0, 2));
    csp.add_variable(Variable::new_range("C", 0, 2));
    csp.add_not_equal(0, 1, "A != B");
    csp.add_not_equal(1, 2, "B != C");
    csp.add_not_equal(0, 2, "A != C");
    csp
}

/// X + Y = 10, X in {1..9}, Y in {1..9}.
/// AC-3 will verify arc consistency (no reductions — every value has support).
fn sum_example() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("X", 1, 9));
    csp.add_variable(Variable::new_range("Y", 1, 9));
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "X + Y = 10".to_string(),
        check: Box::new(|x, y| x + y == 10),
    });
    csp
}

/// X + Y = 10, X >= 7. After AC-3: X in {7,8,9}, Y in {1,2,3}.
fn constrained_sum() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("X", 7, 9));
    csp.add_variable(Variable::new_range("Y", 1, 9));
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "X + Y = 10".to_string(),
        check: Box::new(|x, y| x + y == 10),
    });
    csp
}

/// Chain: A < B, B < C, C < D. All in {1..6}.
/// AC-3 will reduce: A loses 4,5,6 (can't be < anything if at max).
/// B loses 1 and 6. C loses 1,2 and 6. D loses 1,2,3.
fn chain_ordering() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("A", 1, 6));
    csp.add_variable(Variable::new_range("B", 1, 6));
    csp.add_variable(Variable::new_range("C", 1, 6));
    csp.add_variable(Variable::new_range("D", 1, 6));
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "A < B".to_string(),
        check: Box::new(|a, b| a < b),
    });
    csp.add_constraint(BinaryConstraint {
        var_i: 1,
        var_j: 2,
        name: "B < C".to_string(),
        check: Box::new(|b, c| b < c),
    });
    csp.add_constraint(BinaryConstraint {
        var_i: 2,
        var_j: 3,
        name: "C < D".to_string(),
        check: Box::new(|c, d| c < d),
    });
    csp
}

/// Infeasible: X in {1,2}, Y in {1,2}, X != Y, X > Y.
/// AC-3 detects: X=1 needs Y<1 → no support. X=2 needs Y<2 and Y!=2 → Y=1.
/// Then Y=1, X=2 is the only option. But wait: X!=Y is satisfied (2!=1).
/// Actually this IS feasible: X=2, Y=1. Let's make it truly infeasible:
/// X in {1,2}, Y in {1,2}, X = Y, X != Y. Contradiction.
fn infeasible_example() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_set("X", &[1, 2]));
    csp.add_variable(Variable::new_set("Y", &[1, 2]));
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "X = Y".to_string(),
        check: Box::new(|x, y| x == y),
    });
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "X != Y".to_string(),
        check: Box::new(|x, y| x != y),
    });
    csp
}

/// 4-Queens as a quick test: 4 variables (rows), domains {0..3} (columns).
/// Constraints: queens can't share column or diagonal.
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
                check: Box::new(move |qi, qj| {
                    qi != qj && (qi - qj).abs() != diff
                }),
            });
        }
    }
    csp
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Phase 2: AC-3 Arc Consistency Algorithm                ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // --- Example 1: X + Y = 10 (full domains) ---
    println!("━━━ Example 1: X + Y = 10 (full domains) ━━━\n");
    let mut csp = sum_example();
    let mut stats = CSPResult::new();
    println!("  Before AC-3:");
    print_domains(&csp);

    let consistent = ac3(&mut csp, &mut stats);
    println!("\n  After AC-3: consistent = {}", consistent);
    print_domains(&csp);
    println!("  Propagations (revise calls): {}", stats.propagations);
    println!("  Expected: no reductions (every value in X has support in Y)\n");

    // --- Example 2: X + Y = 10, X >= 7 ---
    println!("━━━ Example 2: X + Y = 10, X in {{7,8,9}} ━━━\n");
    let mut csp = constrained_sum();
    let mut stats = CSPResult::new();
    println!("  Before AC-3:");
    print_domains(&csp);

    let consistent = ac3(&mut csp, &mut stats);
    println!("\n  After AC-3: consistent = {}", consistent);
    print_domains(&csp);
    println!("  Propagations: {}", stats.propagations);
    println!("  Expected: Y reduced to {{1, 2, 3}}\n");

    // --- Example 3: Chain ordering A < B < C < D ---
    println!("━━━ Example 3: Chain A < B < C < D, all in {{1..6}} ━━━\n");
    let mut csp = chain_ordering();
    let mut stats = CSPResult::new();
    println!("  Before AC-3:");
    print_domains(&csp);

    let consistent = ac3(&mut csp, &mut stats);
    println!("\n  After AC-3: consistent = {}", consistent);
    print_domains(&csp);
    println!("  Propagations: {}", stats.propagations);
    println!("  Expected: A={{1,2,3}}, B={{2,3,4}}, C={{3,4,5}}, D={{4,5,6}}\n");

    // --- Example 4: Map coloring (3 regions, 3 colors) ---
    println!("━━━ Example 4: Map Coloring (A!=B, B!=C, A!=C) ━━━\n");
    let mut csp = map_coloring_3();
    let mut stats = CSPResult::new();
    println!("  Before AC-3:");
    print_domains(&csp);

    let consistent = ac3(&mut csp, &mut stats);
    println!("\n  After AC-3: consistent = {}", consistent);
    print_domains(&csp);
    println!("  Propagations: {}", stats.propagations);
    println!("  Expected: all domains remain {{0,1,2}} (all-different with 3 vars, 3 colors");
    println!("  is arc-consistent but not solved — needs search)\n");

    // --- Example 5: Infeasible (X = Y AND X != Y) ---
    println!("━━━ Example 5: Infeasible (X = Y AND X != Y) ━━━\n");
    let mut csp = infeasible_example();
    let mut stats = CSPResult::new();
    println!("  Before AC-3:");
    print_domains(&csp);

    let consistent = ac3(&mut csp, &mut stats);
    println!("\n  After AC-3: consistent = {}", consistent);
    print_domains(&csp);
    println!("  Propagations: {}", stats.propagations);
    println!("  Expected: consistent = false (domains become empty)\n");

    // --- Example 6: 4-Queens ---
    println!("━━━ Example 6: 4-Queens (AC-3 only, no search) ━━━\n");
    let mut csp = four_queens();
    let mut stats = CSPResult::new();
    println!("  Before AC-3:");
    print_domains(&csp);

    let consistent = ac3(&mut csp, &mut stats);
    println!("\n  After AC-3: consistent = {}", consistent);
    print_domains(&csp);
    println!("  Propagations: {}", stats.propagations);
    if csp.is_solved() {
        println!("  AC-3 alone solved the 4-Queens! (Unlikely — usually needs search.)");
    } else {
        println!("  AC-3 reduced domains but didn't solve it.");
        println!("  Backtracking search (Phase 3) will find the solution.");
    }

    println!("\n━━━ Phase 2 Complete ━━━");
    println!("  You implemented AC-3: the standard propagation algorithm for CP.");
    println!("  AC-3 enforces arc consistency in O(e*d^3) time.");
    println!("  Next: Phase 3 combines AC-3 with backtracking search (MAC).");
}
