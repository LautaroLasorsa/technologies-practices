// Phase 1: Domain Representation & Revise
//
// This binary introduces the core data structures of Constraint Programming:
// variables with finite domains, binary constraints, and the REVISE operation.
//
// REVISE is the atomic building block of arc consistency. Given a constraint
// arc (X_i, X_j), it removes values from D_i that have no "support" — no
// compatible value in D_j that satisfies the constraint.
//
// We also implement node consistency: filtering each variable's domain against
// unary-style constraints (constraints where one variable is already fixed).

#[path = "common.rs"]
mod common;

use common::*;

/// Revise the domain of variable `var_i` with respect to variable `var_j`
/// under the given constraint check function.
///
/// For each value v in D_i, check if there exists at least one value w in D_j
/// such that check(v, w) is satisfied. If no such w exists (v has no "support"),
/// remove v from D_i.
///
/// Returns true if D_i was modified (any value removed).
///
/// Note: This operates on a mutable slice of Domains rather than on &mut CSP
/// to avoid borrow checker issues (we need to read D_j while modifying D_i,
/// and also need to read the constraint's check function). In AC-3 (Phase 2),
/// domains are extracted from the CSP before the loop and written back after.
///
/// # TODO(human): Implement the REVISE operation
///
/// This is the building block of arc consistency — the most fundamental
/// operation in Constraint Programming.
///
/// "Support" = a value in the other domain that makes the constraint happy.
/// If a value has no support, it can NEVER be part of a solution for this
/// constraint, so removing it is sound (doesn't eliminate any solutions).
///
/// Implementation steps:
///   1. Copy the values in D_i to a temporary Vec (you can't mutate the
///      domain while iterating over it):
///        let values_i: Vec<i32> = domains[var_i].iter().copied().collect();
///   2. For each value v in the copy:
///      - Check if ANY value w in D_j satisfies check(v, w).
///      - If no w satisfies, remove v from D_i and set revised = true.
///   3. Return revised.
///
/// Use .iter().any(|&w| ...) to check for support efficiently —
/// you can stop as soon as one support is found.
pub fn revise(
    domains: &mut [Domain],
    var_i: usize,
    var_j: usize,
    check: &dyn Fn(i32, i32) -> bool,
) -> bool {
    todo!("TODO(human): Implement REVISE — remove unsupported values from D_i")
}

/// Apply node consistency: for each variable, remove values that violate
/// constraints where the OTHER variable's domain is already a singleton.
///
/// This is a simplified form of propagation: if X_j = {5} (only one value),
/// then for constraint (X_i, X_j), we can immediately remove any value v
/// from D_i where constraint(v, 5) is false.
///
/// # TODO(human): Implement node consistency
///
/// Node consistency is the simplest form of constraint propagation.
/// When a variable has been assigned (domain size 1), we can use its
/// constraints to filter other variables' domains immediately.
///
/// Implementation steps:
///   1. Extract domains from csp.variables into a Vec<Domain>.
///   2. Loop with a `changed` flag (repeat until no more changes):
///      For each constraint c in csp.constraints:
///        - If domains[c.var_j] has exactly one value w:
///            filter domains[c.var_i], keeping only v where (c.check)(v, w).
///        - If domains[c.var_i] has exactly one value v:
///            filter domains[c.var_j], keeping only w where (c.check)(v, w).
///        - If any domain changed, set changed = true.
///   3. Write domains back to csp.variables.
///
/// To filter a BTreeSet in place: collect values to remove into a Vec,
/// then remove them. Or rebuild with .retain() (BTreeSet::retain is available).
///
/// Note: This is weaker than AC-3 (Phase 2) because it only propagates
/// from singleton domains. AC-3 also removes values that lack support
/// even when both domains have multiple values.
pub fn node_consistency(csp: &mut CSP) {
    todo!("TODO(human): Implement node consistency — propagate from singleton domains")
}

// ---------------------------------------------------------------------------
// Example problems for Phase 1
// ---------------------------------------------------------------------------

/// Simple arithmetic CSP: X + Y = 10, X in {1..9}, Y in {1..9}.
fn arithmetic_example() -> CSP {
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

/// Inequality CSP: A != B, A in {1,2,3}, B in {1,2,3}, plus B = 2 (fixed).
fn inequality_example() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_set("A", &[1, 2, 3]));
    csp.add_variable(Variable::new_set("B", &[2])); // B is fixed to 2
    csp.add_not_equal(0, 1, "A != B");
    csp
}

/// Ordering CSP: P < Q, Q < R, P in {1..5}, Q in {1..5}, R in {1..5}.
fn ordering_example() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("P", 1, 5));
    csp.add_variable(Variable::new_range("Q", 1, 5));
    csp.add_variable(Variable::new_range("R", 1, 5));
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "P < Q".to_string(),
        check: Box::new(|p, q| p < q),
    });
    csp.add_constraint(BinaryConstraint {
        var_i: 1,
        var_j: 2,
        name: "Q < R".to_string(),
        check: Box::new(|q, r| q < r),
    });
    csp
}

/// More constrained: X + Y = 10, X >= 7. X in {7,8,9}, Y in {1..9}.
/// After revising Y w.r.t. X: Y in {1,2,3}.
fn constrained_arithmetic() -> CSP {
    let mut csp = CSP::new();
    csp.add_variable(Variable::new_range("X", 7, 9)); // X >= 7
    csp.add_variable(Variable::new_range("Y", 1, 9));
    csp.add_constraint(BinaryConstraint {
        var_i: 0,
        var_j: 1,
        name: "X + Y = 10".to_string(),
        check: Box::new(|x, y| x + y == 10),
    });
    csp
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Phase 1: Domain Representation & REVISE                ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // --- Example 1: REVISE on X + Y = 10 with X >= 7 ---
    println!("━━━ Example 1: REVISE on X + Y = 10 (X >= 7) ━━━\n");
    println!("  X in {{7, 8, 9}}, Y in {{1..9}}");
    println!("  Constraint: X + Y = 10\n");

    let mut csp = constrained_arithmetic();
    let mut domains = csp.save_domains();
    println!("  Before REVISE(Y, X):");
    print_domains(&csp);

    // Revise Y with respect to X: remove values from Y that have no support in X.
    // Since X in {7,8,9} and X + Y = 10, Y must be in {1,2,3}.
    // Note: constraint is X + Y = 10 with var_i=X(0), var_j=Y(1).
    // We want to revise Y(1) w.r.t. X(0). Since Y is var_j in the constraint,
    // we swap: check(y, x) should be check(x, y) => use |y, x| x + y == 10.
    let check = &csp.constraints[0].check;
    let changed = revise(&mut domains, 1, 0, &|y, x| check(x, y));
    csp.restore_domains(&domains);

    println!("\n  After REVISE(Y, X): changed = {}", changed);
    print_domains(&csp);
    println!("  Expected: Y in {{1, 2, 3}} (only values with support in X)\n");

    // --- Example 2: REVISE on A != B where B = {2} ---
    println!("━━━ Example 2: REVISE on A != B (B fixed to 2) ━━━\n");

    let mut csp = inequality_example();
    let mut domains = csp.save_domains();
    println!("  Before REVISE(A, B):");
    print_domains(&csp);

    let check = &csp.constraints[0].check;
    let changed = revise(&mut domains, 0, 1, &|a, b| check(a, b));
    csp.restore_domains(&domains);

    println!("\n  After REVISE(A, B): changed = {}", changed);
    print_domains(&csp);
    println!("  Expected: A in {{1, 3}} (value 2 has no support: 2 != 2 is false)\n");

    // --- Example 3: Node consistency on P < Q, Q < R ---
    println!("━━━ Example 3: Node Consistency on P < Q < R ━━━\n");

    let mut csp = ordering_example();
    println!("  Before node consistency:");
    print_domains(&csp);
    print_constraints(&csp);

    node_consistency(&mut csp);
    println!("\n  After node consistency:");
    print_domains(&csp);
    println!("  Note: Node consistency alone may not reduce much here");
    println!("  (it only propagates from singletons). AC-3 (Phase 2) will do more.\n");

    // --- Example 4: Full REVISE chain on X + Y = 10 ---
    println!("━━━ Example 4: REVISE both directions on X + Y = 10 ━━━\n");

    let mut csp = arithmetic_example();
    let mut domains = csp.save_domains();
    println!("  Initial: X in {{1..9}}, Y in {{1..9}}, constraint: X + Y = 10");
    print_domains(&csp);

    // Revise X w.r.t. Y (forward: X is var_i)
    let check = &csp.constraints[0].check;
    let c1 = revise(&mut domains, 0, 1, &|x, y| check(x, y));
    csp.restore_domains(&domains);
    println!("\n  After REVISE(X, Y): changed = {}", c1);
    print_domains(&csp);

    // Revise Y w.r.t. X (reverse: Y is var_j, so swap args)
    let mut domains = csp.save_domains();
    let check = &csp.constraints[0].check;
    let c2 = revise(&mut domains, 1, 0, &|y, x| check(x, y));
    csp.restore_domains(&domains);
    println!("\n  After REVISE(Y, X): changed = {}", c2);
    print_domains(&csp);
    println!("  Both domains remain {{1..9}} because every value has support:");
    println!("  e.g., X=1 is supported by Y=9 (1+9=10), X=9 by Y=1 (9+1=10).\n");

    println!("━━━ Phase 1 Complete ━━━");
    println!("  You implemented REVISE (the building block of AC-3)");
    println!("  and node consistency (propagation from singletons).");
    println!("  Next: Phase 2 assembles REVISE into the full AC-3 algorithm.");
}
