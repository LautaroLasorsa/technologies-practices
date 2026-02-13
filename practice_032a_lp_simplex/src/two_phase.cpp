// ============================================================================
// Phase 4: Two-Phase Simplex Method
// ============================================================================
//
// The basic simplex algorithm (Phases 1-3) requires an initial BFS, which is
// easy when all constraints are "<=" (slack variables provide an identity
// basis). But many real LPs have ">=" or "=" constraints where no obvious
// initial BFS exists.
//
// The TWO-PHASE SIMPLEX METHOD solves this by:
//   Phase I:  Find a feasible starting point (or prove infeasibility)
//   Phase II: Optimize the original objective from that starting point
//
// Phase I introduces ARTIFICIAL VARIABLES (one per constraint that lacks a
// natural slack) and minimizes their sum. If the minimum is 0, we found a
// BFS for the original problem. If > 0, the original problem is infeasible.
//
// Test problems include:
//   1. A diet problem with ">=" constraints (minimum nutrition requirements)
//   2. A mixed-constraint problem (some <=, some >=, some =)
//   3. An infeasible problem (to verify detection)
//
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>

// ─── Data types ──────────────────────────────────────────────────────────────

struct Tableau {
    Eigen::MatrixXd data;
    std::vector<int> basis;
    int m;
    int n;
};

enum class SimplexStatus {
    OPTIMAL,
    UNBOUNDED,
    INFEASIBLE,
    MAX_ITERATIONS
};

struct SimplexResult {
    SimplexStatus status;
    Eigen::VectorXd solution;
    double objective_value;
    int iterations;
};

// ─── Provided: utilities ─────────────────────────────────────────────────────

void print_tableau(const Tableau& tab, const std::string& label = "") {
    if (!label.empty()) std::cout << "\n--- " << label << " ---\n";
    int cols = tab.n + 1;

    std::cout << std::setw(8) << "Basis";
    for (int j = 0; j < tab.n; ++j)
        std::cout << std::setw(10) << ("x" + std::to_string(j + 1));
    std::cout << std::setw(10) << "RHS" << "\n";
    std::cout << std::string(8 + cols * 10, '-') << "\n";

    std::cout << std::setw(8) << "z";
    for (int j = 0; j < cols; ++j)
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << tab.data(0, j);
    std::cout << "\n";

    for (int i = 1; i <= tab.m; ++i) {
        std::string lbl = "x" + std::to_string(tab.basis[i - 1] + 1);
        std::cout << std::setw(8) << lbl;
        for (int j = 0; j < cols; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << tab.data(i, j);
        std::cout << "\n";
    }
}

void print_result(const SimplexResult& result, int num_decision, const std::string& label = "") {
    if (!label.empty()) std::cout << "\n=== " << label << " ===\n";
    switch (result.status) {
        case SimplexStatus::OPTIMAL:
            std::cout << "Status: OPTIMAL\n";
            std::cout << "Iterations: " << result.iterations << "\n";
            std::cout << "Objective value: " << result.objective_value << "\n";
            std::cout << "Decision variables:\n";
            for (int j = 0; j < num_decision; ++j) {
                std::cout << "  x" << (j + 1) << " = " << std::fixed
                          << std::setprecision(4) << result.solution[j] << "\n";
            }
            break;
        case SimplexStatus::UNBOUNDED:
            std::cout << "Status: UNBOUNDED\n";
            break;
        case SimplexStatus::INFEASIBLE:
            std::cout << "Status: INFEASIBLE\n";
            std::cout << "The constraint system has no feasible solution.\n";
            break;
        case SimplexStatus::MAX_ITERATIONS:
            std::cout << "Status: MAX ITERATIONS\n";
            break;
    }
}

// ─── Provided: simplex primitives ────────────────────────────────────────────

int select_pivot_column(const Tableau& tab) {
    int best_col = -1;
    double best_val = -1e-10;
    for (int j = 0; j < tab.n; ++j) {
        if (tab.data(0, j) < best_val) {
            best_val = tab.data(0, j);
            best_col = j;
        }
    }
    return best_col;
}

int select_pivot_row(const Tableau& tab, int pivot_col) {
    int best_row = -1;
    double best_ratio = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= tab.m; ++i) {
        double coeff = tab.data(i, pivot_col);
        if (coeff > 1e-10) {
            double ratio = tab.data(i, tab.n) / coeff;
            if (ratio < best_ratio - 1e-12) {
                best_ratio = ratio;
                best_row = i;
            }
        }
    }
    return best_row;
}

void pivot(Tableau& tab, int pivot_row, int pivot_col) {
    tab.data.row(pivot_row) /= tab.data(pivot_row, pivot_col);
    for (int i = 0; i <= tab.m; ++i) {
        if (i == pivot_row) continue;
        double factor = tab.data(i, pivot_col);
        tab.data.row(i) -= factor * tab.data.row(pivot_row);
    }
    tab.basis[pivot_row - 1] = pivot_col;
}

/// Run simplex from a given tableau (used internally by both phases).
SimplexStatus run_simplex(Tableau& tab, int max_iter = 200) {
    for (int iter = 0; iter < max_iter; ++iter) {
        int col = select_pivot_column(tab);
        if (col == -1) return SimplexStatus::OPTIMAL;
        int row = select_pivot_row(tab, col);
        if (row == -1) return SimplexStatus::UNBOUNDED;
        pivot(tab, row, col);
    }
    return SimplexStatus::MAX_ITERATIONS;
}

// ─── TODO(human): Phase I — find an initial BFS ─────────────────────────────

/// Construct and solve the Phase I auxiliary problem to find an initial BFS.
///
/// TODO(human): Implement Phase I of the two-phase simplex method.
///
/// CONTEXT — WHY WE NEED PHASE I:
///   When a constraint is "a^T x >= b" or "a^T x = b", there is no natural
///   slack variable to serve as an initial basic variable. For ">=" we get
///   a surplus variable with coefficient -1, which can't be in a starting
///   basis (it would be negative). For "=" there's no slack at all.
///
///   Phase I creates a GUARANTEED feasible starting problem by adding
///   artificial variables a_i >= 0 to every constraint that needs one,
///   then minimizing their sum. If the minimum is 0, all artificials left
///   the basis and we have a valid BFS for the original problem.
///
/// WHAT YOU RECEIVE:
///   - A: the constraint matrix in STANDARD FORM (m x n), already includes
///     slack/surplus columns, with all b >= 0 (rows with negative b have
///     been multiplied by -1).
///   - b: the RHS vector (all entries >= 0).
///   - needs_artificial: a vector of booleans, one per constraint. It is
///     true if that constraint needs an artificial variable (i.e., it was
///     originally ">=" or "=", so no natural slack serves as basis).
///
/// STEP 1 — Count artificial variables.
///   Let num_art = number of true entries in needs_artificial.
///   The Phase I problem has n + num_art variables total.
///
/// STEP 2 — Build the Phase I tableau.
///   Create a (m+1) x (n + num_art + 1) matrix:
///     - Rows 1..m: copy A into the first n columns. For each constraint i
///       that needs an artificial, add a +1 in the artificial's column.
///     - RHS column: copy b.
///     - Row 0 (objective): the Phase I objective minimizes the sum of
///       artificial variables: min a_1 + a_2 + ... + a_k.
///       So the objective coefficients are 0 for original variables and
///       +1 for each artificial variable.
///
/// STEP 3 — Set up the initial basis.
///   For constraints with a natural slack (needs_artificial[i] == false),
///     basis[i] = the slack variable's column index.
///   For constraints needing an artificial (needs_artificial[i] == true),
///     basis[i] = the artificial variable's column index.
///
/// STEP 4 — FIX THE OBJECTIVE ROW (critical step!).
///   The initial objective row has +1 for artificial variables. But since
///   artificial variables are IN the basis, their reduced costs should be 0.
///   You must subtract the rows of basic artificial variables from row 0:
///     for each constraint i where basis[i] is an artificial:
///       tab.data.row(0) -= tab.data.row(i + 1)
///   This ensures the objective row correctly reflects reduced costs.
///   (This is equivalent to computing c_bar = c_B * B^{-1} * A - c.)
///
/// STEP 5 — Solve the Phase I LP.
///   Call run_simplex(tab). The optimal value is -tab.data(0, n + num_art).
///   If the optimal value > epsilon (e.g., 1e-8), the original LP is
///   INFEASIBLE — return an empty/invalid Tableau.
///
/// STEP 6 — Prepare the tableau for Phase II.
///   If feasible (Phase I optimal value ~ 0):
///     a) Drop the artificial variable columns from the tableau (keep only
///        the first n columns and the RHS column).
///     b) The basis vector and constraint rows remain valid.
///     c) Return this reduced tableau — it contains a valid BFS for the
///        original n-variable problem.
///
/// RETURN: A pair (bool, Tableau).
///   - bool = true if feasible, false if infeasible.
///   - Tableau = the Phase I result tableau (with artificials removed),
///     ready for Phase II. Only valid if bool == true.
///
/// HINT on identifying slack columns:
///   If you set up the standard form such that slack variables for <=
///   constraints are columns num_original, num_original+1, ..., then
///   for constraint i, if needs_artificial[i] is false, the basis variable
///   is whatever slack column corresponds to that constraint. You may
///   need to pass additional info or use a convention. A simple approach:
///   scan the A matrix for each constraint row to find a column with +1
///   that forms part of an identity submatrix (only that row has a nonzero).
std::pair<bool, Tableau> phase_one(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const std::vector<bool>& needs_artificial,
    const std::vector<int>& slack_basis_columns
) {
    // TODO(human): implement Phase I of the two-phase simplex method
    throw std::runtime_error("TODO(human): phase_one not implemented");
}

// ─── TODO(human): Two-phase simplex ─────────────────────────────────────────

/// Solve an arbitrary LP using the two-phase simplex method.
///
/// TODO(human): Implement the full two-phase simplex procedure.
///
/// INPUT FORMAT:
///   - c_obj: objective coefficients for the ORIGINAL decision variables only
///            (length = num_original). This is for MINIMIZATION.
///   - A_ineq: matrix of inequality/equality constraint coefficients
///             (m x num_original), one row per constraint.
///   - b_rhs: RHS vector (length m).
///   - con_types: constraint types, one per row: -1 for <=, 0 for =, +1 for >=.
///
/// STEP 1 — Convert to standard form (add slacks/surplus).
///   This is similar to Phase 1's to_standard_form(), but now you also
///   track which constraints need artificial variables:
///     - "<=" constraints: add slack (+1), slack is a natural basis variable.
///       needs_artificial = false.
///     - ">=" constraints: add surplus (-1), surplus CANNOT be basis.
///       needs_artificial = true.
///     - "=" constraints: no slack at all. needs_artificial = true.
///   Make sure b >= 0 (if b_i < 0, multiply row by -1 and flip type).
///   Build the full A matrix (m x n_total where n_total = num_original + num_slack).
///   Build the full c vector (original costs + 0 for slacks).
///
/// STEP 2 — Check if Phase I is needed.
///   If all constraints are "<=", every constraint has a natural slack basis
///   variable and we can skip Phase I entirely.
///   Otherwise, call phase_one() to get a feasible starting tableau.
///
/// STEP 3 — Phase II: optimize the original objective.
///   Take the tableau from Phase I (or build one if Phase I was skipped).
///   Replace the objective row (row 0) with the ORIGINAL objective:
///     - Set row 0 to zeros.
///     - For each variable j in 0..n-1: row 0, col j = c[j].
///     - RHS of row 0 = 0.
///   But basic variables must have zero reduced cost! So fix the objective:
///     for each basic variable basis[i] in row (i+1):
///       tab.data.row(0) -= c[basis[i]] * tab.data.row(i + 1)
///   Now run_simplex(tab) to find the optimal solution.
///
/// STEP 4 — Extract the solution.
///   Same as Phase 3: solution vector from basis, objective from row 0 RHS.
///
/// RETURN: SimplexResult with status, solution, objective_value, iterations.
///
/// MATHEMATICAL INSIGHT:
///   The two-phase method cleanly separates FEASIBILITY from OPTIMALITY.
///   Phase I answers: "Does a feasible solution exist?" by solving a
///   different (easier) LP. Phase II answers: "What is the best feasible
///   solution?" starting from the point Phase I found. This separation
///   is elegant and mirrors how commercial solvers work (though they use
///   more sophisticated techniques like crash starts and presolve).
///
/// ALTERNATIVE — BIG-M METHOD:
///   Instead of two phases, add artificial variables with a huge cost M
///   in the objective: min c^T x + M * sum(artificials). If M is large
///   enough, the solver drives artificials to 0. Advantage: one LP solve.
///   Disadvantage: choosing M is tricky — too small and artificials stay
///   in the solution, too large and you get numerical instability. The
///   two-phase method avoids this M-tuning entirely.
SimplexResult two_phase_simplex(
    const Eigen::VectorXd& c_obj,
    const Eigen::MatrixXd& A_ineq,
    const Eigen::VectorXd& b_rhs,
    const std::vector<int>& con_types,
    bool verbose = true
) {
    // TODO(human): implement the full two-phase simplex method
    throw std::runtime_error("TODO(human): two_phase_simplex not implemented");
}

// ─── Test problems ───────────────────────────────────────────────────────────

/// Test 1: Diet problem with >= constraints
///   Minimize  2*x1 + 3*x2 + 7*x3    (cost of foods)
///   s.t.   x1 + 2*x2 +  x3 >= 14    (min calories)
///            x1 +  x2 + 3*x3 >= 14    (min protein)
///          3*x1 +  x2 +  x3 >= 14    (min vitamins)
///   x1, x2, x3 >= 0
void test_diet_problem() {
    std::cout << "\n============================================\n";
    std::cout << " Test 1: Diet Problem (all >= constraints)\n";
    std::cout << "============================================\n";
    std::cout << "Minimize 2*x1 + 3*x2 + 7*x3\n";
    std::cout << "s.t. x1 + 2*x2 +  x3 >= 14\n";
    std::cout << "      x1 +  x2 + 3*x3 >= 14\n";
    std::cout << "     3*x1 +  x2 +  x3 >= 14\n";

    int num_orig = 3;
    Eigen::VectorXd c(num_orig);
    c << 2.0, 3.0, 7.0;

    Eigen::MatrixXd A(3, num_orig);
    A << 1.0, 2.0, 1.0,
         1.0, 1.0, 3.0,
         3.0, 1.0, 1.0;

    Eigen::VectorXd b(3);
    b << 14.0, 14.0, 14.0;

    std::vector<int> types = {1, 1, 1};  // all >=

    SimplexResult result = two_phase_simplex(c, A, b, types, true);
    print_result(result, num_orig, "Diet Problem Result");
}

/// Test 2: Mixed constraints
///   Minimize  x1 + 2*x2
///   s.t.   x1 + x2  = 10       (equality)
///          x1 - x2 <= 4        (upper bound)
///          x1       >= 2        (lower bound)
///   x1, x2 >= 0
///   Expected: x1=2, x2=8, z=18
void test_mixed_constraints() {
    std::cout << "\n============================================\n";
    std::cout << " Test 2: Mixed Constraints (=, <=, >=)\n";
    std::cout << "============================================\n";
    std::cout << "Minimize x1 + 2*x2\n";
    std::cout << "s.t. x1 + x2  = 10\n";
    std::cout << "     x1 - x2 <= 4\n";
    std::cout << "     x1      >= 2\n";
    std::cout << "Expected: x1=2, x2=8, z=18\n";

    int num_orig = 2;
    Eigen::VectorXd c(num_orig);
    c << 1.0, 2.0;

    Eigen::MatrixXd A(3, num_orig);
    A << 1.0,  1.0,
         1.0, -1.0,
         1.0,  0.0;

    Eigen::VectorXd b(3);
    b << 10.0, 4.0, 2.0;

    std::vector<int> types = {0, -1, 1};  // =, <=, >=

    SimplexResult result = two_phase_simplex(c, A, b, types, true);
    print_result(result, num_orig, "Mixed Constraints Result");
}

/// Test 3: Infeasible problem
///   Minimize  x1 + x2
///   s.t.  x1 + x2 <= 4
///         x1 + x2 >= 8
///   x1, x2 >= 0
///   Clearly infeasible: can't be both <= 4 and >= 8 simultaneously.
void test_infeasible() {
    std::cout << "\n============================================\n";
    std::cout << " Test 3: Infeasible Problem\n";
    std::cout << "============================================\n";
    std::cout << "Minimize x1 + x2\n";
    std::cout << "s.t. x1 + x2 <= 4\n";
    std::cout << "     x1 + x2 >= 8\n";
    std::cout << "Expected: INFEASIBLE\n";

    int num_orig = 2;
    Eigen::VectorXd c(num_orig);
    c << 1.0, 1.0;

    Eigen::MatrixXd A(2, num_orig);
    A << 1.0, 1.0,
         1.0, 1.0;

    Eigen::VectorXd b(2);
    b << 4.0, 8.0;

    std::vector<int> types = {-1, 1};  // <=, >=

    SimplexResult result = two_phase_simplex(c, A, b, types, true);
    print_result(result, num_orig, "Infeasible Problem Result");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "==========================================\n";
    std::cout << " Phase 4: Two-Phase Simplex Method\n";
    std::cout << "==========================================\n";

    test_diet_problem();
    test_mixed_constraints();
    test_infeasible();

    return 0;
}
