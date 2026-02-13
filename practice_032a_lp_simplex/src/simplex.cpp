// ============================================================================
// Phase 3: Full Simplex Solver Loop
// ============================================================================
//
// This phase brings together everything from Phases 1 and 2 into a complete
// simplex solver that iterates from the initial BFS to the optimal solution.
// You will implement the main loop that repeatedly selects pivots and detects
// termination conditions (optimality or unboundedness).
//
// The solver is tested on three problems of increasing difficulty:
//   1. Production planning (2 variables, 2 constraints) — from earlier phases
//   2. Diet problem (3 variables, 3 constraints)
//   3. A degenerate problem to observe cycling potential
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
    MAX_ITERATIONS
};

struct SimplexResult {
    SimplexStatus status;
    Eigen::VectorXd solution;      // full variable values (length n)
    double objective_value;         // optimal objective (for the minimization)
    int iterations;                 // number of pivots performed
};

// ─── Provided: pretty-print ──────────────────────────────────────────────────

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

void print_result(const SimplexResult& result, int num_original, const std::string& label = "") {
    if (!label.empty()) std::cout << "\n=== " << label << " ===\n";
    switch (result.status) {
        case SimplexStatus::OPTIMAL:
            std::cout << "Status: OPTIMAL\n";
            std::cout << "Iterations: " << result.iterations << "\n";
            std::cout << "Objective value: " << result.objective_value << "\n";
            std::cout << "Decision variables:\n";
            for (int j = 0; j < num_original; ++j) {
                std::cout << "  x" << (j + 1) << " = " << std::fixed
                          << std::setprecision(4) << result.solution[j] << "\n";
            }
            break;
        case SimplexStatus::UNBOUNDED:
            std::cout << "Status: UNBOUNDED (objective -> -infinity)\n";
            std::cout << "Iterations before detection: " << result.iterations << "\n";
            break;
        case SimplexStatus::MAX_ITERATIONS:
            std::cout << "Status: MAX ITERATIONS reached (" << result.iterations << ")\n";
            std::cout << "Possible cycling detected.\n";
            break;
    }
}

// ─── Provided: Pivot helpers (re-implement or copy from Phase 2) ─────────────

/// Select the pivot column using Dantzig's rule (most negative reduced cost).
/// Returns -1 if optimal (no negative reduced costs).
int select_pivot_column(const Tableau& tab) {
    int best_col = -1;
    double best_val = -1e-10;  // tolerance
    for (int j = 0; j < tab.n; ++j) {
        if (tab.data(0, j) < best_val) {
            best_val = tab.data(0, j);
            best_col = j;
        }
    }
    return best_col;
}

/// Select the pivot row using the minimum ratio test.
/// Returns -1 if unbounded (no positive entry in pivot column).
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

/// Perform the pivot operation (Gauss-Jordan elimination step).
void pivot(Tableau& tab, int pivot_row, int pivot_col) {
    // Normalize the pivot row
    double piv = tab.data(pivot_row, pivot_col);
    tab.data.row(pivot_row) /= piv;

    // Eliminate all other rows
    for (int i = 0; i <= tab.m; ++i) {
        if (i == pivot_row) continue;
        double factor = tab.data(i, pivot_col);
        tab.data.row(i) -= factor * tab.data.row(pivot_row);
    }

    // Update basis
    tab.basis[pivot_row - 1] = pivot_col;
}

// ─── Provided: Build tableau from standard form data ─────────────────────────

/// Build an initial simplex tableau from standard form components.
/// Assumes slack variables form the initial basis (all constraints from <=).
Tableau build_tableau(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
                      const Eigen::VectorXd& b, int num_original) {
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());

    Tableau tab;
    tab.m = m;
    tab.n = n;
    tab.data = Eigen::MatrixXd::Zero(m + 1, n + 1);

    // Objective row
    tab.data.block(0, 0, 1, n) = c.transpose();
    tab.data(0, n) = 0.0;

    // Constraint rows
    tab.data.block(1, 0, m, n) = A;
    tab.data.block(1, n, m, 1) = b;

    // Initial basis: slack variables
    tab.basis.resize(m);
    for (int i = 0; i < m; ++i) {
        tab.basis[i] = num_original + i;
    }

    return tab;
}

// ─── TODO(human): Full simplex solver loop ───────────────────────────────────

/// Run the simplex algorithm to optimality (or detect unboundedness/cycling).
///
/// TODO(human): Implement the complete simplex iteration loop.
///
/// THE SIMPLEX ALGORITHM (high-level):
///   Starting from an initial BFS (encoded in the tableau), repeat:
///     1. CHECK OPTIMALITY: call select_pivot_column().
///        If it returns -1, all reduced costs >= 0 → current BFS is optimal.
///     2. CHECK UNBOUNDEDNESS: call select_pivot_row(pivot_col).
///        If it returns -1, no constraint limits the entering variable
///        → the LP is unbounded (objective → -infinity).
///     3. PIVOT: call pivot(tab, pivot_row, pivot_col) to move to the
///        adjacent vertex.
///     4. Increment iteration counter.
///     5. (Optional) Print the tableau at each iteration for learning.
///   Repeat until optimal, unbounded, or max_iterations reached.
///
/// EXTRACTING THE SOLUTION:
///   When the algorithm terminates with OPTIMAL status:
///     - Create a solution vector of length n, initialized to 0.
///     - For each basic variable: solution[basis[i]] = tab.data(i+1, n)
///       (the RHS of the constraint row where this variable is basic).
///     - Non-basic variables have value 0 (they are not in the basis).
///     - The objective value is -tab.data(0, n) (because we stored -z
///       in the objective row's RHS).
///
/// WHY THE LOOP TERMINATES:
///   Under non-degeneracy (all BFS are distinct), each pivot strictly
///   improves the objective. Since there are finitely many vertices
///   (at most C(n,m) = n! / (m!(n-m)!) basic feasible solutions),
///   the algorithm must terminate. Under degeneracy, Bland's rule
///   guarantees termination, but we use a max_iterations safeguard.
///
/// PARAMETERS:
///   - tab: the initial simplex tableau (modified in-place during solving)
///   - verbose: if true, print the tableau at each iteration
///   - max_iter: safety limit to prevent infinite loops (default 100)
///
/// RETURN: A SimplexResult with status, solution vector, objective, iterations.
///
/// IMPLEMENTATION OUTLINE:
///   SimplexResult result;
///   int iter = 0;
///   while (iter < max_iter) {
///       int col = select_pivot_column(tab);
///       if (col == -1) { /* OPTIMAL — extract solution and return */ }
///       int row = select_pivot_row(tab, col);
///       if (row == -1) { /* UNBOUNDED — return */ }
///       if (verbose) { /* print iteration info */ }
///       pivot(tab, row, col);
///       iter++;
///   }
///   /* MAX_ITERATIONS — return */
///
/// HINT: To extract the solution, note that non-basic variables are 0
/// and basic variable values are in the RHS column of their row.
SimplexResult solve(Tableau& tab, bool verbose = true, int max_iter = 100) {
    // TODO(human): implement the full simplex iteration loop
    throw std::runtime_error("TODO(human): solve not implemented");
}

// ─── Test problems ───────────────────────────────────────────────────────────

/// Problem 1: Production planning (from Phases 1-2)
///   Maximize  5*x1 + 4*x2
///   s.t.  6*x1 + 4*x2 <= 24
///           x1 + 2*x2 <=  6
///   Standard form: min -5*x1 - 4*x2, with 2 slacks
///   Expected optimal: x1=3, x2=1.5, z=21
void test_production_planning() {
    std::cout << "\n============================================\n";
    std::cout << " Test 1: Production Planning\n";
    std::cout << "============================================\n";
    std::cout << "Maximize 5*x1 + 4*x2\n";
    std::cout << "s.t. 6*x1 + 4*x2 <= 24\n";
    std::cout << "       x1 + 2*x2 <=  6\n";
    std::cout << "Expected: x1=3, x2=1.5, max z=21\n";

    Eigen::VectorXd c(4);
    c << -5.0, -4.0, 0.0, 0.0;

    Eigen::MatrixXd A(2, 4);
    A << 6.0, 4.0, 1.0, 0.0,
         1.0, 2.0, 0.0, 1.0;

    Eigen::VectorXd b(2);
    b << 24.0, 6.0;

    Tableau tab = build_tableau(c, A, b, 2);
    SimplexResult result = solve(tab, true);
    print_result(result, 2, "Production Planning Result");

    // Since we minimized -c^T x, the max objective is -result.objective_value
    if (result.status == SimplexStatus::OPTIMAL) {
        std::cout << "Original maximization objective: " << -result.objective_value << "\n";
    }
}

/// Problem 2: Diet problem (3 variables, 3 constraints)
///   Minimize  2*x1 + 3*x2 + 5*x3   (cost per unit of food 1, 2, 3)
///   s.t.  x1 + 2*x2 + 3*x3 >= 12   (minimum vitamin A)
///         2*x1 +  x2 +  x3 >= 8    (minimum vitamin B)
///         x1, x2, x3 >= 0
///
///   Converting >= to <= by multiplying by -1:
///     -x1 - 2*x2 - 3*x3 <= -12
///     -2*x1 - x2  -  x3 <= -8
///   Then adding slacks and ensuring b >= 0 by flipping signs:
///     x1 + 2*x2 + 3*x3 - s1      = 12   (note: surplus, not slack)
///     2*x1 +  x2 +  x3      - s2 = 8
///
///   BUT: this has no obvious initial BFS (slack vars would be negative).
///   For Phase 3, we reformulate with <= constraints by changing perspective:
///
///   Alternative formulation (resource allocation view):
///   Minimize 2*x1 + 3*x2 + 5*x3
///   s.t.  x1 +  x2 +  x3 <= 10   (budget constraint)
///         x1 + 2*x2 + 3*x3 <= 18  (shelf space)
///         2*x1 + x2 +  x3 <= 12   (prep time)
///   Expected: feasible with at least one binding constraint
void test_resource_allocation() {
    std::cout << "\n============================================\n";
    std::cout << " Test 2: Resource Allocation\n";
    std::cout << "============================================\n";
    std::cout << "Minimize 2*x1 + 3*x2 + 5*x3\n";
    std::cout << "s.t. x1 +  x2 +  x3 <= 10\n";
    std::cout << "     x1 + 2*x2 + 3*x3 <= 18\n";
    std::cout << "    2*x1 +  x2 +  x3 <= 12\n";

    int num_orig = 3;
    int num_slack = 3;
    int n = num_orig + num_slack;

    Eigen::VectorXd c(n);
    c << 2.0, 3.0, 5.0, 0.0, 0.0, 0.0;

    Eigen::MatrixXd A(3, n);
    A << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
         1.0, 2.0, 3.0, 0.0, 1.0, 0.0,
         2.0, 1.0, 1.0, 0.0, 0.0, 1.0;

    Eigen::VectorXd b(3);
    b << 10.0, 18.0, 12.0;

    Tableau tab = build_tableau(c, A, b, num_orig);
    SimplexResult result = solve(tab, true);
    print_result(result, num_orig, "Resource Allocation Result");
}

/// Problem 3: Degenerate LP (to observe zero-improvement pivots)
///   Minimize -x1 - x2
///   s.t.  x1       <= 4
///              x2   <= 4
///         x1 + x2   <= 6
///   Note: at the optimum (x1=4, x2=2) or (x1=2, x2=4), the third
///   constraint is active. The simplex may encounter degenerate pivots
///   where the objective doesn't change but the basis rotates.
void test_degenerate() {
    std::cout << "\n============================================\n";
    std::cout << " Test 3: Degenerate LP\n";
    std::cout << "============================================\n";
    std::cout << "Minimize -x1 - x2\n";
    std::cout << "s.t. x1      <= 4\n";
    std::cout << "          x2 <= 4\n";
    std::cout << "     x1 + x2 <= 6\n";
    std::cout << "Expected: x1+x2=6 with z=-6\n";

    int num_orig = 2;
    int num_slack = 3;
    int n = num_orig + num_slack;

    Eigen::VectorXd c(n);
    c << -1.0, -1.0, 0.0, 0.0, 0.0;

    Eigen::MatrixXd A(3, n);
    A << 1.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 1.0;

    Eigen::VectorXd b(3);
    b << 4.0, 4.0, 6.0;

    Tableau tab = build_tableau(c, A, b, num_orig);
    SimplexResult result = solve(tab, true);
    print_result(result, num_orig, "Degenerate LP Result");

    if (result.status == SimplexStatus::OPTIMAL) {
        std::cout << "Original maximization objective: " << -result.objective_value << "\n";
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "==========================================\n";
    std::cout << " Phase 3: Full Simplex Solver\n";
    std::cout << "==========================================\n";

    test_production_planning();
    test_resource_allocation();
    test_degenerate();

    return 0;
}
