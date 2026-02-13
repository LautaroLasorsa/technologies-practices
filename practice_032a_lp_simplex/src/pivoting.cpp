// ============================================================================
// Phase 2: Simplex Pivoting Mechanics
// ============================================================================
//
// This phase teaches the atomic operation of the simplex algorithm: the PIVOT.
// A pivot is a single step that moves from one vertex (basic feasible solution)
// to an adjacent vertex along an edge of the feasible polytope. Each pivot:
//
//   1. Selects an ENTERING variable (non-basic -> basic)  — via reduced costs
//   2. Selects a LEAVING variable (basic -> non-basic)    — via minimum ratio test
//   3. Performs ROW OPERATIONS to update the tableau       — Gauss-Jordan elimination
//
// You will implement all three sub-operations and execute a single pivot step
// on a hardcoded tableau to see the mechanics in action.
//
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>

// ─── Tableau struct (same as Phase 1) ────────────────────────────────────────

struct Tableau {
    Eigen::MatrixXd data;   // (m+1) x (n+1) matrix
    std::vector<int> basis; // basis[i] = variable index in row i+1
    int m;                  // number of constraints
    int n;                  // number of variables
};

// ─── Provided: pretty-print ──────────────────────────────────────────────────

void print_tableau(const Tableau& tab) {
    std::cout << "\n=== Simplex Tableau ===\n";
    int cols = tab.n + 1;

    // Header
    std::cout << std::setw(8) << "Basis";
    for (int j = 0; j < tab.n; ++j)
        std::cout << std::setw(10) << ("x" + std::to_string(j + 1));
    std::cout << std::setw(10) << "RHS" << "\n";
    std::cout << std::string(8 + cols * 10, '-') << "\n";

    // Objective row
    std::cout << std::setw(8) << "z";
    for (int j = 0; j < cols; ++j)
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << tab.data(0, j);
    std::cout << "\n";

    // Constraint rows
    for (int i = 1; i <= tab.m; ++i) {
        std::string label = "x" + std::to_string(tab.basis[i - 1] + 1);
        std::cout << std::setw(8) << label;
        for (int j = 0; j < cols; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << tab.data(i, j);
        std::cout << "\n";
    }
}

// ─── Provided: hardcoded initial tableau ─────────────────────────────────────
//
// This is the initial tableau for the production problem from Phase 1:
//   Minimize -5*x1 - 4*x2 + 0*s1 + 0*s2
//   s.t.  6*x1 + 4*x2 + s1      = 24
//           x1 + 2*x2      + s2 =  6
//   Basis = {s1 (index 2), s2 (index 3)}

Tableau make_initial_tableau() {
    Tableau tab;
    tab.m = 2;  // 2 constraints
    tab.n = 4;  // x1, x2, s1, s2

    // (m+1) x (n+1) = 3 x 5
    tab.data = Eigen::MatrixXd::Zero(3, 5);

    // Row 0 (objective): [-5, -4, 0, 0 | 0]
    //   Reduced costs for minimizing -5x1 - 4x2.
    //   Negative values mean the objective can be improved by bringing
    //   that variable into the basis.
    tab.data(0, 0) = -5.0;  // c_bar for x1
    tab.data(0, 1) = -4.0;  // c_bar for x2
    tab.data(0, 2) =  0.0;  // c_bar for s1 (basic)
    tab.data(0, 3) =  0.0;  // c_bar for s2 (basic)
    tab.data(0, 4) =  0.0;  // -z (current objective value)

    // Row 1 (constraint 1, basic var = s1):
    //   6*x1 + 4*x2 + 1*s1 + 0*s2 = 24
    tab.data(1, 0) = 6.0;
    tab.data(1, 1) = 4.0;
    tab.data(1, 2) = 1.0;
    tab.data(1, 3) = 0.0;
    tab.data(1, 4) = 24.0;

    // Row 2 (constraint 2, basic var = s2):
    //   1*x1 + 2*x2 + 0*s1 + 1*s2 = 6
    tab.data(2, 0) = 1.0;
    tab.data(2, 1) = 2.0;
    tab.data(2, 2) = 0.0;
    tab.data(2, 3) = 1.0;
    tab.data(2, 4) = 6.0;

    // Initial basis: row 1 -> s1 (index 2), row 2 -> s2 (index 3)
    tab.basis = {2, 3};

    return tab;
}

// ─── TODO(human): Select pivot column (entering variable) ────────────────────

/// Find the column index of the variable that should ENTER the basis.
///
/// TODO(human): Implement the entering variable selection rule.
///
/// THE RULE (Dantzig's rule / most negative reduced cost):
///   Scan the objective row (row 0) across columns 0..n-1 (exclude RHS).
///   Find the column j with the MOST NEGATIVE reduced cost (c_bar_j).
///   If all reduced costs are >= 0, the current solution is OPTIMAL.
///
/// WHY THIS WORKS:
///   The reduced cost c_bar_j represents the rate of change of the objective
///   function per unit increase of variable x_j. In minimization:
///     - c_bar_j < 0 means increasing x_j would DECREASE the objective (good!)
///     - c_bar_j >= 0 means increasing x_j would not improve the objective
///   Choosing the most negative value is a greedy heuristic — it picks the
///   variable with the steepest descent direction. This is NOT guaranteed
///   to minimize the total number of pivots (that's NP-hard to determine),
///   but it works well in practice.
///
/// BLAND'S RULE (alternative for anti-cycling):
///   Instead of "most negative", pick the SMALLEST INDEX j with c_bar_j < 0.
///   This guarantees finite termination even under degeneracy, but may
///   require more pivots. For this practice, use Dantzig's rule.
///
/// RETURN VALUE:
///   - The column index j (0-based) of the entering variable, or
///   - (-1) if no negative reduced cost exists (current solution is optimal)
///
/// IMPLEMENTATION:
///   1. Initialize best_col = -1, best_val = 0.0 (or a small tolerance like -1e-10)
///   2. Loop j = 0 to n-1:
///        if tab.data(0, j) < best_val:
///          best_val = tab.data(0, j)
///          best_col = j
///   3. Return best_col
///
/// NOTE: Using a small negative tolerance (e.g., -1e-10 instead of 0)
/// helps avoid cycling due to floating-point imprecision. Values very
/// close to zero should be treated as zero.
int select_pivot_column(const Tableau& tab) {
    // TODO(human): implement Dantzig's pivot column selection rule
    throw std::runtime_error("TODO(human): select_pivot_column not implemented");
}

// ─── TODO(human): Select pivot row (leaving variable) ────────────────────────

/// Find the row index of the variable that should LEAVE the basis.
///
/// TODO(human): Implement the minimum ratio test.
///
/// THE MINIMUM RATIO TEST:
///   Given the entering column (pivot_col), for each constraint row i (1..m):
///     - If tab.data(i, pivot_col) > 0:
///         Compute ratio = tab.data(i, n) / tab.data(i, pivot_col)
///         (i.e., RHS / coefficient in the pivot column)
///     - If tab.data(i, pivot_col) <= 0:
///         Skip this row (negative or zero entries cannot limit the increase)
///   Choose the row i with the SMALLEST non-negative ratio.
///
/// WHY THE MINIMUM RATIO TEST:
///   When we increase the entering variable x_j from 0, the basic variables
///   change according to:  x_B = b - A_pivot_col * x_j
///   The basic variable in row i hits zero when:
///     x_j = b_i / A(i, pivot_col)
///   We must stop at the FIRST basic variable that hits zero (the minimum
///   ratio) to maintain feasibility (all variables >= 0). Going further
///   would make that basic variable negative, violating the constraint.
///
/// UNBOUNDEDNESS DETECTION:
///   If NO row has a positive entry in the pivot column, the entering
///   variable can increase to +infinity without any basic variable hitting
///   zero. This means the LP is UNBOUNDED (objective -> -infinity).
///   Return -1 in this case.
///
/// TIE-BREAKING (DEGENERACY):
///   If multiple rows achieve the same minimum ratio, the pivot is
///   DEGENERATE — the objective value won't change. Degeneracy can cause
///   cycling (revisiting the same basis). For now, break ties by choosing
///   the smallest row index (a simple form of Bland's rule for rows).
///
/// RETURN VALUE:
///   - The row index i (1-based, i.e., 1..m) of the leaving variable, or
///   - (-1) if the problem is unbounded
///
/// IMPLEMENTATION:
///   1. Initialize best_row = -1, best_ratio = +infinity
///   2. Loop i = 1 to m:
///        coeff = tab.data(i, pivot_col)
///        if coeff > 1e-10:  // positive (with tolerance)
///          ratio = tab.data(i, n) / coeff  // RHS / pivot-column entry
///          if ratio < best_ratio (or ratio == best_ratio and i < best_row):
///            best_ratio = ratio
///            best_row = i
///   3. Return best_row (or -1 if no positive coefficient found)
int select_pivot_row(const Tableau& tab, int pivot_col) {
    // TODO(human): implement the minimum ratio test
    throw std::runtime_error("TODO(human): select_pivot_row not implemented");
}

// ─── TODO(human): Perform the pivot operation ────────────────────────────────

/// Execute a pivot: make the pivot element 1 and eliminate all other entries
/// in the pivot column.
///
/// TODO(human): Implement the pivot (Gauss-Jordan elimination step).
///
/// THE PIVOT OPERATION is identical to a step of Gauss-Jordan elimination:
///
///   Given pivot position (pivot_row, pivot_col):
///
///   STEP 1 — NORMALIZE the pivot row.
///     Divide the entire pivot row by the pivot element:
///       tab.data.row(pivot_row) /= tab.data(pivot_row, pivot_col)
///     After this, tab.data(pivot_row, pivot_col) == 1.0.
///
///   STEP 2 — ELIMINATE all other rows in the pivot column.
///     For every row i != pivot_row (INCLUDING row 0, the objective):
///       factor = tab.data(i, pivot_col)
///       tab.data.row(i) -= factor * tab.data.row(pivot_row)
///     After this, tab.data(i, pivot_col) == 0.0 for all i != pivot_row.
///
///   STEP 3 — UPDATE the basis tracking.
///     The leaving variable (previously basic in pivot_row) is replaced by
///     the entering variable:
///       tab.basis[pivot_row - 1] = pivot_col
///     (pivot_row is 1-based for constraint rows, basis is 0-indexed)
///
/// WHY THIS IS GAUSS-JORDAN:
///   The simplex method is, at its core, a structured form of Gaussian
///   elimination on the augmented system [A | b]. Each pivot transforms
///   the tableau so that the basis columns form an identity submatrix.
///   The reduced costs in row 0 are updated simultaneously, maintaining
///   the optimality conditions automatically.
///
/// NUMERICAL NOTE:
///   In production solvers, pivoting is done with partial pivoting or
///   LU-based approaches to maintain numerical stability. For this
///   educational implementation, direct row operations on the dense
///   tableau are fine for small problems.
///
/// HINT: Eigen makes this very clean:
///   tab.data.row(pivot_row) /= tab.data(pivot_row, pivot_col);
///   for (int i = 0; i <= tab.m; ++i) { ... }
void pivot(Tableau& tab, int pivot_row, int pivot_col) {
    // TODO(human): implement the Gauss-Jordan pivot operation
    throw std::runtime_error("TODO(human): pivot not implemented");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "==========================================\n";
    std::cout << " Phase 2: Simplex Pivoting Mechanics\n";
    std::cout << "==========================================\n";

    Tableau tab = make_initial_tableau();

    std::cout << "\n--- BEFORE PIVOT ---\n";
    print_tableau(tab);

    // Step 1: Select entering variable
    int pivot_col = select_pivot_column(tab);
    if (pivot_col == -1) {
        std::cout << "\nAlready optimal! No pivot needed.\n";
        return 0;
    }
    std::cout << "\nEntering variable: x" << (pivot_col + 1)
              << " (reduced cost = " << tab.data(0, pivot_col) << ")\n";

    // Step 2: Select leaving variable
    int pivot_row = select_pivot_row(tab, pivot_col);
    if (pivot_row == -1) {
        std::cout << "\nProblem is UNBOUNDED!\n";
        return 1;
    }
    std::cout << "Leaving variable: x" << (tab.basis[pivot_row - 1] + 1)
              << " (row " << pivot_row << ", ratio = "
              << tab.data(pivot_row, tab.n) / tab.data(pivot_row, pivot_col) << ")\n";
    std::cout << "Pivot element: tab[" << pivot_row << "][" << pivot_col << "] = "
              << tab.data(pivot_row, pivot_col) << "\n";

    // Step 3: Perform the pivot
    pivot(tab, pivot_row, pivot_col);

    std::cout << "\n--- AFTER PIVOT ---\n";
    print_tableau(tab);

    // Show the new BFS
    std::cout << "\n--- Current Basic Feasible Solution ---\n";
    std::cout << "Basic variables:\n";
    for (int i = 0; i < tab.m; ++i) {
        std::cout << "  x" << (tab.basis[i] + 1) << " = " << tab.data(i + 1, tab.n) << "\n";
    }
    std::cout << "Non-basic variables = 0\n";
    std::cout << "Objective value: " << -tab.data(0, tab.n) << "\n";

    // Show what a second pivot would look like
    std::cout << "\n--- Checking for another pivot ---\n";
    int next_col = select_pivot_column(tab);
    if (next_col == -1) {
        std::cout << "All reduced costs >= 0. Solution is OPTIMAL!\n";
    } else {
        std::cout << "Next entering variable would be: x" << (next_col + 1)
                  << " (reduced cost = " << tab.data(0, next_col) << ")\n";
        std::cout << "(Performing the second pivot is left to Phase 3)\n";
    }

    return 0;
}
