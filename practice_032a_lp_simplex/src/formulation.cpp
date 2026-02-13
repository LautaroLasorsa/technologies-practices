// ============================================================================
// Phase 1: LP Formulation & Standard Form
// ============================================================================
//
// This phase teaches the foundational step of any LP solver: converting a
// human-readable optimization problem into the standard form that the simplex
// algorithm requires. You will learn to:
//
//   1. Represent an LP in matrix/vector notation using Eigen
//   2. Convert inequality constraints to equalities by adding slack variables
//   3. Build the initial simplex tableau from the standard-form LP
//
// Sample problem (production planning):
//   Maximize  5*x1 + 4*x2
//   Subject to:
//     6*x1 + 4*x2 <= 24    (machine hours)
//       x1 + 2*x2 <=  6    (raw material)
//     x1, x2 >= 0
//
// In standard form (minimize, equalities, non-negative variables):
//   Minimize  -5*x1 - 4*x2 + 0*s1 + 0*s2
//   Subject to:
//     6*x1 + 4*x2 + s1      = 24
//       x1 + 2*x2      + s2 =  6
//     x1, x2, s1, s2 >= 0
//
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>

// ─── Data types ──────────────────────────────────────────────────────────────

/// Represents a constraint in the original (possibly inequality) form.
/// type: -1 means "<=", 0 means "=", +1 means ">="
struct Constraint {
    Eigen::VectorXd coefficients;  // LHS coefficients (one per original variable)
    double rhs;                     // right-hand side value
    int type;                       // -1: <=, 0: =, +1: >=
};

/// An LP in its original (human-readable) form.
/// sense: +1 = maximize, -1 = minimize
struct OriginalLP {
    Eigen::VectorXd objective;          // objective coefficients
    std::vector<Constraint> constraints;
    int sense;                          // +1 = maximize, -1 = minimize
};

/// An LP in standard form: minimize c^T x  s.t.  Ax = b, x >= 0.
/// Also tracks which variables are slacks (for tableau interpretation).
struct StandardLP {
    Eigen::VectorXd c;      // objective coefficients (length n)
    Eigen::MatrixXd A;      // constraint matrix (m x n)
    Eigen::VectorXd b;      // RHS vector (length m)
    int num_original;       // number of original decision variables
    int num_slack;          // number of slack/surplus variables added
};

/// The simplex tableau in augmented form.
///
/// Layout (m+1 rows, n+1 columns):
///
///   Row 0:       [ reduced costs (c_bar) ... | -z ]   (objective row)
///   Rows 1..m:   [ A coefficients ...        |  b ]   (constraint rows)
///
/// The last column stores the RHS values (b for constraints, -z for objective).
struct Tableau {
    Eigen::MatrixXd data;   // (m+1) x (n+1) matrix
    std::vector<int> basis; // basis[i] = index of basic variable in row i+1
    int m;                  // number of constraints
    int n;                  // number of variables (including slacks)
};

// ─── Provided: pretty-print functions ────────────────────────────────────────

void print_original_lp(const OriginalLP& lp) {
    std::cout << "\n=== Original LP ===\n";
    std::cout << (lp.sense == 1 ? "Maximize" : "Minimize") << "  ";
    for (int j = 0; j < lp.objective.size(); ++j) {
        if (j > 0 && lp.objective[j] >= 0) std::cout << "+ ";
        std::cout << lp.objective[j] << "*x" << (j + 1) << " ";
    }
    std::cout << "\nSubject to:\n";
    for (size_t i = 0; i < lp.constraints.size(); ++i) {
        const auto& con = lp.constraints[i];
        std::cout << "  ";
        for (int j = 0; j < con.coefficients.size(); ++j) {
            if (j > 0 && con.coefficients[j] >= 0) std::cout << "+ ";
            std::cout << con.coefficients[j] << "*x" << (j + 1) << " ";
        }
        std::string op = (con.type == -1) ? "<=" : (con.type == 0 ? "=" : ">=");
        std::cout << op << " " << con.rhs << "\n";
    }
    std::cout << "  x_i >= 0 for all i\n";
}

void print_standard_lp(const StandardLP& lp) {
    std::cout << "\n=== Standard Form LP ===\n";
    std::cout << "Minimize  ";
    int n = static_cast<int>(lp.c.size());
    for (int j = 0; j < n; ++j) {
        if (j > 0 && lp.c[j] >= 0) std::cout << "+ ";
        if (j < lp.num_original)
            std::cout << lp.c[j] << "*x" << (j + 1) << " ";
        else
            std::cout << lp.c[j] << "*s" << (j - lp.num_original + 1) << " ";
    }
    std::cout << "\nSubject to:\n";
    int m = static_cast<int>(lp.A.rows());
    for (int i = 0; i < m; ++i) {
        std::cout << "  ";
        for (int j = 0; j < n; ++j) {
            if (j > 0 && lp.A(i, j) >= 0) std::cout << "+ ";
            if (j < lp.num_original)
                std::cout << lp.A(i, j) << "*x" << (j + 1) << " ";
            else
                std::cout << lp.A(i, j) << "*s" << (j - lp.num_original + 1) << " ";
        }
        std::cout << "= " << lp.b[i] << "\n";
    }
    std::cout << "  All variables >= 0\n";
    std::cout << "  (" << lp.num_original << " original + "
              << lp.num_slack << " slack = " << n << " total variables)\n";
}

void print_tableau(const Tableau& tab) {
    std::cout << "\n=== Simplex Tableau ===\n";
    int rows = tab.m + 1;
    int cols = tab.n + 1;

    // Header
    std::cout << std::setw(8) << "Basis";
    for (int j = 0; j < tab.n; ++j)
        std::cout << std::setw(10) << ("x" + std::to_string(j + 1));
    std::cout << std::setw(10) << "RHS" << "\n";
    std::cout << std::string(8 + cols * 10, '-') << "\n";

    // Objective row (row 0)
    std::cout << std::setw(8) << "z";
    for (int j = 0; j < cols; ++j)
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << tab.data(0, j);
    std::cout << "\n";

    // Constraint rows
    for (int i = 1; i <= tab.m; ++i) {
        std::string basis_label = "x" + std::to_string(tab.basis[i - 1] + 1);
        std::cout << std::setw(8) << basis_label;
        for (int j = 0; j < cols; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << tab.data(i, j);
        std::cout << "\n";
    }
}

// ─── Provided: build the sample problem ──────────────────────────────────────

OriginalLP make_production_problem() {
    // Maximize 5*x1 + 4*x2
    // s.t.  6*x1 + 4*x2 <= 24
    //         x1 + 2*x2 <=  6
    //       x1, x2 >= 0
    OriginalLP lp;
    lp.sense = 1;  // maximize
    lp.objective = Eigen::VectorXd(2);
    lp.objective << 5.0, 4.0;

    Constraint c1;
    c1.coefficients = Eigen::VectorXd(2);
    c1.coefficients << 6.0, 4.0;
    c1.rhs = 24.0;
    c1.type = -1;  // <=

    Constraint c2;
    c2.coefficients = Eigen::VectorXd(2);
    c2.coefficients << 1.0, 2.0;
    c2.rhs = 6.0;
    c2.type = -1;  // <=

    lp.constraints = {c1, c2};
    return lp;
}

// ─── TODO(human): Convert to standard form ───────────────────────────────────

/// Convert an LP from its original (inequality) form to standard form.
///
/// TODO(human): Implement this conversion. Here is what you need to do:
///
/// STEP 1 — Handle the objective sense.
///   If the original problem is a maximization (sense == +1), negate the
///   objective coefficients to convert to minimization:
///     max c^T x  <==>  min (-c)^T x
///   This is because the simplex algorithm works with minimization. The
///   optimal value of the original max problem is the negation of the
///   min problem's optimal value.
///
/// STEP 2 — Count how many slack/surplus variables you need.
///   Each inequality constraint needs one extra variable:
///     - A "<=" constraint (type == -1) gets a SLACK variable s_i >= 0:
///         a^T x <= b  becomes  a^T x + s_i = b
///     - A ">=" constraint (type == +1) gets a SURPLUS variable s_i >= 0:
///         a^T x >= b  becomes  a^T x - s_i = b
///     - An "=" constraint (type == 0) needs no extra variable.
///   Count the total number of inequality constraints to determine how many
///   slack/surplus variables to add.
///
/// STEP 3 — Build the new objective vector c (length = num_original + num_slack).
///   The original variables keep their (possibly negated) coefficients.
///   All slack/surplus variables have coefficient 0 in the objective
///   (they don't contribute to cost — they are bookkeeping).
///
/// STEP 4 — Build the constraint matrix A (m x n_total) and RHS vector b.
///   For each constraint:
///     - Copy the original coefficients into the first num_original columns.
///     - For "<=" constraints: put +1 in the column of this constraint's slack.
///     - For ">=" constraints: put -1 in the column of this constraint's surplus.
///     - For "=" constraints: no extra column entry.
///     - Copy the RHS value.
///   IMPORTANT: If a RHS value is negative, multiply the entire row (both A
///   and b) by -1 and flip the inequality direction. The simplex method
///   requires b >= 0 for the initial basic feasible solution to work.
///
/// STEP 5 — Return a StandardLP with all the pieces assembled.
///
/// HINT: Use Eigen::MatrixXd::Zero(rows, cols) to initialize the A matrix
/// with zeros. Keep a running "slack column index" starting at num_original
/// that you increment each time you process an inequality constraint.
///
/// MATHEMATICAL FOUNDATION:
///   The standard form  min c^T x, Ax = b, x >= 0  is the universal format
///   for LP solvers. Every LP can be converted to this form. The slack
///   variables transform the feasible region from a set of half-spaces
///   (inequalities) into the intersection of hyperplanes (equalities) in
///   a higher-dimensional space, while keeping all variables non-negative.
///   Geometrically, the slack variable measures "how far" a constraint is
///   from being tight (active).
StandardLP to_standard_form(const OriginalLP& original) {
    // TODO(human): implement the 5-step conversion described above
    throw std::runtime_error("TODO(human): to_standard_form not implemented");
}

// ─── TODO(human): Build the initial simplex tableau ──────────────────────────

/// Construct the initial simplex tableau from a standard-form LP.
///
/// TODO(human): Build the tableau matrix and identify the initial basis.
///
/// The simplex tableau is a compact representation that combines the
/// constraint system and objective function into a single matrix for
/// efficient row operations. The layout is:
///
///   Row 0 (objective): [ c_1  c_2  ...  c_n  | 0 ]
///   Row i (constraint): [ A_i1  A_i2 ... A_in | b_i ]
///
/// WHERE the objective row stores the REDUCED COSTS (c_bar), not the raw
/// costs. For the initial tableau where slack variables form the basis:
///   - Reduced cost of a non-basic variable x_j = c_j (its original cost)
///   - Reduced cost of a basic variable = 0
///   - The RHS of the objective row = current objective value = 0
///     (because all decision variables are 0, all slacks equal b)
///
/// THE INITIAL BASIS:
///   When all constraints came from "<=" inequalities (Phase 1 of this
///   practice), the slack variables form a natural identity submatrix in A.
///   The initial BFS is: original variables = 0, slack_i = b_i.
///   So basis[i] = index of the slack variable for constraint i.
///
/// IMPLEMENTATION STEPS:
///   1. Create a (m+1) x (n+1) matrix initialized to zero.
///   2. Row 0, columns 0..n-1: copy the objective coefficients c.
///      Row 0, column n (RHS): set to 0 (initial objective value).
///   3. Rows 1..m, columns 0..n-1: copy the rows of A.
///      Rows 1..m, column n: copy the elements of b.
///   4. Identify the basis: for each constraint i (1-indexed row i+1),
///      find which slack variable column has a 1 in row i and 0 elsewhere.
///      For a problem converted from all "<=" constraints, this is simply
///      basis[i] = num_original + i.
///   5. Return the Tableau struct.
///
/// HINT: Use Eigen's block operations:
///   data.block(startRow, startCol, numRows, numCols)
///   data.row(i).head(n)  — first n elements of row i
///
/// WHY THE TABLEAU MATTERS:
///   The tableau is the "workspace" of the simplex algorithm. Each pivot
///   operation modifies the tableau in-place via elementary row operations,
///   exactly like Gaussian elimination. The reduced costs in row 0 tell
///   us whether the current solution is optimal (all >= 0 for minimization).
Tableau setup_initial_tableau(const StandardLP& std_lp) {
    // TODO(human): implement the tableau construction described above
    throw std::runtime_error("TODO(human): setup_initial_tableau not implemented");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "========================================\n";
    std::cout << " Phase 1: LP Formulation & Standard Form\n";
    std::cout << "========================================\n";

    // Step 1: Show the original problem
    OriginalLP original = make_production_problem();
    print_original_lp(original);

    // Step 2: Convert to standard form
    std::cout << "\n--- Converting to standard form ---\n";
    StandardLP standard = to_standard_form(original);
    print_standard_lp(standard);

    // Step 3: Build the initial tableau
    std::cout << "\n--- Building initial simplex tableau ---\n";
    Tableau tab = setup_initial_tableau(standard);
    print_tableau(tab);

    // Verification: the initial BFS should have x1=0, x2=0, s1=24, s2=6
    std::cout << "\n--- Initial Basic Feasible Solution ---\n";
    std::cout << "Decision variables: x1 = 0, x2 = 0\n";
    std::cout << "Slack variables: s1 = " << tab.data(1, tab.n) << ", s2 = " << tab.data(2, tab.n) << "\n";
    std::cout << "Objective value: " << -tab.data(0, tab.n) << "\n";
    std::cout << "(Objective is 0 because no decision variable is in the basis yet)\n";

    return 0;
}
