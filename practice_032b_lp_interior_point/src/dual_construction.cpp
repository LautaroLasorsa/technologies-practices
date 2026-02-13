// ============================================================================
// Practice 032b — Phase 1: Dual Construction & Duality Verification
// ============================================================================
// Given a primal LP, mechanically construct its dual and verify weak/strong
// duality using known optimal solutions.
//
// Primal form used here (minimization):
//   minimize   c^T x
//   subject to A x >= b
//              x >= 0
//
// Corresponding dual (maximization):
//   maximize   b^T y
//   subject to A^T y <= c
//              y >= 0
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ─── Data structures ─────────────────────────────────────────────────────────

/// Represents a primal LP in the form:
///   minimize   c^T x
///   subject to A x >= b
///              x >= 0
struct LP {
    MatrixXd A;       // m x n constraint matrix
    VectorXd b;       // m-vector: RHS of constraints
    VectorXd c;       // n-vector: objective coefficients
    std::vector<std::string> var_names;        // names for x_1..x_n
    std::vector<std::string> constraint_names; // names for constraints 1..m

    int num_vars() const { return static_cast<int>(c.size()); }
    int num_constraints() const { return static_cast<int>(b.size()); }
};

/// Represents the dual LP:
///   maximize   b^T y
///   subject to A^T y <= c
///              y >= 0
struct DualLP {
    MatrixXd A_dual;  // n x m constraint matrix (= A^T from primal)
    VectorXd b_dual;  // n-vector: RHS of dual constraints (= c from primal)
    VectorXd c_dual;  // m-vector: dual objective coefficients (= b from primal)
    std::vector<std::string> var_names;        // names for y_1..y_m
    std::vector<std::string> constraint_names; // names for dual constraints 1..n
};

// ─── Pretty-print ────────────────────────────────────────────────────────────

void print_primal(const LP& lp) {
    std::cout << "=== PRIMAL LP ===\n";
    std::cout << "minimize  ";
    for (int j = 0; j < lp.num_vars(); ++j) {
        if (j > 0 && lp.c(j) >= 0) std::cout << " + ";
        else if (j > 0 && lp.c(j) < 0) std::cout << " - ";
        double coeff = (j == 0) ? lp.c(j) : std::abs(lp.c(j));
        std::cout << coeff << "*" << lp.var_names[j];
    }
    std::cout << "\nsubject to:\n";
    for (int i = 0; i < lp.num_constraints(); ++i) {
        std::cout << "  " << std::setw(12) << lp.constraint_names[i] << ":  ";
        for (int j = 0; j < lp.num_vars(); ++j) {
            if (j > 0 && lp.A(i, j) >= 0) std::cout << " + ";
            else if (j > 0 && lp.A(i, j) < 0) std::cout << " - ";
            double coeff = (j == 0) ? lp.A(i, j) : std::abs(lp.A(i, j));
            std::cout << coeff << "*" << lp.var_names[j];
        }
        std::cout << " >= " << lp.b(i) << "\n";
    }
    std::cout << "  All variables >= 0\n\n";
}

void print_dual(const DualLP& dual) {
    std::cout << "=== DUAL LP ===\n";
    std::cout << "maximize  ";
    for (int j = 0; j < static_cast<int>(dual.c_dual.size()); ++j) {
        if (j > 0 && dual.c_dual(j) >= 0) std::cout << " + ";
        else if (j > 0 && dual.c_dual(j) < 0) std::cout << " - ";
        double coeff = (j == 0) ? dual.c_dual(j) : std::abs(dual.c_dual(j));
        std::cout << coeff << "*" << dual.var_names[j];
    }
    std::cout << "\nsubject to:\n";
    int num_constraints = static_cast<int>(dual.b_dual.size());
    int num_vars = static_cast<int>(dual.c_dual.size());
    for (int i = 0; i < num_constraints; ++i) {
        std::cout << "  " << std::setw(12) << dual.constraint_names[i] << ":  ";
        for (int j = 0; j < num_vars; ++j) {
            if (j > 0 && dual.A_dual(i, j) >= 0) std::cout << " + ";
            else if (j > 0 && dual.A_dual(i, j) < 0) std::cout << " - ";
            double coeff = (j == 0) ? dual.A_dual(i, j) : std::abs(dual.A_dual(i, j));
            std::cout << coeff << "*" << dual.var_names[j];
        }
        std::cout << " <= " << dual.b_dual(i) << "\n";
    }
    std::cout << "  All variables >= 0\n\n";
}

// ─── TODO(human) implementations ────────────────────────────────────────────

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Construct the dual LP from the primal LP                   ║
// ║                                                                         ║
// ║ Given a primal LP:                                                      ║
// ║   minimize   c^T x                                                      ║
// ║   subject to A x >= b,   x >= 0                                         ║
// ║                                                                         ║
// ║ Construct the dual LP:                                                  ║
// ║   maximize   b^T y                                                      ║
// ║   subject to A^T y <= c,  y >= 0                                        ║
// ║                                                                         ║
// ║ Mechanical rules (memorize this table!):                                ║
// ║                                                                         ║
// ║   1. Primal minimize  →  Dual maximize (and vice versa)                 ║
// ║   2. Primal objective c becomes the RHS of dual constraints             ║
// ║   3. Primal RHS b becomes the dual objective coefficients               ║
// ║   4. Primal constraint matrix A becomes A^T in the dual                 ║
// ║   5. Primal ">=" constraint  →  Dual variable y_i >= 0                  ║
// ║      (our primal has all >=, so all dual vars are >= 0)                 ║
// ║   6. Primal variable x_j >= 0  →  Dual constraint j is "<="            ║
// ║      (our primal has all x >= 0, so all dual constraints are <=)        ║
// ║                                                                         ║
// ║ Steps to implement:                                                     ║
// ║   a) Set dual.A_dual = primal.A transposed  (n x m matrix)             ║
// ║   b) Set dual.c_dual = primal.b  (the primal RHS becomes dual obj)     ║
// ║   c) Set dual.b_dual = primal.c  (the primal obj becomes dual RHS)     ║
// ║   d) Generate variable names: y_1, y_2, ... (one per primal constraint)║
// ║   e) Generate constraint names from primal variable names               ║
// ║                                                                         ║
// ║ Example:                                                                ║
// ║   Primal: min [2,3]^T x, s.t. [[1,2],[3,1]] x >= [4,5], x >= 0        ║
// ║   Dual:   max [4,5]^T y, s.t. [[1,3],[2,1]] y <= [2,3], y >= 0        ║
// ║                                                                         ║
// ║ Hint: Eigen's .transpose() gives you A^T directly.                      ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
DualLP construct_dual(const LP& primal) {
    (void)primal;
    throw std::runtime_error("TODO(human): implement construct_dual");
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Verify weak duality                                        ║
// ║                                                                         ║
// ║ Weak duality theorem states:                                            ║
// ║   For any feasible primal x and feasible dual y:                        ║
// ║     dual_obj = b^T y  <=  c^T x = primal_obj                           ║
// ║                                                                         ║
// ║ (For a primal minimization / dual maximization pair.)                   ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   primal_obj — value of c^T x for some feasible primal solution         ║
// ║   dual_obj   — value of b^T y for some feasible dual solution           ║
// ║   tol        — numerical tolerance for floating-point comparison         ║
// ║                                                                         ║
// ║ Return true if dual_obj <= primal_obj + tol.                            ║
// ║                                                                         ║
// ║ Why this matters: weak duality gives you a CERTIFICATE of optimality.   ║
// ║ If you find primal_obj == dual_obj, BOTH must be optimal. This is how   ║
// ║ solvers prove optimality — they solve both primal and dual, and when    ║
// ║ the gap closes, they stop.                                              ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
bool verify_weak_duality(double primal_obj, double dual_obj, double tol = 1e-8) {
    (void)primal_obj;
    (void)dual_obj;
    (void)tol;
    throw std::runtime_error("TODO(human): implement verify_weak_duality");
}

/// Verify strong duality: at optimality, primal_obj == dual_obj.
bool verify_strong_duality(double primal_obj, double dual_obj, double tol = 1e-6) {
    return std::abs(primal_obj - dual_obj) < tol;
}

// ─── A minimal LP solver for verification ────────────────────────────────────
// This is a bare-bones solver (brute-force vertex enumeration for small LPs)
// used only to verify duality. NOT for production use.
//
// For a system Ax >= b, x >= 0 with n variables and m constraints, the
// feasible region has at most C(m+n, n) vertices. We enumerate all possible
// bases by choosing n constraints from the m+n constraints (m original + n
// non-negativity), solving the resulting square system, and keeping the
// feasible solutions.

struct SolveResult {
    bool feasible;
    double obj_value;
    VectorXd solution;
};

/// Solve minimize c^T x s.t. Ax >= b, x >= 0 by vertex enumeration.
/// Only practical for small problems (n,m <= 6).
SolveResult solve_by_enumeration(const LP& lp) {
    int n = lp.num_vars();
    int m = lp.num_constraints();
    int total = m + n; // m original constraints + n non-negativity constraints

    // Build the full constraint system: [A; I] x >= [b; 0]
    // A vertex is where n constraints are active (tight).
    // Active constraint i (i < m): A.row(i) * x = b(i)
    // Active constraint m+j (j < n): x(j) = 0

    SolveResult best{false, std::numeric_limits<double>::max(), VectorXd::Zero(n)};

    // Enumerate all subsets of size n from {0, ..., total-1}
    std::vector<int> indices(n);
    // Initialize to first combination
    for (int i = 0; i < n; ++i) indices[i] = i;

    auto next_combination = [&]() -> bool {
        int k = n - 1;
        while (k >= 0 && indices[k] == total - n + k) --k;
        if (k < 0) return false;
        ++indices[k];
        for (int j = k + 1; j < n; ++j) indices[j] = indices[j - 1] + 1;
        return true;
    };

    do {
        // Build n x n system from active constraints
        MatrixXd M(n, n);
        VectorXd rhs(n);
        for (int row = 0; row < n; ++row) {
            int idx = indices[row];
            if (idx < m) {
                M.row(row) = lp.A.row(idx);
                rhs(row) = lp.b(idx);
            } else {
                // Non-negativity constraint x_{idx-m} = 0
                M.row(row).setZero();
                M(row, idx - m) = 1.0;
                rhs(row) = 0.0;
            }
        }

        // Solve M * x = rhs
        if (std::abs(M.determinant()) < 1e-12) continue; // Singular
        VectorXd x = M.colPivHouseholderQr().solve(rhs);

        // Check feasibility: x >= -tol and Ax >= b - tol
        bool feasible = true;
        for (int j = 0; j < n; ++j) {
            if (x(j) < -1e-8) { feasible = false; break; }
        }
        if (feasible) {
            VectorXd Ax = lp.A * x;
            for (int i = 0; i < m; ++i) {
                if (Ax(i) < lp.b(i) - 1e-8) { feasible = false; break; }
            }
        }

        if (feasible) {
            double obj = lp.c.dot(x);
            if (!best.feasible || obj < best.obj_value) {
                best.feasible = true;
                best.obj_value = obj;
                best.solution = x;
            }
        }
    } while (next_combination());

    return best;
}

/// Solve the dual maximize c_dual^T y s.t. A_dual * y <= b_dual, y >= 0
/// by converting to a minimization and using enumeration.
SolveResult solve_dual_by_enumeration(const DualLP& dual) {
    // Convert max b^T y, A^T y <= c, y >= 0
    // to     min -b^T y, -A^T y >= -c, y >= 0
    LP as_min;
    as_min.A = -dual.A_dual;
    as_min.b = -dual.b_dual;
    as_min.c = -dual.c_dual;
    as_min.var_names = dual.var_names;
    as_min.constraint_names = dual.constraint_names;

    auto result = solve_by_enumeration(as_min);
    if (result.feasible) {
        result.obj_value = -result.obj_value; // flip back to maximization
    }
    return result;
}

// ─── Test problems ───────────────────────────────────────────────────────────

LP make_test_problem_1() {
    // Classic 2-variable LP:
    //   minimize  -3x1 - 5x2         (equivalently, maximize 3x1 + 5x2)
    //   subject to  x1        <= 4    →  -x1       >= -4
    //                    2x2  <= 12   →      -2x2  >= -12
    //               3x1 + 2x2 <= 18  →  -3x1 -2x2 >= -18
    //               x1, x2 >= 0
    //
    // Optimal primal: x = (2, 6), obj = -3*2 + -5*6 = -36
    // Dual: maximize -4y1 -12y2 -18y3, s.t. ...

    LP lp;
    lp.A.resize(3, 2);
    lp.A << -1,  0,
             0, -2,
            -3, -2;
    lp.b.resize(3);
    lp.b << -4, -12, -18;
    lp.c.resize(2);
    lp.c << -3, -5;
    lp.var_names = {"x1", "x2"};
    lp.constraint_names = {"resource_A", "resource_B", "resource_C"};
    return lp;
}

LP make_test_problem_2() {
    // 3-variable LP:
    //   minimize  -2x1 - 3x2 - x3
    //   subject to  x1 +  x2 + x3 <= 10   →  -x1 - x2 - x3 >= -10
    //               2x1 +  x2      <= 8    →  -2x1 - x2      >= -8
    //                      x2 + x3 <= 7    →       - x2 - x3 >= -7
    //               x1, x2, x3 >= 0
    //
    // Optimal: x = (0, 1, 6), obj = -2*0 -3*1 -1*6 = -9

    LP lp;
    lp.A.resize(3, 3);
    lp.A << -1, -1, -1,
            -2, -1,  0,
             0, -1, -1;
    lp.b.resize(3);
    lp.b << -10, -8, -7;
    lp.c.resize(3);
    lp.c << -2, -3, -1;
    lp.var_names = {"x1", "x2", "x3"};
    lp.constraint_names = {"capacity", "labor", "material"};
    return lp;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================================================================\n";
    std::cout << "  Phase 1: Dual Construction & Duality Verification\n";
    std::cout << "================================================================\n\n";

    // ── Test Problem 1 ──────────────────────────────────────────────────
    std::cout << "--- Test Problem 1 (2 variables, 3 constraints) ---\n\n";
    LP primal1 = make_test_problem_1();
    print_primal(primal1);

    DualLP dual1 = construct_dual(primal1);
    print_dual(dual1);

    // Solve both
    auto primal_result = solve_by_enumeration(primal1);
    auto dual_result = solve_dual_by_enumeration(dual1);

    if (primal_result.feasible) {
        std::cout << "Primal optimal solution: x = [";
        for (int i = 0; i < primal1.num_vars(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << primal_result.solution(i);
        }
        std::cout << "]\n";
        std::cout << "Primal optimal value: " << primal_result.obj_value << "\n";
    } else {
        std::cout << "Primal: INFEASIBLE\n";
    }

    if (dual_result.feasible) {
        std::cout << "Dual optimal solution: y = [";
        for (int i = 0; i < static_cast<int>(dual_result.solution.size()); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << dual_result.solution(i);
        }
        std::cout << "]\n";
        std::cout << "Dual optimal value: " << dual_result.obj_value << "\n";
    } else {
        std::cout << "Dual: INFEASIBLE\n";
    }

    if (primal_result.feasible && dual_result.feasible) {
        std::cout << "\nWeak duality holds: "
                  << (verify_weak_duality(primal_result.obj_value, dual_result.obj_value)
                      ? "YES" : "NO") << "\n";
        std::cout << "Strong duality holds: "
                  << (verify_strong_duality(primal_result.obj_value, dual_result.obj_value)
                      ? "YES" : "NO") << "\n";
        std::cout << "Duality gap: "
                  << std::abs(primal_result.obj_value - dual_result.obj_value) << "\n";
    }

    // ── Test Problem 2 ──────────────────────────────────────────────────
    std::cout << "\n--- Test Problem 2 (3 variables, 3 constraints) ---\n\n";
    LP primal2 = make_test_problem_2();
    print_primal(primal2);

    DualLP dual2 = construct_dual(primal2);
    print_dual(dual2);

    auto primal_result2 = solve_by_enumeration(primal2);
    auto dual_result2 = solve_dual_by_enumeration(dual2);

    if (primal_result2.feasible) {
        std::cout << "Primal optimal solution: x = [";
        for (int i = 0; i < primal2.num_vars(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << primal_result2.solution(i);
        }
        std::cout << "]\n";
        std::cout << "Primal optimal value: " << primal_result2.obj_value << "\n";
    }

    if (dual_result2.feasible) {
        std::cout << "Dual optimal solution: y = [";
        for (int i = 0; i < static_cast<int>(dual_result2.solution.size()); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << dual_result2.solution(i);
        }
        std::cout << "]\n";
        std::cout << "Dual optimal value: " << dual_result2.obj_value << "\n";
    }

    if (primal_result2.feasible && dual_result2.feasible) {
        std::cout << "\nWeak duality holds: "
                  << (verify_weak_duality(primal_result2.obj_value, dual_result2.obj_value)
                      ? "YES" : "NO") << "\n";
        std::cout << "Strong duality holds: "
                  << (verify_strong_duality(primal_result2.obj_value, dual_result2.obj_value)
                      ? "YES" : "NO") << "\n";
        std::cout << "Duality gap: "
                  << std::abs(primal_result2.obj_value - dual_result2.obj_value) << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Phase 1 complete.\n";
    std::cout << "================================================================\n";

    return 0;
}
