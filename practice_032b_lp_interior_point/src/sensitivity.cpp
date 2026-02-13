// ============================================================================
// Practice 032b — Phase 3: Sensitivity Analysis
// ============================================================================
// Perturb the RHS of constraints, re-solve, and observe how the optimal value
// changes. Compare the rate of change to the dual variable (shadow price).
// Also compute the allowable range over which the basis stays optimal.
//
// Key insight: sensitivity analysis is the PRACTICAL use of duality.
// Instead of asking "what is the optimal solution?", you ask:
//   "How much would I pay for one more unit of resource X?"
//   "How far can demand change before I need a different production plan?"
//
// We use the vertex enumeration solver from Phase 1 (small problems only).
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ─── LP struct (same as Phase 1) ─────────────────────────────────────────────

struct LP {
    MatrixXd A;       // m x n (Ax >= b form)
    VectorXd b;       // m-vector
    VectorXd c;       // n-vector
    std::vector<std::string> var_names;
    std::vector<std::string> constraint_names;

    int num_vars() const { return static_cast<int>(c.size()); }
    int num_constraints() const { return static_cast<int>(b.size()); }
};

struct SolveResult {
    bool feasible;
    double obj_value;
    VectorXd solution;
};

// ─── Vertex enumeration solver (from Phase 1) ───────────────────────────────

SolveResult solve_by_enumeration(const LP& lp) {
    int n = lp.num_vars();
    int m = lp.num_constraints();
    int total = m + n;

    SolveResult best{false, std::numeric_limits<double>::max(), VectorXd::Zero(n)};

    std::vector<int> indices(n);
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
        MatrixXd M(n, n);
        VectorXd rhs(n);
        for (int row = 0; row < n; ++row) {
            int idx = indices[row];
            if (idx < m) {
                M.row(row) = lp.A.row(idx);
                rhs(row) = lp.b(idx);
            } else {
                M.row(row).setZero();
                M(row, idx - m) = 1.0;
                rhs(row) = 0.0;
            }
        }

        if (std::abs(M.determinant()) < 1e-12) continue;
        VectorXd x = M.colPivHouseholderQr().solve(rhs);

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

// ─── Test problem ────────────────────────────────────────────────────────────

LP make_sensitivity_problem() {
    // Same production problem as Phase 2:
    //   maximize  5*x1 + 8*x2
    //   subject to  2*x1 + 3*x2 <= 120  (labor)
    //               4*x1 + 2*x2 <= 100  (wood)
    //               1*x1 + 2*x2 <= 80   (machine)
    //               x1, x2 >= 0
    //
    // In our min >= form:
    //   minimize -5*x1 - 8*x2
    //   -2x1 - 3x2 >= -120
    //   -4x1 - 2x2 >= -100
    //   -1x1 - 2x2 >= -80

    LP lp;
    lp.A.resize(3, 2);
    lp.A << -2, -3,
            -4, -2,
            -1, -2;
    lp.b.resize(3);
    lp.b << -120, -100, -80;
    lp.c.resize(2);
    lp.c << -5, -8;
    lp.var_names = {"x1", "x2"};
    lp.constraint_names = {"labor", "wood", "machine"};
    return lp;
}

// ─── TODO(human) implementations ────────────────────────────────────────────

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): RHS sensitivity analysis                                   ║
// ║                                                                         ║
// ║ For each constraint i, perturb b_i by +delta and -delta, re-solve       ║
// ║ the LP, and compute the numerical derivative:                           ║
// ║                                                                         ║
// ║   dZ/db_i ≈ (Z(b_i + delta) - Z(b_i - delta)) / (2 * delta)           ║
// ║                                                                         ║
// ║ This should approximately equal the dual variable y_i (shadow price).   ║
// ║                                                                         ║
// ║ IMPORTANT: Our LP is in the form Ax >= b, where we've negated the       ║
// ║ original <= constraints. So b = -b_orig. When we say "perturb b_i      ║
// ║ by delta", we mean perturb the ORIGINAL RHS by delta. Since             ║
// ║ b = -b_orig, perturbing b_orig by +delta means changing b by -delta.    ║
// ║ So for labor hours: b_orig = 120, increasing to 121 means b changes    ║
// ║ from -120 to -121.                                                      ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   lp    — the LP (in min, >= form)                                      ║
// ║   delta — perturbation amount (try 0.1 or 1.0)                          ║
// ║                                                                         ║
// ║ Returns: vector of m numerical derivatives dZ/db_orig_i                 ║
// ║                                                                         ║
// ║ Steps for each constraint i:                                            ║
// ║   1. Create a copy of lp. Perturb b(i) by -delta (since b = -b_orig,   ║
// ║      decreasing b by delta means increasing b_orig by delta).           ║
// ║   2. Solve the perturbed LP. Record Z_plus = obj value.                 ║
// ║   3. Perturb b(i) by +delta instead. Solve. Record Z_minus.            ║
// ║   4. Compute dZ/db_orig = (Z_plus - Z_minus) / (2*delta).              ║
// ║      (Note the sign: Z_plus corresponds to b_orig + delta, etc.)        ║
// ║   5. Store in result vector.                                            ║
// ║                                                                         ║
// ║ The numerical derivative should match y_i, validating the theorem       ║
// ║ that shadow prices equal marginal values of resources.                  ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
std::vector<double> sensitivity_rhs(const LP& lp, double delta = 1.0) {
    (void)lp;
    (void)delta;
    throw std::runtime_error("TODO(human): implement sensitivity_rhs");
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Allowable RHS range                                        ║
// ║                                                                         ║
// ║ For constraint `constraint_idx`, find the range [b_low, b_high] of      ║
// ║ ORIGINAL RHS values (b_orig) over which the current optimal basis       ║
// ║ (set of binding constraints) remains the same.                          ║
// ║                                                                         ║
// ║ Within this range, the shadow price y_i is valid and the optimal value  ║
// ║ changes linearly: Z*(b_orig + delta) = Z* + y_i * delta.               ║
// ║ Outside this range, a different set of constraints becomes binding      ║
// ║ and the shadow price changes.                                           ║
// ║                                                                         ║
// ║ Approach (numerical, suitable for small LPs):                           ║
// ║   1. Solve the original LP, get the optimal solution x* and which       ║
// ║      constraints are binding (|Ax - b| < tol).                          ║
// ║   2. For the target constraint, gradually increase b_orig (decrease b)  ║
// ║      by small steps, re-solve, and check if the same set of            ║
// ║      constraints is binding. Stop when the basis changes.               ║
// ║   3. Do the same for decreasing b_orig (increasing b).                  ║
// ║   4. Return the range [b_low_orig, b_high_orig].                        ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   lp             — the LP                                               ║
// ║   constraint_idx — which constraint to analyze (0-indexed)              ║
// ║   step           — step size for the search (default 0.5)               ║
// ║   max_steps      — maximum steps in each direction (default 200)        ║
// ║                                                                         ║
// ║ Returns: pair (b_low_orig, b_high_orig)                                 ║
// ║                                                                         ║
// ║ Helper: to find binding constraints for a solution x, compute           ║
// ║ slack = Ax - b and mark constraints where |slack| < 1e-6.              ║
// ║                                                                         ║
// ║ Note: this brute-force approach works for our small LPs. In practice,  ║
// ║ solvers compute allowable ranges algebraically using the inverse of     ║
// ║ the basis matrix: the range is determined by how far you can perturb    ║
// ║ b before B^{-1} b has a negative component (basis becomes infeasible). ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
std::pair<double, double> allowable_range(
    const LP& lp,
    int constraint_idx,
    double step = 0.5,
    int max_steps = 200)
{
    (void)lp;
    (void)constraint_idx;
    (void)step;
    (void)max_steps;
    throw std::runtime_error("TODO(human): implement allowable_range");
}

// ─── Helper: identify binding constraints ────────────────────────────────────

std::vector<bool> find_binding(const LP& lp, const VectorXd& x, double tol = 1e-6) {
    VectorXd slack = lp.A * x - lp.b;
    std::vector<bool> binding(lp.num_constraints());
    for (int i = 0; i < lp.num_constraints(); ++i) {
        binding[i] = std::abs(slack(i)) < tol;
    }
    return binding;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================================================================\n";
    std::cout << "  Phase 3: Sensitivity Analysis\n";
    std::cout << "================================================================\n\n";

    LP lp = make_sensitivity_problem();

    // Solve original LP
    auto result = solve_by_enumeration(lp);
    if (!result.feasible) {
        std::cout << "ERROR: LP is infeasible!\n";
        return 1;
    }

    std::cout << "Original LP solution:\n";
    for (int j = 0; j < lp.num_vars(); ++j) {
        std::cout << "  " << lp.var_names[j] << " = " << result.solution(j) << "\n";
    }
    std::cout << "  Objective (min form): " << result.obj_value << "\n";
    std::cout << "  Profit (max form):    " << -result.obj_value << "\n\n";

    // Show binding constraints
    auto binding = find_binding(lp, result.solution);
    VectorXd slack = lp.A * result.solution - lp.b;
    std::cout << "Constraint status:\n";
    for (int i = 0; i < lp.num_constraints(); ++i) {
        std::cout << "  " << lp.constraint_names[i]
                  << ": slack = " << slack(i)
                  << (binding[i] ? " (BINDING)" : " (slack)") << "\n";
    }
    std::cout << "\n";

    // Known dual variables (shadow prices) from Phase 2:
    VectorXd y_star(3);
    y_star << 2.0, 0.0, 1.0;
    std::cout << "Known shadow prices (from duality):\n";
    for (int i = 0; i < lp.num_constraints(); ++i) {
        std::cout << "  y_" << lp.constraint_names[i] << " = " << y_star(i) << "\n";
    }
    std::cout << "\n";

    // ── RHS Sensitivity ──────────────────────────────────────────────────
    std::cout << "--- RHS Sensitivity Analysis ---\n\n";

    auto derivatives = sensitivity_rhs(lp, 0.1);

    std::cout << "\nComparison: numerical dZ/db vs shadow price y:\n";
    std::cout << std::setw(15) << "Resource"
              << std::setw(15) << "dZ/db_orig"
              << std::setw(15) << "Shadow Price"
              << std::setw(15) << "Match?" << "\n";
    std::cout << std::string(60, '-') << "\n";
    for (int i = 0; i < lp.num_constraints(); ++i) {
        bool match = std::abs(derivatives[i] - y_star(i)) < 0.1;
        std::cout << std::setw(15) << lp.constraint_names[i]
                  << std::setw(15) << derivatives[i]
                  << std::setw(15) << y_star(i)
                  << std::setw(15) << (match ? "YES" : "NO") << "\n";
    }
    std::cout << "\n";

    // ── Allowable Ranges ─────────────────────────────────────────────────
    std::cout << "--- Allowable RHS Ranges ---\n\n";
    std::cout << "(Range of original RHS where current basis stays optimal)\n\n";

    for (int i = 0; i < lp.num_constraints(); ++i) {
        auto [low, high] = allowable_range(lp, i, 0.5, 200);
        std::cout << "  " << lp.constraint_names[i]
                  << ": b_orig in [" << low << ", " << high << "]"
                  << "  (current = " << -lp.b(i) << ")\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Phase 3 complete.\n";
    std::cout << "================================================================\n";

    return 0;
}
