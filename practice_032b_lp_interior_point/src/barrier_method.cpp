// ============================================================================
// Practice 032b — Phase 4: Barrier Method (Interior Point)
// ============================================================================
// Implement a primal-dual path-following interior point method for LP.
//
// Problem form (standard equality form for IPM):
//   minimize   c^T x
//   subject to A x = b
//              x >= 0
//
// The barrier subproblem:
//   minimize   c^T x  -  mu * SUM_i ln(x_i)
//   subject to A x = b
//
// KKT conditions for the barrier problem:
//   A^T lambda + s = c       (dual feasibility)
//   A x = b                   (primal feasibility)
//   X S e = mu * e            (perturbed complementarity)
//   x > 0, s > 0             (strict positivity)
//
// where X = diag(x), S = diag(s), e = (1,...,1)^T
//
// The algorithm reduces mu toward 0, taking Newton steps at each mu value.
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ─── Data structures ─────────────────────────────────────────────────────────

/// State of the barrier/IPM solver at each iteration.
struct BarrierState {
    VectorXd x;       // primal variables (n-vector, x > 0)
    VectorXd lambda;  // dual variables / Lagrange multipliers (m-vector)
    VectorXd s;       // dual slacks (n-vector, s > 0)
    double mu;        // barrier parameter (drives to 0)
};

/// Result of the barrier solver.
struct BarrierResult {
    bool converged;
    int iterations;
    double obj_value;
    VectorXd x;         // optimal primal
    VectorXd lambda;     // optimal dual
    VectorXd s;          // optimal dual slack
    double final_mu;
    double duality_gap;  // x^T s at termination
};

// ─── Utility functions ───────────────────────────────────────────────────────

/// Print a vector with label.
void print_vec(const std::string& label, const VectorXd& v) {
    std::cout << label << " = [";
    for (int i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::setw(8) << v(i);
    }
    std::cout << "]\n";
}

/// Compute duality gap = x^T s.
double duality_gap(const VectorXd& x, const VectorXd& s) {
    return x.dot(s);
}

/// Find the maximum step length alpha in (0, 1] such that
/// v + alpha * dv > 0 for all components.
double max_step(const VectorXd& v, const VectorXd& dv, double tau = 0.995) {
    double alpha = 1.0;
    for (int i = 0; i < v.size(); ++i) {
        if (dv(i) < 0) {
            alpha = std::min(alpha, -tau * v(i) / dv(i));
        }
    }
    return alpha;
}

// ─── TODO(human) implementations ────────────────────────────────────────────

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Compute the barrier objective value                        ║
// ║                                                                         ║
// ║ The barrier objective adds a logarithmic penalty to keep x > 0:         ║
// ║                                                                         ║
// ║   f(x) = c^T x  -  mu * SUM_{i=1}^{n} ln(x_i)                         ║
// ║                                                                         ║
// ║ As mu → 0, the barrier vanishes and f(x) → c^T x (the true LP obj).    ║
// ║ For mu > 0, the barrier pushes x away from the boundary x_i = 0.       ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   c  — objective coefficients (n-vector)                                ║
// ║   x  — current point (n-vector, all components > 0)                     ║
// ║   mu — barrier parameter (positive scalar)                              ║
// ║                                                                         ║
// ║ Returns: c^T x - mu * SUM ln(x_i)                                      ║
// ║                                                                         ║
// ║ Hint: use c.dot(x) for c^T x. For the log sum, iterate over            ║
// ║ components: for (int i = 0; i < x.size(); ++i) sum += std::log(x(i))   ║
// ║                                                                         ║
// ║ Edge case: if any x(i) <= 0, the barrier is +infinity (return a very    ║
// ║ large number like 1e30 to signal infeasibility).                        ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
double barrier_objective(const VectorXd& c, const VectorXd& x, double mu) {
    (void)c;
    (void)x;
    (void)mu;
    throw std::runtime_error("TODO(human): implement barrier_objective");
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Compute the Newton step for the KKT system                 ║
// ║                                                                         ║
// ║ The KKT system for the barrier problem is:                              ║
// ║                                                                         ║
// ║   A^T lambda + s = c         → residual r_dual = c - A^T lambda - s     ║
// ║   A x = b                     → residual r_primal = b - A x             ║
// ║   X S e = mu * e              → residual r_comp = mu*e - X*S*e          ║
// ║                                                                         ║
// ║ The Newton system (linearized KKT) is:                                  ║
// ║                                                                         ║
// ║   | 0    A^T   I  | | dx      |   | r_dual   |                         ║
// ║   | A    0     0  | | dlambda | = | r_primal |                          ║
// ║   | S    0     X  | | ds      |   | r_comp   |                          ║
// ║                                                                         ║
// ║ This is an (n + m + n) x (n + m + n) system. For small problems, we     ║
// ║ can assemble and solve it directly using Eigen.                         ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   A      — constraint matrix (m x n)                                    ║
// ║   b      — RHS (m-vector)                                               ║
// ║   c      — objective (n-vector)                                         ║
// ║   x      — current primal (n-vector, x > 0)                            ║
// ║   lambda — current dual (m-vector)                                      ║
// ║   s      — current dual slack (n-vector, s > 0)                         ║
// ║   mu     — current barrier parameter                                    ║
// ║                                                                         ║
// ║ Returns: tuple (dx, dlambda, ds) as VectorXd's                          ║
// ║                                                                         ║
// ║ Steps:                                                                  ║
// ║   1. Compute residuals:                                                 ║
// ║      r_dual   = c - A^T * lambda - s                                    ║
// ║      r_primal = b - A * x                                               ║
// ║      r_comp   = mu * ones - x.cwiseProduct(s)   (element-wise x_i*s_i)  ║
// ║                                                                         ║
// ║   2. Build the (2n+m) x (2n+m) block matrix K:                         ║
// ║      K = | 0_{nxn}   A^T      I_n    |                                 ║
// ║          | A         0_{mxm}  0_{mxn} |                                 ║
// ║          | S_diag    0_{nxm}  X_diag  |                                 ║
// ║      where S_diag = diag(s), X_diag = diag(x)                          ║
// ║                                                                         ║
// ║   3. Build the RHS vector: [r_dual; r_primal; r_comp]                   ║
// ║                                                                         ║
// ║   4. Solve K * [dx; dlambda; ds] = rhs using Eigen:                     ║
// ║      VectorXd step = K.colPivHouseholderQr().solve(rhs);               ║
// ║                                                                         ║
// ║   5. Extract dx (first n), dlambda (next m), ds (last n) from step.    ║
// ║                                                                         ║
// ║ Hint for building K in Eigen:                                           ║
// ║   MatrixXd K = MatrixXd::Zero(2*n + m, 2*n + m);                       ║
// ║   K.block(0, n, n, m) = A.transpose();   // A^T in top-right           ║
// ║   K.block(0, n+m, n, n) = MatrixXd::Identity(n,n);  // I_n            ║
// ║   K.block(n, 0, m, n) = A;               // A in middle-left           ║
// ║   K.block(n+m, 0, n, n) = s.asDiagonal();  // S_diag in bottom-left   ║
// ║   K.block(n+m, n+m, n, n) = x.asDiagonal(); // X_diag in bottom-right ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
struct NewtonDirection {
    VectorXd dx;
    VectorXd dlambda;
    VectorXd ds;
};

NewtonDirection newton_step(
    const MatrixXd& A,
    const VectorXd& b,
    const VectorXd& c,
    const VectorXd& x,
    const VectorXd& lambda,
    const VectorXd& s,
    double mu)
{
    (void)A;
    (void)b;
    (void)c;
    (void)x;
    (void)lambda;
    (void)s;
    (void)mu;
    throw std::runtime_error("TODO(human): implement newton_step");
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Barrier method main loop (path-following IPM)              ║
// ║                                                                         ║
// ║ This is the full interior point solver. Starting from a strictly        ║
// ║ feasible interior point, repeatedly:                                    ║
// ║   1. Compute the Newton direction (dx, dlambda, ds)                     ║
// ║   2. Line search to maintain x > 0 and s > 0                           ║
// ║   3. Update (x, lambda, s)                                              ║
// ║   4. Reduce mu (the barrier parameter)                                  ║
// ║   5. Check convergence via duality gap                                  ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   A         — constraint matrix (m x n), for Ax = b                     ║
// ║   b         — RHS (m-vector)                                            ║
// ║   c         — objective (n-vector)                                      ║
// ║   x0        — initial strictly feasible primal (x0 > 0, A*x0 = b)      ║
// ║   mu_init   — initial barrier parameter (e.g., 10.0)                    ║
// ║   sigma     — centering parameter, mu_new = sigma * mu (e.g., 0.2)     ║
// ║   tol       — convergence tolerance on duality gap (e.g., 1e-8)        ║
// ║   max_iter  — maximum iterations (e.g., 100)                           ║
// ║                                                                         ║
// ║ Algorithm:                                                              ║
// ║   1. Initialize: x = x0, compute initial lambda and s:                  ║
// ║      - s = c - A^T * lambda (choose lambda = 0 initially)              ║
// ║      - If any s_i <= 0, shift: s = s + (|min(s)| + 1) * ones          ║
// ║        (This is a heuristic to get a starting s > 0)                    ║
// ║   2. Compute mu = x^T s / n (average complementarity)                   ║
// ║   3. For iter = 0 to max_iter:                                          ║
// ║      a. Set target mu = sigma * (x^T s / n)                             ║
// ║      b. Compute Newton direction: (dx, dlambda, ds) via newton_step()   ║
// ║      c. Step length: alpha_p = max_step(x, dx)                          ║
// ║                      alpha_d = max_step(s, ds)                          ║
// ║      d. Update: x += alpha_p * dx                                       ║
// ║                 lambda += alpha_d * dlambda                              ║
// ║                 s += alpha_d * ds                                        ║
// ║      e. Compute duality gap = x^T s                                     ║
// ║      f. Print iteration info: iter, obj, gap, mu, alpha_p, alpha_d      ║
// ║      g. If gap < tol: converged!                                        ║
// ║   4. Return BarrierResult with final x, lambda, s, obj, gap.            ║
// ║                                                                         ║
// ║ Note on starting point: finding a strictly feasible (x0, s0) is itself  ║
// ║ a nontrivial problem. For this exercise, we provide x0 for each test   ║
// ║ problem. In production solvers, a "Phase I" method or self-dual         ║
// ║ embedding is used.                                                      ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
BarrierResult solve_barrier(
    const MatrixXd& A,
    const VectorXd& b,
    const VectorXd& c,
    const VectorXd& x0,
    double sigma = 0.2,
    double tol = 1e-8,
    int max_iter = 100)
{
    (void)A;
    (void)b;
    (void)c;
    (void)x0;
    (void)sigma;
    (void)tol;
    (void)max_iter;
    throw std::runtime_error("TODO(human): implement solve_barrier");
}

// ─── Test problems (in equality standard form Ax = b, x >= 0) ───────────────

/// Test 1: Simple 2-variable LP
///   maximize  3x1 + 5x2
///   s.t.      x1        <= 4
///                  2x2  <= 12
///             3x1 + 2x2 <= 18
///             x1, x2 >= 0
///
/// Standard form (add slacks x3, x4, x5):
///   minimize  -3x1 - 5x2
///   s.t.      x1      + x3           = 4
///                  2x2      + x4      = 12
///             3x1 + 2x2          + x5 = 18
///   x1..x5 >= 0
///
/// Known optimal: x1=2, x2=6, x3=2, x4=0, x5=0, obj = -36
///
struct TestProblem {
    MatrixXd A;
    VectorXd b;
    VectorXd c;
    VectorXd x0;       // strictly feasible starting point (x0 > 0, Ax0 = b)
    double known_opt;   // known optimal objective value
    std::string name;
};

TestProblem make_test1() {
    TestProblem p;
    p.name = "Production LP (2 vars + 3 slacks)";

    // 3 constraints, 5 variables (2 original + 3 slack)
    p.A.resize(3, 5);
    p.A << 1, 0, 1, 0, 0,
           0, 2, 0, 1, 0,
           3, 2, 0, 0, 1;

    p.b.resize(3);
    p.b << 4, 12, 18;

    p.c.resize(5);
    p.c << -3, -5, 0, 0, 0;

    // Starting point: x = (1, 1, 3, 10, 13) — strictly feasible
    // Check: 1+3=4 ✓, 2+10=12 ✓, 3+2+13=18 ✓, all > 0 ✓
    p.x0.resize(5);
    p.x0 << 1, 1, 3, 10, 13;

    p.known_opt = -36.0;  // min(-3*2 - 5*6) = -36

    return p;
}

/// Test 2: 3-variable LP
///   maximize  2x1 + 3x2 + x3
///   s.t.      x1 + x2 + x3 <= 10
///             2x1 + x2      <= 8
///                  x2 + x3  <= 7
///   x1..x3 >= 0
///
/// Standard form (add slacks x4, x5, x6):
///   minimize  -2x1 - 3x2 - x3
///   s.t.      x1 + x2 + x3 + x4          = 10
///             2x1 + x2           + x5      = 8
///                   x2 + x3           + x6 = 7
///   x1..x6 >= 0
///
/// Known optimal: x1=0, x2=1, x3=6, obj = -9
TestProblem make_test2() {
    TestProblem p;
    p.name = "Resource LP (3 vars + 3 slacks)";

    p.A.resize(3, 6);
    p.A << 1, 1, 1, 1, 0, 0,
           2, 1, 0, 0, 1, 0,
           0, 1, 1, 0, 0, 1;

    p.b.resize(3);
    p.b << 10, 8, 7;

    p.c.resize(6);
    p.c << -2, -3, -1, 0, 0, 0;

    // Starting point: x = (1, 1, 1, 7, 5, 5) — strictly feasible
    // Check: 1+1+1+7=10 ✓, 2+1+5=8 ✓, 1+1+5=7 ✓, all > 0 ✓
    p.x0.resize(6);
    p.x0 << 1, 1, 1, 7, 5, 5;

    p.known_opt = -9.0;  // min(-2*0 - 3*1 - 1*6) = -9

    return p;
}

/// Test 3: Degenerate LP (to test robustness)
///   minimize  x1 + x2
///   s.t.      x1 + x2 = 1
///             x1, x2 >= 0
///
/// Known optimal: any (x1, x2) with x1 + x2 = 1, x >= 0. obj = 1.
/// The barrier method will find the analytic center: x1 = x2 = 0.5.
TestProblem make_test3() {
    TestProblem p;
    p.name = "Degenerate LP (analytic center)";

    p.A.resize(1, 2);
    p.A << 1, 1;

    p.b.resize(1);
    p.b << 1;

    p.c.resize(2);
    p.c << 1, 1;

    p.x0.resize(2);
    p.x0 << 0.5, 0.5;

    p.known_opt = 1.0;

    return p;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================================================================\n";
    std::cout << "  Phase 4: Barrier Method (Interior Point)\n";
    std::cout << "================================================================\n\n";

    std::vector<TestProblem> problems = {make_test1(), make_test2(), make_test3()};

    for (size_t p = 0; p < problems.size(); ++p) {
        const auto& prob = problems[p];
        std::cout << "--- Test " << (p + 1) << ": " << prob.name << " ---\n\n";

        std::cout << "Problem size: " << prob.A.rows() << " constraints, "
                  << prob.A.cols() << " variables\n";
        std::cout << "Known optimal value: " << prob.known_opt << "\n\n";

        // Verify starting point
        VectorXd residual = prob.A * prob.x0 - prob.b;
        std::cout << "Starting point feasibility (||Ax0 - b||): "
                  << residual.norm() << "\n";
        std::cout << "Starting point min component: " << prob.x0.minCoeff() << "\n";
        std::cout << "Starting objective: " << prob.c.dot(prob.x0) << "\n\n";

        // Solve
        std::cout << "Running barrier method...\n";
        std::cout << std::setw(6) << "Iter"
                  << std::setw(14) << "Objective"
                  << std::setw(14) << "Gap"
                  << std::setw(14) << "Mu"
                  << std::setw(10) << "alpha_p"
                  << std::setw(10) << "alpha_d" << "\n";
        std::cout << std::string(68, '-') << "\n";

        auto result = solve_barrier(prob.A, prob.b, prob.c, prob.x0,
                                    0.2, 1e-8, 100);

        std::cout << "\n";
        if (result.converged) {
            std::cout << "CONVERGED in " << result.iterations << " iterations\n";
        } else {
            std::cout << "DID NOT CONVERGE after " << result.iterations << " iterations\n";
        }

        std::cout << "Final objective: " << result.obj_value << "\n";
        std::cout << "Known optimal:   " << prob.known_opt << "\n";
        std::cout << "Difference:      " << std::abs(result.obj_value - prob.known_opt) << "\n";
        std::cout << "Duality gap:     " << result.duality_gap << "\n";

        print_vec("  x*", result.x);
        print_vec("  lambda*", result.lambda);
        print_vec("  s*", result.s);

        std::cout << "\n";
    }

    std::cout << "================================================================\n";
    std::cout << "  Phase 4 complete.\n";
    std::cout << "================================================================\n";

    return 0;
}
