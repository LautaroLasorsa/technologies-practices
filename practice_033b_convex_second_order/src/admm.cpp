// Practice 033b -- Phase 2: ADMM (Alternating Direction Method of Multipliers)
// Solve a box-constrained quadratic program by splitting into simple subproblems.

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ============================================================================
// Data structures
// ============================================================================

/// Result of ADMM: final solution, convergence history, residuals.
struct ADMMResult {
    VectorXd x_final;                    ///< Final primal variable x
    VectorXd z_final;                    ///< Final split variable z
    VectorXd u_final;                    ///< Final scaled dual variable u
    std::vector<double> objectives;      ///< f(x_k) at each iteration
    std::vector<double> primal_residuals; ///< ||x - z|| at each iteration
    std::vector<double> dual_residuals;   ///< rho * ||z_k - z_{k-1}|| at each iteration
    int iterations;
    bool converged;
};

// ============================================================================
// Problem setup: box-constrained quadratic program
// ============================================================================

/// A box-constrained QP:
///   minimize    (1/2) x^T P x + q^T x
///   subject to  lower <= x <= upper
///
/// We solve this via ADMM in consensus form:
///   minimize    f(x) + g(z)
///   subject to  x = z
///
/// where f(x) = (1/2) x^T P x + q^T x  (quadratic objective, smooth)
///       g(z) = indicator of {z : lower <= z <= upper}  (box constraint)
///
/// The augmented Lagrangian is:
///   L_rho(x, z, u) = f(x) + g(z) + (rho/2) ||x - z + u||^2
///
/// where u is the scaled dual variable (y/rho in unscaled form).
struct BoxConstrainedQP {
    MatrixXd P;      ///< Positive definite quadratic term (n x n)
    VectorXd q;      ///< Linear term (n x 1)
    VectorXd lower;  ///< Lower bounds (n x 1)
    VectorXd upper;  ///< Upper bounds (n x 1)

    int dim() const { return static_cast<int>(q.size()); }

    /// Unconstrained optimum: x* = -P^{-1} q
    VectorXd unconstrained_optimum() const {
        return P.ldlt().solve(-q);
    }

    /// Objective value: (1/2) x^T P x + q^T x
    double objective(const VectorXd& x) const {
        return 0.5 * x.dot(P * x) + q.dot(x);
    }

    /// Gradient: P*x + q
    VectorXd gradient(const VectorXd& x) const {
        return P * x + q;
    }

    /// Check if x satisfies box constraints
    bool is_feasible(const VectorXd& x, double tol = 1e-8) const {
        for (int i = 0; i < dim(); ++i) {
            if (x(i) < lower(i) - tol || x(i) > upper(i) + tol) return false;
        }
        return true;
    }

    /// Project onto box constraints (for comparison / verification)
    VectorXd project(const VectorXd& x) const {
        return x.cwiseMax(lower).cwiseMin(upper);
    }
};

/// Create a random box-constrained QP where the unconstrained optimum
/// lies OUTSIDE the box, so constraints are active.
BoxConstrainedQP make_box_qp(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    BoxConstrainedQP prob;

    // Generate a random PD matrix: P = A^T A + mu * I
    MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = normal(rng);
    prob.P = A.transpose() * A + 1.0 * MatrixXd::Identity(n, n);

    // Linear term that pushes optimum outside the box
    prob.q = VectorXd(n);
    for (int i = 0; i < n; ++i) prob.q(i) = normal(rng) * 5.0;

    // Box constraints: [-1, 1] for each variable
    prob.lower = VectorXd::Constant(n, -1.0);
    prob.upper = VectorXd::Constant(n,  1.0);

    return prob;
}

// ============================================================================
// TODO(human): ADMM x-update
// ============================================================================

/// ADMM x-update: minimize over x the augmented Lagrangian.
///
/// TODO(human): Implement the x-update step of ADMM.
///
///   The augmented Lagrangian (in scaled form) is:
///     L_rho(x, z, u) = (1/2) x^T P x + q^T x + (rho/2) ||x - z + u||^2
///
///   Taking the derivative w.r.t. x and setting to zero:
///     P*x + q + rho*(x - z + u) = 0
///     (P + rho*I) * x = -q + rho*(z - u)
///
///   This is a symmetric positive definite linear system (because P is PD
///   and rho > 0, so P + rho*I is strictly PD). Solve it with Eigen's
///   LDLT decomposition, which is efficient for SPD matrices:
///
///     VectorXd rhs = -q + rho * (z - u);
///     VectorXd x = (P + rho * MatrixXd::Identity(n, n)).ldlt().solve(rhs);
///
///   Performance note: in a real implementation, you would pre-factor
///   (P + rho*I) once and reuse it across iterations (since P and rho
///   don't change). Here, re-factoring each iteration is fine for learning.
///
///   The rho*I term is critical: it makes the system well-conditioned even
///   if P is nearly singular, and it controls the trade-off between
///   minimizing f(x) and satisfying the consensus constraint x = z.
///
/// @param P    Quadratic term (n x n, positive definite)
/// @param q    Linear term (n x 1)
/// @param rho  ADMM penalty parameter
/// @param z    Current z variable (n x 1)
/// @param u    Current scaled dual variable (n x 1)
/// @return     Updated x variable
VectorXd admm_x_update(const MatrixXd& P, const VectorXd& q,
                        double rho, const VectorXd& z, const VectorXd& u) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): ADMM z-update (projection onto box constraints)
// ============================================================================

/// ADMM z-update: project (x + u) onto the box [lower, upper].
///
/// TODO(human): Implement the z-update step of ADMM.
///
///   The z-update minimizes over z:
///     g(z) + (rho/2) ||x_{k+1} - z + u_k||^2
///
///   where g(z) is the indicator function of the box {z : lower <= z <= upper}:
///     g(z) = 0     if lower <= z <= upper
///     g(z) = +inf  otherwise
///
///   Minimizing "indicator + quadratic" is equivalent to projecting the
///   unconstrained minimizer (x + u) onto the box. The projection is:
///     z_i = clamp(x_i + u_i, lower_i, upper_i)
///         = max(lower_i, min(upper_i, x_i + u_i))
///
///   This is the simplest possible z-update: just element-wise clamping.
///   For more complex g (e.g., L1 norm), the z-update would be soft-thresholding.
///   For polyhedral constraints, it would be a QP. The beauty of ADMM is that
///   each subproblem can be solved independently using the best available method.
///
///   With Eigen, you can do this concisely:
///     (x + u).cwiseMax(lower).cwiseMin(upper)
///
/// @param x      Current x variable after x-update (n x 1)
/// @param u      Current scaled dual variable (n x 1)
/// @param lower  Lower bounds (n x 1)
/// @param upper  Upper bounds (n x 1)
/// @return       Updated z variable (projected onto box)
VectorXd admm_z_update(const VectorXd& x, const VectorXd& u,
                        const VectorXd& lower, const VectorXd& upper) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Full ADMM loop
// ============================================================================

/// ADMM solver for box-constrained QP.
///
/// TODO(human): Implement the complete ADMM iteration loop.
///
///   ADMM alternates three updates per iteration:
///
///     1. x-update: x_{k+1} = argmin_x { f(x) + (rho/2)||x - z_k + u_k||^2 }
///        -> Solve (P + rho*I) x = -q + rho*(z_k - u_k)
///        -> Use admm_x_update()
///
///     2. z-update: z_{k+1} = argmin_z { g(z) + (rho/2)||x_{k+1} - z + u_k||^2 }
///        -> Project x_{k+1} + u_k onto box constraints
///        -> Use admm_z_update()
///
///     3. Dual update: u_{k+1} = u_k + (x_{k+1} - z_{k+1})
///        -> This is gradient ascent on the dual variable
///        -> It penalizes violation of the consensus constraint x = z
///
///   Convergence is monitored via two residuals:
///
///     Primal residual: r_k = ||x_{k+1} - z_{k+1}||
///       -> Measures constraint violation (how far x and z are from agreement)
///       -> Should decrease to zero
///
///     Dual residual:   s_k = rho * ||z_{k+1} - z_k||
///       -> Measures how much the z variable changed (proxy for dual optimality)
///       -> When z stops changing, we're near the dual optimum
///       -> The rho scaling makes it comparable in magnitude to the primal residual
///
///   Stop when BOTH residuals are small:
///     ||r_k|| < tol  AND  ||s_k|| < tol
///
///   The penalty parameter rho controls convergence speed:
///     - Large rho: x-update emphasizes x ≈ z (fast constraint satisfaction, slow objective decrease)
///     - Small rho: x-update emphasizes minimizing f (fast objective decrease, slow consensus)
///     - Typical starting values: rho = 1.0 or rho = 1/n
///     - Advanced: adaptive rho (increase if primal residual >> dual, decrease if dual >> primal)
///
///   Initialize: z_0 = 0, u_0 = 0 (or any feasible z_0)
///
///   Algorithm:
///     z = VectorXd::Zero(n), u = VectorXd::Zero(n)
///     for k = 0, ..., max_iter-1:
///         x = admm_x_update(P, q, rho, z, u)
///         z_old = z
///         z = admm_z_update(x, u, lower, upper)
///         u = u + (x - z)
///
///         primal_res = ||x - z||
///         dual_res = rho * ||z - z_old||
///
///         record objective(x), primal_res, dual_res
///
///         if primal_res < tol AND dual_res < tol:
///             converged = true; break
///
/// @param P         Quadratic term (n x n)
/// @param q         Linear term (n x 1)
/// @param lower     Lower bounds (n x 1)
/// @param upper     Upper bounds (n x 1)
/// @param rho       ADMM penalty parameter
/// @param max_iter  Maximum iterations
/// @param tol       Convergence tolerance on both residuals
/// @return          ADMMResult with solution and convergence history
ADMMResult admm_solve(const MatrixXd& P, const VectorXd& q,
                      const VectorXd& lower, const VectorXd& upper,
                      double rho, int max_iter, double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// Print helpers
// ============================================================================

void print_admm_header() {
    std::cout << std::setw(6)  << "iter"
              << std::setw(16) << "objective"
              << std::setw(16) << "primal_res"
              << std::setw(16) << "dual_res"
              << std::endl;
    std::cout << std::string(54, '-') << std::endl;
}

void print_admm_row(int iter, double obj, double primal_res, double dual_res) {
    std::cout << std::setw(6)  << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << obj
              << std::setw(16) << primal_res
              << std::setw(16) << dual_res
              << std::endl;
}

// ============================================================================
// main: solve box-constrained QP via ADMM
// ============================================================================

int main() {
    std::cout << "=== Phase 2: ADMM for Box-Constrained QP ===" << std::endl;
    std::cout << std::endl;

    // --- Problem setup ---
    int n = 10;
    auto prob = make_box_qp(n);

    VectorXd x_unc = prob.unconstrained_optimum();
    double f_unc = prob.objective(x_unc);
    bool unc_feasible = prob.is_feasible(x_unc);

    std::cout << "Problem: n=" << n << ", box constraints [-1, 1]" << std::endl;
    std::cout << "Unconstrained optimum: f* = " << std::scientific << std::setprecision(4) << f_unc << std::endl;
    std::cout << "Unconstrained optimum feasible? " << (unc_feasible ? "YES (constraints inactive)" : "NO (constraints active)")
              << std::endl;

    if (!unc_feasible) {
        std::cout << "Unconstrained x*: [";
        for (int i = 0; i < std::min(n, 5); ++i) {
            std::cout << std::fixed << std::setprecision(3) << x_unc(i);
            if (i < std::min(n, 5) - 1) std::cout << ", ";
        }
        if (n > 5) std::cout << ", ...";
        std::cout << "]" << std::endl;
        std::cout << "Some components violate [-1, 1] -> ADMM will find the constrained optimum." << std::endl;
    }
    std::cout << std::endl;

    // --- Solve via ADMM ---
    double rho = 1.0;
    int max_iter = 500;
    double tol = 1e-6;

    std::cout << "Running ADMM (rho=" << rho << ", max_iter=" << max_iter << ", tol=" << tol << ")..." << std::endl;
    std::cout << std::endl;

    try {
        auto result = admm_solve(prob.P, prob.q, prob.lower, prob.upper,
                                  rho, max_iter, tol);

        // Print convergence log
        print_admm_header();
        int print_first = std::min(10, result.iterations);
        for (int i = 0; i < print_first; ++i) {
            print_admm_row(i, result.objectives[i], result.primal_residuals[i], result.dual_residuals[i]);
        }
        if (result.iterations > 15) {
            std::cout << "  ..." << std::endl;
            for (int i = result.iterations - 5; i < result.iterations; ++i) {
                print_admm_row(i, result.objectives[i], result.primal_residuals[i], result.dual_residuals[i]);
            }
        }
        std::cout << std::endl;

        std::cout << "Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "ADMM objective:         " << std::scientific << std::setprecision(6)
                  << prob.objective(result.x_final) << std::endl;
        std::cout << "Unconstrained objective: " << f_unc << std::endl;
        std::cout << "Feasible? " << (prob.is_feasible(result.x_final) ? "YES" : "NO") << std::endl;
        std::cout << std::endl;

        // Show solution
        std::cout << "--- Solution comparison ---" << std::endl;
        std::cout << std::setw(6)  << "i"
                  << std::setw(12) << "x_admm"
                  << std::setw(12) << "x_unc"
                  << std::setw(12) << "lower"
                  << std::setw(12) << "upper"
                  << std::setw(10) << "active?"
                  << std::endl;
        std::cout << std::string(62, '-') << std::endl;

        for (int i = 0; i < n; ++i) {
            double xi = result.x_final(i);
            bool at_lower = std::abs(xi - prob.lower(i)) < 1e-4;
            bool at_upper = std::abs(xi - prob.upper(i)) < 1e-4;
            std::string active = at_lower ? "lower" : (at_upper ? "upper" : "");

            std::cout << std::setw(6)  << i
                      << std::setw(12) << std::fixed << std::setprecision(4) << xi
                      << std::setw(12) << x_unc(i)
                      << std::setw(12) << prob.lower(i)
                      << std::setw(12) << prob.upper(i)
                      << std::setw(10) << active
                      << std::endl;
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    // --- Experiment: different rho values ---
    std::cout << std::endl;
    std::cout << "--- Effect of rho on convergence speed ---" << std::endl;
    std::cout << std::setw(10) << "rho"
              << std::setw(12) << "iterations"
              << std::setw(16) << "objective"
              << std::endl;
    std::cout << std::string(38, '-') << std::endl;

    for (double rho_test : {0.01, 0.1, 1.0, 10.0, 100.0}) {
        try {
            auto result = admm_solve(prob.P, prob.q, prob.lower, prob.upper,
                                      rho_test, 2000, 1e-6);
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << rho_test
                      << std::setw(12) << result.iterations
                      << std::setw(16) << std::scientific << std::setprecision(6)
                      << prob.objective(result.x_final)
                      << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << rho_test
                      << "  ERROR: " << e.what() << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. ADMM splits constrained problems into: linear solve + projection + dual update" << std::endl;
    std::cout << "  2. Each subproblem is trivial; together they solve the hard constrained problem" << std::endl;
    std::cout << "  3. Primal residual = constraint violation, dual residual = optimality gap" << std::endl;
    std::cout << "  4. rho affects speed (not correctness): too small or too large both slow convergence" << std::endl;
    std::cout << "  5. ADMM is robust: converges under mild conditions, no line search needed" << std::endl;

    return 0;
}
