// Practice 033a — Phase 2: Backtracking Line Search
// Implement Armijo backtracking line search and compare with fixed-step GD.

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <limits>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ============================================================================
// Data structures
// ============================================================================

struct GDResult {
    VectorXd x_final;
    std::vector<VectorXd> trajectory;
    std::vector<double> f_values;
    std::vector<double> grad_norms;
    int iterations;
    bool converged;
};

/// General interface for a differentiable function.
/// Implemented as a struct with std::function members for flexibility.
struct DifferentiableFunction {
    std::function<double(const VectorXd&)> eval;
    std::function<VectorXd(const VectorXd&)> gradient;
    std::string name;
};

// ============================================================================
// Test functions
// ============================================================================

/// Convex quadratic: f(x) = 0.5 * x^T Q x + q^T x
DifferentiableFunction make_quadratic(const MatrixXd& Q, const VectorXd& q) {
    DifferentiableFunction func;
    func.name = "Quadratic";
    func.eval = [Q, q](const VectorXd& x) -> double {
        return 0.5 * x.dot(Q * x) + q.dot(x);
    };
    func.gradient = [Q, q](const VectorXd& x) -> VectorXd {
        return Q * x + q;
    };
    return func;
}

/// Smooth strongly-convex function (modified Rosenbrock):
///   f(x, y) = 100*(y - x^2)^2 + (1 - x)^2 + 0.5*(x^2 + y^2)
///
/// The standard Rosenbrock is non-convex, but adding 0.5*||x||^2
/// makes this strongly convex while keeping the narrow curved valley
/// that makes optimization challenging.
///
/// This tests whether your line search can handle varying curvature:
///   - Very steep walls perpendicular to the valley
///   - Very flat floor along the valley
DifferentiableFunction make_regularized_rosenbrock() {
    DifferentiableFunction func;
    func.name = "Regularized Rosenbrock";

    func.eval = [](const VectorXd& x) -> double {
        double a = x(1) - x(0) * x(0);
        double b = 1.0 - x(0);
        return 100.0 * a * a + b * b + 0.5 * x.squaredNorm();
    };

    func.gradient = [](const VectorXd& x) -> VectorXd {
        VectorXd grad(2);
        double a = x(1) - x(0) * x(0);
        // df/dx0 = -400*x0*(x1 - x0^2) - 2*(1 - x0) + x0
        grad(0) = -400.0 * x(0) * a - 2.0 * (1.0 - x(0)) + x(0);
        // df/dx1 = 200*(x1 - x0^2) + x1
        grad(1) = 200.0 * a + x(1);
        return grad;
    };

    return func;
}

/// Log-sum-exp function: f(x) = log(sum_i exp(a_i^T x + b_i))
/// This is a smooth convex approximation to max(a_i^T x + b_i).
/// Gradient: sum_i (exp(a_i^T x + b_i) / sum_j exp(a_j^T x + b_j)) * a_i
DifferentiableFunction make_log_sum_exp(const MatrixXd& A, const VectorXd& b) {
    DifferentiableFunction func;
    func.name = "Log-Sum-Exp";

    func.eval = [A, b](const VectorXd& x) -> double {
        VectorXd z = A * x + b;
        double max_z = z.maxCoeff();  // Numerical stability trick
        double sum = 0.0;
        for (int i = 0; i < z.size(); ++i) {
            sum += std::exp(z(i) - max_z);
        }
        return max_z + std::log(sum);
    };

    func.gradient = [A, b](const VectorXd& x) -> VectorXd {
        VectorXd z = A * x + b;
        double max_z = z.maxCoeff();
        VectorXd exp_z(z.size());
        for (int i = 0; i < z.size(); ++i) {
            exp_z(i) = std::exp(z(i) - max_z);
        }
        double sum = exp_z.sum();
        VectorXd weights = exp_z / sum;  // Softmax weights
        return A.transpose() * weights;
    };

    return func;
}

// ============================================================================
// Print helpers
// ============================================================================

void print_header() {
    std::cout << std::setw(6) << "iter"
              << std::setw(16) << "f(x)"
              << std::setw(16) << "||grad||"
              << std::setw(12) << "alpha"
              << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

void print_iteration(int iter, double f_val, double grad_norm, double alpha) {
    std::cout << std::setw(6) << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << f_val
              << std::setw(16) << grad_norm
              << std::setw(12) << std::fixed << std::setprecision(6) << alpha
              << std::endl;
}

// ============================================================================
// Fixed-step gradient descent (provided for comparison)
// ============================================================================

GDResult gd_fixed_step(const DifferentiableFunction& func,
                       const VectorXd& x0,
                       double alpha,
                       int max_iter,
                       double tol) {
    GDResult result;
    result.converged = false;

    VectorXd x = x0;
    for (int k = 0; k < max_iter; ++k) {
        VectorXd grad = func.gradient(x);
        double grad_norm = grad.norm();
        double f_val = func.eval(x);

        result.trajectory.push_back(x);
        result.f_values.push_back(f_val);
        result.grad_norms.push_back(grad_norm);

        if (grad_norm < tol) {
            result.converged = true;
            result.iterations = k;
            result.x_final = x;
            return result;
        }

        // Check for divergence
        if (std::isnan(f_val) || std::isinf(f_val) || f_val > 1e20) {
            result.iterations = k;
            result.x_final = x;
            return result;
        }

        x = x - alpha * grad;
    }

    result.iterations = max_iter;
    result.x_final = x;
    return result;
}

// ============================================================================
// TODO(human): Implement backtracking line search
// ============================================================================

/// Backtracking line search using the Armijo sufficient decrease condition.
///
/// TODO(human): Implement the Armijo backtracking algorithm.
///
///   Backtracking line search (Armijo condition):
///     Start with alpha = alpha_init
///     While f(x + alpha * d) > f(x) + c1 * alpha * gradient(x)^T d:
///         alpha = beta * alpha  (shrink step)
///
///   Parameters:
///     x:          current point
///     direction:  search direction d (typically d = -gradient for steepest descent)
///     alpha_init: starting step size (e.g., 1.0)
///     beta:       shrink factor in (0, 1), typically 0.5
///     c1:         sufficient decrease parameter in (0, 1), typically 1e-4
///
///   The Armijo condition ensures "sufficient decrease" — we don't accept
///   a step unless it reduces f by at least c1 * alpha * directional_derivative.
///   This prevents tiny steps that barely improve and huge steps that overshoot.
///
///   The directional derivative is: gradient(x)^T * d
///   For steepest descent (d = -gradient), this equals -||gradient||^2 (always negative).
///
///   Algorithm:
///     1. Compute f_current = func.eval(x)
///     2. Compute directional_derivative = func.gradient(x).dot(direction)
///     3. Set alpha = alpha_init
///     4. While f(x + alpha * direction) > f_current + c1 * alpha * directional_derivative:
///          alpha = beta * alpha
///        (add a max-iteration guard, e.g., 50 backtracks, to prevent infinite loops)
///     5. Return alpha
///
///   Why it works: The Armijo condition is a relaxation of "decrease f".
///   The parameter c1 controls how much decrease we require:
///     c1 = 0   -> accept any decrease (too loose, can stall)
///     c1 = 1   -> require full linear decrease (too strict, tiny steps)
///     c1 = 1e-4 -> practical sweet spot (almost any decrease is fine)
double backtracking_line_search(const DifferentiableFunction& func,
                                const VectorXd& x,
                                const VectorXd& direction,
                                double alpha_init = 1.0,
                                double beta = 0.5,
                                double c1 = 1e-4) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Implement GD with line search
// ============================================================================

/// Gradient descent using backtracking line search for step size selection.
///
/// TODO(human): Implement gradient descent with Armijo line search.
///
///   This is like vanilla GD, but instead of using a fixed alpha, you call
///   backtracking_line_search() at each iteration to find a good alpha.
///
///   Algorithm:
///     1. Set x = x0
///     2. For k = 0, 1, ..., max_iter-1:
///        a. Compute grad = func.gradient(x)
///        b. If ||grad|| < tol, stop (converged)
///        c. Set direction = -grad (steepest descent direction)
///        d. Find alpha = backtracking_line_search(func, x, direction)
///        e. Update: x = x + alpha * direction (= x - alpha * grad)
///        f. Record x, f(x), ||grad||
///     3. Return result
///
///   Key advantages over fixed step:
///     - No need to know L (the Lipschitz constant)
///     - Adapts to local curvature (takes bigger steps in flat regions)
///     - More robust: won't diverge if you start far from optimum
///
///   Key disadvantage:
///     - Extra function evaluations per iteration (typically 1-5 backtracks)
///     - For a quadratic, this overhead may not be worth it (L is easy to compute)
///     - For complex functions (neural networks), function evals are expensive
GDResult gd_with_line_search(const DifferentiableFunction& func,
                             const VectorXd& x0,
                             int max_iter,
                             double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// main: compare fixed-step vs line-search GD
// ============================================================================

int main() {
    std::cout << "=== Phase 2: Backtracking Line Search ===" << std::endl;
    std::cout << std::endl;

    // --- Experiment 1: Quadratic (kappa = 50) ---
    std::cout << "--- Experiment 1: Quadratic (kappa = 50) ---" << std::endl;
    MatrixXd Q(2, 2);
    double theta = 0.5;
    MatrixXd R(2, 2);
    R << std::cos(theta), -std::sin(theta),
         std::sin(theta),  std::cos(theta);
    MatrixXd D = MatrixXd::Zero(2, 2);
    D(0, 0) = 1.0;   // mu
    D(1, 1) = 50.0;  // L
    Q = R * D * R.transpose();

    VectorXd q(2);
    q << 1.0, -2.0;

    auto func_quad = make_quadratic(Q, q);
    double L_quad = 50.0;

    VectorXd x0(2);
    x0 << 5.0, 5.0;

    std::cout << "  Fixed step (alpha = 1/L = " << 1.0 / L_quad << "):" << std::endl;
    try {
        auto result_fixed = gd_fixed_step(func_quad, x0, 1.0 / L_quad, 2000, 1e-8);
        std::cout << "    Converged: " << (result_fixed.converged ? "yes" : "no")
                  << " in " << result_fixed.iterations << " iterations" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "    ERROR: " << e.what() << std::endl;
    }

    std::cout << "  Line search:" << std::endl;
    try {
        auto result_ls = gd_with_line_search(func_quad, x0, 2000, 1e-8);
        std::cout << "    Converged: " << (result_ls.converged ? "yes" : "no")
                  << " in " << result_ls.iterations << " iterations" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "    ERROR: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // --- Experiment 2: Regularized Rosenbrock ---
    std::cout << "--- Experiment 2: Regularized Rosenbrock ---" << std::endl;
    auto func_rosen = make_regularized_rosenbrock();

    VectorXd x0_rosen(2);
    x0_rosen << -1.0, 1.0;

    std::cout << "  Fixed step (alpha = 0.001):" << std::endl;
    try {
        auto result_fixed = gd_fixed_step(func_rosen, x0_rosen, 0.001, 10000, 1e-6);
        std::cout << "    Converged: " << (result_fixed.converged ? "yes" : "no")
                  << " in " << result_fixed.iterations << " iterations"
                  << ", f(x) = " << func_rosen.eval(result_fixed.x_final) << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "    ERROR: " << e.what() << std::endl;
    }

    std::cout << "  Fixed step (alpha = 0.01) — may diverge:" << std::endl;
    try {
        auto result_fixed = gd_fixed_step(func_rosen, x0_rosen, 0.01, 10000, 1e-6);
        std::cout << "    Converged: " << (result_fixed.converged ? "yes" : "no")
                  << " in " << result_fixed.iterations << " iterations"
                  << ", f(x) = " << func_rosen.eval(result_fixed.x_final) << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "    ERROR: " << e.what() << std::endl;
    }

    std::cout << "  Line search:" << std::endl;
    try {
        auto result_ls = gd_with_line_search(func_rosen, x0_rosen, 10000, 1e-6);
        std::cout << "    Converged: " << (result_ls.converged ? "yes" : "no")
                  << " in " << result_ls.iterations << " iterations"
                  << ", f(x) = " << func_rosen.eval(result_ls.x_final) << std::endl;
        std::cout << "    x_final = [" << result_ls.x_final.transpose() << "]" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "    ERROR: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // --- Experiment 3: Log-Sum-Exp ---
    std::cout << "--- Experiment 3: Log-Sum-Exp (5 terms in R^3) ---" << std::endl;
    MatrixXd A(5, 3);
    A << 1.0,  2.0,  0.5,
        -1.0,  0.5,  1.0,
         0.5, -1.0,  2.0,
         2.0,  0.0, -1.0,
        -0.5,  1.0,  0.5;

    VectorXd b_lse(5);
    b_lse << 0.1, -0.2, 0.3, -0.1, 0.0;

    auto func_lse = make_log_sum_exp(A, b_lse);

    VectorXd x0_lse(3);
    x0_lse << 2.0, -1.0, 3.0;

    std::cout << "  Line search:" << std::endl;
    try {
        auto result = gd_with_line_search(func_lse, x0_lse, 5000, 1e-8);
        std::cout << "    Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "    x_final = [" << result.x_final.transpose() << "]" << std::endl;
        std::cout << "    f(x_final) = " << func_lse.eval(result.x_final) << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "    ERROR: " << e.what() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. Line search adapts step size automatically (no need to know L)" << std::endl;
    std::cout << "  2. On functions with varying curvature (Rosenbrock), line search" << std::endl;
    std::cout << "     is much more robust than any fixed step" << std::endl;
    std::cout << "  3. Trade-off: extra function evaluations per iteration" << std::endl;
    std::cout << "  4. Armijo c1=1e-4 is very permissive — almost any decrease is accepted" << std::endl;

    return 0;
}
