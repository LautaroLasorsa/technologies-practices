// Practice 033a — Phase 4: Nesterov Accelerated Gradient
// Compare vanilla GD with Nesterov's method on ill-conditioned problems.

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <iomanip>

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

/// Convex quadratic: f(x) = 0.5 * x^T Q x + q^T x
struct QuadraticFunction {
    MatrixXd Q;
    VectorXd q;

    double eval(const VectorXd& x) const {
        return 0.5 * x.dot(Q * x) + q.dot(x);
    }

    VectorXd gradient(const VectorXd& x) const {
        return Q * x + q;
    }

    VectorXd optimal_point() const {
        return Q.ldlt().solve(-q);
    }

    double optimal_value() const {
        return eval(optimal_point());
    }

    double lipschitz_constant() const {
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(Q);
        return solver.eigenvalues().maxCoeff();
    }

    double strong_convexity() const {
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(Q);
        return solver.eigenvalues().minCoeff();
    }

    double condition_number() const {
        return lipschitz_constant() / strong_convexity();
    }
};

// ============================================================================
// Create ill-conditioned quadratic
// ============================================================================

/// Create an n-dimensional quadratic with specified condition number kappa.
/// Eigenvalues: mu = 1, L = kappa (so condition_number = kappa).
/// Uses a random rotation to make the problem non-axis-aligned.
QuadraticFunction make_ill_conditioned(int n, double kappa, unsigned seed = 42) {
    // Deterministic random for reproducibility
    srand(seed);

    // Random orthogonal matrix via QR
    MatrixXd A = MatrixXd::Random(n, n);
    Eigen::HouseholderQR<MatrixXd> qr(A);
    MatrixXd R_orth = qr.householderQ();

    // Eigenvalues: linearly space from 1 to kappa
    VectorXd eigenvalues = VectorXd::LinSpaced(n, 1.0, kappa);
    MatrixXd D = eigenvalues.asDiagonal();

    QuadraticFunction func;
    func.Q = R_orth * D * R_orth.transpose();
    func.Q = (func.Q + func.Q.transpose()) / 2.0;  // Exact symmetry

    // Linear term that puts the optimum away from origin
    func.q = VectorXd::Ones(n) * 2.0;

    return func;
}

// ============================================================================
// Vanilla gradient descent (provided for comparison)
// ============================================================================

GDResult gradient_descent(const QuadraticFunction& func,
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

        x = x - alpha * grad;
    }

    result.iterations = max_iter;
    result.x_final = x;
    result.f_values.push_back(func.eval(x));
    result.grad_norms.push_back(func.gradient(x).norm());
    return result;
}

// ============================================================================
// TODO(human): Implement Nesterov accelerated gradient descent
// ============================================================================

/// Nesterov's Accelerated Gradient Descent.
///
/// TODO(human): Implement Nesterov's method.
///
///   Nesterov's Accelerated Gradient Descent:
///     y_0 = x_0, t_0 = 1
///     For k = 0, 1, 2, ...
///       x_{k+1} = y_k - (1/L) * gradient(y_k)
///       t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
///       y_{k+1} = x_{k+1} + ((t_k - 1) / t_{k+1}) * (x_{k+1} - x_k)
///
///   The "momentum" term ((t_k-1)/t_{k+1}) * (x_{k+1} - x_k) is the magic:
///     - It looks ahead using the previous trajectory direction
///     - Achieves O(1/k^2) convergence vs O(1/k) for vanilla GD
///     - For strongly convex: O((sqrt(kappa) - 1)/(sqrt(kappa) + 1))^k
///       vs O((kappa-1)/(kappa+1))^k for vanilla GD
///
///   On ill-conditioned problems (large kappa = L/mu), the speedup is dramatic:
///     kappa = 1000 -> vanilla GD needs ~1000 iters, Nesterov ~31 iters (sqrt(kappa))
///
///   Algorithm in detail:
///     1. Initialize: x = x0, y = x0, t = 1.0
///     2. For k = 0, 1, ..., max_iter-1:
///        a. Compute grad = func.gradient(y)     <-- gradient at y, NOT at x!
///        b. Compute x_new = y - (1.0/L) * grad  <-- gradient step from y
///        c. Compute t_new = (1.0 + sqrt(1.0 + 4.0*t*t)) / 2.0
///        d. Compute momentum = (t - 1.0) / t_new
///        e. Compute y_new = x_new + momentum * (x_new - x)  <-- the "look-ahead"
///        f. Record trajectory, f_values, grad_norms (use x_new and func.eval(x_new))
///        g. Stopping: if func.gradient(x_new).norm() < tol, converged
///        h. Update: x = x_new, y = y_new, t = t_new
///     3. Return result
///
///   Critical detail: the gradient is evaluated at y (the "look-ahead" point),
///   not at x. This is what distinguishes Nesterov from Polyak's heavy ball.
///   The sequence {t_k} generates the optimal momentum coefficients.
///
///   The momentum coefficient starts near 0 and grows towards 1:
///     k=0: t=1.0, momentum=0.0
///     k=1: t=1.618, momentum=0.382
///     k=5: t=5.2, momentum=0.81
///     k=20: t=19.5, momentum=0.95
///   This gradual increase prevents oscillation in early iterations.
GDResult nesterov_accelerated(const QuadraticFunction& func,
                              const VectorXd& x0,
                              double L,
                              int max_iter,
                              double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// Print helpers
// ============================================================================

void print_convergence_comparison(const std::string& method,
                                  const GDResult& result,
                                  double f_star) {
    std::cout << "  " << method << ":" << std::endl;
    std::cout << "    Converged: " << (result.converged ? "yes" : "no")
              << " in " << result.iterations << " iterations" << std::endl;
    if (!result.f_values.empty()) {
        double final_gap = result.f_values.back() - f_star;
        std::cout << "    Final f(x) - f* = " << std::scientific
                  << std::setprecision(4) << final_gap << std::endl;
    }
}

void print_convergence_table(const std::string& label,
                             const GDResult& result,
                             double f_star,
                             int max_rows = 20) {
    std::cout << "  " << label << " convergence:" << std::endl;
    std::cout << std::setw(8) << "iter"
              << std::setw(16) << "f(x) - f*"
              << std::setw(16) << "||grad||"
              << std::endl;
    std::cout << "  " << std::string(40, '-') << std::endl;

    int n = static_cast<int>(result.f_values.size());
    int step = std::max(1, n / max_rows);

    for (int i = 0; i < n; i += step) {
        std::cout << std::setw(8) << i
                  << std::setw(16) << std::scientific << std::setprecision(4)
                  << result.f_values[i] - f_star
                  << std::setw(16) << result.grad_norms[i]
                  << std::endl;
    }
    // Always print the last row
    if ((n - 1) % step != 0 && n > 0) {
        std::cout << std::setw(8) << n - 1
                  << std::setw(16) << std::scientific << std::setprecision(4)
                  << result.f_values[n - 1] - f_star
                  << std::setw(16) << result.grad_norms[n - 1]
                  << std::endl;
    }
}

// ============================================================================
// main: GD vs Nesterov showdown
// ============================================================================

int main() {
    std::cout << "=== Phase 4: Nesterov Accelerated Gradient ===" << std::endl;
    std::cout << std::endl;

    // --- Experiment 1: Moderate condition number (kappa = 100) ---
    std::cout << "--- Experiment 1: kappa = 100 (2D) ---" << std::endl;
    auto func_100 = make_ill_conditioned(2, 100.0);

    double L = func_100.lipschitz_constant();
    double mu = func_100.strong_convexity();
    double kappa = func_100.condition_number();
    double f_star = func_100.optimal_value();
    VectorXd x_star = func_100.optimal_point();

    std::cout << "  L = " << L << ", mu = " << mu << ", kappa = " << kappa << std::endl;
    std::cout << "  f* = " << f_star << std::endl;
    std::cout << std::endl;

    VectorXd x0(2);
    x0 << 10.0, 10.0;
    double tol = 1e-10;

    try {
        auto result_gd = gradient_descent(func_100, x0, 1.0 / L, 5000, tol);
        print_convergence_comparison("Vanilla GD", result_gd, f_star);
    } catch (const std::runtime_error& e) {
        std::cerr << "  GD ERROR: " << e.what() << std::endl;
    }

    try {
        auto result_nag = nesterov_accelerated(func_100, x0, L, 5000, tol);
        print_convergence_comparison("Nesterov", result_nag, f_star);
    } catch (const std::runtime_error& e) {
        std::cerr << "  Nesterov ERROR: " << e.what() << std::endl;
    }

    std::cout << "  Expected: GD ~" << static_cast<int>(kappa)
              << " iters, Nesterov ~" << static_cast<int>(std::sqrt(kappa))
              << " iters" << std::endl;
    std::cout << std::endl;

    // --- Experiment 2: High condition number (kappa = 1000) ---
    std::cout << "--- Experiment 2: kappa = 1000 (2D) — the dramatic case ---" << std::endl;
    auto func_1000 = make_ill_conditioned(2, 1000.0);

    L = func_1000.lipschitz_constant();
    mu = func_1000.strong_convexity();
    kappa = func_1000.condition_number();
    f_star = func_1000.optimal_value();

    std::cout << "  L = " << L << ", mu = " << mu << ", kappa = " << kappa << std::endl;
    std::cout << std::endl;

    try {
        auto result_gd = gradient_descent(func_1000, x0, 1.0 / L, 20000, tol);
        print_convergence_comparison("Vanilla GD", result_gd, f_star);

        auto result_nag = nesterov_accelerated(func_1000, x0, L, 20000, tol);
        print_convergence_comparison("Nesterov", result_nag, f_star);

        // Print ratio
        if (result_gd.iterations > 0 && result_nag.iterations > 0) {
            std::cout << "  Speedup: " << std::fixed << std::setprecision(1)
                      << static_cast<double>(result_gd.iterations) / result_nag.iterations
                      << "x fewer iterations with Nesterov" << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }

    std::cout << "  Expected: GD ~" << 1000
              << " iters, Nesterov ~" << static_cast<int>(std::sqrt(1000.0))
              << " iters" << std::endl;
    std::cout << std::endl;

    // --- Experiment 3: Higher dimension (n=20, kappa=500) ---
    std::cout << "--- Experiment 3: n=20, kappa=500 ---" << std::endl;
    auto func_20d = make_ill_conditioned(20, 500.0);

    L = func_20d.lipschitz_constant();
    kappa = func_20d.condition_number();
    f_star = func_20d.optimal_value();

    std::cout << "  n = 20, L = " << L << ", kappa = " << kappa << std::endl;

    VectorXd x0_20d = VectorXd::Constant(20, 5.0);

    try {
        auto result_gd = gradient_descent(func_20d, x0_20d, 1.0 / L, 20000, tol);
        print_convergence_comparison("Vanilla GD", result_gd, f_star);

        auto result_nag = nesterov_accelerated(func_20d, x0_20d, L, 20000, tol);
        print_convergence_comparison("Nesterov", result_nag, f_star);

        if (result_gd.iterations > 0 && result_nag.iterations > 0) {
            std::cout << "  Speedup: " << std::fixed << std::setprecision(1)
                      << static_cast<double>(result_gd.iterations) / result_nag.iterations
                      << "x" << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // --- Experiment 4: Convergence trace for kappa=100 ---
    std::cout << "--- Experiment 4: Detailed convergence trace (kappa=100) ---" << std::endl;

    try {
        auto result_gd = gradient_descent(func_100, x0, 1.0 / func_100.lipschitz_constant(),
                                          5000, 1e-10);
        print_convergence_table("Vanilla GD", result_gd, func_100.optimal_value());
        std::cout << std::endl;

        auto result_nag = nesterov_accelerated(func_100, x0, func_100.lipschitz_constant(),
                                               5000, 1e-10);
        print_convergence_table("Nesterov", result_nag, func_100.optimal_value());
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. Nesterov converges in O(sqrt(kappa)) vs O(kappa) for vanilla GD" << std::endl;
    std::cout << "  2. For kappa=1000: ~32 vs ~1000 iterations (31x speedup)" << std::endl;
    std::cout << "  3. The momentum term 'looks ahead' using trajectory direction" << std::endl;
    std::cout << "  4. Nesterov is provably optimal: no first-order method can do better" << std::endl;
    std::cout << "     on smooth convex functions (Nesterov 1983 lower bound)" << std::endl;
    std::cout << "  5. In ML: Adam/AdaGrad incorporate similar momentum ideas + adaptivity" << std::endl;

    return 0;
}
