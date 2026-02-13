// Practice 033a — Phase 1: Gradient Descent
// Implement gradient descent with fixed step size on convex quadratics.

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

/// Result of a gradient descent run: final point, trajectory, and convergence info.
struct GDResult {
    VectorXd x_final;                ///< Final iterate
    std::vector<VectorXd> trajectory; ///< All iterates x_0, x_1, ..., x_k
    std::vector<double> f_values;     ///< f(x_k) at each iteration
    std::vector<double> grad_norms;   ///< ||gradient(x_k)|| at each iteration
    int iterations;                   ///< Number of iterations performed
    bool converged;                   ///< Whether ||gradient|| < tol was achieved
};

/// A convex quadratic function: f(x) = 0.5 * x^T Q x + q^T x
/// where Q is symmetric positive definite.
///
/// Gradient: grad f(x) = Q*x + q
/// Optimal solution: x* = -Q^{-1} q
/// Lipschitz constant of gradient: L = max eigenvalue of Q
/// Strong convexity parameter: mu = min eigenvalue of Q
struct QuadraticFunction {
    MatrixXd Q;  ///< Symmetric positive definite matrix
    VectorXd q;  ///< Linear term

    /// Evaluate f(x) = 0.5 * x^T Q x + q^T x
    double eval(const VectorXd& x) const {
        return 0.5 * x.dot(Q * x) + q.dot(x);
    }

    /// Gradient: grad f(x) = Q*x + q
    VectorXd gradient(const VectorXd& x) const {
        return Q * x + q;
    }

    /// Optimal value: f(x*) = -0.5 * q^T Q^{-1} q
    double optimal_value() const {
        VectorXd x_star = Q.ldlt().solve(-q);
        return eval(x_star);
    }

    /// Optimal point: x* = -Q^{-1} q
    VectorXd optimal_point() const {
        return Q.ldlt().solve(-q);
    }

    /// Lipschitz constant L = largest eigenvalue of Q
    double lipschitz_constant() const {
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(Q);
        return solver.eigenvalues().maxCoeff();
    }

    /// Strong convexity parameter mu = smallest eigenvalue of Q
    double strong_convexity() const {
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(Q);
        return solver.eigenvalues().minCoeff();
    }

    /// Condition number kappa = L / mu
    double condition_number() const {
        return lipschitz_constant() / strong_convexity();
    }
};

// ============================================================================
// Helpers: create quadratics with specified condition numbers
// ============================================================================

/// Create a 2D quadratic with eigenvalues lambda_min and lambda_max.
/// The resulting Q has condition number kappa = lambda_max / lambda_min.
/// We rotate the eigenvectors by angle theta to make the problem non-axis-aligned.
QuadraticFunction make_quadratic_2d(double lambda_min, double lambda_max,
                                     double theta = 0.3) {
    // Rotation matrix
    MatrixXd R(2, 2);
    R << std::cos(theta), -std::sin(theta),
         std::sin(theta),  std::cos(theta);

    // Diagonal eigenvalue matrix
    MatrixXd D = MatrixXd::Zero(2, 2);
    D(0, 0) = lambda_min;
    D(1, 1) = lambda_max;

    // Q = R * D * R^T  (rotate eigenvalues)
    QuadraticFunction func;
    func.Q = R * D * R.transpose();

    // Linear term: shift the optimum away from the origin
    func.q = VectorXd(2);
    func.q << 1.0, -2.0;

    return func;
}

/// Create an n-dimensional quadratic with eigenvalues linearly spaced
/// from lambda_min to lambda_max.
QuadraticFunction make_quadratic_nd(int n, double lambda_min, double lambda_max) {
    // Random orthogonal matrix via QR decomposition of a random matrix
    MatrixXd A = MatrixXd::Random(n, n);
    Eigen::HouseholderQR<MatrixXd> qr(A);
    MatrixXd R_orth = qr.householderQ();

    // Eigenvalues linearly spaced
    VectorXd eigenvalues = VectorXd::LinSpaced(n, lambda_min, lambda_max);
    MatrixXd D = eigenvalues.asDiagonal();

    QuadraticFunction func;
    func.Q = R_orth * D * R_orth.transpose();
    // Ensure exact symmetry
    func.Q = (func.Q + func.Q.transpose()) / 2.0;

    func.q = VectorXd::Ones(n);
    return func;
}

// ============================================================================
// Print helpers
// ============================================================================

void print_header() {
    std::cout << std::setw(6) << "iter"
              << std::setw(16) << "f(x)"
              << std::setw(16) << "||grad||"
              << std::setw(16) << "f(x)-f*"
              << std::endl;
    std::cout << std::string(54, '-') << std::endl;
}

void print_iteration(int iter, double f_val, double grad_norm, double f_star) {
    std::cout << std::setw(6) << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << f_val
              << std::setw(16) << grad_norm
              << std::setw(16) << f_val - f_star
              << std::endl;
}

// ============================================================================
// TODO(human): Implement gradient descent
// ============================================================================

/// Gradient descent with fixed step size.
///
/// TODO(human): Implement the gradient descent loop.
///
///   Implement gradient descent:
///     x_{k+1} = x_k - alpha * gradient(x_k)
///
///   Stop when ||gradient|| < tol or max_iter reached.
///   Store the trajectory (all x_k values) for analysis.
///
///   Key insight: the step size alpha must satisfy alpha < 2/L where L is the
///   Lipschitz constant of the gradient (= largest eigenvalue of Q for quadratics).
///   Too large -> diverges. Too small -> painfully slow.
///
///   For a quadratic f(x) = 0.5 * x^T Q x + q^T x:
///     gradient = Q*x + q
///     Optimal step: alpha = 1/L where L = max eigenvalue of Q
///
///   Algorithm:
///     1. Set x = x0
///     2. For k = 0, 1, ..., max_iter-1:
///        a. Compute grad = func.gradient(x)
///        b. If ||grad|| < tol, stop (converged)
///        c. Record x, f(x), ||grad|| in the result
///        d. Update: x = x - alpha * grad
///     3. Record the final iterate
///
///   Try running with: alpha = 0.001, 0.01, 0.1, 1/L, 2/L + epsilon
///   Observe: convergence speed vs divergence
///
///   Expected behavior on a quadratic with L=10, mu=1 (kappa=10):
///     alpha = 1/L = 0.1  -> converges in ~25 iterations
///     alpha = 0.01       -> converges in ~250 iterations (10x slower)
///     alpha = 0.2 (= 2/L)-> barely converges (oscillates)
///     alpha = 0.21 (> 2/L)-> DIVERGES (f(x) grows unbounded)
GDResult gradient_descent(const QuadraticFunction& func,
                          const VectorXd& x0,
                          double alpha,
                          int max_iter,
                          double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// main: run experiments
// ============================================================================

int main() {
    std::cout << "=== Phase 1: Gradient Descent on Convex Quadratics ===" << std::endl;
    std::cout << std::endl;

    // --- Experiment 1: Well-conditioned quadratic (kappa = 10) ---
    std::cout << "--- Experiment 1: Well-conditioned quadratic (kappa = 10) ---" << std::endl;
    auto func_easy = make_quadratic_2d(1.0, 10.0);

    double L = func_easy.lipschitz_constant();
    double mu = func_easy.strong_convexity();
    double kappa = func_easy.condition_number();
    VectorXd x_star = func_easy.optimal_point();
    double f_star = func_easy.optimal_value();

    std::cout << "  L = " << L << ", mu = " << mu << ", kappa = " << kappa << std::endl;
    std::cout << "  x* = [" << x_star.transpose() << "]" << std::endl;
    std::cout << "  f* = " << f_star << std::endl;
    std::cout << std::endl;

    VectorXd x0(2);
    x0 << 5.0, 5.0;

    // Test with optimal step size: alpha = 1/L
    std::cout << "  Step size: alpha = 1/L = " << 1.0 / L << std::endl;
    print_header();
    try {
        auto result = gradient_descent(func_easy, x0, 1.0 / L, 500, 1e-8);
        // Print first 10 iterations and last 5
        for (int i = 0; i < std::min(10, result.iterations); ++i) {
            print_iteration(i, result.f_values[i], result.grad_norms[i], f_star);
        }
        if (result.iterations > 15) {
            std::cout << "  ..." << std::endl;
            for (int i = result.iterations - 5; i < result.iterations; ++i) {
                print_iteration(i, result.f_values[i], result.grad_norms[i], f_star);
            }
        }
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "  Final x = [" << result.x_final.transpose() << "]" << std::endl;
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
        std::cout << std::endl;
    }

    // Test with too-small step size
    std::cout << "  Step size: alpha = 0.01 (too small)" << std::endl;
    try {
        auto result = gradient_descent(func_easy, x0, 0.01, 500, 1e-8);
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
        std::cout << std::endl;
    }

    // Test with step size near divergence boundary
    std::cout << "  Step size: alpha = 2/L + 0.01 (should diverge)" << std::endl;
    try {
        auto result = gradient_descent(func_easy, x0, 2.0 / L + 0.01, 20, 1e-8);
        print_header();
        for (int i = 0; i < std::min(10, result.iterations); ++i) {
            print_iteration(i, result.f_values[i], result.grad_norms[i], f_star);
        }
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
        std::cout << std::endl;
    }

    // --- Experiment 2: Ill-conditioned quadratic (kappa = 100) ---
    std::cout << "--- Experiment 2: Ill-conditioned quadratic (kappa = 100) ---" << std::endl;
    auto func_hard = make_quadratic_2d(1.0, 100.0);

    L = func_hard.lipschitz_constant();
    kappa = func_hard.condition_number();
    f_star = func_hard.optimal_value();

    std::cout << "  L = " << L << ", kappa = " << kappa << std::endl;
    std::cout << "  alpha = 1/L = " << 1.0 / L << std::endl;
    std::cout << std::endl;

    try {
        auto result = gradient_descent(func_hard, x0, 1.0 / L, 2000, 1e-8);
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "  (Compare: kappa=10 needed ~25 iters, kappa=100 needs ~"
                  << static_cast<int>(kappa) << " iters)" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }

    // --- Experiment 3: Higher-dimensional (n=10, kappa=50) ---
    std::cout << std::endl;
    std::cout << "--- Experiment 3: n=10, kappa=50 ---" << std::endl;
    auto func_10d = make_quadratic_nd(10, 1.0, 50.0);

    L = func_10d.lipschitz_constant();
    kappa = func_10d.condition_number();
    f_star = func_10d.optimal_value();

    std::cout << "  L = " << L << ", kappa = " << kappa << std::endl;

    VectorXd x0_10d = VectorXd::Constant(10, 5.0);
    try {
        auto result = gradient_descent(func_10d, x0_10d, 1.0 / L, 2000, 1e-8);
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "  f(x_final) = " << func_10d.eval(result.x_final)
                  << ", f* = " << f_star << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. Step size alpha must be in (0, 2/L) for convergence" << std::endl;
    std::cout << "  2. Optimal fixed step: alpha = 1/L" << std::endl;
    std::cout << "  3. Iterations to converge ~ O(kappa) for strongly convex" << std::endl;
    std::cout << "  4. Higher condition number -> slower convergence" << std::endl;

    return 0;
}
