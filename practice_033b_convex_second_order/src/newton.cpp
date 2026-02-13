// Practice 033b -- Phase 3: Newton's Method
// Implement pure Newton and damped Newton (with backtracking line search)
// for logistic regression. Compare quadratic convergence with GD.

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <algorithm>
#include <limits>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ============================================================================
// Data structures
// ============================================================================

/// Result of Newton's method: solution, convergence history.
struct NewtonResult {
    VectorXd x_final;                     ///< Final iterate
    std::vector<double> objectives;       ///< f(x_k) at each iteration
    std::vector<double> grad_norms;       ///< ||gradient(x_k)|| at each iteration
    std::vector<double> newton_decrements; ///< lambda^2 = g^T H^{-1} g at each iteration
    int iterations;
    bool converged;
};

/// Result of gradient descent (for comparison).
struct GDResult {
    VectorXd x_final;
    std::vector<double> objectives;
    std::vector<double> grad_norms;
    int iterations;
    bool converged;
};

// ============================================================================
// Logistic regression: objective, gradient, Hessian
// ============================================================================

/// Logistic regression loss with L2 regularization:
///   f(w) = (1/n) sum_{i=1}^{n} log(1 + exp(-y_i * x_i^T w)) + (mu/2) ||w||^2
///
/// where (x_i, y_i) are data points (features, label), y_i in {-1, +1}.
/// The L2 term ensures strong convexity (positive definite Hessian).
struct LogisticRegression {
    MatrixXd X;       ///< Data matrix (n_samples x n_features)
    VectorXd y;       ///< Labels (n_samples x 1), values in {-1, +1}
    double mu;        ///< L2 regularization strength

    int n_samples() const { return static_cast<int>(X.rows()); }
    int n_features() const { return static_cast<int>(X.cols()); }

    /// Objective: (1/n) sum log(1 + exp(-y_i * x_i^T w)) + (mu/2) ||w||^2
    double eval(const VectorXd& w) const {
        int n = n_samples();
        double loss = 0.0;
        for (int i = 0; i < n; ++i) {
            double margin = y(i) * X.row(i).dot(w);
            // log(1 + exp(-margin)), numerically stable
            if (margin > 0) {
                loss += std::log(1.0 + std::exp(-margin));
            } else {
                loss += -margin + std::log(1.0 + std::exp(margin));
            }
        }
        return loss / n + 0.5 * mu * w.squaredNorm();
    }

    /// Gradient: (1/n) sum -y_i * sigma(-y_i * x_i^T w) * x_i + mu * w
    /// where sigma(t) = 1 / (1 + exp(-t)) is the sigmoid function.
    VectorXd gradient(const VectorXd& w) const {
        int n = n_samples();
        int d = n_features();
        VectorXd grad = VectorXd::Zero(d);

        for (int i = 0; i < n; ++i) {
            double margin = y(i) * X.row(i).dot(w);
            // sigmoid(-margin) = 1 / (1 + exp(margin))
            double sig;
            if (margin > 0) {
                sig = 1.0 / (1.0 + std::exp(margin));
            } else {
                double em = std::exp(-margin);  // this is exp(|margin|)
                sig = em / (1.0 + em);
            }
            // Note: sig = sigmoid(-margin) = sigmoid(-y_i * x_i^T w)
            // The gradient contribution is -y_i * sig * x_i
            grad -= (y(i) * sig / n) * X.row(i).transpose();
        }
        grad += mu * w;
        return grad;
    }

    /// Hessian: (1/n) sum sigma_i * (1 - sigma_i) * x_i * x_i^T + mu * I
    /// where sigma_i = sigmoid(y_i * x_i^T w).
    ///
    /// The Hessian of logistic loss is always PSD. With mu > 0, H is PD.
    /// This is what makes Newton's method work well: the Hessian is well-behaved.
    MatrixXd hessian(const VectorXd& w) const {
        int n = n_samples();
        int d = n_features();
        MatrixXd H = mu * MatrixXd::Identity(d, d);

        for (int i = 0; i < n; ++i) {
            double margin = y(i) * X.row(i).dot(w);
            double sig;
            if (margin > 0) {
                double em = std::exp(-margin);
                sig = 1.0 / (1.0 + em);
            } else {
                double ep = std::exp(margin);
                sig = ep / (1.0 + ep);
            }
            // sigma_i * (1 - sigma_i) is always in [0, 0.25]
            double weight = sig * (1.0 - sig) / n;
            H += weight * X.row(i).transpose() * X.row(i);
        }
        return H;
    }
};

/// Generate synthetic binary classification data.
/// Features are drawn from N(0, 1). Labels are determined by a true weight
/// vector w_true: y_i = sign(x_i^T w_true + noise).
LogisticRegression generate_logistic_data(int n_samples, int n_features,
                                           double mu = 0.01, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    LogisticRegression lr;
    lr.mu = mu;

    // Generate features
    lr.X = MatrixXd(n_samples, n_features);
    for (int i = 0; i < n_samples; ++i)
        for (int j = 0; j < n_features; ++j)
            lr.X(i, j) = normal(rng);

    // True weight vector
    VectorXd w_true(n_features);
    for (int j = 0; j < n_features; ++j)
        w_true(j) = normal(rng);
    w_true.normalize();
    w_true *= 3.0;  // scale for reasonable separation

    // Generate labels with some noise
    lr.y = VectorXd(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double margin = lr.X.row(i).dot(w_true) + normal(rng) * 0.5;
        lr.y(i) = (margin > 0) ? 1.0 : -1.0;
    }

    return lr;
}

// ============================================================================
// Provided: Gradient descent (for comparison)
// ============================================================================

/// Simple GD with backtracking line search on logistic regression.
GDResult gd_with_backtracking(const LogisticRegression& lr, const VectorXd& x0,
                               int max_iter, double tol) {
    GDResult result;
    VectorXd x = x0;

    for (int k = 0; k < max_iter; ++k) {
        double fval = lr.eval(x);
        VectorXd g = lr.gradient(x);
        double gnorm = g.norm();

        result.objectives.push_back(fval);
        result.grad_norms.push_back(gnorm);

        if (gnorm < tol) {
            result.x_final = x;
            result.iterations = k;
            result.converged = true;
            return result;
        }

        // Backtracking line search
        double alpha = 1.0;
        double c1 = 1e-4;
        double rho = 0.5;
        double descent = g.squaredNorm();  // -g^T * (-g) = ||g||^2

        while (alpha > 1e-16) {
            VectorXd x_new = x - alpha * g;
            if (lr.eval(x_new) <= fval - c1 * alpha * descent) {
                break;
            }
            alpha *= rho;
        }

        x = x - alpha * g;
    }

    result.x_final = x;
    result.iterations = max_iter;
    result.converged = false;
    return result;
}

// ============================================================================
// TODO(human): Pure Newton's method
// ============================================================================

/// Newton's method for logistic regression.
///
/// TODO(human): Implement pure Newton's method.
///
///   Newton's method uses the local quadratic approximation:
///     f(x + dx) ≈ f(x) + g^T dx + (1/2) dx^T H dx
///
///   where g = gradient(x), H = hessian(x). Setting the gradient of this
///   approximation to zero gives the Newton equation:
///     H * dx = -g
///
///   The solution dx = -H^{-1} g is the Newton direction.
///   The update is: x_{k+1} = x_k + dx
///
///   Solving the Newton system:
///   Since H is symmetric positive definite (for logistic regression with L2),
///   use Eigen's LDLT decomposition which is the most efficient factorization
///   for SPD matrices:
///     VectorXd dx = -H.ldlt().solve(g);
///
///   Newton decrement (convergence measure):
///     lambda^2 = g^T * H^{-1} * g = -g^T * dx
///
///   When lambda^2 / 2 < tol, we are within tol of the optimal value.
///   This is a better stopping criterion than ||g|| because it accounts
///   for the curvature (Hessian) -- in poorly scaled problems, ||g|| can
///   be small even when x is far from optimal.
///
///   Convergence: QUADRATIC near the optimum.
///     ||x_{k+1} - x*|| <= C * ||x_k - x*||^2
///
///   This means the number of correct digits DOUBLES each iteration:
///     Error:  10^{-1} -> 10^{-2} -> 10^{-4} -> 10^{-8} -> 10^{-16}
///     (5 iterations for machine precision from a reasonable starting point)
///
///   WARNING: Pure Newton may diverge if started far from the optimum!
///   The quadratic model may be a poor approximation, causing the step
///   to overshoot or move in a bad direction. Damped Newton (next exercise)
///   fixes this with a line search.
///
///   Algorithm:
///     x = x0
///     for k = 0, ..., max_iter-1:
///         g = gradient(x)
///         H = hessian(x)
///         dx = -H.ldlt().solve(g)      [Newton direction]
///         lambda_sq = -g.dot(dx)        [Newton decrement squared]
///
///         record f(x), ||g||, lambda_sq
///
///         if lambda_sq / 2.0 < tol:
///             converged; break
///
///         x = x + dx                    [full Newton step]
///
/// @param lr        Logistic regression problem (provides eval, gradient, hessian)
/// @param x0        Starting point
/// @param max_iter  Maximum Newton iterations
/// @param tol       Convergence tolerance on lambda^2 / 2
/// @return          NewtonResult with solution and convergence history
NewtonResult newton_method(const LogisticRegression& lr, const VectorXd& x0,
                            int max_iter, double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Damped Newton (Newton + backtracking line search)
// ============================================================================

/// Damped Newton's method: Newton direction + backtracking line search.
///
/// TODO(human): Implement damped Newton's method.
///
///   Pure Newton takes the full step x + dx, which may overshoot when
///   far from the optimum. Damped Newton adds a line search:
///     x_{k+1} = x_k + t * dx
///   where t in (0, 1] is found by backtracking.
///
///   This combines the best of both worlds:
///   - Far from optimum (damped phase): t < 1, linear convergence (like GD)
///   - Near optimum (pure phase): t = 1 is always accepted, quadratic convergence
///
///   The transition from damped to pure phase typically happens within a few
///   iterations. After that, Newton converges quadratically.
///
///   Backtracking line search on Newton direction:
///     Start with t = 1.0 (try full Newton step first)
///     While f(x + t*dx) > f(x) + c1 * t * g^T * dx:
///         t = beta * t
///
///   Parameters: c1 = 0.01 (Armijo), beta = 0.5 (halving)
///   Note: g^T * dx < 0 always (dx is a descent direction when H is PD),
///   so c1 * t * g^T * dx is negative -- we check sufficient decrease.
///
///   Algorithm:
///     x = x0
///     for k = 0, ..., max_iter-1:
///         g = gradient(x)
///         H = hessian(x)
///         dx = -H.ldlt().solve(g)
///         lambda_sq = -g.dot(dx)
///
///         record f(x), ||g||, lambda_sq
///
///         if lambda_sq / 2.0 < tol:
///             converged; break
///
///         // Backtracking line search
///         t = 1.0
///         f_x = f(x)
///         while f(x + t*dx) > f_x + c1 * t * g.dot(dx):
///             t *= beta
///
///         x = x + t * dx
///
///   The damped Newton method is globally convergent (converges from any
///   starting point for a convex function with bounded sublevel sets) AND
///   locally quadratically convergent (retains Newton's fast convergence
///   near the optimum).
///
/// @param lr        Logistic regression problem
/// @param x0        Starting point
/// @param max_iter  Maximum iterations
/// @param tol       Convergence tolerance on lambda^2 / 2
/// @return          NewtonResult with solution and convergence history
NewtonResult damped_newton(const LogisticRegression& lr, const VectorXd& x0,
                            int max_iter, double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// Print helpers
// ============================================================================

void print_newton_header() {
    std::cout << std::setw(6)  << "iter"
              << std::setw(16) << "f(x)"
              << std::setw(16) << "||grad||"
              << std::setw(16) << "lambda^2/2"
              << std::endl;
    std::cout << std::string(54, '-') << std::endl;
}

void print_newton_row(int iter, double fval, double gnorm, double newton_dec) {
    std::cout << std::setw(6)  << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << fval
              << std::setw(16) << gnorm
              << std::setw(16) << newton_dec / 2.0
              << std::endl;
}

void print_gd_header() {
    std::cout << std::setw(6)  << "iter"
              << std::setw(16) << "f(x)"
              << std::setw(16) << "||grad||"
              << std::endl;
    std::cout << std::string(38, '-') << std::endl;
}

void print_gd_row(int iter, double fval, double gnorm) {
    std::cout << std::setw(6)  << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << fval
              << std::setw(16) << gnorm
              << std::endl;
}

// ============================================================================
// main: compare GD vs Newton on logistic regression
// ============================================================================

int main() {
    std::cout << "=== Phase 3: Newton's Method for Logistic Regression ===" << std::endl;
    std::cout << std::endl;

    // --- Problem setup ---
    int n_samples = 200;
    int n_features = 20;
    double mu = 0.01;  // L2 regularization

    auto lr = generate_logistic_data(n_samples, n_features, mu);

    std::cout << "Logistic regression: n_samples=" << n_samples
              << ", n_features=" << n_features << ", mu=" << mu << std::endl;
    std::cout << std::endl;

    VectorXd x0 = VectorXd::Zero(n_features);
    double tol = 1e-10;

    // --- Experiment 1: Gradient Descent ---
    std::cout << "--- Gradient Descent (with backtracking line search) ---" << std::endl;
    auto gd_result = gd_with_backtracking(lr, x0, 500, tol);

    print_gd_header();
    for (int i = 0; i < std::min(5, gd_result.iterations); ++i) {
        print_gd_row(i, gd_result.objectives[i], gd_result.grad_norms[i]);
    }
    if (gd_result.iterations > 10) {
        std::cout << "  ..." << std::endl;
        for (int i = gd_result.iterations - 3; i < gd_result.iterations; ++i) {
            print_gd_row(i, gd_result.objectives[i], gd_result.grad_norms[i]);
        }
    }
    std::cout << "GD converged: " << (gd_result.converged ? "yes" : "no")
              << " in " << gd_result.iterations << " iterations" << std::endl;
    std::cout << "GD final objective: " << std::scientific << std::setprecision(10)
              << gd_result.objectives.back() << std::endl;
    std::cout << std::endl;

    // --- Experiment 2: Pure Newton ---
    std::cout << "--- Pure Newton's Method ---" << std::endl;
    try {
        auto newton_result = newton_method(lr, x0, 50, tol);

        print_newton_header();
        for (int i = 0; i < newton_result.iterations; ++i) {
            print_newton_row(i, newton_result.objectives[i], newton_result.grad_norms[i],
                             newton_result.newton_decrements[i]);
        }
        std::cout << "Newton converged: " << (newton_result.converged ? "yes" : "no")
                  << " in " << newton_result.iterations << " iterations" << std::endl;
        std::cout << "Newton final objective: " << std::scientific << std::setprecision(10)
                  << newton_result.objectives.back() << std::endl;
        std::cout << std::endl;

        // Show quadratic convergence: ratio of successive Newton decrements
        if (newton_result.iterations > 2) {
            std::cout << "--- Quadratic convergence check ---" << std::endl;
            std::cout << "  If convergence is quadratic, lambda_{k+1}^2 / (lambda_k^2)^2 ~ constant" << std::endl;
            for (int i = 1; i < newton_result.iterations; ++i) {
                double prev = newton_result.newton_decrements[i - 1];
                double curr = newton_result.newton_decrements[i];
                if (prev > 1e-15) {
                    std::cout << "  iter " << i << ": lambda^2 = " << std::scientific << std::setprecision(4)
                              << curr << ", ratio = " << curr / (prev * prev) << std::endl;
                }
            }
            std::cout << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << std::endl;
    }

    // --- Experiment 3: Damped Newton ---
    std::cout << "--- Damped Newton (Newton + backtracking) ---" << std::endl;
    try {
        // Start from a point far from optimum to show damping effect
        VectorXd x0_far = VectorXd::Constant(n_features, 5.0);
        auto damped_result = damped_newton(lr, x0_far, 50, tol);

        print_newton_header();
        for (int i = 0; i < damped_result.iterations; ++i) {
            print_newton_row(i, damped_result.objectives[i], damped_result.grad_norms[i],
                             damped_result.newton_decrements[i]);
        }
        std::cout << "Damped Newton converged: " << (damped_result.converged ? "yes" : "no")
                  << " in " << damped_result.iterations << " iterations" << std::endl;
        std::cout << "Final objective: " << std::scientific << std::setprecision(10)
                  << damped_result.objectives.back() << std::endl;
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << std::endl;
    }

    // --- Summary ---
    std::cout << "--- Comparison summary ---" << std::endl;
    std::cout << "  GD iterations:     " << gd_result.iterations << std::endl;
    try {
        auto nr = newton_method(lr, x0, 50, tol);
        std::cout << "  Newton iterations: " << nr.iterations << std::endl;
        std::cout << "  Speedup: ~" << gd_result.iterations / std::max(1, nr.iterations) << "x fewer iterations" << std::endl;
        std::cout << "  (But each Newton iteration costs O(n^3) vs O(n) for GD)" << std::endl;
    } catch (...) {
        std::cout << "  Newton: not yet implemented" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. Newton uses Hessian (second derivative) for quadratic convergence" << std::endl;
    std::cout << "  2. Convergence: error SQUARES each iteration (10^-3 -> 10^-6 -> 10^-12)" << std::endl;
    std::cout << "  3. Newton decrement lambda^2/2 is a natural stopping criterion" << std::endl;
    std::cout << "  4. Pure Newton may diverge far from optimum; damped Newton adds safety" << std::endl;
    std::cout << "  5. Cost: O(n^3) per iteration (Hessian solve) vs O(n) for GD" << std::endl;
    std::cout << "  6. Newton wins when n is moderate and high precision is needed" << std::endl;

    return 0;
}
