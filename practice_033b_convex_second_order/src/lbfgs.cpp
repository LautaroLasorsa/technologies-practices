// Practice 033b -- Phase 4: L-BFGS (Limited-Memory BFGS)
// Implement the two-loop recursion and full L-BFGS solver. Compare with GD
// on high-dimensional Rosenbrock and logistic regression.

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

/// Result of L-BFGS: solution, convergence history.
struct LBFGSResult {
    VectorXd x_final;                ///< Final iterate
    std::vector<double> objectives;  ///< f(x_k) at each iteration
    std::vector<double> grad_norms;  ///< ||gradient(x_k)|| at each iteration
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
// Test functions: Rosenbrock (n-dimensional) and logistic regression
// ============================================================================

/// Pair of (eval, gradient) function objects for a smooth objective.
struct SmoothFunction {
    std::function<double(const VectorXd&)> eval;
    std::function<VectorXd(const VectorXd&)> grad;
};

/// Create the n-dimensional generalized Rosenbrock function.
///
///   f(x) = sum_{i=0}^{n-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
///
/// The minimum is at x* = (1, 1, ..., 1) with f(x*) = 0.
/// This is a classic benchmark: narrow curved valley makes it hard for GD.
SmoothFunction make_rosenbrock_nd(int n) {
    SmoothFunction fn;

    fn.eval = [n](const VectorXd& x) -> double {
        double f = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            double t1 = x(i + 1) - x(i) * x(i);
            double t2 = 1.0 - x(i);
            f += 100.0 * t1 * t1 + t2 * t2;
        }
        return f;
    };

    fn.grad = [n](const VectorXd& x) -> VectorXd {
        VectorXd g = VectorXd::Zero(n);
        for (int i = 0; i < n - 1; ++i) {
            double t1 = x(i + 1) - x(i) * x(i);
            // df/dx_i = -400 * x_i * (x_{i+1} - x_i^2) + 2*(x_i - 1)
            g(i) += -400.0 * x(i) * t1 + 2.0 * (x(i) - 1.0);
            // df/dx_{i+1} = 200 * (x_{i+1} - x_i^2)
            g(i + 1) += 200.0 * t1;
        }
        return g;
    };

    return fn;
}

/// Create logistic regression objective (eval + gradient only, no Hessian needed).
///
///   f(w) = (1/n) sum log(1 + exp(-y_i * x_i^T w)) + (mu/2) ||w||^2
///
/// Reuses the same numerically-stable pattern as newton.cpp.
SmoothFunction make_logistic_regression(const MatrixXd& X, const VectorXd& y,
                                         double mu) {
    int n = static_cast<int>(X.rows());
    int d = static_cast<int>(X.cols());

    SmoothFunction fn;

    fn.eval = [X, y, mu, n](const VectorXd& w) -> double {
        double loss = 0.0;
        for (int i = 0; i < n; ++i) {
            double margin = y(i) * X.row(i).dot(w);
            if (margin > 0) {
                loss += std::log(1.0 + std::exp(-margin));
            } else {
                loss += -margin + std::log(1.0 + std::exp(margin));
            }
        }
        return loss / n + 0.5 * mu * w.squaredNorm();
    };

    fn.grad = [X, y, mu, n, d](const VectorXd& w) -> VectorXd {
        VectorXd g = VectorXd::Zero(d);
        for (int i = 0; i < n; ++i) {
            double margin = y(i) * X.row(i).dot(w);
            double sig;
            if (margin > 0) {
                sig = 1.0 / (1.0 + std::exp(margin));
            } else {
                double em = std::exp(-margin);
                sig = em / (1.0 + em);
            }
            g -= (y(i) * sig / n) * X.row(i).transpose();
        }
        g += mu * w;
        return g;
    };

    return fn;
}

/// Generate synthetic logistic regression data (same as newton.cpp).
std::pair<MatrixXd, VectorXd> generate_logistic_data(int n_samples, int n_features,
                                                       unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd X(n_samples, n_features);
    for (int i = 0; i < n_samples; ++i)
        for (int j = 0; j < n_features; ++j)
            X(i, j) = normal(rng);

    VectorXd w_true(n_features);
    for (int j = 0; j < n_features; ++j)
        w_true(j) = normal(rng);
    w_true.normalize();
    w_true *= 3.0;

    VectorXd y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double margin = X.row(i).dot(w_true) + normal(rng) * 0.5;
        y(i) = (margin > 0) ? 1.0 : -1.0;
    }

    return {X, y};
}

// ============================================================================
// Provided: Gradient descent (for comparison)
// ============================================================================

/// Simple gradient descent with fixed-step backtracking line search.
GDResult gradient_descent(const SmoothFunction& func, const VectorXd& x0,
                           double alpha_init, int max_iter, double tol) {
    GDResult result;
    VectorXd x = x0;

    for (int k = 0; k < max_iter; ++k) {
        double fval = func.eval(x);
        VectorXd g = func.grad(x);
        double gnorm = g.norm();

        result.objectives.push_back(fval);
        result.grad_norms.push_back(gnorm);

        if (gnorm < tol) {
            result.x_final = x;
            result.iterations = k;
            result.converged = true;
            return result;
        }

        // Backtracking line search on -g direction
        double alpha = alpha_init;
        double c1 = 1e-4;
        double rho = 0.5;
        double descent = g.squaredNorm();

        while (alpha > 1e-16) {
            VectorXd x_new = x - alpha * g;
            if (func.eval(x_new) <= fval - c1 * alpha * descent) {
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
// Provided: Backtracking line search (Armijo condition)
// ============================================================================

/// Backtracking line search for arbitrary descent direction.
///
/// Finds step size alpha such that:
///   f(x + alpha * direction) <= f(x) + c1 * alpha * grad^T * direction
///
/// @param func       Objective function (eval only)
/// @param x          Current point
/// @param direction  Descent direction (must have grad^T * direction < 0)
/// @param alpha_init Initial step size (typically 1.0 for quasi-Newton)
/// @param beta       Backtracking factor (0 < beta < 1, typically 0.5)
/// @param c1         Sufficient decrease parameter (typically 1e-4)
/// @return           Step size satisfying the Armijo condition
double backtracking_line_search(const SmoothFunction& func, const VectorXd& x,
                                 const VectorXd& direction, double alpha_init,
                                 double beta, double c1) {
    double fval = func.eval(x);
    VectorXd g = func.grad(x);
    double directional_derivative = g.dot(direction);  // should be negative

    double alpha = alpha_init;
    while (alpha > 1e-16) {
        if (func.eval(x + alpha * direction) <= fval + c1 * alpha * directional_derivative) {
            return alpha;
        }
        alpha *= beta;
    }
    return alpha;
}

// ============================================================================
// TODO(human): L-BFGS two-loop recursion
// ============================================================================

/// Compute the L-BFGS search direction using the two-loop recursion.
///
/// TODO(human): L-BFGS Two-Loop Recursion
///
/// This is the algorithmic core of L-BFGS. Given the current gradient g and
/// the last m pairs of (s_k, y_k) where:
///   s_k = x_{k+1} - x_k   (step difference)
///   y_k = g_{k+1} - g_k    (gradient difference)
///
/// Compute the search direction d = -H_k * g without forming H_k explicitly.
///
/// Algorithm (Nocedal & Wright, Algorithm 7.4):
///
///   int m = s_history.size();  // number of stored pairs
///   VectorXd q = gradient;
///   std::vector<double> alpha(m);
///   std::vector<double> rho(m);
///
///   // First loop: iterate backward through history (i = m-1 down to 0)
///   for (int i = m - 1; i >= 0; --i) {
///       rho[i] = 1.0 / s_history[i].dot(y_history[i]);
///       alpha[i] = rho[i] * s_history[i].dot(q);
///       q = q - alpha[i] * y_history[i];
///   }
///
///   // Initial Hessian approximation: H_0 = gamma * I
///   // gamma = (s_{m-1}^T y_{m-1}) / (y_{m-1}^T y_{m-1})
///   double gamma = s_history.back().dot(y_history.back())
///                / y_history.back().dot(y_history.back());
///   VectorXd r = gamma * q;
///
///   // Second loop: iterate forward through history (i = 0 up to m-1)
///   for (int i = 0; i < m; ++i) {
///       double beta = rho[i] * y_history[i].dot(r);
///       r = r + s_history[i] * (alpha[i] - beta);
///   }
///
///   return -r;  // search direction (negative because we minimize)
///
/// This computes H_k * g in O(m*n) time using only m stored pairs.
/// The gamma scaling adapts the initial Hessian estimate to the local curvature.
/// Typical m = 5-20; more memory rarely helps beyond m=20.
///
/// @param gradient   Current gradient g_k
/// @param s_history  History of step differences s_i = x_{i+1} - x_i
/// @param y_history  History of gradient differences y_i = g_{i+1} - g_i
/// @return           Search direction d = -H_k * g_k
VectorXd lbfgs_direction(const VectorXd& gradient,
                          const std::vector<VectorXd>& s_history,
                          const std::vector<VectorXd>& y_history) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Full L-BFGS solver
// ============================================================================

/// L-BFGS optimizer with Armijo backtracking line search.
///
/// TODO(human): Complete L-BFGS Solver
///
/// Algorithm:
///   1. Initialize x = x0, compute g = gradient(x)
///   2. For k = 0, 1, 2, ...
///      a. If ||g|| < tol -> converged
///      b. If k == 0 or history is empty:
///           direction = -g  (steepest descent for first step)
///         Else:
///           direction = lbfgs_direction(g, s_history, y_history)
///
///      c. alpha = backtracking_line_search(func, x, direction, 1.0, 0.5, 1e-4)
///         (Wolfe conditions preferred, but Armijo works for learning)
///
///      d. x_new = x + alpha * direction
///         g_new = gradient(x_new)
///
///      e. s = x_new - x         // step difference
///         y = g_new - g         // gradient difference
///
///      f. Curvature check: if s.dot(y) > 1e-10:
///           Push (s, y) onto history
///           If history.size() > m: remove oldest pair
///         (Skip the update if curvature is non-positive -- safeguard)
///
///      g. x = x_new, g = g_new
///
/// The curvature check (s^T y > 0) ensures the Hessian approximation
/// stays positive definite. This can fail near non-convex regions.
///
/// Memory parameter m: typically 5-20. Larger m = better Hessian
/// approximation but more memory and compute per iteration.
///
/// @param func      Smooth objective function (eval + grad)
/// @param x0        Starting point
/// @param m         Memory parameter (number of stored (s,y) pairs)
/// @param max_iter  Maximum iterations
/// @param tol       Convergence tolerance on ||gradient||
/// @return          LBFGSResult with solution and convergence history
LBFGSResult lbfgs_solve(const SmoothFunction& func, const VectorXd& x0,
                          int m, int max_iter, double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// Print helpers
// ============================================================================

void print_header(const std::string& label) {
    std::cout << std::setw(6) << "iter"
              << std::setw(16) << "f(x)"
              << std::setw(16) << "||grad||"
              << "  " << label
              << std::endl;
    std::cout << std::string(42 + static_cast<int>(label.size()), '-') << std::endl;
}

void print_row(int iter, double fval, double gnorm) {
    std::cout << std::setw(6) << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << fval
              << std::setw(16) << gnorm
              << std::endl;
}

void print_summary(const std::string& name, int iterations, bool converged,
                    double final_obj, double final_gnorm) {
    std::cout << name << ": "
              << (converged ? "converged" : "NOT converged")
              << " in " << iterations << " iterations"
              << ", f=" << std::scientific << std::setprecision(6) << final_obj
              << ", ||g||=" << final_gnorm
              << std::endl;
}

void print_first_last(const std::vector<double>& objectives,
                       const std::vector<double>& grad_norms,
                       int iterations, int first_n = 5, int last_n = 3) {
    int show_first = std::min(first_n, iterations);
    for (int i = 0; i < show_first; ++i) {
        print_row(i, objectives[i], grad_norms[i]);
    }
    if (iterations > first_n + last_n) {
        std::cout << "  ..." << std::endl;
        for (int i = iterations - last_n; i < iterations; ++i) {
            print_row(i, objectives[i], grad_norms[i]);
        }
    } else if (iterations > show_first) {
        for (int i = show_first; i < iterations; ++i) {
            print_row(i, objectives[i], grad_norms[i]);
        }
    }
}

// ============================================================================
// main: L-BFGS experiments
// ============================================================================

int main() {
    std::cout << "=== Phase 4: L-BFGS (Limited-Memory BFGS) ===" << std::endl;
    std::cout << std::endl;

    double tol = 1e-6;

    // ---------------------------------------------------------------
    // Experiment 1: GD vs L-BFGS on 20D Rosenbrock
    // ---------------------------------------------------------------
    {
        std::cout << "--- Experiment 1: 20D Rosenbrock ---" << std::endl;
        int n = 20;
        auto func = make_rosenbrock_nd(n);
        VectorXd x0 = VectorXd::Constant(n, -1.0);  // standard start: all -1

        std::cout << "Start: x0 = (-1, -1, ..., -1), f(x0) = "
                  << std::fixed << std::setprecision(1) << func.eval(x0) << std::endl;
        std::cout << "Optimum: x* = (1, 1, ..., 1), f(x*) = 0" << std::endl;
        std::cout << std::endl;

        // GD
        std::cout << "Gradient Descent (backtracking):" << std::endl;
        print_header("GD");
        auto gd = gradient_descent(func, x0, 1.0, 10000, tol);
        print_first_last(gd.objectives, gd.grad_norms, gd.iterations);
        print_summary("GD", gd.iterations, gd.converged,
                       gd.objectives.back(), gd.grad_norms.back());
        std::cout << std::endl;

        // L-BFGS (m=10)
        std::cout << "L-BFGS (m=10):" << std::endl;
        print_header("L-BFGS");
        try {
            auto lbfgs = lbfgs_solve(func, x0, 10, 10000, tol);
            print_first_last(lbfgs.objectives, lbfgs.grad_norms, lbfgs.iterations);
            print_summary("L-BFGS", lbfgs.iterations, lbfgs.converged,
                           lbfgs.objectives.back(), lbfgs.grad_norms.back());

            // Check solution quality
            VectorXd x_star = VectorXd::Ones(n);
            std::cout << "||x_lbfgs - x*|| = " << std::scientific << std::setprecision(4)
                      << (lbfgs.x_final - x_star).norm() << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    }

    // ---------------------------------------------------------------
    // Experiment 2: GD vs L-BFGS on 50D Rosenbrock
    // ---------------------------------------------------------------
    {
        std::cout << "--- Experiment 2: 50D Rosenbrock ---" << std::endl;
        int n = 50;
        auto func = make_rosenbrock_nd(n);
        VectorXd x0 = VectorXd::Constant(n, -1.0);

        std::cout << "Start: f(x0) = "
                  << std::fixed << std::setprecision(1) << func.eval(x0)
                  << " (n=" << n << ")" << std::endl;
        std::cout << std::endl;

        // GD
        std::cout << "Gradient Descent (backtracking):" << std::endl;
        auto gd = gradient_descent(func, x0, 1.0, 50000, tol);
        print_summary("GD", gd.iterations, gd.converged,
                       gd.objectives.back(), gd.grad_norms.back());
        std::cout << std::endl;

        // L-BFGS (m=10)
        std::cout << "L-BFGS (m=10):" << std::endl;
        try {
            auto lbfgs = lbfgs_solve(func, x0, 10, 50000, tol);
            print_summary("L-BFGS", lbfgs.iterations, lbfgs.converged,
                           lbfgs.objectives.back(), lbfgs.grad_norms.back());

            VectorXd x_star = VectorXd::Ones(n);
            std::cout << "||x_lbfgs - x*|| = " << std::scientific << std::setprecision(4)
                      << (lbfgs.x_final - x_star).norm() << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    }

    // ---------------------------------------------------------------
    // Experiment 3: L-BFGS on logistic regression
    // ---------------------------------------------------------------
    {
        std::cout << "--- Experiment 3: L-BFGS on Logistic Regression ---" << std::endl;
        int n_samples = 200;
        int n_features = 20;
        double mu = 0.01;

        auto [X, y] = generate_logistic_data(n_samples, n_features);
        auto func = make_logistic_regression(X, y, mu);

        VectorXd x0 = VectorXd::Zero(n_features);

        std::cout << "Logistic regression: n_samples=" << n_samples
                  << ", n_features=" << n_features << ", mu=" << mu << std::endl;
        std::cout << "(Newton comparison from Phase 3 uses the same data with seed=42)" << std::endl;
        std::cout << std::endl;

        // GD
        std::cout << "Gradient Descent:" << std::endl;
        auto gd = gradient_descent(func, x0, 1.0, 5000, 1e-10);
        print_summary("GD", gd.iterations, gd.converged,
                       gd.objectives.back(), gd.grad_norms.back());
        std::cout << std::endl;

        // L-BFGS
        std::cout << "L-BFGS (m=10):" << std::endl;
        print_header("L-BFGS");
        try {
            auto lbfgs = lbfgs_solve(func, x0, 10, 5000, 1e-10);
            print_first_last(lbfgs.objectives, lbfgs.grad_norms, lbfgs.iterations);
            print_summary("L-BFGS", lbfgs.iterations, lbfgs.converged,
                           lbfgs.objectives.back(), lbfgs.grad_norms.back());
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    }

    // ---------------------------------------------------------------
    // Experiment 4: Effect of memory parameter m
    // ---------------------------------------------------------------
    {
        std::cout << "--- Experiment 4: Effect of Memory Parameter m ---" << std::endl;
        int n = 50;
        auto func = make_rosenbrock_nd(n);
        VectorXd x0 = VectorXd::Constant(n, -1.0);

        std::cout << "50D Rosenbrock, comparing m = {3, 10, 20}" << std::endl;
        std::cout << std::endl;

        std::vector<int> m_values = {3, 10, 20};
        for (int m : m_values) {
            std::string label = "L-BFGS (m=" + std::to_string(m) + ")";
            try {
                auto lbfgs = lbfgs_solve(func, x0, m, 50000, tol);
                print_summary(label, lbfgs.iterations, lbfgs.converged,
                               lbfgs.objectives.back(), lbfgs.grad_norms.back());
            } catch (const std::runtime_error& e) {
                std::cout << label << ": ERROR -- " << e.what() << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // ---------------------------------------------------------------
    // Key takeaways
    // ---------------------------------------------------------------
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. L-BFGS uses only gradient info but approximates the inverse Hessian" << std::endl;
    std::cout << "  2. Two-loop recursion computes H_k*g in O(m*n) -- no matrix stored" << std::endl;
    std::cout << "  3. Superlinear convergence: much faster than GD, close to Newton" << std::endl;
    std::cout << "  4. Memory m=5-20 is usually sufficient; more rarely helps" << std::endl;
    std::cout << "  5. L-BFGS is the default for large-scale smooth optimization" << std::endl;
    std::cout << "  6. Cost: O(m*n) per iteration vs O(n^3) for Newton vs O(n) for GD" << std::endl;

    return 0;
}
