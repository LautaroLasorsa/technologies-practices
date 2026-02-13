// Practice 033b -- Phase 1: Proximal Gradient Descent / ISTA for LASSO
// Implement soft-thresholding (proximal operator for L1) and the ISTA algorithm
// to solve the LASSO (L1-regularized least squares) problem.

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ============================================================================
// Data structures
// ============================================================================

/// Result of ISTA: final solution, convergence history, sparsity info.
struct ISTAResult {
    VectorXd x_final;                ///< Final iterate
    std::vector<double> objectives;  ///< LASSO objective at each iteration
    std::vector<double> grad_norms;  ///< ||gradient of smooth part|| at each iteration
    std::vector<int> nnz_counts;     ///< Number of nonzeros in x at each iteration
    int iterations;                  ///< Total iterations
    bool converged;                  ///< Whether stopping criterion was met
};

// ============================================================================
// Problem setup: synthetic sparse signal recovery
// ============================================================================

/// Generate a synthetic LASSO problem for sparse signal recovery.
///
/// The true signal x_true is k-sparse (only k out of n components are nonzero).
/// We observe b = A * x_true + noise, and try to recover x_true by solving:
///   minimize (1/2)||Ax - b||^2 + lambda * ||x||_1
///
/// With appropriate lambda, L1 regularization recovers the sparse signal.
struct LASSOProblem {
    MatrixXd A;          ///< Measurement matrix (m x n), m < n (underdetermined)
    VectorXd b;          ///< Observations: b = A * x_true + noise
    double lambda;       ///< Regularization strength
    VectorXd x_true;     ///< Ground truth sparse signal (for comparison)

    /// Lipschitz constant of the smooth gradient: L = ||A^T A||_op = largest eigenvalue of A^T A
    double lipschitz_constant() const {
        // For m x n matrix A with m < n, compute largest singular value squared
        Eigen::JacobiSVD<MatrixXd> svd(A);
        double sigma_max = svd.singularValues()(0);
        return sigma_max * sigma_max;
    }
};

/// Create a synthetic sparse signal recovery problem.
///   m = number of measurements (rows of A)
///   n = number of variables (columns of A, dimension of x)
///   k = sparsity level (number of nonzeros in x_true)
///   noise_std = standard deviation of observation noise
///   lambda = L1 regularization strength
LASSOProblem make_lasso_problem(int m, int n, int k, double noise_std, double lambda,
                                 unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    LASSOProblem prob;

    // Measurement matrix: random Gaussian, normalized columns
    prob.A = MatrixXd(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            prob.A(i, j) = normal(rng) / std::sqrt(static_cast<double>(m));

    // True sparse signal: k random nonzero entries
    prob.x_true = VectorXd::Zero(n);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < k; ++i) {
        prob.x_true(indices[i]) = normal(rng) * 3.0;  // nonzero entries ~ N(0, 9)
    }

    // Observations with noise
    VectorXd noise(m);
    for (int i = 0; i < m; ++i) noise(i) = normal(rng) * noise_std;
    prob.b = prob.A * prob.x_true + noise;

    prob.lambda = lambda;
    return prob;
}

// ============================================================================
// Provided: LASSO objective and smooth gradient
// ============================================================================

/// Full LASSO objective: f(x) + g(x) = (1/2)||Ax - b||^2 + lambda * ||x||_1
double lasso_objective(const MatrixXd& A, const VectorXd& b,
                       const VectorXd& x, double lambda) {
    VectorXd residual = A * x - b;
    return 0.5 * residual.squaredNorm() + lambda * x.lpNorm<1>();
}

/// Gradient of the smooth part f(x) = (1/2)||Ax - b||^2.
/// grad f(x) = A^T (Ax - b)
///
/// This is the part we take a gradient step on. The non-smooth L1 part
/// is handled by the proximal operator (soft-thresholding).
VectorXd lasso_smooth_gradient(const MatrixXd& A, const VectorXd& b,
                                const VectorXd& x) {
    return A.transpose() * (A * x - b);
}

// ============================================================================
// TODO(human): Implement soft-thresholding (proximal operator for L1 norm)
// ============================================================================

/// Soft-thresholding operator: the proximal operator of threshold * ||.||_1
///
/// TODO(human): Implement the soft-thresholding operator.
///
///   Soft-thresholding (proximal operator of lambda * ||.||_1):
///     prox_{t * ||.||_1}(v)_i = sign(v_i) * max(|v_i| - t, 0)
///
///   where t = threshold (= step_size * lambda in ISTA).
///
///   This is THE fundamental operation that produces sparsity in L1 methods.
///   It shrinks each component of v toward zero by exactly t units:
///     - If v_i > t:   result_i = v_i - t     (shrink positive values)
///     - If v_i < -t:  result_i = v_i + t     (shrink negative values)
///     - If |v_i| <= t: result_i = 0           (small values become EXACTLY zero)
///
///   Why does this produce sparsity?
///   The L1 norm ||x||_1 has non-smooth "kinks" at zero for each coordinate.
///   These kinks act like attractors: the soft-thresholding operator pulls
///   small coefficients to exactly zero. In contrast, L2 regularization
///   (ridge regression) shrinks toward zero but never reaches it.
///
///   Geometrically: the L1 ball {x : ||x||_1 <= r} is a diamond/cross-polytope
///   with corners on the coordinate axes. The proximal operator projects the
///   gradient step toward these corners, making coordinates exactly zero.
///
///   Connection to LASSO: in the LASSO objective (1/2)||Ax-b||^2 + lambda*||x||_1,
///   soft-thresholding with t = alpha*lambda (step_size * regularization) acts
///   as automatic variable selection. Variables whose contribution to reducing
///   ||Ax - b||^2 is smaller than lambda get zeroed out entirely.
///
///   Implementation hint with Eigen:
///     v.array().sign() * (v.array().abs() - threshold).max(0.0)
///   This applies the formula element-wise using Eigen's array operations.
///
/// @param v         Input vector (typically: x - alpha * gradient)
/// @param threshold Shrinkage amount (= step_size * lambda)
/// @return          Soft-thresholded vector
VectorXd soft_threshold(const VectorXd& v, double threshold) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Implement ISTA (Iterative Shrinkage-Thresholding Algorithm)
// ============================================================================

/// ISTA: proximal gradient descent for LASSO.
///
/// TODO(human): Implement the ISTA loop.
///
///   ISTA solves: minimize (1/2)||Ax - b||^2 + lambda * ||x||_1
///   by splitting into smooth part f(x) = (1/2)||Ax - b||^2
///                   and non-smooth part g(x) = lambda * ||x||_1
///
///   Each iteration performs two half-steps:
///     1. Gradient step on the smooth part: v = x_k - alpha * grad_f(x_k)
///     2. Proximal step on the non-smooth part: x_{k+1} = prox_{alpha*g}(v)
///                                                       = soft_threshold(v, alpha * lambda)
///
///   Combined: x_{k+1} = soft_threshold(x_k - alpha * A^T(A*x_k - b), alpha * lambda)
///
///   Step size: alpha must satisfy alpha <= 1/L where L = ||A^T A||_op
///   (largest eigenvalue of A^T A). Use the provided step_size parameter.
///
///   Convergence: O(1/k) on the LASSO objective value.
///   For LASSO, this means the iterate x_k satisfies:
///     lasso_obj(x_k) - lasso_obj(x*) <= C / k
///
///   Stopping criterion: stop when the change in iterates is small:
///     ||x_{k+1} - x_k|| / max(1, ||x_k||) < tol
///   OR when max_iter is reached.
///
///   At each iteration, record:
///     - The LASSO objective value: (1/2)||Ax - b||^2 + lambda * ||x||_1
///     - The gradient norm of the smooth part: ||A^T(Ax - b)||
///     - The number of nonzeros in x (count entries with |x_i| > 1e-10)
///
///   Algorithm pseudocode:
///     x = x0
///     for k = 0, ..., max_iter-1:
///         record objective, gradient norm, nnz count
///         grad = A^T(A*x - b)                        [smooth gradient]
///         v = x - step_size * grad                    [gradient step]
///         x_new = soft_threshold(v, step_size * lambda) [proximal step]
///         if ||x_new - x|| / max(1, ||x||) < tol:
///             x = x_new; break
///         x = x_new
///     record final stats
///
/// @param A          Measurement matrix (m x n)
/// @param b          Observation vector (m x 1)
/// @param lambda     L1 regularization strength
/// @param x0         Initial point
/// @param step_size  Step size alpha (should be <= 1/L)
/// @param max_iter   Maximum iterations
/// @param tol        Convergence tolerance on relative iterate change
/// @return           ISTAResult with solution and convergence history
ISTAResult ista(const MatrixXd& A, const VectorXd& b, double lambda,
                const VectorXd& x0, double step_size,
                int max_iter, double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// Print helpers
// ============================================================================

void print_ista_header() {
    std::cout << std::setw(6)  << "iter"
              << std::setw(16) << "objective"
              << std::setw(16) << "||grad_f||"
              << std::setw(8)  << "nnz"
              << std::endl;
    std::cout << std::string(46, '-') << std::endl;
}

void print_ista_row(int iter, double obj, double grad_norm, int nnz) {
    std::cout << std::setw(6)  << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << obj
              << std::setw(16) << grad_norm
              << std::setw(8)  << nnz
              << std::endl;
}

// ============================================================================
// main: sparse signal recovery via ISTA
// ============================================================================

int main() {
    std::cout << "=== Phase 1: Proximal Gradient / ISTA for LASSO ===" << std::endl;
    std::cout << std::endl;

    // --- Problem setup ---
    // m=50 measurements, n=200 variables, k=10 nonzeros in true signal
    // This is underdetermined (m < n), so ordinary least squares has infinitely
    // many solutions. L1 regularization finds the SPARSE solution.
    int m = 50, n = 200, k = 10;
    double noise_std = 0.1;
    double lambda = 0.1;

    auto prob = make_lasso_problem(m, n, k, noise_std, lambda);
    double L = prob.lipschitz_constant();
    double step_size = 0.95 / L;  // slightly less than 1/L for safety

    std::cout << "Problem: m=" << m << " measurements, n=" << n << " variables, k=" << k << " sparse" << std::endl;
    std::cout << "Lambda = " << lambda << ", L = " << L << ", step_size = " << step_size << std::endl;
    std::cout << "True signal: " << k << " nonzeros out of " << n << " ("
              << std::fixed << std::setprecision(1) << 100.0 * k / n << "% sparse)" << std::endl;
    std::cout << std::endl;

    // --- Run ISTA ---
    VectorXd x0 = VectorXd::Zero(n);  // start from zero
    int max_iter = 2000;
    double tol = 1e-8;

    std::cout << "Running ISTA (max_iter=" << max_iter << ", tol=" << tol << ")..." << std::endl;
    std::cout << std::endl;

    try {
        auto result = ista(prob.A, prob.b, lambda, x0, step_size, max_iter, tol);

        // Print convergence log (first 10 + last 5 iterations)
        print_ista_header();
        int print_first = std::min(10, result.iterations);
        for (int i = 0; i < print_first; ++i) {
            print_ista_row(i, result.objectives[i], result.grad_norms[i], result.nnz_counts[i]);
        }
        if (result.iterations > 15) {
            std::cout << "  ..." << std::endl;
            for (int i = result.iterations - 5; i < result.iterations; ++i) {
                print_ista_row(i, result.objectives[i], result.grad_norms[i], result.nnz_counts[i]);
            }
        }
        std::cout << std::endl;

        std::cout << "Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "Final objective: " << std::scientific << std::setprecision(6)
                  << result.objectives.back() << std::endl;

        // Count nonzeros in solution
        int nnz_solution = 0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(result.x_final(i)) > 1e-10) ++nnz_solution;
        }
        std::cout << "Nonzeros in solution: " << nnz_solution << " (true signal has " << k << ")" << std::endl;

        // Compare with true signal: recovery error
        double recovery_error = (result.x_final - prob.x_true).norm() / prob.x_true.norm();
        std::cout << "Relative recovery error ||x - x_true|| / ||x_true|| = "
                  << std::scientific << std::setprecision(4) << recovery_error << std::endl;
        std::cout << std::endl;

        // Print the largest components of the recovered signal vs true signal
        std::cout << "--- Top-10 components comparison ---" << std::endl;
        std::cout << std::setw(6) << "idx"
                  << std::setw(14) << "x_recovered"
                  << std::setw(14) << "x_true"
                  << std::endl;
        std::cout << std::string(34, '-') << std::endl;

        // Sort by magnitude of recovered signal
        std::vector<int> sorted_idx(n);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
            return std::abs(result.x_final(a)) > std::abs(result.x_final(b));
        });

        for (int i = 0; i < std::min(15, n); ++i) {
            int idx = sorted_idx[i];
            double x_rec = result.x_final(idx);
            double x_true = prob.x_true(idx);
            if (std::abs(x_rec) < 1e-10 && std::abs(x_true) < 1e-10) continue;
            std::cout << std::setw(6) << idx
                      << std::setw(14) << std::fixed << std::setprecision(4) << x_rec
                      << std::setw(14) << x_true
                      << std::endl;
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. Soft-thresholding = proximal operator of L1 norm" << std::endl;
    std::cout << "  2. ISTA alternates: gradient step (smooth) + prox step (non-smooth)" << std::endl;
    std::cout << "  3. L1 regularization produces SPARSE solutions (exact zeros)" << std::endl;
    std::cout << "  4. Lambda controls sparsity: larger lambda = fewer nonzeros" << std::endl;
    std::cout << "  5. Convergence: O(1/k) on objective value (same rate as GD)" << std::endl;

    return 0;
}
