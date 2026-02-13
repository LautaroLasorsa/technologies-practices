// Practice 033a — Phase 3: Projected Gradient Descent
// Implement projection operators and projected GD for constrained optimization.

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <algorithm>

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

/// Differentiable function interface
struct DifferentiableFunction {
    std::function<double(const VectorXd&)> eval;
    std::function<VectorXd(const VectorXd&)> gradient;
    std::string name;
};

/// Projection operator: takes a point and returns its projection onto a set.
using ProjectionFn = std::function<VectorXd(const VectorXd&)>;

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

// ============================================================================
// Projection helpers (provided)
// ============================================================================

/// Identity projection (no constraint — returns x unchanged).
/// Useful for testing: projected GD with identity = unconstrained GD.
VectorXd project_identity(const VectorXd& x) {
    return x;
}

/// Project onto the probability simplex: {x >= 0, sum(x) = 1}
/// Algorithm: sort, find threshold, clip. O(n log n).
/// This is provided as a reference — it's a more complex projection.
VectorXd project_simplex(const VectorXd& x) {
    int n = x.size();
    VectorXd sorted = x;
    std::sort(sorted.data(), sorted.data() + n, std::greater<double>());

    double cumsum = 0.0;
    double threshold = 0.0;
    for (int i = 0; i < n; ++i) {
        cumsum += sorted(i);
        double t = (cumsum - 1.0) / (i + 1);
        if (sorted(i) - t > 0) {
            threshold = t;
        }
    }

    VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = std::max(0.0, x(i) - threshold);
    }
    return result;
}

// ============================================================================
// Print helpers
// ============================================================================

void print_header() {
    std::cout << std::setw(6) << "iter"
              << std::setw(16) << "f(x)"
              << std::setw(16) << "||grad||"
              << std::setw(16) << "||x-x_prev||"
              << std::endl;
    std::cout << std::string(54, '-') << std::endl;
}

void print_iteration(int iter, double f_val, double grad_norm, double step_norm) {
    std::cout << std::setw(6) << iter
              << std::setw(16) << std::scientific << std::setprecision(4) << f_val
              << std::setw(16) << grad_norm
              << std::setw(16) << step_norm
              << std::endl;
}

// ============================================================================
// TODO(human): Implement projection onto box constraints
// ============================================================================

/// Project x onto the box constraint set {x : lower <= x <= upper}.
///
/// TODO(human): Implement box projection.
///
///   Box projection: for each component i, clamp x_i to [lower_i, upper_i].
///   This is the simplest projection — just clipping each coordinate independently.
///
///   Mathematically: proj(x)_i = max(lower_i, min(upper_i, x_i))
///
///   Equivalently, using Eigen: result = x.cwiseMax(lower).cwiseMin(upper)
///
///   This is O(n) and trivially parallelizable. Box constraints arise everywhere:
///     - Variable bounds in optimization (0 <= x <= 1)
///     - Physical constraints (temperature between 0 and 100)
///     - Pixel values in image processing (0 to 255)
///
///   The box set B = {x : l <= x <= u} is convex because it's the intersection
///   of 2n half-spaces (one for each bound on each coordinate).
///
///   Verify your implementation: project([3, -1, 0.5], [0, 0, 0], [1, 1, 1])
///   should return [1, 0, 0.5] (clamp 3->1, clamp -1->0, keep 0.5).
VectorXd project_box(const VectorXd& x, const VectorXd& lower, const VectorXd& upper) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Implement projection onto L2-ball
// ============================================================================

/// Project x onto the L2-ball {z : ||z - center|| <= radius}.
///
/// TODO(human): Implement L2-ball projection.
///
///   L2-ball projection:
///     - If x is already inside the ball (||x - center|| <= radius), return x unchanged.
///     - If x is outside, project to the closest point on the ball surface:
///         proj(x) = center + radius * (x - center) / ||x - center||
///
///   This is like "pulling" x towards the center until it's on the boundary.
///   The direction from center to x is preserved, only the distance changes.
///
///   Geometrically: draw a ray from the center through x; the projection is
///   where this ray intersects the ball surface.
///
///   Edge case: if x == center (zero-norm direction), return center (or any point
///   on the boundary — but this is degenerate and unlikely in practice).
///
///   This is O(n) — just a norm computation and a scaling.
///
///   L2-ball constraints arise in:
///     - Regularization (||weights|| <= C in SVMs)
///     - Trust region methods (step size bounded by radius)
///     - Robust optimization (uncertainty sets)
///
///   Verify: project([3, 0], center=[0, 0], radius=1) should return [1, 0].
///   Verify: project([0.5, 0], center=[0, 0], radius=1) should return [0.5, 0] (inside).
VectorXd project_l2_ball(const VectorXd& x, const VectorXd& center, double radius) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// TODO(human): Implement projected gradient descent
// ============================================================================

/// Projected gradient descent: GD with projection after each step.
///
/// TODO(human): Implement projected gradient descent.
///
///   For constrained optimization: minimize f(x) subject to x in C,
///   projected GD modifies the standard update:
///
///     x_{k+1} = project_C( x_k - alpha * gradient(f(x_k)) )
///
///   The projection ensures feasibility at every iteration. The algorithm:
///
///     1. Set x = project(x0) to ensure starting point is feasible
///     2. For k = 0, 1, ..., max_iter-1:
///        a. Compute grad = func.gradient(x)
///        b. Compute candidate: z = x - alpha * grad
///        c. Project: x_new = project(z)
///        d. Record x, f(x), ||grad||
///        e. Stopping criterion: ||x_new - x|| < tol  (not ||grad|| < tol!)
///           Why? At a constrained optimum, the gradient is NOT zero — it points
///           into the infeasible region. The correct criterion is that the
///           projected gradient step makes negligible progress.
///        f. Set x = x_new
///     3. Return result
///
///   Convergence: same rate as unconstrained GD when projection is exact and cheap.
///   The step size alpha follows the same rules (alpha < 2/L for quadratics).
///
///   Important subtlety: the stopping criterion for constrained problems is
///   NOT ||gradient|| < tol (the gradient may be large at a boundary optimum).
///   Instead, use ||x_{k+1} - x_k|| < tol, which measures actual progress.
GDResult projected_gradient_descent(const DifferentiableFunction& func,
                                    const ProjectionFn& project,
                                    const VectorXd& x0,
                                    double alpha,
                                    int max_iter,
                                    double tol) {
    throw std::runtime_error("TODO(human): not implemented");
}

// ============================================================================
// main: constrained optimization experiments
// ============================================================================

int main() {
    std::cout << "=== Phase 3: Projected Gradient Descent ===" << std::endl;
    std::cout << std::endl;

    // Create a 2D quadratic whose unconstrained optimum is at x* = [-1, 3]
    // (outside typical constraint sets)
    MatrixXd Q(2, 2);
    Q << 4.0, 1.0,
         1.0, 2.0;
    VectorXd q(2);
    q << 2.0, -10.0;

    auto func = make_quadratic(Q, q);

    // Compute unconstrained optimum for reference
    VectorXd x_star_free = Q.ldlt().solve(-q);
    double f_star_free = func.eval(x_star_free);
    std::cout << "Unconstrained optimum: x* = [" << x_star_free.transpose()
              << "], f* = " << f_star_free << std::endl;

    // Lipschitz constant
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(Q);
    double L = solver.eigenvalues().maxCoeff();
    double alpha = 1.0 / L;
    std::cout << "L = " << L << ", alpha = 1/L = " << alpha << std::endl;
    std::cout << std::endl;

    VectorXd x0(2);
    x0 << 5.0, 5.0;

    // --- Experiment 1: Box constraints [0, 2] x [0, 2] ---
    std::cout << "--- Experiment 1: Box constraints [0, 2] x [0, 2] ---" << std::endl;
    VectorXd lower(2), upper(2);
    lower << 0.0, 0.0;
    upper << 2.0, 2.0;

    ProjectionFn proj_box = [&lower, &upper](const VectorXd& x) -> VectorXd {
        return project_box(x, lower, upper);
    };

    try {
        auto result = projected_gradient_descent(func, proj_box, x0, alpha, 1000, 1e-8);
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "  x_final = [" << result.x_final.transpose() << "]" << std::endl;
        std::cout << "  f(x_final) = " << func.eval(result.x_final) << std::endl;
        std::cout << "  (Unconstrained optimum was at [" << x_star_free.transpose()
                  << "], outside the box)" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // --- Experiment 2: L2-ball ||x|| <= 1 ---
    std::cout << "--- Experiment 2: L2-ball ||x|| <= 1 ---" << std::endl;
    VectorXd center = VectorXd::Zero(2);
    double radius = 1.0;

    ProjectionFn proj_ball = [&center, radius](const VectorXd& x) -> VectorXd {
        return project_l2_ball(x, center, radius);
    };

    try {
        auto result = projected_gradient_descent(func, proj_ball, x0, alpha, 1000, 1e-8);
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "  x_final = [" << result.x_final.transpose() << "]" << std::endl;
        std::cout << "  ||x_final|| = " << result.x_final.norm() << std::endl;
        std::cout << "  f(x_final) = " << func.eval(result.x_final) << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // --- Experiment 3: Simplex constraint (provided projection) ---
    std::cout << "--- Experiment 3: Simplex {x >= 0, sum(x) = 1} ---" << std::endl;

    ProjectionFn proj_simplex = [](const VectorXd& x) -> VectorXd {
        return project_simplex(x);
    };

    try {
        auto result = projected_gradient_descent(func, proj_simplex, x0, alpha, 1000, 1e-8);
        std::cout << "  Converged: " << (result.converged ? "yes" : "no")
                  << " in " << result.iterations << " iterations" << std::endl;
        std::cout << "  x_final = [" << result.x_final.transpose() << "]" << std::endl;
        std::cout << "  sum(x_final) = " << result.x_final.sum() << std::endl;
        std::cout << "  f(x_final) = " << func.eval(result.x_final) << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // --- Experiment 4: Compare constrained vs unconstrained ---
    std::cout << "--- Experiment 4: Unconstrained vs constrained solutions ---" << std::endl;

    ProjectionFn proj_none = [](const VectorXd& x) -> VectorXd {
        return project_identity(x);
    };

    try {
        auto result_free = projected_gradient_descent(func, proj_none, x0, alpha, 1000, 1e-8);
        std::cout << "  Unconstrained:  x = [" << result_free.x_final.transpose()
                  << "], f = " << func.eval(result_free.x_final)
                  << " (" << result_free.iterations << " iters)" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR (unconstrained): " << e.what() << std::endl;
    }

    try {
        auto result_box = projected_gradient_descent(func, proj_box, x0, alpha, 1000, 1e-8);
        std::cout << "  Box [0,2]x[0,2]: x = [" << result_box.x_final.transpose()
                  << "], f = " << func.eval(result_box.x_final)
                  << " (" << result_box.iterations << " iters)" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR (box): " << e.what() << std::endl;
    }

    try {
        auto result_ball = projected_gradient_descent(func, proj_ball, x0, alpha, 1000, 1e-8);
        std::cout << "  L2-ball ||x||<=1: x = [" << result_ball.x_final.transpose()
                  << "], f = " << func.eval(result_ball.x_final)
                  << " (" << result_ball.iterations << " iters)" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "  ERROR (ball): " << e.what() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Key takeaways ===" << std::endl;
    std::cout << "  1. Projected GD = unconstrained GD + project after each step" << std::endl;
    std::cout << "  2. Constraint tightens the feasible region -> higher objective value" << std::endl;
    std::cout << "  3. Stopping criterion: ||x_{k+1} - x_k|| < tol (NOT ||grad|| < tol)" << std::endl;
    std::cout << "  4. Projection cost matters: box O(n), L2-ball O(n), simplex O(n log n)" << std::endl;

    return 0;
}
