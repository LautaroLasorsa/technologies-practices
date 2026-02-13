// ============================================================================
// Phase 2: Policy Iteration with Exact Policy Evaluation
// ============================================================================
//
// Policy iteration is the alternative to value iteration for solving MDPs.
// Instead of iterating Bellman backups hundreds of times, it:
//   1. Evaluates the current policy EXACTLY by solving a linear system
//   2. Improves the policy greedily (one-step lookahead)
//   3. Repeats until the policy stabilizes (typically 3-5 iterations)
//
// The key insight: for a fixed policy pi, the value function V^pi satisfies
// a system of linear equations (not a fixed-point iteration). Eigen can
// solve this in O(|S|^3) — expensive per step but far fewer steps needed.
//
// You will implement both policy_evaluation() and policy_iteration(),
// then compare with value iteration on the same gridworld.
//
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// ─── MDP definition (same as Phase 1) ───────────────────────────────────────

struct MDP {
    int n_states;
    int n_actions;
    std::vector<std::vector<std::vector<double>>> P;
    std::vector<std::vector<double>> R;
    double gamma;
    std::vector<bool> terminal;
};

// ─── Provided: print helpers (same as Phase 1) ─────────────────────────────

void print_value_function(const VectorXd& V, int rows, int cols,
                          const std::vector<bool>& terminal) {
    std::cout << "\nValue Function:\n";
    std::cout << std::string(cols * 10 + 1, '-') << "\n";
    for (int r = 0; r < rows; ++r) {
        std::cout << "|";
        for (int c = 0; c < cols; ++c) {
            int s = r * cols + c;
            if (terminal[s]) {
                std::cout << std::setw(8) << "  TERM" << " |";
            } else {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                          << V(s) << " |";
            }
        }
        std::cout << "\n" << std::string(cols * 10 + 1, '-') << "\n";
    }
}

void print_policy(const std::vector<int>& policy, int rows, int cols,
                   const std::vector<bool>& terminal) {
    const std::string arrows[] = {"  ^  ", "  v  ", "  <  ", "  >  "};
    std::cout << "\nPolicy:\n";
    std::cout << std::string(cols * 8 + 1, '-') << "\n";
    for (int r = 0; r < rows; ++r) {
        std::cout << "|";
        for (int c = 0; c < cols; ++c) {
            int s = r * cols + c;
            if (terminal[s]) {
                std::cout << "  TERM |";
            } else {
                std::cout << " " << arrows[policy[s]] << " |";
            }
        }
        std::cout << "\n" << std::string(cols * 8 + 1, '-') << "\n";
    }
}

// ─── Provided: build the gridworld MDP (same as Phase 1) ───────────────────

MDP build_gridworld(int rows = 4, int cols = 4) {
    int n_states = rows * cols;
    int n_actions = 4;
    double gamma = 0.9;
    double step_cost = -0.04;
    double goal_reward = 1.0;
    double trap_reward = -1.0;

    int goal_state = 0 * cols + 3;
    int trap_state = 1 * cols + 1;

    MDP mdp;
    mdp.n_states = n_states;
    mdp.n_actions = n_actions;
    mdp.gamma = gamma;
    mdp.terminal.assign(n_states, false);
    mdp.terminal[goal_state] = true;
    mdp.terminal[trap_state] = true;

    mdp.P.assign(n_states, std::vector<std::vector<double>>(
        n_actions, std::vector<double>(n_states, 0.0)));
    mdp.R.assign(n_states, std::vector<double>(n_actions, step_cost));

    for (int a = 0; a < n_actions; ++a) {
        mdp.R[goal_state][a] = goal_reward;
        mdp.R[trap_state][a] = trap_reward;
    }

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    int perp[4][2] = {{2, 3}, {2, 3}, {0, 1}, {0, 1}};

    auto clamp_move = [&](int r, int c, int action) -> int {
        int nr = r + dr[action];
        int nc = c + dc[action];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
            return r * cols + c;
        return nr * cols + nc;
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int s = r * cols + c;
            if (mdp.terminal[s]) {
                for (int a = 0; a < n_actions; ++a)
                    mdp.P[s][a][s] = 1.0;
                continue;
            }
            for (int a = 0; a < n_actions; ++a) {
                mdp.P[s][a][clamp_move(r, c, a)] += 0.8;
                mdp.P[s][a][clamp_move(r, c, perp[a][0])] += 0.1;
                mdp.P[s][a][clamp_move(r, c, perp[a][1])] += 0.1;
            }
        }
    }

    return mdp;
}

// ─── Provided: value iteration (for comparison) ─────────────────────────────

/// Value iteration (provided, so you can compare results with policy iteration).
std::pair<VectorXd, std::vector<int>> value_iteration(
    const MDP& mdp, double tol = 1e-6, int max_iter = 1000)
{
    int S = mdp.n_states;
    int A = mdp.n_actions;
    VectorXd V = VectorXd::Zero(S);
    std::vector<int> policy(S, 0);

    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd V_new = VectorXd::Zero(S);
        for (int s = 0; s < S; ++s) {
            if (mdp.terminal[s]) continue;
            double best_q = -1e18;
            int best_a = 0;
            for (int a = 0; a < A; ++a) {
                double q = mdp.R[s][a];
                for (int sp = 0; sp < S; ++sp)
                    q += mdp.gamma * mdp.P[s][a][sp] * V(sp);
                if (q > best_q) { best_q = q; best_a = a; }
            }
            V_new(s) = best_q;
            policy[s] = best_a;
        }
        double delta = (V_new - V).cwiseAbs().maxCoeff();
        V = V_new;
        if (delta < tol) {
            std::cout << "  Value iteration converged in " << (iter + 1)
                      << " iterations (delta=" << std::scientific << delta << ")\n";
            break;
        }
    }
    return {V, policy};
}

// ─── TODO(human): Policy Evaluation (exact, via linear system) ──────────────

/// Compute V^pi: the value function for a fixed policy pi.
///
/// TODO(human): Build and solve the linear system (I - gamma * P^pi) V = R^pi.
///
/// POLICY EVALUATION — THE LINEAR SYSTEM:
///   For a fixed policy pi, V^pi satisfies:
///     V^pi(s) = R(s, pi(s)) + gamma * sum_{s'} P(s'|s, pi(s)) * V^pi(s')
///
///   In matrix form:
///     V = R^pi + gamma * P^pi * V
///     (I - gamma * P^pi) * V = R^pi
///
///   Build the matrices:
///     P_pi(s, s') = P(s' | s, pi(s))  — the |S| x |S| transition matrix
///       under policy pi. Row s contains the transition distribution when
///       taking action pi(s) in state s.
///     R_pi(s) = R(s, pi(s))  — the |S|-length reward vector under policy pi.
///
///   Then solve: (I - gamma * P_pi) * V = R_pi
///     using Eigen: V = A.colPivHouseholderQr().solve(R_pi)
///     where A = I - gamma * P_pi
///
///   WHY THIS WORKS:
///     The matrix (I - gamma * P_pi) is always invertible when gamma < 1.
///     The spectral radius of P_pi is at most 1 (it's a stochastic matrix),
///     so the spectral radius of gamma * P_pi is at most gamma < 1,
///     which means (I - gamma * P_pi) is non-singular.
///
///   COST: O(|S|^3) for the linear solve. For |S|=16 (our gridworld), this
///   is trivial. For |S|=10000, you'd use iterative policy evaluation instead.
///
/// PARAMETERS:
///   - mdp: the MDP
///   - policy: vector of length n_states, policy[s] = action to take in state s
///
/// RETURNS: V^pi as an Eigen VectorXd of length n_states
VectorXd policy_evaluation(const MDP& mdp, const std::vector<int>& policy) {
    // TODO(human): build P_pi, R_pi, solve (I - gamma*P_pi)*V = R_pi
    throw std::runtime_error("TODO(human): policy_evaluation not implemented");
}

// ─── TODO(human): Policy Iteration ──────────────────────────────────────────

/// Run policy iteration: alternate exact evaluation and greedy improvement.
///
/// TODO(human): Implement the policy iteration loop.
///
/// POLICY ITERATION ALGORITHM:
///   1. Initialize policy pi(s) = 0 for all states (arbitrary).
///   2. EVALUATE: V = policy_evaluation(mdp, pi)  — exact value of current policy.
///   3. IMPROVE: For each non-terminal state s:
///        pi_new(s) = argmax_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V(s') ]
///      This is the "greedy" policy w.r.t. the current value function.
///   4. CHECK CONVERGENCE: If pi_new == pi for all states, the policy is optimal.
///      (Policy improvement theorem: if the greedy policy equals the current
///       policy, both are optimal.)
///   5. Otherwise: set pi = pi_new, go to step 2.
///
/// WHY IT CONVERGES:
///   - Each improvement step strictly improves V (or leaves it unchanged).
///   - There are finitely many deterministic policies (|A|^|S|).
///   - Therefore, the algorithm must terminate.
///   - In practice, it converges in 3-10 iterations for most MDPs.
///
/// COMPARISON WITH VALUE ITERATION:
///   - Value iteration: many cheap iterations (O(|S|^2 * |A|) per iter)
///   - Policy iteration: few expensive iterations (O(|S|^3) per iter for eval)
///   - For small |S|: policy iteration wins (fewer total operations)
///   - For large |S|: value iteration wins (avoid cubic solve)
///
/// PARAMETERS:
///   - mdp: the MDP to solve
///   - max_iter: safety limit on number of policy improvement rounds
///
/// RETURNS: pair of (optimal value function V*, optimal policy pi*)
std::pair<VectorXd, std::vector<int>> policy_iteration(
    const MDP& mdp, int max_iter = 100)
{
    // TODO(human): implement the evaluate-improve loop described above
    throw std::runtime_error("TODO(human): policy_iteration not implemented");
}

// ─── main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=============================================\n";
    std::cout << " Phase 2: Policy Iteration with Exact Eval\n";
    std::cout << "=============================================\n";

    int rows = 4, cols = 4;
    MDP mdp = build_gridworld(rows, cols);

    // --- Run policy iteration ---
    std::cout << "\n--- Policy Iteration ---\n";
    auto [V_pi, pol_pi] = policy_iteration(mdp);
    print_value_function(V_pi, rows, cols, mdp.terminal);
    print_policy(pol_pi, rows, cols, mdp.terminal);

    // --- Run value iteration for comparison ---
    std::cout << "\n--- Value Iteration (for comparison) ---\n";
    auto [V_vi, pol_vi] = value_iteration(mdp);
    print_value_function(V_vi, rows, cols, mdp.terminal);
    print_policy(pol_vi, rows, cols, mdp.terminal);

    // --- Compare results ---
    std::cout << "\n--- Comparison ---\n";
    double max_v_diff = (V_pi - V_vi).cwiseAbs().maxCoeff();
    std::cout << "Max |V_pi - V_vi| = " << std::scientific << max_v_diff << "\n";

    bool policies_match = true;
    for (int s = 0; s < mdp.n_states; ++s) {
        if (!mdp.terminal[s] && pol_pi[s] != pol_vi[s]) {
            policies_match = false;
            std::cout << "  Policy differs at state " << s
                      << ": PI=" << pol_pi[s] << " VI=" << pol_vi[s] << "\n";
        }
    }
    if (policies_match) {
        std::cout << "Policies match exactly!\n";
    }

    // --- Demonstrate policy evaluation on a suboptimal policy ---
    std::cout << "\n--- Policy Evaluation: 'always go right' ---\n";
    std::vector<int> right_policy(mdp.n_states, 3);  // action 3 = RIGHT
    VectorXd V_right = policy_evaluation(mdp, right_policy);
    print_value_function(V_right, rows, cols, mdp.terminal);
    std::cout << "V*(3,0) = " << std::fixed << std::setprecision(4) << V_pi(3 * cols + 0)
              << "  vs  V^right(3,0) = " << V_right(3 * cols + 0)
              << "  (optimal is higher)\n";

    return 0;
}
