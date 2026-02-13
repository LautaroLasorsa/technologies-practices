// ============================================================================
// Phase 1: Value Iteration on a Stochastic Gridworld
// ============================================================================
//
// This phase implements the most fundamental MDP algorithm: value iteration.
// The testbed is a 4x4 gridworld with a "slippery floor" — when the agent
// chooses to move in a direction, there's an 80% chance of going there and
// a 10% chance of slipping to each perpendicular direction. This stochasticity
// makes the problem a true MDP (not just shortest-path search).
//
// The agent starts anywhere and wants to reach the goal cell (bottom-right)
// while avoiding a penalty cell. Each step costs -0.04 to encourage finding
// the goal quickly. The goal gives +1 reward, and a trap cell gives -1.
//
// You will implement value_iteration(): the Bellman backup loop that
// converges to the optimal value function V* and extracts the optimal policy.
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

// ─── MDP definition ─────────────────────────────────────────────────────────

/// Generic finite MDP: (S, A, P, R, gamma).
///
/// Transition probabilities: P[s][a][s'] = probability of reaching s' from s
///   under action a. Each P[s][a] is a distribution over S (sums to 1).
///
/// Rewards: R[s][a] = immediate reward for taking action a in state s.
///   (State-action reward model. Some texts use R(s,a,s') but the expectation
///   over s' gives R(s,a) = sum_{s'} P(s'|s,a) * r(s,a,s'), which is what
///   we store directly.)
struct MDP {
    int n_states;
    int n_actions;
    // P[s][a][s'] — transition probability
    std::vector<std::vector<std::vector<double>>> P;
    // R[s][a] — expected immediate reward
    std::vector<std::vector<double>> R;
    double gamma;  // discount factor
    // Terminal states (no further transitions — value is 0 from these)
    std::vector<bool> terminal;
};

// ─── Provided: print helpers ────────────────────────────────────────────────

/// Print value function as a grid (for gridworld visualization).
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

/// Print policy as a grid with arrow symbols.
/// Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
void print_policy(const std::vector<int>& policy, int rows, int cols,
                   const std::vector<bool>& terminal) {
    const std::string arrows[] = {"  ^  ", "  v  ", "  <  ", "  >  "};
    std::cout << "\nOptimal Policy:\n";
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

// ─── Provided: build the gridworld MDP ──────────────────────────────────────

/// Construct a 4x4 gridworld MDP with slippery floor.
///
/// Grid layout (0-indexed):
///   (0,0) (0,1) (0,2) (0,3)
///   (1,0) (1,1) (1,2) (1,3)
///   (2,0) (2,1) (2,2) (2,3)
///   (3,0) (3,1) (3,2) (3,3)
///
/// Special cells:
///   (0,3) = GOAL (+1 reward, terminal)
///   (1,1) = TRAP (-1 reward, terminal)
///   All other cells: step cost = -0.04
///
/// Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
///
/// Slippery floor: 80% intended direction, 10% each perpendicular.
///   e.g., if action=RIGHT: 80% right, 10% up, 10% down.
///   Hitting a wall means staying in place (bounce back).
///
/// Discount factor: gamma = 0.9
MDP build_gridworld(int rows = 4, int cols = 4) {
    int n_states = rows * cols;
    int n_actions = 4;  // UP, DOWN, LEFT, RIGHT
    double gamma = 0.9;
    double step_cost = -0.04;
    double goal_reward = 1.0;
    double trap_reward = -1.0;

    int goal_state = 0 * cols + 3;  // (0,3)
    int trap_state = 1 * cols + 1;  // (1,1)

    MDP mdp;
    mdp.n_states = n_states;
    mdp.n_actions = n_actions;
    mdp.gamma = gamma;
    mdp.terminal.assign(n_states, false);
    mdp.terminal[goal_state] = true;
    mdp.terminal[trap_state] = true;

    // Initialize P and R
    mdp.P.assign(n_states, std::vector<std::vector<double>>(
        n_actions, std::vector<double>(n_states, 0.0)));
    mdp.R.assign(n_states, std::vector<double>(n_actions, step_cost));

    // Set terminal state rewards
    for (int a = 0; a < n_actions; ++a) {
        mdp.R[goal_state][a] = goal_reward;
        mdp.R[trap_state][a] = trap_reward;
    }

    // Direction vectors: UP, DOWN, LEFT, RIGHT
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    // Perpendicular directions for each action
    // UP(0) -> perp: LEFT(2), RIGHT(3)
    // DOWN(1) -> perp: LEFT(2), RIGHT(3)
    // LEFT(2) -> perp: UP(0), DOWN(1)
    // RIGHT(3) -> perp: UP(0), DOWN(1)
    int perp[4][2] = {{2, 3}, {2, 3}, {0, 1}, {0, 1}};

    auto clamp_move = [&](int r, int c, int action) -> int {
        int nr = r + dr[action];
        int nc = c + dc[action];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
            return r * cols + c;  // bounce back (stay in place)
        }
        return nr * cols + nc;
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int s = r * cols + c;
            if (mdp.terminal[s]) {
                // Terminal states: self-loop with probability 1 for all actions
                for (int a = 0; a < n_actions; ++a) {
                    mdp.P[s][a][s] = 1.0;
                }
                continue;
            }
            for (int a = 0; a < n_actions; ++a) {
                // 80% intended direction
                int s_intended = clamp_move(r, c, a);
                mdp.P[s][a][s_intended] += 0.8;

                // 10% each perpendicular
                int s_perp1 = clamp_move(r, c, perp[a][0]);
                mdp.P[s][a][s_perp1] += 0.1;

                int s_perp2 = clamp_move(r, c, perp[a][1]);
                mdp.P[s][a][s_perp2] += 0.1;
            }
        }
    }

    return mdp;
}

// ─── TODO(human): Value Iteration ───────────────────────────────────────────

/// Run value iteration on the given MDP.
///
/// TODO(human): Implement the Bellman backup loop.
///
/// VALUE ITERATION ALGORITHM:
///   Initialize V_0(s) = 0 for all states.
///   Repeat until convergence:
///     For each non-terminal state s:
///       For each action a:
///         Compute Q(s,a) = R(s,a) + gamma * sum_{s'} P(s'|s,a) * V_k(s')
///       V_{k+1}(s) = max_a Q(s,a)
///       policy(s)  = argmax_a Q(s,a)
///     Check convergence: delta = max_s |V_{k+1}(s) - V_k(s)|
///     If delta < tol: stop.
///
/// CONVERGENCE RATE:
///   ||V_{k+1} - V*||_inf <= gamma * ||V_k - V*||_inf
///   For gamma=0.9, error shrinks by 10% per iteration.
///   Typical convergence: 50-200 iterations for tol=1e-6.
///
/// IMPLEMENTATION NOTES:
///   - Use Eigen::VectorXd for V (size n_states). Initialize to zero.
///   - Skip terminal states in the update (their value stays at 0).
///   - For Q(s,a): iterate over all s' from 0 to n_states-1,
///     accumulating P[s][a][s'] * V(s'). Most P entries are 0 in a
///     gridworld (sparse), but the loop is fine for 16 states.
///   - Track the best action per state to build the policy vector.
///   - Print iteration count and delta at convergence.
///
/// PARAMETERS:
///   - mdp: the MDP to solve
///   - tol: convergence tolerance (stop when max change < tol)
///   - max_iter: safety limit
///
/// RETURNS: pair of (optimal value function V*, optimal policy pi*)
std::pair<VectorXd, std::vector<int>> value_iteration(
    const MDP& mdp, double tol = 1e-6, int max_iter = 1000)
{
    // TODO(human): implement the Bellman backup loop described above
    throw std::runtime_error("TODO(human): value_iteration not implemented");
}

// ─── main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=============================================\n";
    std::cout << " Phase 1: Value Iteration on 4x4 Gridworld\n";
    std::cout << "=============================================\n";

    std::cout << "\nGridworld layout:\n";
    std::cout << "  (0,0) (0,1) (0,2) (0,3)=GOAL(+1)\n";
    std::cout << "  (1,0) (1,1)=TRAP(-1) (1,2) (1,3)\n";
    std::cout << "  (2,0) (2,1) (2,2) (2,3)\n";
    std::cout << "  (3,0) (3,1) (3,2) (3,3)\n";
    std::cout << "\nSlippery floor: 80% intended, 10% each perpendicular\n";
    std::cout << "Step cost: -0.04, Discount: 0.9\n";

    int rows = 4, cols = 4;
    MDP mdp = build_gridworld(rows, cols);

    std::cout << "\n--- Running Value Iteration ---\n";
    auto [V, policy] = value_iteration(mdp, 1e-6, 1000);

    print_value_function(V, rows, cols, mdp.terminal);
    print_policy(policy, rows, cols, mdp.terminal);

    // Show convergence with different tolerances
    std::cout << "\n--- Convergence comparison ---\n";
    for (double tol : {1e-2, 1e-4, 1e-6, 1e-8}) {
        auto [V_test, _] = value_iteration(mdp, tol, 10000);
        // Count iterations (re-run to get iteration count printed)
        // The function prints iteration count internally
        std::cout << "  tol=" << std::scientific << tol
                  << "  V(3,0)=" << std::fixed << std::setprecision(6)
                  << V_test(3 * cols + 0) << "\n";
    }

    // Show effect of different discount factors
    std::cout << "\n--- Effect of discount factor gamma ---\n";
    for (double g : {0.5, 0.7, 0.9, 0.95, 0.99}) {
        MDP mdp_g = build_gridworld(rows, cols);
        mdp_g.gamma = g;
        auto [V_g, pol_g] = value_iteration(mdp_g, 1e-6, 5000);
        std::cout << "  gamma=" << std::fixed << std::setprecision(2) << g
                  << "  V(3,0)=" << std::setprecision(4) << V_g(3 * cols + 0)
                  << "  V(0,0)=" << std::setprecision(4) << V_g(0) << "\n";
    }

    return 0;
}
