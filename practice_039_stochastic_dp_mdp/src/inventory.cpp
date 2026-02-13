// ============================================================================
// Phase 3: Inventory Management MDP
// ============================================================================
//
// This phase applies MDP tools to a classic OR problem: how much inventory
// to order when customer demand is uncertain. The MDP is fully constructed
// for you — your job is to solve it and, more importantly, ANALYZE the
// optimal policy to extract managerial insights.
//
// The inventory model:
//   - State s: current inventory level (0, 1, ..., max_inventory)
//   - Action a: units to order (0, 1, ..., max_order), with s + a <= max_inv
//   - Demand D: random (Poisson-like distribution with given probabilities)
//   - Transition: next_inventory = max(0, s + a - D)
//   - Reward: revenue - ordering cost - holding cost - stockout penalty
//
// The key result: for many inventory models, the optimal policy has the
// elegant (s, S) structure — order up to level S when inventory falls
// below threshold s. You will verify this by analyzing the solved MDP.
//
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <numeric>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// ─── MDP definition ─────────────────────────────────────────────────────────

struct MDP {
    int n_states;
    int n_actions;
    std::vector<std::vector<std::vector<double>>> P;
    std::vector<std::vector<double>> R;
    double gamma;
    std::vector<bool> terminal;
};

// ─── Inventory parameters ───────────────────────────────────────────────────

struct InventoryParams {
    int max_inventory;        // maximum inventory capacity
    int max_order;            // maximum units to order per period
    double holding_cost;      // cost per unit held per period
    double stockout_penalty;  // penalty per unit of unmet demand
    double order_cost;        // cost per unit ordered
    double selling_price;     // revenue per unit sold
    std::vector<double> demand_probs;  // demand_probs[d] = P(demand = d)
};

// ─── Provided: build inventory MDP from parameters ──────────────────────────

/// Convert inventory parameters into a generic MDP.
///
/// State space: {0, 1, ..., max_inventory}
///   State s represents having s units in stock at the start of the period.
///
/// Action space: {0, 1, ..., max_order}
///   Action a represents ordering a units. Feasibility: s + a <= max_inventory.
///   Infeasible actions get reward -1e9 and self-loop (they'll never be chosen).
///
/// Transition dynamics (for feasible action a in state s):
///   After ordering, inventory becomes s + a (before demand realization).
///   Demand D is drawn from demand_probs.
///   Next state s' = max(0, s + a - D) = max(0, on_hand - D).
///   If D > on_hand: stockout (lost sales).
///   If D <= on_hand: leftover inventory carried to next period.
///
/// Reward R(s, a):
///   Expected over demand distribution:
///     R(s,a) = sum_d P(D=d) * [
///       selling_price * min(d, s+a)            (revenue from sales)
///       - order_cost * a                        (ordering cost, deterministic)
///       - holding_cost * max(0, s+a-d)          (holding cost on leftover)
///       - stockout_penalty * max(0, d - s - a)  (penalty for unmet demand)
///     ]
MDP build_inventory_mdp(const InventoryParams& params) {
    int S = params.max_inventory + 1;  // states: 0, 1, ..., max_inventory
    int A = params.max_order + 1;      // actions: 0, 1, ..., max_order
    int max_demand = static_cast<int>(params.demand_probs.size());

    MDP mdp;
    mdp.n_states = S;
    mdp.n_actions = A;
    mdp.gamma = 0.95;  // long-horizon inventory planning
    mdp.terminal.assign(S, false);  // no terminal states (infinite horizon)

    mdp.P.assign(S, std::vector<std::vector<double>>(
        A, std::vector<double>(S, 0.0)));
    mdp.R.assign(S, std::vector<double>(A, 0.0));

    for (int s = 0; s < S; ++s) {
        for (int a = 0; a < A; ++a) {
            int on_hand = s + a;  // inventory after ordering

            // Feasibility check: can't exceed capacity
            if (on_hand > params.max_inventory) {
                mdp.R[s][a] = -1e9;  // infeasible — huge penalty
                mdp.P[s][a][s] = 1.0;  // self-loop
                continue;
            }

            double expected_reward = 0.0;

            for (int d = 0; d < max_demand; ++d) {
                double prob = params.demand_probs[d];
                if (prob < 1e-15) continue;

                int sold = std::min(d, on_hand);
                int leftover = std::max(0, on_hand - d);
                int unmet = std::max(0, d - on_hand);

                // Revenue - ordering cost - holding cost - stockout penalty
                double reward =
                    params.selling_price * sold
                    - params.order_cost * a
                    - params.holding_cost * leftover
                    - params.stockout_penalty * unmet;

                expected_reward += prob * reward;

                // Transition: next state = leftover inventory
                int next_s = leftover;  // already clamped to [0, max_inv]
                mdp.P[s][a][next_s] += prob;
            }

            mdp.R[s][a] = expected_reward;
        }
    }

    return mdp;
}

// ─── Provided: Poisson-like demand probabilities ────────────────────────────

/// Generate truncated Poisson probabilities for demand.
/// demand_probs[d] = P(D=d) for d = 0, 1, ..., max_demand-1.
/// Truncated: remaining probability mass goes to max_demand-1.
std::vector<double> poisson_demand(double lambda, int max_demand) {
    std::vector<double> probs(max_demand, 0.0);
    double cumulative = 0.0;
    for (int d = 0; d < max_demand - 1; ++d) {
        // Poisson PMF: e^{-lambda} * lambda^d / d!
        double log_prob = -lambda + d * std::log(lambda);
        double factorial = 0.0;
        for (int k = 1; k <= d; ++k) factorial += std::log(k);
        log_prob -= factorial;
        probs[d] = std::exp(log_prob);
        cumulative += probs[d];
    }
    probs[max_demand - 1] = 1.0 - cumulative;  // truncation
    return probs;
}

// ─── Provided: value iteration (for solving the inventory MDP) ──────────────

std::pair<VectorXd, std::vector<int>> value_iteration(
    const MDP& mdp, double tol = 1e-6, int max_iter = 2000)
{
    int S = mdp.n_states;
    int A = mdp.n_actions;
    VectorXd V = VectorXd::Zero(S);
    std::vector<int> policy(S, 0);

    int iter = 0;
    for (; iter < max_iter; ++iter) {
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
                      << " iterations\n";
            break;
        }
    }
    if (iter == max_iter) {
        std::cout << "  Value iteration: max iterations reached (" << max_iter << ")\n";
    }
    return {V, policy};
}

// ─── Provided: print inventory policy ───────────────────────────────────────

void print_inventory_policy(const std::vector<int>& policy, int max_inv) {
    std::cout << "\nOptimal Inventory Policy:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::setw(15) << "Inventory" << std::setw(15) << "Order Qty"
              << std::setw(15) << "Order-up-to" << "\n";
    std::cout << std::string(40, '-') << "\n";
    for (int s = 0; s <= max_inv; ++s) {
        std::cout << std::setw(15) << s
                  << std::setw(15) << policy[s]
                  << std::setw(15) << (s + policy[s]) << "\n";
    }
}

// ─── TODO(human): Analyze the optimal inventory policy ──────────────────────

/// Analyze the solved inventory MDP to extract managerial insights.
///
/// TODO(human): Implement the analysis of the optimal inventory policy.
///
/// ANALYSIS TASKS:
///   1. Print the policy table: for each inventory level s, show the optimal
///      order quantity a = policy[s] and the resulting order-up-to level s+a.
///      (Already done by print_inventory_policy — call it here.)
///
///   2. Identify the REORDER POINT (little-s):
///      The highest inventory level where the policy still orders something.
///      Formally: reorder_point = max { s : policy[s] > 0 }.
///      Below this level, you should order; at or above, you don't.
///
///   3. Identify the ORDER-UP-TO LEVEL (big-S):
///      When the policy does order, what level does it bring inventory to?
///      Formally: S = s + policy[s] for any s where policy[s] > 0.
///      If the policy is truly (s,S), this value should be the same
///      regardless of which s < reorder_point you look at.
///
///   4. Verify the (s, S) STRUCTURE:
///      Check whether the policy follows the pattern:
///        - For all states below the reorder point: order up to exactly S
///        - For all states at or above the reorder point: order nothing (0)
///      Print whether the (s, S) structure holds, and what s and S are.
///      The (s, S) result (Scarf, 1960) says this structure is optimal
///      for inventory problems with fixed ordering costs — but it often
///      emerges even without fixed costs in discounted infinite-horizon models.
///
///   5. Compute EXPECTED PERFORMANCE:
///      Print V(s) for a few representative starting inventory levels
///      (e.g., s=0, s=reorder_point, s=big_S, s=max_inventory).
///      V(s) represents the expected total discounted profit starting from
///      inventory level s under the optimal policy.
///      Also print the expected profit per period: V(s) * (1 - gamma),
///      which gives the long-run average reward.
///
/// WHY THIS MATTERS:
///   The gap between "solving an MDP" and "extracting business value" is
///   where practitioners struggle. A warehouse manager doesn't want a
///   16-entry lookup table — they want: "reorder when stock drops below 5,
///   order enough to bring it to 12." The (s, S) analysis gives them exactly
///   that. Discovering this structure from the MDP solution (not assuming it)
///   validates the theory and builds confidence in the model.
///
/// PARAMETERS:
///   - mdp: the solved MDP
///   - V: optimal value function
///   - policy: optimal policy
///   - params: inventory parameters (for context: costs, prices)
void analyze_optimal_policy(const MDP& mdp, const VectorXd& V,
                            const std::vector<int>& policy,
                            const InventoryParams& params)
{
    // TODO(human): implement the 5-step analysis described above
    throw std::runtime_error("TODO(human): analyze_optimal_policy not implemented");
}

// ─── main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=============================================\n";
    std::cout << " Phase 3: Inventory Management MDP\n";
    std::cout << "=============================================\n";

    // --- Define inventory parameters ---
    InventoryParams params;
    params.max_inventory = 15;
    params.max_order = 10;
    params.holding_cost = 1.0;       // $1 per unit per period
    params.stockout_penalty = 5.0;   // $5 per unit of unmet demand
    params.order_cost = 2.0;         // $2 per unit ordered
    params.selling_price = 8.0;      // $8 per unit sold
    params.demand_probs = poisson_demand(4.0, params.max_inventory + 1);

    std::cout << "\nInventory Parameters:\n";
    std::cout << "  Max inventory: " << params.max_inventory << "\n";
    std::cout << "  Max order: " << params.max_order << "\n";
    std::cout << "  Holding cost: $" << params.holding_cost << "/unit/period\n";
    std::cout << "  Stockout penalty: $" << params.stockout_penalty << "/unit\n";
    std::cout << "  Order cost: $" << params.order_cost << "/unit\n";
    std::cout << "  Selling price: $" << params.selling_price << "/unit\n";
    std::cout << "  Mean demand: 4.0 (Poisson)\n";

    std::cout << "\nDemand distribution:\n";
    for (int d = 0; d < static_cast<int>(params.demand_probs.size()); ++d) {
        if (params.demand_probs[d] > 0.001) {
            std::cout << "  P(D=" << d << ") = " << std::fixed
                      << std::setprecision(4) << params.demand_probs[d] << "\n";
        }
    }

    // --- Build and solve the MDP ---
    std::cout << "\n--- Building Inventory MDP ---\n";
    MDP mdp = build_inventory_mdp(params);
    std::cout << "  States: " << mdp.n_states << " (inventory 0.." << params.max_inventory << ")\n";
    std::cout << "  Actions: " << mdp.n_actions << " (order 0.." << params.max_order << ")\n";
    std::cout << "  Discount: " << mdp.gamma << "\n";

    std::cout << "\n--- Solving with Value Iteration ---\n";
    auto [V, policy] = value_iteration(mdp, 1e-8, 5000);

    // --- Analyze the optimal policy ---
    std::cout << "\n--- Analyzing Optimal Policy ---\n";
    analyze_optimal_policy(mdp, V, policy, params);

    // --- Sensitivity analysis: vary stockout penalty ---
    std::cout << "\n--- Sensitivity: Varying Stockout Penalty ---\n";
    std::cout << std::setw(15) << "Penalty" << std::setw(15) << "Reorder Pt"
              << std::setw(15) << "Order-up-to" << "\n";
    std::cout << std::string(45, '-') << "\n";

    for (double penalty : {1.0, 3.0, 5.0, 10.0, 20.0}) {
        InventoryParams p2 = params;
        p2.stockout_penalty = penalty;
        MDP mdp2 = build_inventory_mdp(p2);
        auto [V2, pol2] = value_iteration(mdp2, 1e-8, 5000);

        // Find reorder point and order-up-to level
        int reorder_pt = -1;
        int order_up_to = -1;
        for (int s = params.max_inventory; s >= 0; --s) {
            if (pol2[s] > 0) {
                reorder_pt = s;
                order_up_to = s + pol2[s];
                break;
            }
        }
        std::cout << std::setw(12) << "$" << std::fixed << std::setprecision(0)
                  << penalty << std::setw(15) << reorder_pt
                  << std::setw(15) << order_up_to << "\n";
    }

    return 0;
}
