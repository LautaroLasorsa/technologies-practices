# Practice 039: Stochastic Dynamic Programming — MDPs & Value Iteration

## Technologies

- **C++17** — Modern C++ with structured bindings, constexpr, std::optional
- **Eigen 3.4** — Header-only linear algebra library (matrices, vectors, linear system solves), fetched via CMake FetchContent
- **CMake 3.16+** — Build system with FetchContent for dependency management

## Stack

- C++17
- Eigen 3.4.0 (header-only, fetched via FetchContent)

## Theoretical Context

### Dynamic Programming in the OR Context

**Dynamic Programming (DP)** in operations research is a framework for **sequential decision-making over time**. Unlike the DP familiar from competitive programming (which optimizes over subproblems of a fixed input), OR-style DP involves a **decision-maker** who:

1. **Observes** the current state of a system (inventory level, machine condition, portfolio value).
2. **Chooses an action** from a set of feasible actions (order quantity, repair/replace, buy/sell).
3. **Receives a reward** (or incurs a cost) that depends on the state and action.
4. **Transitions** to a new state — possibly **stochastically** (random demand, machine failure, market movement).
5. **Repeats** at the next time step.

The goal is to find a **policy** (a rule mapping states to actions) that maximizes the total expected reward (or minimizes total expected cost) over a planning horizon, which may be finite or infinite.

Richard Bellman formalized this framework in the 1950s. The key insight is the **Principle of Optimality**: an optimal policy has the property that, regardless of how you arrived at a state, the remaining decisions from that state onward must also be optimal. This recursive structure is what makes DP work.

### Markov Decision Process (MDP)

An **MDP** is the mathematical formalization of sequential decision-making under uncertainty. It is defined by the tuple **(S, A, P, R, gamma)**:

| Component | Symbol | Definition |
|-----------|--------|------------|
| **States** | S | Finite set of states the system can be in |
| **Actions** | A | Finite set of actions available (may depend on state) |
| **Transition probabilities** | P(s'\|s,a) | Probability of moving to state s' when taking action a in state s |
| **Rewards** | R(s,a) | Immediate reward received when taking action a in state s |
| **Discount factor** | gamma in [0,1) | How much future rewards are worth relative to immediate rewards |

The **Markov property** is critical: the transition probability P(s'|s,a) depends only on the current state s and action a, not on the history of how we got to s. This makes the problem tractable — we only need to track the current state, not the entire trajectory.

The **discount factor** gamma serves two purposes: (1) it makes the infinite-horizon problem well-defined (total reward is bounded), and (2) it models the practical preference for immediate rewards over distant ones (time value of money, risk of system termination).

### The Bellman Equation

The **optimal value function** V*(s) gives the maximum expected discounted reward achievable starting from state s. It satisfies the **Bellman optimality equation**:

```
V*(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V*(s') ]
```

This says: the value of state s is the best action's immediate reward plus the discounted expected value of wherever we end up. This is a **fixed-point equation** — V* is the unique function satisfying it (for gamma < 1).

The **optimal policy** pi*(s) is the action that achieves the maximum:

```
pi*(s) = argmax_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V*(s') ]
```

### Value Iteration

**Value iteration** is the simplest algorithm for solving MDPs. It computes V* by iterating the Bellman backup operator:

```
V_0(s) = 0  for all s  (or any initial guess)

V_{k+1}(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V_k(s') ]
```

**Convergence guarantee:** Value iteration converges to V* at a geometric rate:

```
||V_{k+1} - V*||_inf <= gamma * ||V_k - V*||_inf
```

So for gamma = 0.9, the error decreases by a factor of 0.9 each iteration. After k iterations, the error is at most gamma^k * ||V_0 - V*||. In practice, we stop when `max_s |V_{k+1}(s) - V_k(s)| < tol`.

**Complexity per iteration:** O(|S|^2 * |A|) — for each state, for each action, sum over all successor states.

**Total complexity:** O(|S|^2 * |A| * log(1/tol) / log(1/gamma)) iterations to reach tolerance tol.

### Policy Iteration

**Policy iteration** is an alternative algorithm that often converges in far fewer iterations:

1. **Initialize:** Start with an arbitrary policy pi_0 (e.g., always take action 0).
2. **Policy Evaluation:** Compute V^{pi} exactly — the value function for the current policy pi. Since pi is fixed, V^{pi} satisfies:
   ```
   V^pi(s) = R(s, pi(s)) + gamma * sum_{s'} P(s'|s, pi(s)) * V^pi(s')
   ```
   This is a **system of linear equations** (one per state), solvable exactly.
3. **Policy Improvement:** For each state, compute the greedy action:
   ```
   pi_{k+1}(s) = argmax_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V^{pi_k}(s') ]
   ```
4. **Convergence check:** If pi_{k+1} == pi_k for all states, the policy is optimal. Stop.
5. **Repeat** from step 2 with the new policy.

**Convergence guarantee:** Policy iteration converges in at most |A|^|S| steps (the number of deterministic policies). In practice, it typically converges in 3-10 iterations, even for large MDPs.

**Trade-off vs value iteration:** Each policy iteration step is more expensive (O(|S|^3) for the linear solve) but the number of iterations is much smaller. For small-to-medium MDPs, policy iteration is usually faster overall.

### Policy Evaluation as a Linear System

For a fixed policy pi, the value function V^pi satisfies:

```
V^pi(s) = R(s, pi(s)) + gamma * sum_{s'} P(s'|s, pi(s)) * V^pi(s')
```

In matrix form, let:
- **R^pi** be the vector where R^pi(s) = R(s, pi(s))
- **P^pi** be the matrix where P^pi(s, s') = P(s'|s, pi(s))

Then: `V = R^pi + gamma * P^pi * V`

Rearranging: `(I - gamma * P^pi) * V = R^pi`

This is a standard linear system `Ax = b` where A = (I - gamma * P^pi) and b = R^pi. The matrix (I - gamma * P^pi) is always invertible when gamma < 1 (because the spectral radius of gamma * P^pi is at most gamma < 1). Solve with Eigen's `.colPivHouseholderQr().solve()` or direct `.inverse()` for small systems.

### OR Applications of MDPs

MDPs model a wide range of OR problems:

| Application | State | Action | Stochasticity |
|-------------|-------|--------|---------------|
| **Inventory management** | Current stock level | Order quantity | Random customer demand |
| **Machine maintenance** | Machine condition (good/degraded/failed) | Repair, replace, or do nothing | Random failure/degradation |
| **Revenue management** | Time remaining + seats available | Price to charge | Random customer arrivals |
| **Resource allocation** | Available resources | How to allocate | Random task arrivals |
| **Queueing control** | Number of customers in queue | Number of servers to activate | Random arrivals and service times |

### Inventory Management MDP

The **inventory management problem** is a classic OR application of MDPs:

- **State s:** Current inventory level (0, 1, 2, ..., max_inventory).
- **Action a:** Number of units to order (0, 1, ..., max_order), subject to s + a <= max_inventory.
- **Demand D:** Random variable (e.g., Poisson distribution). Realized after ordering decision.
- **Transition:** Next inventory = max(0, s + a - D). Excess demand is lost (no backlogging in our model).
- **Reward:** R(s, a) = revenue from sales - ordering cost - holding cost - stockout penalty.
  - Revenue: selling_price * min(D, s + a)
  - Ordering cost: order_cost * a (plus possible fixed cost for a > 0)
  - Holding cost: holding_cost * max(0, s + a - D) (cost of unsold inventory)
  - Stockout penalty: stockout_penalty * max(0, D - s - a) (cost of unmet demand)

**The (s, S) policy:** A fundamental result in inventory theory (due to Scarf, 1960) states that for many inventory models, the optimal policy has a simple structure: there exists a **reorder point** s and an **order-up-to level** S such that:
- If inventory < s: order enough to bring inventory up to S.
- If inventory >= s: don't order.

This (s, S) structure emerges naturally from solving the MDP — we don't impose it, we discover it.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **MDP** | (S, A, P, R, gamma) — mathematical model for sequential decision-making under uncertainty |
| **State** | Current configuration of the system |
| **Action** | Decision made by the agent at each state |
| **Transition probability** | P(s'\|s,a) — probability of reaching s' from s under action a |
| **Reward** | R(s,a) — immediate payoff for taking action a in state s |
| **Discount factor** | gamma in [0,1) — weight on future vs immediate rewards |
| **Value function** | V(s) — expected total discounted reward from state s onward |
| **Policy** | pi: S -> A — mapping from states to actions |
| **Bellman equation** | V*(s) = max_a [R(s,a) + gamma * sum P(s'\|s,a) V*(s')] — fixed-point characterization of optimality |
| **Bellman backup** | One application of the Bellman operator: update V using current estimates |
| **Value iteration** | Iterate Bellman backups until convergence; rate gamma^k |
| **Policy iteration** | Alternate exact policy evaluation (linear solve) and greedy improvement |
| **Policy evaluation** | Solve (I - gamma P^pi) V = R^pi for the value of a fixed policy |
| **(s, S) policy** | Inventory: order up to S when stock falls below s; optimal for many models |
| **Contraction mapping** | The Bellman operator is a gamma-contraction in L-infinity norm, guaranteeing unique fixed point |

## Description

Implement value iteration, policy iteration (with exact linear-system policy evaluation), and apply them to two MDP problems: a stochastic gridworld and an inventory management model. Analyze the optimal inventory policy to discover the (s, S) structure.

### What you'll build

1. **Value iteration engine** — Bellman backup loop with convergence detection on a 4x4 slippery gridworld
2. **Policy iteration engine** — Exact policy evaluation via Eigen linear solve, policy improvement loop
3. **Inventory management MDP** — Model stochastic demand, solve for optimal ordering policy, identify (s, S) structure

## Instructions

### Phase 1: Value Iteration on Gridworld (~30 min)

**File:** `src/value_iteration.cpp`

This phase teaches the most fundamental MDP algorithm. The gridworld is a standard testbed: states are grid cells, actions are compass directions, and the "slippery floor" introduces stochasticity (80% chance of moving in the intended direction, 10% chance of each perpendicular direction).

**What you implement:**
- `value_iteration()` — The Bellman backup loop. For each state, compute the Q-value for every action (immediate reward + discounted expected future value), take the max, and update the value function. Repeat until the maximum change across all states falls below a tolerance.

**Why it matters:** Value iteration is the workhorse algorithm for small-to-medium MDPs. The Bellman backup is the atomic operation of all DP-based methods. Understanding convergence (geometric rate gamma^k) and how the value function propagates information from goal states backward through the state space is essential for both OR (inventory, scheduling) and RL (Q-learning is value iteration with samples).

### Phase 2: Policy Iteration with Exact Evaluation (~30 min)

**File:** `src/policy_iteration.cpp`

This phase teaches the alternative to value iteration that exploits linear algebra. Instead of iterating Bellman backups hundreds of times, policy iteration evaluates each candidate policy *exactly* by solving a linear system, then improves the policy greedily. It typically converges in 3-5 iterations.

**What you implement:**
- `policy_evaluation()` — Build the transition matrix P^pi and reward vector R^pi for the current policy, then solve (I - gamma * P^pi) V = R^pi using Eigen.
- `policy_iteration()` — The outer loop: evaluate, improve, check convergence.

**Why it matters:** Policy iteration demonstrates the power of exact computation vs iteration. The linear system (I - gamma P^pi) V = R^pi is one of the most important equations in sequential decision-making. Understanding it connects MDPs to linear algebra (the matrix is always invertible for gamma < 1 because it's strictly diagonally dominant after a suitable scaling argument). Policy iteration also shows why the number of *policies* matters (finite convergence) vs the number of *value functions* (infinite).

### Phase 3: Inventory Management MDP (~30 min)

**File:** `src/inventory.cpp`

This phase applies MDP tools to a real OR problem: how much inventory to order when demand is uncertain. The MDP is fully constructed for you (state transitions encode demand probabilities, costs, and stockout penalties). You solve it with value iteration (provided) and then analyze the resulting policy.

**What you implement:**
- `analyze_optimal_policy()` — Extract managerial insights: identify the reorder point, the order-up-to level, verify the (s, S) policy structure, and compute expected long-run performance. This is the "so what" of the MDP solution — translating math into actionable decisions.

**Why it matters:** The gap between "I can solve an MDP" and "I can extract business value from the solution" is where most practitioners struggle. The (s, S) policy is one of the most celebrated results in inventory theory — it says the optimal policy has a simple, interpretable structure even though the underlying optimization problem is complex. Discovering this structure from the MDP solution (rather than assuming it) is a powerful demonstration of what DP can do.

## Motivation

MDPs are the mathematical framework for **sequential decision-making under uncertainty** — the core problem in both operations research and reinforcement learning. Understanding Bellman equations, value iteration, and policy iteration is foundational for:

- **Operations Research:** Inventory management, maintenance scheduling, revenue management, and resource allocation all use MDP models. The Bellman equation is as fundamental to OR as the simplex method is to linear programming.
- **Reinforcement Learning:** RL algorithms (Q-learning, SARSA, actor-critic) are all approximate methods for solving MDPs when the model (P, R) is unknown. Understanding the exact algorithms (value/policy iteration) is prerequisite to understanding the approximate ones.
- **System design:** Many production systems (warehouse automation, dynamic pricing, predictive maintenance) use MDP-based controllers. Understanding the theory helps design the state space, action space, and reward function correctly.
- **Interviews:** MDP-related questions appear in quant trading (optimal execution, market-making), tech (recommendation systems, ad bidding), and logistics (fleet routing, inventory) interviews.

The connection between OR and RL is direct: OR assumes you know the model and solves it exactly; RL assumes you don't know the model and learns it from data. This practice builds the exact-model side, which makes RL click when you encounter it later.

## Commands

All commands are run from the `practice_039_stochastic_dp_mdp/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build 2>&1"` | Configure the project (fetches Eigen via FetchContent on first run) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases 2>&1"` | Build all three phase executables at once |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_value_iteration 2>&1"` | Build Phase 1: Value iteration on gridworld |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_policy_iteration 2>&1"` | Build Phase 2: Policy iteration with exact evaluation |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_inventory 2>&1"` | Build Phase 3: Inventory management MDP |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\phase1_value_iteration.exe` | Run Phase 1: Value iteration on 4x4 slippery gridworld |
| `build\Release\phase2_policy_iteration.exe` | Run Phase 2: Policy iteration, compare with value iteration |
| `build\Release\phase3_inventory.exe` | Run Phase 3: Inventory management — solve and analyze optimal policy |

## References

- [Bellman, R. (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Original formulation of the principle of optimality
- [Puterman, M. (2005). Markov Decision Processes: Discrete Stochastic Dynamic Programming](https://www.wiley.com/en-us/Markov+Decision+Processes-p-9780471727828) — The definitive MDP reference for OR
- [Sutton & Barto (2018). Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) — Ch. 4 covers value/policy iteration with excellent intuition
- [Bertsekas, D. (2012). Dynamic Programming and Optimal Control](https://www.athenasc.com/dpbook.html) — Rigorous treatment of stochastic DP with OR applications
- [Scarf, H. (1960). The Optimality of (s, S) Policies in the Dynamic Inventory Problem](https://www.jstor.org/stable/24900281) — Foundational result proving (s, S) optimality
- [Eigen Documentation](https://eigen.tuxfamily.org/dox/) — Matrix operations and linear system solvers

## State

`not-started`
