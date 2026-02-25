# Practice 072 — RL-based Adaptive Load Balancing

## Technologies

- **Gymnasium** (0.29+) — Standard RL environment interface (successor to OpenAI Gym)
- **Stable-Baselines3** (2.1+) — Battle-tested RL algorithm implementations (PPO, DQN, A2C, SAC)
- **NumPy** — Numerical arrays for observation/action spaces and simulation math
- **Matplotlib** — Plotting latency distributions, training curves, comparison charts
- **Pandas** — Tabular results aggregation and analysis

## Stack

Python 3.12 (uv). No Docker needed.

## Theoretical Context

### Load Balancing in Distributed Systems

Load balancing is the process of distributing incoming requests across a pool of K backend servers to optimize system performance. The fundamental goal is twofold: **minimize request latency** (each individual request should be served as fast as possible) and **maximize throughput** (the system as a whole should handle as many requests per second as possible). A secondary goal is **fairness** — no single server should be overwhelmed while others sit idle.

In production systems, a load balancer sits between clients and a fleet of servers. Every incoming request triggers a **routing decision**: which of the K servers should handle this request? This decision must be made in real-time, typically in microseconds, and the quality of these decisions directly determines user-perceived latency, server utilization, and overall system reliability.

Load balancing matters because modern distributed systems rarely consist of identical, equally-loaded servers. Real deployments involve **heterogeneous hardware** (some servers have more CPU/RAM), **variable request complexity** (a search query is cheaper than a report generation), **non-uniform degradation** (a server's disk might be failing, increasing its I/O latency), and **dynamic traffic patterns** (Black Friday spikes, diurnal patterns, viral content).

### Traditional Load Balancing Algorithms

**Round-Robin** assigns requests to servers in a fixed cyclic order (server 0, 1, 2, ..., K-1, 0, 1, ...). It is trivially simple — O(1) decision time, no state needed. However, it is completely blind to server load. If server 2 is processing a complex query that takes 10x longer, round-robin keeps sending it new requests anyway.

**Weighted Round-Robin** assigns static weights to servers (e.g., server with 8 cores gets weight 2, server with 4 cores gets weight 1). This handles heterogeneous hardware at provisioning time but cannot adapt to runtime conditions — if the high-weight server starts degrading, the weights remain stale.

**Least-Connections** always routes to the server with the fewest active (in-flight) connections. This is a significant improvement — it is **reactive**, adapting to the current state. If a server is slow, it accumulates connections and receives fewer new ones. However, it treats all connections as equal (a lightweight health-check connection counts the same as a heavy batch job) and it reacts only after load has already accumulated.

**Random** picks a server uniformly at random. Surprisingly effective at scale (by the law of large numbers, load distributes roughly evenly) and requires zero state. But it has high variance for small server pools and completely ignores server state.

**Consistent Hashing** maps requests to servers based on a hash of the request key. This ensures that the same key always reaches the same server (useful for caching) but provides no load-aware balancing.

**Power of Two Choices** picks two random servers and routes to the one with fewer connections. This simple strategy achieves exponentially better load balance than pure random (from O(log n) max load to O(log log n)) and is used in production systems like Envoy and Netflix Zuul.

### Why Traditional Algorithms Fail in Dynamic Environments

Traditional algorithms break down when the environment is non-stationary — that is, when the distribution of request types, server capacities, and traffic patterns changes over time. Consider these scenarios:

1. **Server degradation**: A server's SSD starts failing, increasing its p99 latency from 5ms to 500ms. Round-robin and random are oblivious. Least-connections will eventually route fewer requests, but only after many requests have already experienced high latency.

2. **Traffic spikes**: A sudden 10x traffic increase overwhelms the system. The optimal strategy might be to drop low-priority requests or temporarily route everything to the fastest servers. Traditional algorithms have no notion of priority or adaptive throttling.

3. **Heterogeneous request types**: Some requests are cheap (cache hits), others are expensive (database joins). If a server receives several expensive requests in a row, its queue grows. Least-connections sees the queue but does not know which server will drain fastest.

4. **Correlated failures**: When servers share a network switch that fails, multiple servers become unreachable simultaneously. Traditional algorithms detect this only through timeouts — a reactive strategy that wastes requests.

The core limitation is that traditional algorithms use **fixed rules** that cannot learn patterns or predict future states. They react to symptoms (high queue length) rather than anticipating problems (this server's latency has been trending upward).

### Why Reinforcement Learning is a Good Fit

Load balancing is naturally formulated as a **sequential decision problem**: at each timestep, the agent (load balancer) **observes** the current system state (queue lengths, CPU utilization, recent latencies), **chooses an action** (which server to route to), and **receives a reward** (negative latency — lower latency means higher reward). This is precisely the Markov Decision Process (MDP) framework that RL algorithms are designed to solve.

Key properties that make RL attractive for load balancing:

- **Non-stationary environment**: Traffic patterns change throughout the day. RL agents can continuously adapt their policy, unlike static rules.
- **Delayed feedback**: A routing decision's quality is only known after the request completes. RL handles delayed rewards natively through value function estimation.
- **Complex state interactions**: The optimal server choice depends on the joint state of all servers (not just one metric). RL can learn to weigh multiple features simultaneously.
- **No explicit model needed**: Model-free RL (PPO, DQN) does not require a mathematical model of server behavior — it learns directly from experience.

The trade-off: RL agents need **training data** (thousands of episodes of interaction), may be **unstable** during early training, and their decisions are less interpretable than "route to least-loaded server." In practice, RL-based load balancing is most valuable when the environment is sufficiently complex that hand-tuned heuristics leave significant performance on the table.

### RL Fundamentals Refresher

**Markov Decision Process (MDP)**: A tuple (S, A, P, R, gamma) where S is the state space, A is the action space, P(s'|s,a) is the transition function, R(s,a) is the reward function, and gamma is the discount factor.

**Policy (pi)**: A mapping from states to actions (or action probabilities). The goal of RL is to find the optimal policy pi* that maximizes expected cumulative reward.

**Value Function V(s)**: The expected cumulative reward starting from state s and following policy pi. Used by value-based methods (DQN) to evaluate how good a state is.

**Action-Value Function Q(s,a)**: The expected cumulative reward of taking action a in state s and then following pi. DQN directly approximates Q with a neural network.

**Policy Gradient**: Instead of estimating Q, directly parameterize the policy pi_theta and optimize theta by gradient ascent on expected reward. PPO is a policy gradient method.

### Gymnasium Custom Environments

[Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym) provides a standard interface for RL environments. To create a custom environment, you subclass `gymnasium.Env` and implement:

- **`__init__`**: Define `self.observation_space` (what the agent sees) and `self.action_space` (what it can do) using `gymnasium.spaces` objects (Box, Discrete, MultiDiscrete, etc.).
- **`reset(seed, options)`**: Return the environment to its initial state. Returns `(observation, info)`.
- **`step(action)`**: Apply the action, advance the simulation, return `(observation, reward, terminated, truncated, info)`.
- **`_get_observation()`**: Helper to construct the current observation from internal state.

The `observation_space` and `action_space` declarations are not just documentation — Stable-Baselines3 reads them to configure its neural network input/output dimensions. A `Box(low, high, shape, dtype)` space describes continuous observations; `Discrete(n)` describes a choice among n options.

### State Design for Load Balancing

The observation (what the RL agent sees at each timestep) should capture enough information for the agent to make a good routing decision. For load balancing, useful per-server features include:

| Feature | What it captures |
|---------|-----------------|
| **Queue length** | How many requests are waiting — direct measure of current load |
| **CPU utilization** | Processing capacity being used — correlates with response time |
| **Average recent latency** | How fast the server has been responding — captures degradation |
| **Processing rate** | Server's throughput capacity — varies across heterogeneous hardware |

The observation is typically a 2D array of shape `(num_servers, num_features)` flattened to 1D for the neural network. Each row is one server's state vector.

### Reward Design

The reward signal is the most critical design decision in RL. For load balancing:

- **Primary reward**: Negative latency of the routed request: `reward = -latency`. This directly incentivizes the agent to minimize latency.
- **Penalty for queue overflow**: If a server's queue is full and the agent routes there anyway, apply a large penalty (e.g., -10). This teaches the agent to avoid overloaded servers.
- **Throughput bonus** (optional): Small positive reward for each successfully processed request. Encourages the agent to keep the system moving.

Reward shaping matters: if the penalty for overflow is too large relative to normal latency, the agent becomes overly conservative. If too small, it ignores queue limits. Empirical tuning is necessary.

### PPO and DQN: Algorithm Overview

**DQN (Deep Q-Network)**: A value-based algorithm that approximates Q(s,a) with a neural network. For each state, it estimates the expected return of each action and picks the one with the highest Q-value. Key features: experience replay buffer (stores past transitions for batch training), target network (stabilizes training), epsilon-greedy exploration. **Works only with discrete action spaces** — perfect for load balancing where actions are server indices {0, 1, ..., K-1}.

**PPO (Proximal Policy Optimization)**: A policy gradient algorithm that directly optimizes the policy network. The key innovation is the **clipped surrogate objective**: it limits how much the policy can change in one update, preventing catastrophically large steps. Key features: no replay buffer (uses on-policy data), handles both discrete and continuous actions, generally more stable than vanilla policy gradient. PPO is the default choice in many RL applications due to its robustness and simplicity of tuning.

For load balancing with K servers:
- Action space: `Discrete(K)` — choose which server gets the request
- Both PPO and DQN work. PPO is typically more stable; DQN can be more sample-efficient.
- [Stable-Baselines3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [DQN docs](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)

### Evaluation Metrics

| Metric | Formula / Description | Why it matters |
|--------|----------------------|----------------|
| **Mean latency** | Average response time across all requests | Overall system performance |
| **p99 latency** | 99th percentile response time | Tail latency — what the worst-affected users experience |
| **Throughput** | Requests successfully processed per unit time | System capacity utilization |
| **Dropped request rate** | Fraction of requests that hit a full queue | Availability — should be near zero |
| **Jain's fairness index** | J = (sum x_i)^2 / (n * sum x_i^2) where x_i is server utilization. Range [0, 1]; 1 = perfectly fair | Whether load is distributed evenly across servers |

### Real-World Context

Production load balancers operate at massive scale:

- **Google Maglev**: A distributed software load balancer handling millions of packets per second. Uses consistent hashing with connection tracking. [Eisenbud et al., NSDI 2016]
- **AWS Application Load Balancer (ALB)**: Supports weighted target groups, path-based routing, and health checks. Uses least-outstanding-requests by default.
- **Envoy Proxy**: Open-source L7 proxy used by Istio service mesh. Supports round-robin, least-request, ring-hash, random, and Maglev. [envoyproxy.io](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/overview)

RL-based approaches in research:

- **DeepRM** (Mao et al., 2016): Pioneering work applying deep RL to multi-resource cluster scheduling. Trained a policy gradient agent that learned to pack jobs across multiple resource dimensions. The agent discovered strategies like withholding large jobs to favor many small jobs, reducing average slowdown. [Paper](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf)
- **Decima** (Mao et al., 2019): RL-based scheduler for data-processing DAGs (Spark jobs). Uses graph neural networks to handle variable-size job graphs. Outperforms hand-tuned heuristics by 21-45%.
- **Park** (Mao et al., 2019): Platform for RL in systems — provides Gymnasium-compatible environments for load balancing, congestion control, job scheduling, and more.

### References

- [Gymnasium Documentation: Create a Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Stable-Baselines3: Using Custom Environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [Stable-Baselines3: Examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
- [PPO Paper: Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [DQN Paper: Mnih et al., 2015](https://arxiv.org/abs/1312.5602)
- [DeepRM: Resource Management with Deep RL (Mao et al., 2016)](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf)
- [Jain's Fairness Index (Wikipedia)](https://en.wikipedia.org/wiki/Fairness_measure)
- [Envoy Load Balancing Overview](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/overview)
- [RL-based Adaptive Load Balancing for Dynamic Cloud Environments (arXiv:2409.04896)](https://arxiv.org/html/2409.04896v1)

## Description

Build a custom Gymnasium environment that simulates a fleet of backend servers with variable processing rates, queue depths, and latency characteristics. Implement traditional load balancing baselines (round-robin, random, least-connections), then train an RL agent (PPO/DQN) using Stable-Baselines3 to learn an adaptive routing policy. Compare the RL agent against baselines on mean latency, p99 latency, throughput, and fairness. Finally, test all policies under dynamic scenarios — server crashes, traffic spikes, and heterogeneous degradation — to demonstrate where RL's adaptability shines.

### What you'll learn

1. **Custom Gymnasium environments** — implementing the Env interface (reset, step, observation/action spaces) for a non-trivial simulation
2. **Reward engineering** — how reward function design shapes agent behavior
3. **Stable-Baselines3 training pipeline** — wrapping environments, configuring PPO/DQN, monitoring training
4. **Systematic evaluation** — comparing policies with statistical rigor using multiple metrics
5. **RL vs heuristics trade-offs** — when RL adds value and when simpler methods suffice

## Instructions

### Phase 1: Environment Design & Baseline Policies (~30 min)

1. Install dependencies: `uv sync`
2. **Exercise 1 (`src/00_load_balancer_env.py`):** Implement the custom Gymnasium environment that simulates a load balancer routing requests to K backend servers. This is the foundation — you need to define what the agent observes (per-server queue length, CPU utilization, average latency), what it can do (choose a server index), and how the simulation advances (requests arrive, get queued, get processed, latency is computed). The environment must pass `gymnasium.utils.env_checker.check_env`. Understanding how to translate a real-world system into the MDP formalism (state, action, reward, transition) is the core RL skill this exercise teaches.
3. **Exercise 2 (`src/01_baseline_policies.py`):** Implement three classic load balancing policies and an evaluation harness. Baselines serve two purposes: they give you a performance floor to compare RL against, and they force you to think about what information each policy uses (round-robin uses nothing, least-connections uses queue state). The evaluation function you build here will be reused for all subsequent comparisons.

### Phase 2: RL Training (~30 min)

1. **Exercise 3 (`src/02_train_agent.py`):** Train an RL agent using Stable-Baselines3. You'll learn to wrap a custom environment for SB3, choose between PPO and DQN, configure hyperparameters (learning rate, batch size, network architecture), and monitor training progress. The key insight: the agent starts worse than random (exploring randomly with a bad policy) but should converge to outperform baselines after sufficient training.

### Phase 3: Evaluation & Comparison (~20 min)

1. **Exercise 4 (`src/03_evaluation.py`):** Systematically compare the trained RL agent against all baselines. You'll collect per-step metrics across multiple episodes, compute aggregate statistics (mean, p99, throughput), and create comparison visualizations. This exercise teaches the importance of rigorous evaluation — a single episode is not enough due to stochastic simulation, and mean latency alone hides tail behavior.

### Phase 4: Dynamic Scenarios (~30 min)

1. **Exercise 5 (`src/04_dynamic_scenarios.py`):** Extend the environment with dynamic events (server crashes, traffic spikes, capacity degradation) and test how each policy adapts. This is where RL should shine — static policies cannot change their behavior when conditions change, but a well-trained RL agent has learned to respond to state changes. The exercise also teaches environment extension patterns (subclassing, event scheduling).

## Motivation

- **Bridges ML and systems**: RL for systems (scheduling, load balancing, congestion control) is an active research area with direct industrial applications. Understanding how to formulate a systems problem as an MDP is a transferable skill.
- **Custom Gymnasium practice**: Building a non-trivial custom environment is a common task in applied RL that goes beyond toy examples.
- **Complements distributed systems practices**: Practices 014 (SAGA), 049 (Raft), and 052 (Resilience) cover rule-based distributed patterns. This practice introduces the ML-based alternative.
- **Industry relevance**: Companies like Google, Meta, and Netflix actively research RL-based infrastructure optimization. DeepRM, Decima, and CacheLib use RL for resource management.

## Commands

All commands run from `practice_072_rl_load_balancing/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (gymnasium, stable-baselines3, numpy, matplotlib, pandas) |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/00_load_balancer_env.py` | Exercise 1: Test the custom Gymnasium environment (runs env_checker + random steps) |
| `uv run python src/01_baseline_policies.py` | Exercise 2: Run baseline load balancing policies (round-robin, random, least-connections) |
| `uv run python src/02_train_agent.py` | Exercise 3: Train RL agent with stable-baselines3 (PPO or DQN) |
| `uv run python src/03_evaluation.py` | Exercise 4: Compare RL agent vs baselines (mean/p99 latency, throughput, fairness) |
| `uv run python src/04_dynamic_scenarios.py` | Exercise 5: Test all policies under dynamic conditions (crashes, spikes, degradation) |

### Utility

| Command | Description |
|---------|-------------|
| `python clean.py` | Remove generated files (models/, plots/, logs/, data/, caches) |

## State

`not-started`
