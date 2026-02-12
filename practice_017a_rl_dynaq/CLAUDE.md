# Practice 017a: World Model RL -- Dyna-Q Basics

## Technologies

- **Gymnasium** -- Standard RL environment interface (FrozenLake, CliffWalking)
- **Dyna-Q** -- Sutton & Barto's tabular model-based RL algorithm (Chapter 8)
- **NumPy** -- Q-table and model storage
- **Matplotlib** -- Learning curve visualization

## Stack

- Python 3.12+ (uv)

## Theoretical Context

### What is Dyna-Q and what problem does it solve?

**Dyna-Q** is a model-based reinforcement learning algorithm that addresses the **sample efficiency problem** in RL. Pure model-free methods like Q-learning learn exclusively from real environment interactions, which can be expensive or slow (imagine training a robot — each interaction takes time and risks wear). Dyna-Q learns a **model of the environment** (how states transition given actions) and uses that model to generate **simulated experience** for additional learning, extracting more value from each real interaction.

The core insight: if you can predict what would happen after taking action `a` in state `s`, you can "imagine" trajectories without actually executing them in the real environment, then update your policy based on those imagined outcomes. This is the foundation of **planning via mental simulation**.

### How Dyna-Q works internally

Dyna-Q combines three components in each timestep:

1. **Direct RL** (acting): The agent interacts with the real environment, takes an action, observes a reward and next state, and updates its Q-values using the standard Q-learning rule.

2. **Model learning** (remembering): The observed transition `(s, a, r, s', done)` is stored in an **environment model** — in tabular Dyna-Q, this is just a lookup table mapping `(state, action) → (reward, next_state, done)`.

3. **Planning** (dreaming): The agent samples `n` random previously-observed `(s, a)` pairs from the model, retrieves the predicted `(r, s', done)`, and performs Q-learning updates on those **simulated transitions** as if they were real. This is the "Dyna" part — dynamic programming interleaved with learning.

The planning step is computationally cheap (just table lookups and Q-updates) but provides `n` extra learning steps per real step, dramatically accelerating convergence. The tradeoff: the model must be accurate. If the model is wrong, the agent trains on incorrect simulations.

### Key concepts

| Concept | Description |
|---------|-------------|
| **Model-free RL** | Learns value functions or policies directly from experience without modeling environment dynamics (e.g., Q-learning, SARSA) |
| **Model-based RL** | Learns a model of the environment (transition function, reward function), then uses that model for planning or policy improvement |
| **Tabular model** | A lookup table storing observed transitions; no generalization to unseen states |
| **Planning** | Using a learned model to simulate trajectories and update value estimates without real environment interaction |
| **Sample efficiency** | How many real environment interactions are needed to learn a good policy; model-based methods are typically more sample-efficient |
| **Dyna architecture** | A family of algorithms (Dyna-Q, Dyna-Q+, Dyna-2) that integrate learning, planning, and acting in a unified loop |
| **Deterministic model** | Assumes each `(s, a)` maps to exactly one `(r, s', done)`; stochastic models would store distributions |

### Ecosystem context

**Alternatives and evolution:**

- **Pure model-free (Q-learning, DQN, PPO)**: No model, learn only from real data. Simple but sample-inefficient. Best when environment interactions are cheap (e.g., simulated games).
- **Pure model-based (AlphaZero, MuZero)**: Learn a model first, then plan extensively via tree search. High computational cost, but dominant in domains with expensive real interaction (e.g., Go, robotics).
- **Hybrid (Dyna-Q, MBPO, Dreamer)**: Learn both a model and a value function/policy. Dyna-Q is the simplest hybrid — it's the conceptual ancestor of modern neural world-model methods like Dreamer (017c).

**When to use Dyna-Q:**
- Discrete state/action spaces (tabular setting)
- Deterministic or near-deterministic environments
- When you want a simple, interpretable model-based baseline before moving to neural approaches

**Limitations:**
- No generalization: the tabular model only "knows" states it has visited
- Assumes determinism: can't handle stochastic transitions without modification
- Scalability: table size explodes with state/action space size (doesn't work for images, continuous control)

## Description

Implement **Dyna-Q** from scratch to understand the core idea behind model-based reinforcement learning: the agent learns an internal model of the environment and uses it for *planning* (simulated updates) alongside *direct RL* (real experience updates). This is the foundation for modern world-model approaches explored in 017b and 017c.

### What you'll learn

1. **Model-free baseline** -- Tabular Q-learning with epsilon-greedy exploration on a discrete grid world
2. **Environment model** -- A learned lookup table that memorizes observed transitions `(s, a) -> (r, s')`
3. **Planning via replay** -- Sampling from the model to perform additional Q-value updates without real interaction
4. **Effect of planning steps** -- How increasing `n` planning steps per real step accelerates learning
5. **Quantitative comparison** -- Plotting cumulative reward curves: Q-learning vs Dyna-Q with varying `n`

### Key algorithm (Sutton & Barto, Chapter 8)

```
Initialize Q(s, a) and Model(s, a) for all s, a
Loop (for each episode):
    s = initial state
    Loop (for each step of episode):
        a = epsilon-greedy(s, Q)
        Take action a, observe r, s'
        Q(s,a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]   # (d) direct RL
        Model(s, a) = (r, s')                                         # (e) model learning
        Repeat n times:                                                # (f) planning
            s_rand = random previously observed state
            a_rand = random action previously taken in s_rand
            r_hat, s_hat = Model(s_rand, a_rand)
            Q(s_rand, a_rand) += alpha * [r_hat + gamma * max_a' Q(s_hat, a') - Q(s_rand, a_rand)]
        s = s'
```

## Instructions

### Phase 1: Understand the Environment (~10 min)

1. Explore `CliffWalking-v1`: 4x12 grid, 48 states, 4 actions (up/right/down/left)
2. Run `uv run python -c "import gymnasium as gym; env = gym.make('CliffWalking-v1'); print(env.observation_space, env.action_space)"` to verify the setup
3. Key question: Why is CliffWalking ideal for comparing Q-learning vs Dyna-Q? (Hint: think about what happens with limited episodes)

### Phase 2: Tabular Q-Learning Agent (~25 min)

1. Open `app/q_learning.py` -- boilerplate is ready, implement the `TODO(human)` sections
2. **User implements:** `select_action` -- epsilon-greedy action selection
3. **User implements:** `update` -- Q-learning update rule: `Q(s,a) += alpha * [r + gamma * max Q(s',a') - Q(s,a)]`
4. **User implements:** `train_episode` -- run one episode collecting (s, a, r, s') transitions
5. Test: `uv run python app/q_learning.py` -- should print episode rewards converging toward -13 (optimal path length on CliffWalking)
6. Key question: What does `max_a' Q(s', a')` mean when `s'` is terminal?

### Phase 3: Tabular Environment Model (~15 min)

1. Open `app/environment_model.py` -- stores observed transitions
2. **User implements:** `update` -- record `(s, a) -> (r, s', done)` in the model
3. **User implements:** `sample` -- pick a random previously-observed `(s, a)` and return `(r, s', done)`
4. Key question: Why does the model only store the *last* observed transition for each `(s, a)` pair? What assumption does this make about the environment?

### Phase 4: Dyna-Q Agent (~20 min)

1. Open `app/dyna_q.py` -- extends Q-learning with model + planning
2. **User implements:** `train_episode` -- like Q-learning but after each real step, do `n` planning steps using the model
3. Test: `uv run python app/dyna_q.py` -- should converge faster than pure Q-learning
4. Key question: If you set `n=0`, how does Dyna-Q differ from Q-learning?

### Phase 5: Compare & Visualize (~15 min)

1. Run `uv run python app/compare.py` -- trains both agents, plots learning curves
2. Observe: Dyna-Q with `n=10` should converge in ~20-30 episodes vs ~100+ for Q-learning
3. Experiment: try `n=0, 5, 10, 50` and observe diminishing returns
4. Discussion: What is the computational tradeoff of more planning steps?

## Motivation

- **Foundation for world models**: Dyna-Q is the simplest model-based RL algorithm and the conceptual ancestor of Dreamer, MuZero, and other world-model approaches
- **Planning vs learning tradeoff**: Understanding when to use model-based vs model-free methods is a fundamental RL design decision
- **Sample efficiency**: Model-based methods extract more value from each real interaction -- critical when real data is expensive (robotics, healthcare, production systems)
- **Bridges to 017b/017c**: This practice builds intuition for neural world models (017b) and latent imagination (017c)

## References

- [Sutton & Barto, Chapter 8 -- Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf)
- [Gymnasium CliffWalking Documentation](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
- [Gymnasium FrozenLake Q-Learning Tutorial](https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/)
- [Dyna-Q Summary (lcalem)](https://lcalem.github.io/blog/2018/12/01/sutton-chap08)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (gymnasium, numpy, matplotlib) |

### Phase 1: Explore the Environment

| Command | Description |
|---------|-------------|
| `uv run python -c "import gymnasium as gym; env = gym.make('CliffWalking-v1'); print(env.observation_space, env.action_space)"` | Verify CliffWalking-v1 environment setup (48 states, 4 actions) |

### Phase 2: Train Q-Learning Agent

| Command | Description |
|---------|-------------|
| `uv run python app/q_learning.py` | Train vanilla Q-Learning agent and print convergence stats |

### Phase 4: Train Dyna-Q Agent

| Command | Description |
|---------|-------------|
| `uv run python app/dyna_q.py` | Train Dyna-Q agent (default n=10 planning steps) |

### Phase 5: Compare & Visualize

| Command | Description |
|---------|-------------|
| `uv run python app/compare.py` | Run full comparison: Q-Learning vs Dyna-Q with varying planning steps, generate plots |
| `uv run python main.py` | Entry point that runs the comparison (calls `app/compare.py:main()`) |

## State

`not-started`
