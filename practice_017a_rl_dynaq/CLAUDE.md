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

Three files, seven small focused TODOs:

1. **`app/q_learning.py`** — the model-free baseline. Three TODOs: `select_action`, `update`, `train_episode`.
2. **`app/environment_model.py`** — the tabular world model. Two TODOs: `update` (store) and `sample` (replay).
3. **`app/dyna_q.py`** — Dyna-Q on top of Q-learning. Two TODOs: `plan` (n simulated Q-updates) and `train_episode` (real step + model feed + planning).

The training harness (`train(...)`), comparison driver (`compare.py`), and plotting (`visualize.py`) are fully scaffolded — you only write the RL mechanics.

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

### Phase 1: Setup & Environment (~10 min)

1. Install dependencies: `uv sync`
2. Explore `CliffWalking-v1`: 4x12 grid, 48 states, 4 actions (up/right/down/left).
3. Sanity check: `uv run python -c "import gymnasium as gym; env = gym.make('CliffWalking-v1'); print(env.observation_space, env.action_space)"`.
4. Key question: Why is CliffWalking ideal for comparing Q-learning vs Dyna-Q? (Hint: think about what happens with limited episodes.)

### Phase 2: Tabular Q-Learning Agent (~25 min)

Open `app/q_learning.py`.

1. **TODO 1 — `select_action()`**: Implement epsilon-greedy action selection. With probability `epsilon` pick a uniform random action (exploration), otherwise `argmax_a Q(state, a)` (exploitation). Same explore/exploit tradeoff as multi-armed bandits — sometimes try a random branch, usually go greedy.
2. **TODO 2 — `update()`**: Implement the one-step Q-learning TD update. Target is `r + gamma * max_a' Q(s', a')`, except on terminal transitions where it collapses to just `r` (no future rewards from the terminal state). This is the core of bootstrapping: the TD error tells you whether your current estimate was too high or too low.
3. **TODO 3 — `train_episode()`**: Roll out a full episode — reset, loop (select → step → update → advance) until `terminated or truncated`, return total undiscounted reward. This is the RL loop that ties action selection and value updates together.
4. Run: `uv run python app/q_learning.py`. Episode rewards should converge toward **-13** (optimal path on CliffWalking).
5. Key question: What does `max_a' Q(s', a')` mean when `s'` is terminal?

### Phase 3: Tabular Environment Model (~15 min)

Open `app/environment_model.py`.

1. **TODO 1 — `EnvironmentModel.update()`**: Record `(s, a) -> (r, s', done)` as a `Transition` in `self.transitions`, and keep the sampling bookkeeping in sync (`observed_states`, `observed_actions[state]`). Think of it as an adjacency list — for every visited node, remember which edges we've explored. This is the first half of model-based RL: memorize the environment so you can replay it later.
2. **TODO 2 — `EnvironmentModel.sample()`**: Return a uniformly-random previously-observed `(s, a, r, s', done)`. Pick a random state from `observed_states`, a random action from `observed_actions[state]`, look up the stored `Transition`. This is how the agent "dreams" — free Q-updates from memory with no real environment interaction.
3. Key question: Why does the model only store the *last* observed transition for each `(s, a)` pair? What assumption does this make about the environment?

### Phase 4: Dyna-Q Agent (~20 min)

Open `app/dyna_q.py`. `DynaQAgent` inherits `select_action` and `update` from `QLearningAgent`, so you only add the two new pieces:

1. **TODO 1 — `DynaQAgent.plan()`**: The planning loop. Repeat `self.n_planning_steps` times: sample from the model, run the inherited Q-`update` on that imagined transition. No-op when the model is empty. This is the computational heart of Dyna — extra cheap Q-updates that piggyback on memorized experience.
2. **TODO 2 — `DynaQAgent.train_episode()`**: Same shape as the Q-learning version, but after every real step, also call `self.model.update(...)` to feed the model, then `self.plan()` to do `n` imagined Q-updates. With `n_planning_steps=0` this must reduce to vanilla Q-learning — that's a good self-check.
3. Run: `uv run python app/dyna_q.py`. Should converge noticeably faster than Q-learning and print a non-zero `model size` at the end.
4. Key question: If you set `n=0`, how does Dyna-Q differ from Q-learning?

### Phase 5: Compare & Visualize (~15 min)

1. Run `uv run python app/compare.py` (or `uv run python main.py`) — trains Q-Learning and Dyna-Q(`n=10`) head-to-head, then sweeps `n ∈ {0, 5, 10, 50}` and saves plots to `plots/`.
2. Observe: Dyna-Q with `n=10` should converge in ~20-30 episodes vs ~100+ for Q-learning.
3. Inspect `plots/learning_curves.png` and `plots/planning_steps.png` — look for diminishing returns as `n` grows.
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
