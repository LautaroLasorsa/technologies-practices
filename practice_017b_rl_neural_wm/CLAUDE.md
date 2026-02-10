# Practice 017b: World Model RL -- Neural World Model

## Technologies

- **PyTorch** -- Neural network framework for building and training the learned dynamics model
- **Gymnasium** -- RL environment toolkit (CartPole-v1 as the target environment)
- **NumPy** -- Numerical operations for replay buffers and data manipulation
- **Matplotlib** -- Visualization of training curves, prediction accuracy, and agent performance

## Stack

- Python 3.12+ (uv)

## Description

Build a **neural world model** that learns environment dynamics from experience, then use it to accelerate policy learning via simulated rollouts. This upgrades the tabular Dyna-Q approach (017a) to continuous state spaces by replacing the lookup table with an MLP that predicts `(next_state, reward, done)` given `(state, action)`.

### What you'll learn

1. **Learned dynamics models** -- Training an MLP to approximate `f(s, a) -> (s', r, done)` from collected transitions
2. **Experience replay** -- Storing real transitions in a buffer and sampling mini-batches for supervised learning
3. **Model accuracy evaluation** -- Measuring prediction error (MSE on states, BCE on done flags) and understanding compounding errors
4. **Model-based planning** -- Using the learned model to generate synthetic rollouts and improve a DQN policy faster than model-free alone
5. **Dyna-style integration** -- Combining real environment steps with simulated model steps in a single training loop
6. **Compounding error problem** -- Why multi-step rollouts with imperfect models degrade and how to mitigate it

### Architecture overview

```
Real Environment (CartPole-v1)
        |
        v
  Data Collection (epsilon-greedy / random)
        |
        v
  Replay Buffer  ------>  Train World Model (MLP)
        |                         |
        v                         v
  Train DQN (real data)    Generate Simulated Rollouts
        |                         |
        v                         v
  Q-Network  <----------  Train DQN (simulated data)
        |
        v
  Select Actions (epsilon-greedy on Q-values)
```

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Review the project structure and run `uv sync`
2. Understand the difference between model-free RL (learn Q directly from environment) and model-based RL (learn a model of the environment, then plan with it)
3. Key question: In Dyna-Q with a tabular model, the model just memorizes transitions. Why can't that work for CartPole's continuous state space?

### Phase 2: Neural World Model (~25 min)

1. Open `app/world_model.py` and study the `WorldModel` class skeleton
2. **User implements:** The MLP architecture in `__init__` -- an encoder that processes (state, one-hot action) through hidden layers, plus separate heads for next-state prediction, reward prediction, and done prediction
3. **User implements:** The `forward` method -- encode input, pass through each head, return predictions
4. **User implements:** The `compute_loss` method -- MSE for state/reward, BCE for done flag, weighted combination
5. Run `uv run python -m app.world_model` to verify the model compiles and shapes are correct
6. Key question: Why use a separate head for `done` with sigmoid+BCE instead of treating it as a regression target?

### Phase 3: Training the World Model (~20 min)

1. Open `app/train_model.py` and study the training scaffold
2. **User implements:** The training loop -- sample batches from the replay buffer, forward pass, compute loss, backprop, optimizer step
3. **User implements:** The validation step -- evaluate prediction MSE on held-out transitions
4. Run data collection first: `uv run python -m app.data_collection` (fully implemented, generates transitions)
5. Train the model: `uv run python -m app.train_model`
6. Inspect the loss curve plot. Key question: If your state prediction MSE is 0.01, does that mean 100-step rollouts will be accurate? Why or why not?

### Phase 4: Planning with the Learned Model (~15 min)

1. Open `app/plan_with_model.py` and study the rollout scaffold
2. **User implements:** `imagine_rollout` -- starting from a real state, use the world model to predict forward N steps, collecting (s, a, r, s', done) tuples
3. **User implements:** `evaluate_model_accuracy` -- compare predicted trajectories to real trajectories and compute per-step error
4. Run `uv run python -m app.plan_with_model` to visualize real vs. predicted trajectories
5. Key question: At what rollout length do predictions start diverging significantly? Why?

### Phase 5: Model-Based Agent (~25 min)

1. Open `app/agent.py` and study the Dyna-style agent scaffold
2. **User implements:** The DQN's `select_action` method (epsilon-greedy on Q-values)
3. **User implements:** The `train_on_batch` method -- standard DQN update (Q-learning with target network)
4. **User implements:** The `_train_on_simulated_data` function -- sample real states from the buffer, predict with the world model, train DQN on simulated transitions
5. **User implements:** The Dyna loop integrating real + simulated steps in `train_dyna_agent`
6. Run `uv run python -m app.agent` and compare learning curves: model-free DQN vs. Dyna-style DQN
7. Key question: How does the ratio of simulated-to-real updates (K) affect sample efficiency vs. stability?

### Phase 6: Analysis & Discussion (~10 min)

1. Experiment with different K values (0, 1, 5, 10) -- plot learning curves
2. Try longer imagined rollouts (1-step vs. 5-step) -- observe compounding error
3. Discussion: What are the failure modes when the world model is inaccurate? How do methods like MBPO (Model-Based Policy Optimization) address this?

## Motivation

- **Sample efficiency matters in production**: Real environment interactions are often expensive (robotics, recommendation systems, simulations). Model-based RL can reduce required samples by 10-100x.
- **Foundation for advanced methods**: Neural world models are the backbone of Dreamer, MuZero, PlaNet, and MBPO -- all state-of-the-art model-based RL algorithms.
- **Practical ML engineering**: Training a dynamics model is a supervised learning problem inside an RL loop. This bridges the gap between standard ML (prediction) and RL (decision-making).
- **Complements 017a (tabular Dyna-Q)**: Moving from tables to neural nets demonstrates how the same algorithmic idea scales to continuous/high-dimensional problems.

## References

- [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/)
- [MBPO: When to Trust Your Model (Janner et al., 2019)](https://arxiv.org/abs/1906.08253)
- [Sutton & Barto, Chapter 8: Planning and Learning](http://incompleteideas.net/book/the-book.html)
- [Model-Based RL -- DI-engine Documentation](https://opendilab.github.io/DI-engine/02_algo/model_based_rl.html)
- [PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [GeeksforGeeks: Model-Based RL](https://www.geeksforgeeks.org/artificial-intelligence/model-based-reinforcement-learning-mbrl-in-ai/)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (torch, gymnasium, numpy, matplotlib) |

### Phase 2: Verify World Model Architecture

| Command | Description |
|---------|-------------|
| `uv run python -m app.world_model` | Run shape tests to verify the WorldModel produces correct output dimensions |

### Phase 3: Data Collection and Model Training

| Command | Description |
|---------|-------------|
| `uv run python -m app.data_collection` | Collect random transitions from CartPole-v1 and save to `data/transitions.npz` |
| `uv run python -m app.train_model` | Train the neural world model on collected data, save checkpoint to `checkpoints/world_model.pt` and loss curve to `plots/training_curve.png` |

### Phase 4: Planning and Model Evaluation

| Command | Description |
|---------|-------------|
| `uv run python -m app.plan_with_model` | Evaluate model predictions vs real trajectories, generate `plots/trajectory_comparison.png` and `plots/compounding_error.png` |

### Phase 5: Model-Based Agent (Dyna-Style DQN)

| Command | Description |
|---------|-------------|
| `uv run python -m app.agent` | Train model-free DQN and Dyna-style DQN, compare learning curves, save plot to `plots/dqn_vs_dyna.png` |

## State

`not-started`
