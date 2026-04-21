# Practice 017c: World Model RL -- Latent Imagination (Dreamer)

## Technologies

- **DreamerV2 architecture** (simplified): RSSM world model, latent imagination, actor-critic in latent space
- **PyTorch**: Neural network framework
- **Gymnasium**: RL environment interface (CartPole-v1, low-dimensional observations)
- **NumPy / Matplotlib**: Numerical computation and training visualization

## Stack

Python 3.12+, PyTorch, Gymnasium

## Theoretical Context

### What is Dreamer and what problem does it solve?

**Dreamer** (and its successors DreamerV2/V3 by Danijar Hafner et al.) is a **latent world model** algorithm that achieves state-of-the-art sample efficiency in model-based RL by learning to **imagine in a compressed latent space** rather than raw observations. The key insight: instead of predicting high-dimensional pixel observations (which compounds errors and is computationally expensive), Dreamer learns a **compact latent representation** of the world's dynamics, then trains an actor-critic policy entirely via **imagination** in that latent space.

The problem solved: **scaling model-based RL to high-dimensional observations and long horizons**. Previous world models (like 017b) struggle with compounding errors in raw state space and can't plan beyond a few steps. Dreamer's latent dynamics model is more robust, enabling hundreds of imagined steps for policy optimization — all without real environment interaction.

### How Dreamer works internally: RSSM, imagination, and actor-critic training

**Architecture overview:**

1. **RSSM (Recurrent State-Space Model)** — the world model:
   - **Observation encoder**: Maps raw observations (e.g., pixels, CartPole state) to an embedding vector.
   - **Recurrent model**: A GRU that maintains a deterministic **recurrent state** `h_t`, capturing sequential dependencies. Input: `(h_{t-1}, z_{t-1}, a_{t-1})`. Output: `h_t`.
   - **Representation model** (posterior): Given `h_t` and the current observation embedding, infers a **stochastic latent state** `z_t` (what *actually* happened). This is `p(z_t | h_t, o_t)`.
   - **Transition model** (prior): Given only `h_t`, predicts what `z_t` *should* be (what the model expects to happen). This is `p(z_t | h_t)`. Trained to match the posterior via KL divergence.
   - **Decoders**: Reconstruct observations and predict rewards from the full latent state `(h_t, z_t)`.

   The RSSM is trained via **reconstruction loss** (observation decoder) + **reward prediction loss** + **KL divergence** between prior and posterior (ensures the prior alone can generate plausible latent trajectories during imagination).

2. **Imagination rollouts** (the "dreaming" phase):
   - Start from a real latent state `(h_0, z_0)` obtained by encoding a real observation.
   - For `H` steps (e.g., 15):
     - Query the actor for an action: `a_t = π(h_t, z_t)`
     - Use the recurrent model to get the next deterministic state: `h_{t+1} = GRU(h_t, z_t, a_t)`
     - Use the transition *prior* (no observations!) to sample the next stochastic state: `z_{t+1} ~ p(z | h_{t+1})`
     - Predict the reward: `r_t = RewardDecoder(h_{t+1}, z_{t+1})`
   - The result: a simulated trajectory of `(h, z, a, r)` tuples generated entirely in latent space, no environment interaction.

3. **Actor-Critic training** (policy optimization in imagination):
   - The **actor** (policy) is trained via REINFORCE on imagined trajectories to maximize **lambda-returns** (a mixture of n-step returns that balances bias-variance).
   - The **critic** (value function) predicts the expected return from a latent state, trained via regression on the lambda-returns computed from imagination.
   - Crucially, **both networks operate only on latent states** `(h, z)` — they never see raw observations. This decouples policy learning from environment interaction.

**Why this works:** The RSSM's recurrent + stochastic structure captures both predictable dynamics (h) and stochastic variability (z). The KL penalty forces the prior to be a good generative model, so imagination rollouts are stable. Training the policy in latent space avoids compounding errors in high-dimensional observation space.

### Key concepts

| Concept | Description |
|---------|-------------|
| **Latent world model** | A dynamics model that operates in a learned low-dimensional latent space, not raw observations |
| **RSSM** | Recurrent State-Space Model: combines a deterministic recurrent path (GRU) with a stochastic latent variable for each timestep |
| **Deterministic state (h)** | The GRU hidden state, capturing sequential information and predictable dynamics |
| **Stochastic state (z)** | A latent variable capturing unpredictable variability (noise, partial observability) at each timestep |
| **Posterior (representation model)** | `p(z_t | h_t, o_t)` — infers what happened given the observation |
| **Prior (transition model)** | `p(z_t | h_t)` — predicts what should happen without seeing the observation; used for imagination |
| **KL divergence regularization** | Trains the prior to match the posterior, ensuring the model can generate plausible latent trajectories during imagination |
| **Imagination** | Unrolling the world model forward in latent space for H steps without environment interaction, using only the prior |
| **Lambda-returns** | A weighted mixture of n-step returns that balances bias (short horizon) and variance (long horizon) for policy optimization |
| **Latent actor-critic** | A policy and value function that operate on latent states `(h, z)`, never on raw observations |

### Ecosystem context

**Dreamer's place in model-based RL:**

| Method | State Representation | Planning Mechanism | Sample Efficiency | Domain |
|--------|----------------------|---------------------|-------------------|--------|
| **Dyna-Q (017a)** | Tabular | Random sampling from model | Medium | Discrete, deterministic |
| **Neural World Model (017b)** | Raw states | Short forward rollouts | Medium | Continuous, low-dim |
| **Dreamer (this practice)** | Learned latent space | Long imagination rollouts + actor-critic | **Very high** | Continuous, pixels, partial obs |
| **MuZero** | Learned latent space | MCTS tree search | Very high | Discrete actions, deterministic (e.g., Go, Atari) |
| **Model-free (SAC, PPO)** | Raw states | None | Low | General (baseline) |

**Why Dreamer is state-of-the-art:**
- **Sample efficiency**: Solves Atari from pixels in ~400K environment steps (vs. 10M+ for model-free PPO/DQN).
- **Generality**: Works across continuous control (DMControl), discrete (Atari), and even procedural generation (Minecraft, Crafter).
- **Scalability**: DreamerV3 uses a single set of hyperparameters across all domains — no per-task tuning.

**Trade-offs:**
- **Complexity**: RSSM + actor-critic + imagination loop is harder to implement and debug than model-free methods.
- **Computational cost**: Training the world model + policy is more expensive than model-free per sample (but total wall-clock time is often lower due to fewer environment steps).
- **When not to use**: If environment interactions are cheap (fast simulators), model-free methods like PPO may be simpler and sufficient.

## Description

Build a **simplified Dreamer agent** that learns to solve CartPole-v1 by *imagining* future trajectories in a learned latent space, then training an actor-critic entirely on those imagined rollouts.

### Core concepts implemented

1. **RSSM (Recurrent State-Space Model)** -- The world model. Combines a deterministic recurrent path (GRU) with a stochastic latent variable to capture both predictable dynamics and environmental uncertainty.
   - **Recurrent model**: GRU that takes previous action + previous stochastic state, produces next deterministic state.
   - **Representation model** (posterior): Given deterministic state + current observation, infers the stochastic latent (what *actually* happened).
   - **Transition model** (prior): Given only the deterministic state, predicts the stochastic latent (what *should* happen without seeing the observation). Trained to match the posterior via KL divergence.

2. **Observation encoder** -- Projects raw observations into an embedding the RSSM can consume. For CartPole (4-dim obs), this is a small MLP (not a CNN -- no pixels here).

3. **Decoders** -- Reconstruct observations and predict rewards from the latent state (deterministic + stochastic concatenated). Provide the reconstruction learning signal for the world model.

4. **Actor-critic in latent space** -- A policy (actor) and value function (critic) that operate *only* on latent states. They never see raw observations during training.

5. **Imagination rollouts** -- Starting from real latent states, unroll the actor + transition model for H steps into the future *without* environment interaction. Compute lambda-returns on these imagined trajectories to train the actor and critic.

### Architecture overview (simplified DreamerV2 for low-dim obs)

```
Real environment loop:
  obs -> Encoder -> embedding
  RSSM posterior(h, embedding) -> z_posterior
  RSSM recurrent(h, prev_action, prev_z) -> h_next
  Decoder(h, z) -> obs_reconstructed, reward_predicted
  Store (obs, action, reward, done) in replay buffer

Imagination loop (no environment):
  Sample initial (h, z) from replay
  For t = 1..H:
    action = Actor(h, z)
    h_next = RSSM.recurrent(h, action, z)
    z_next = RSSM.transition_prior(h_next)  -- no observation needed!
    reward = RewardDecoder(h_next, z_next)
  Compute lambda-returns on imagined rewards + critic values
  Update Actor to maximize lambda-returns
  Update Critic to predict lambda-returns
```

### Simplifications vs. full DreamerV2/V3

| Full Dreamer | This practice |
|---|---|
| CNN encoder for pixel observations | MLP encoder for 4-dim CartPole obs |
| 32 categorical distributions x 32 classes | Single Gaussian stochastic latent |
| Symlog transforms, free bits, unimix | Standard KL divergence |
| Discount predictor (continue model) | Fixed discount factor gamma |
| Large replay buffer with batched sequences | Small buffer, simple sequence sampling |

These simplifications keep the session under 120 minutes while preserving every architectural idea.

## Instructions

### Phase 1: Setup (~2 min)

1. `cd practice_017c_rl_dreamer`
2. `uv sync`

### File structure

```
app/
  encoder.py       -- Observation encoder (1 TODO)
  decoder.py       -- Obs + reward decoders (1 TODO, reward decoder provided)
  rssm.py          -- RSSM world model (5 TODOs: the core of Dreamer)
  actor_critic.py  -- Actor + Critic (3 TODOs: forward + two losses)
  imagination.py   -- Latent imagination rollouts (2 TODOs)
  world_model.py   -- WorldModel wrapper: bundles encoder, RSSM, decoders (scaffolded)
  replay_buffer.py -- Episode replay buffer (scaffolded)
  train.py         -- Training loop orchestrator (scaffolded)
  config.py        -- Hyperparameters (scaffolded)
```

### Phase 2: Encoder & Decoder (~10 min)

1. Open `app/encoder.py`.
2. **TODO — `ObservationEncoder._build_network()`**: Return a 2-layer Linear + ELU MLP mapping `obs_dim -> embedding_dim`. The encoder feeds the RSSM posterior; ELU avoids dead neurons along this critical gradient path.
3. Open `app/decoder.py`.
4. **TODO — `ObservationDecoder._build_network()`**: Use the provided `_build_mlp` helper to map `latent_dim -> obs_dim`. The reward decoder is provided as reference; the reconstruction loss on obs is what forces the RSSM to learn an informative latent.

### Phase 3: RSSM — the world model core (~35 min)

Open `app/rssm.py`. Construction of all sub-networks is scaffolded so you can focus on the dynamics.

1. **TODO 1 — `recurrent_step()`**: The GRU step. `fc([z_{t-1}, a_{t-1}]) -> ELU -> GRUCell(., prev_h)`. This is the deterministic backbone — it carries sequential information across timesteps.
2. **TODO 2 — `posterior()`**: `q(z_t | h_t, embedding_t)`. Mean/log_std heads over `Linear(h, embedding)`, clamp `log_std` to `[-5, 2]`, build a `Normal`, and sample with `rsample()` (reparameterization trick — essential for backprop through the world model).
3. **TODO 3 — `prior()`**: `p(z_t | h_t)` — mirror of the posterior but input is only `h`. This is what makes imagination possible: the prior alone must predict plausible next latents without ever seeing observations.
4. **TODO 4 — `observe_step()`**: Compose `recurrent_step` + `posterior` + `prior`. The step used during *training* with real observations.
5. **TODO 5 — `imagine_step()`**: Compose `recurrent_step` + `prior` only. The step used during *dreaming* — no observations in the loop.

Key question: why do we keep `z` from the posterior during training but sample from the prior during imagination? (Hint: one has access to the ground-truth observation, the other does not.)

### Phase 4: Actor-Critic in latent space (~20 min)

Open `app/actor_critic.py`. The 2-layer MLPs are pre-built; you focus on the distribution wrapping and the two losses.

1. **TODO 1 — `Actor.forward()`**: Compute logits from `latent`, return `Categorical(logits=logits)`. Returning a distribution (not an action) lets callers use `.sample()` or `.log_prob()` / `.entropy()` as needed.
2. **TODO 2 — `Actor.loss()`**: REINFORCE + entropy bonus. `-(log_prob * advantage.detach()).mean() - entropy_weight * entropy.mean()`. The `.detach()` on advantages is critical — the actor must not backprop into the critic's return computation.
3. **TODO 3 — `Critic.loss()`**: `F.mse_loss(critic(latent_states), lambda_returns.detach())`. Targets are fixed regression targets.

### Phase 5: Imagination & Lambda-Returns (~20 min)

Open `app/imagination.py`.

1. **TODO 1 — `compute_lambda_returns()`**: Backward DP computing blended TD / MC returns. Base case `R_H = V_H`; then `R_t = r_t + gamma * ((1 - lambda) * V_{t+1} + lambda * R_{t+1})`. Think of it as a DP pass from right to left.
2. **TODO 2 — `imagine_rollout()`**: The "dreaming" loop. H steps of `actor -> reward_decoder -> rssm.imagine_step`, then bootstrap with the critic at the final state and call `compute_lambda_returns`. No environment interaction.

### Phase 6: Run training (~30 min)

1. `uv run python -m app.train`
2. The orchestrator seed-collects random episodes, then loops: collect → train world model → train actor-critic in imagination → evaluate.
3. Training should reach `reward >= 475` on CartPole-v1 within ~200–400 epochs.
4. The reward curve and world-model loss are saved to `app/training_curves.png`.

Key question: if the critic's loss goes down but evaluation reward stays flat, which component is likely the bottleneck?

## Motivation

- **World models are the frontier of sample-efficient RL.** Dreamer (V1/V2/V3) by Danijar Hafner et al. achieved state-of-the-art across Atari, continuous control, and even Minecraft -- all from pixels, all with orders of magnitude fewer environment interactions than model-free methods.
- **Latent imagination is the key insight.** Instead of learning from expensive real experience, the agent "dreams" thousands of trajectories in a learned latent space. This decouples policy optimization from environment interaction.
- **RSSM is reusable beyond RL.** The recurrent state-space model pattern (deterministic backbone + stochastic latent) appears in video prediction, planning, and any sequential decision-making under uncertainty.
- **Complements 017a (Dyna-Q) and 017b (Neural World Model)** by showing how to scale world-model RL to continuous latent spaces and actor-critic optimization.

### Key references

- [Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (DreamerV1)](https://arxiv.org/abs/1912.01603)
- [Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2)](https://arxiv.org/abs/2010.02193)
- [Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3)](https://arxiv.org/abs/2301.04104)
- [Gymnasium CartPole-v1 documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Commands

All commands run from `practice_017c_rl_dreamer/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (torch, gymnasium, numpy, matplotlib) |

### Training

| Command | Description |
|---------|-------------|
| `uv run python -m app.train` | Run the full Dreamer training loop: seed-collect, train world model, train actor-critic via imagination, evaluate. Saves curves to `app/training_curves.png`. |
