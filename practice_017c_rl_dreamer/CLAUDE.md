# Practice 017c: World Model RL -- Latent Imagination (Dreamer)

## Technologies

- **DreamerV2 architecture** (simplified): RSSM world model, latent imagination, actor-critic in latent space
- **PyTorch**: Neural network framework
- **Gymnasium**: RL environment interface (CartPole-v1, low-dimensional observations)
- **NumPy / Matplotlib**: Numerical computation and training visualization

## Stack

Python 3.12+, PyTorch, Gymnasium

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

### Setup

```bash
cd practice_017c_rl_dreamer
uv sync
```

### File structure

```
app/
  rssm.py          -- RSSM world model (TODO: implement core forward passes)
  encoder.py       -- Observation encoder (TODO: implement forward)
  decoder.py       -- Obs + reward decoders (partially done, TODO for key parts)
  actor_critic.py  -- Actor and Critic networks (TODO: implement forward + loss)
  imagination.py   -- Latent imagination rollouts (TODO: implement rollout loop)
  world_model.py   -- WorldModel wrapper: bundles encoder, RSSM, decoders
  replay_buffer.py -- Simple episode replay buffer (fully implemented)
  train.py         -- Training loop orchestrator (boilerplate implemented)
  config.py        -- Hyperparameters (fully implemented)
```

### Guided implementation order

Work through the TODOs in this order -- each builds on the previous:

1. **`encoder.py`** -- Warmup. Small MLP, straightforward.
2. **`rssm.py`** -- The heart of Dreamer. Implement the three sub-models (recurrent, representation/posterior, transition/prior) and the `observe_step` / `imagine_step` methods.
3. **`decoder.py`** -- Complete the observation and reward decoders.
4. **`actor_critic.py`** -- Actor outputs action distribution; critic estimates state value. Implement forward passes and the loss functions (REINFORCE + entropy for actor, MSE for critic on lambda-returns).
5. **`imagination.py`** -- The "dreaming" loop. Unroll H steps using only the RSSM prior + actor, compute lambda-returns.
6. **Run training** -- `uv run python app/train.py`. Watch the reward curve climb.

### Running

```bash
uv run python app/train.py
```

Training should solve CartPole-v1 (reward >= 475 over 100 episodes) within ~200-400 world-model training epochs, depending on hyperparameters.

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

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (torch, gymnasium, numpy, matplotlib) |

### Training

| Command | Description |
|---------|-------------|
| `uv run python app/train.py` | Run the full Dreamer training loop: collect episodes, train world model, train actor-critic via imagination. Saves training curves to `app/training_curves.png` |
