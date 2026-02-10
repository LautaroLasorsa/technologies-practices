"""Training loop orchestrator for simplified Dreamer on CartPole-v1.

Three-phase training loop (per epoch):
  1. COLLECT: Run the actor in the real environment, store episodes.
  2. WORLD MODEL: Train encoder + RSSM + decoders on replayed sequences.
  3. BEHAVIOR: Imagine rollouts in latent space, train actor + critic.

This module is fully implemented boilerplate. The learning happens in the
modules you implement (rssm.py, encoder.py, decoder.py, actor_critic.py,
imagination.py).
"""

import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from app.config import DreamerConfig
from app.world_model import WorldModel
from app.actor_critic import Actor, Critic
from app.imagination import imagine_rollout
from app.replay_buffer import ReplayBuffer, Episode


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_episode(
    env: gym.Env,
    world_model: WorldModel,
    actor: Actor,
    config: DreamerConfig,
) -> Episode:
    """Run one episode in the real environment using the actor.

    The actor receives latent states from the world model (just like during
    imagination), maintaining consistency between training and collection.

    For the first few episodes (before the world model is trained), actions
    are essentially random since the actor is untrained.
    """
    obs, _ = env.reset()
    h, z = world_model.rssm.initial_state(batch_size=1)
    prev_action = torch.zeros(1, config.action_dim)

    observations = []
    actions = []
    rewards = []

    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            embedding = world_model.encoder(obs_tensor)
            h, z, _, _ = world_model.rssm.observe_step(h, z, prev_action, embedding)
            latent = torch.cat([h, z], dim=-1)
            dist = actor(latent)
            action = dist.sample().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        prev_action = F.one_hot(
            torch.tensor([action]), num_classes=config.action_dim,
        ).float()

    return Episode(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
    )


def train_world_model(
    world_model: WorldModel,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
) -> dict[str, float]:
    """One gradient step on the world model."""
    batch = replay_buffer.sample_batch()
    losses = world_model.compute_loss(
        observations=batch["observations"],
        actions=batch["actions"],
        rewards=batch["rewards"],
    )

    optimizer.zero_grad()
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=100.0)
    optimizer.step()

    return {k: v.item() for k, v in losses.items()}


def train_behavior(
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: DreamerConfig,
) -> dict[str, float]:
    """Train actor + critic on imagined rollouts.

    1. Sample initial latent states from real experience.
    2. Imagine forward using RSSM prior + actor.
    3. Compute lambda-returns.
    4. Update critic to predict lambda-returns.
    5. Update actor to maximize lambda-returns.
    """
    # Get initial states from a real sequence
    batch = replay_buffer.sample_batch()
    with torch.no_grad():
        outputs = world_model.observe_sequence(
            batch["observations"], batch["actions"],
        )
        # Pick a random timestep's states as imagination starting points
        t = random.randint(0, config.sequence_length - 1)
        init_h = outputs["h_states"][:, t]
        init_z = outputs["z_states"][:, t]

    # Imagine rollout
    rollout = imagine_rollout(
        initial_h=init_h,
        initial_z=init_z,
        rssm=world_model.rssm,
        actor=actor,
        reward_decoder=world_model.reward_decoder,
        critic=critic,
        config=config,
    )

    # Train critic
    critic_loss = critic.loss(rollout["latent_states"], rollout["lambda_returns"])
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
    critic_optimizer.step()

    # Train actor
    advantages = rollout["lambda_returns"] - rollout["values"]
    actor_loss = actor.loss(
        rollout["latent_states"],
        rollout["actions"],
        advantages,
    )
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
    actor_optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
    }


def evaluate(
    env: gym.Env,
    world_model: WorldModel,
    actor: Actor,
    config: DreamerConfig,
    num_episodes: int = 5,
) -> float:
    """Evaluate the actor greedily (no exploration noise)."""
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        h, z = world_model.rssm.initial_state(batch_size=1)
        prev_action = torch.zeros(1, config.action_dim)
        episode_reward = 0.0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding = world_model.encoder(obs_tensor)
                h, z, _, _ = world_model.rssm.observe_step(h, z, prev_action, embedding)
                latent = torch.cat([h, z], dim=-1)
                dist = actor(latent)
                action = dist.probs.argmax(dim=-1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            prev_action = F.one_hot(
                torch.tensor([action]), num_classes=config.action_dim,
            ).float()

        total_rewards.append(episode_reward)

    return float(np.mean(total_rewards))


def plot_training(
    reward_history: list[float],
    wm_loss_history: list[float],
) -> None:
    """Plot training curves and save to file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(reward_history, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Evaluation Reward")
    ax1.set_title("Dreamer: Evaluation Reward over Training")
    ax1.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="Solved (475)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(wm_loss_history, linewidth=1.5, color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("World Model Loss")
    ax2.set_title("World Model Total Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("app/training_curves.png", dpi=150)
    plt.close()
    print("Training curves saved to app/training_curves.png")


def main() -> None:
    config = DreamerConfig()
    set_seed(config.seed)
    print(f"Simplified Dreamer on {config.env_name}")
    print(f"Latent dim: {config.latent_dim} (det={config.deterministic_dim}, sto={config.stochastic_dim})")
    print(f"Imagination horizon: {config.imagination_horizon}")
    print()

    env = gym.make(config.env_name)

    world_model = WorldModel(config)
    actor = Actor(config)
    critic = Critic(config)

    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=config.world_model_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    replay_buffer = ReplayBuffer(config)

    reward_history: list[float] = []
    wm_loss_history: list[float] = []

    # --- Seed collection: gather some episodes with random policy ---
    print("Seed collection (5 random episodes)...")
    for _ in range(5):
        episode = collect_episode(env, world_model, actor, config)
        replay_buffer.add_episode(episode)
    print(f"Buffer: {len(replay_buffer)} episodes, {replay_buffer.total_steps} steps")
    print()

    # --- Main training loop ---
    for epoch in range(1, config.total_epochs + 1):
        # Phase 1: Collect experience
        for _ in range(config.collect_episodes_per_epoch):
            episode = collect_episode(env, world_model, actor, config)
            replay_buffer.add_episode(episode)

        # Phase 2: Train world model
        wm_losses = {}
        for _ in range(config.train_steps_per_epoch):
            step_losses = train_world_model(world_model, wm_optimizer, replay_buffer)
            for k, v in step_losses.items():
                wm_losses[k] = wm_losses.get(k, 0.0) + v
        wm_losses = {k: v / config.train_steps_per_epoch for k, v in wm_losses.items()}

        # Phase 3: Train actor-critic via imagination
        behavior_losses = {}
        for _ in range(config.train_steps_per_epoch):
            step_losses = train_behavior(
                world_model, actor, critic,
                actor_optimizer, critic_optimizer,
                replay_buffer, config,
            )
            for k, v in step_losses.items():
                behavior_losses[k] = behavior_losses.get(k, 0.0) + v
        behavior_losses = {k: v / config.train_steps_per_epoch for k, v in behavior_losses.items()}

        # Evaluate
        eval_reward = evaluate(env, world_model, actor, config)
        reward_history.append(eval_reward)
        wm_loss_history.append(wm_losses.get("total", 0.0))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Reward: {eval_reward:6.1f} | "
                f"WM loss: {wm_losses.get('total', 0):7.4f} "
                f"(obs={wm_losses.get('obs_loss', 0):.4f}, "
                f"rew={wm_losses.get('reward_loss', 0):.4f}, "
                f"kl={wm_losses.get('kl_loss', 0):.4f}) | "
                f"Actor: {behavior_losses.get('actor_loss', 0):.4f} | "
                f"Critic: {behavior_losses.get('critic_loss', 0):.4f} | "
                f"Buffer: {replay_buffer.total_steps} steps"
            )

        if eval_reward >= 475.0:
            print(f"\nSolved at epoch {epoch}! Average reward: {eval_reward:.1f}")
            break

    env.close()
    plot_training(reward_history, wm_loss_history)
    print("Done.")


if __name__ == "__main__":
    main()
