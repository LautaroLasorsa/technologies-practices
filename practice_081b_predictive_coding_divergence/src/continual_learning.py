"""
Continual Learning -- Can PCNs remember without forgetting?

Catastrophic forgetting: when a neural network learns task B, it overwrites
the weights needed for task A. This is a major problem for backprop networks
and a key difference from biological learning (humans don't forget how to
walk when they learn to read).

PCNs are hypothesized to forget LESS because:
1. Weight updates are LOCAL -- changing weights for task B only affects
   synapses activated by task B's patterns, leaving task A's synapses intact
2. The inference phase finds task-specific hidden representations that
   minimize interference between tasks
3. Biological brains use predictive coding AND don't catastrophically forget --
   coincidence?

This module runs: train on task 1 -> measure task 1 -> train on task 2 ->
measure both. Forgetting = (task1_baseline - task1_after) / task1_baseline.
A backprop MLP is trained in parallel with the same schedule for comparison.

The only TODO is the PCN training-on-a-loader helper.  Everything else
(schedule, evaluation, forgetting metric, one-hot encoding, dataset loading)
is scaffolded.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork


# -- TODO ------------------------------------------------------------------


def train_pcn_on_loader(
    pcn: PredictiveCodingNetwork,
    loader,
    num_epochs: int,
    num_classes: int = 10,
    device: torch.device = None,
) -> None:
    """Train `pcn` on `loader` for `num_epochs` epochs, in place.

    For each batch: flatten the images to `(batch, input_dim)`, build a
    one-hot target with `create_one_hot`, and call `pcn.train_step(images, one_hot)`.
    This is the core "keep learning without resetting weights" step that
    makes the continual-learning experiment meaningful -- PCN's local
    Hebbian updates only touch synapses with non-zero local error, so
    unused pathways (from task 1) should survive task 2 training better
    than in backprop.
    """
    # TODO(human): implement the PCN training loop.
    raise NotImplementedError("Implement train_pcn_on_loader()")


# -- Scaffolded: experiment orchestrator ----------------------------------


def continual_learning_experiment(
    pcn: PredictiveCodingNetwork,
    task1_loader,
    task2_loader,
    task1_test_loader,
    task2_test_loader,
    num_epochs_per_task: int = 5,
    device: torch.device = None,
) -> dict:
    """Train task1 -> measure baseline -> train task2 (no reset) -> measure both.

    Returns a dict with keys:
        task1_baseline      -- task 1 accuracy after training task 1
        task1_after_task2   -- task 1 accuracy after training task 2 (forgetting)
        task2_accuracy      -- task 2 accuracy after training task 2 (learning)
        forgetting          -- (baseline - after) / max(baseline, 1e-6)

    A forgetting of 0.0 = perfect retention, 1.0 = complete forgetting.
    """
    device = device or torch.device("cpu")

    print("  Training PCN on Task 1...")
    train_pcn_on_loader(pcn, task1_loader, num_epochs_per_task, device=device)
    task1_baseline = evaluate_accuracy(pcn, task1_test_loader, device)
    print(f"  Task 1 baseline: {task1_baseline:.4f}")

    print("  Training PCN on Task 2 (no reset)...")
    train_pcn_on_loader(pcn, task2_loader, num_epochs_per_task, device=device)
    task1_after_task2 = evaluate_accuracy(pcn, task1_test_loader, device)
    task2_accuracy = evaluate_accuracy(pcn, task2_test_loader, device)

    forgetting = (task1_baseline - task1_after_task2) / max(task1_baseline, 1e-6)

    return {
        "task1_baseline": task1_baseline,
        "task1_after_task2": task1_after_task2,
        "task2_accuracy": task2_accuracy,
        "forgetting": forgetting,
    }


def evaluate_accuracy(
    network: PredictiveCodingNetwork,
    test_loader,
    device: torch.device = None,
) -> float:
    """Evaluate classification accuracy using forward pass. Pre-built utility."""
    device = device or torch.device("cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = network.forward(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def create_one_hot(labels: torch.Tensor, num_classes: int = 10, device=None) -> torch.Tensor:
    """Convert integer labels to one-hot. Pre-built utility."""
    device = device or labels.device
    one_hot = torch.zeros(labels.size(0), num_classes, device=device)
    one_hot.scatter_(1, labels.to(device).unsqueeze(1), 1.0)
    return one_hot
