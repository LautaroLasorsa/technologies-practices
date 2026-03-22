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

This experiment tests: train on task 1, then task 2, measure task 1 retention.
Compare PCN vs backprop MLP.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork


def continual_learning_experiment(
    pcn: PredictiveCodingNetwork,
    task1_loader,
    task2_loader,
    task1_test_loader,
    task2_test_loader,
    num_epochs_per_task: int = 5,
    device: torch.device = None,
) -> dict:
    """Run a continual learning experiment: train task1 -> train task2 -> measure both.

    The experiment measures:
    1. Task 1 accuracy after training on task 1 (baseline)
    2. Task 1 accuracy after training on task 2 (forgetting measure)
    3. Task 2 accuracy after training on task 2 (learning measure)

    Forgetting = (baseline - after_task2) / baseline

    Args:
        pcn: The PredictiveCodingNetwork to train
        task1_loader: DataLoader for task 1 training data
        task2_loader: DataLoader for task 2 training data
        task1_test_loader: DataLoader for task 1 test data
        task2_test_loader: DataLoader for task 2 test data
        num_epochs_per_task: Epochs per task
        device: CPU/CUDA

    Returns:
        Dict with accuracy measurements and forgetting metrics:
        {
            "task1_baseline": float,
            "task1_after_task2": float,
            "task2_accuracy": float,
            "forgetting": float,
        }
    """
    # TODO(human): Implement the continual learning experiment.
    #
    # Step 1: Train on Task 1
    #   for epoch in range(num_epochs_per_task):
    #       for images, labels in task1_loader:
    #           images = images.view(images.size(0), -1).to(device)
    #           one_hot = create_one_hot(labels, num_classes=10, device=device)
    #           pcn.train_step(images, one_hot)
    #
    # Step 2: Evaluate on Task 1 (baseline accuracy)
    #   task1_baseline = evaluate_accuracy(pcn, task1_test_loader, device)
    #   Print it so the user can track progress:
    #   print(f"  Task 1 baseline: {task1_baseline:.4f}")
    #
    # Step 3: Train on Task 2 (WITHOUT resetting weights)
    #   Same loop as step 1, but with task2_loader.
    #   This is where catastrophic forgetting would occur -- task 2
    #   training may overwrite the weights that were useful for task 1.
    #
    # Step 4: Evaluate on BOTH tasks after training on task 2
    #   task1_after = evaluate_accuracy(pcn, task1_test_loader, device)
    #   task2_after = evaluate_accuracy(pcn, task2_test_loader, device)
    #
    # Step 5: Compute forgetting metric
    #   forgetting = (task1_baseline - task1_after) / max(task1_baseline, 1e-6)
    #   A forgetting of 0.0 means no forgetting (perfect retention).
    #   A forgetting of 1.0 means complete forgetting (task 1 accuracy = 0).
    #
    # Return the results dict with all four metrics.
    #
    # Think about: Why might local weight updates cause less forgetting?
    # In backprop, the global backward pass adjusts ALL weights based on
    # the gradient -- even weights far from the output that happen to have
    # non-zero gradients. In PCN, weights only update if their local
    # prediction error is non-zero -- if task 2 doesn't activate certain
    # cortical areas, those areas' weights are untouched.
    raise NotImplementedError("TODO(human): Implement continual learning experiment")


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
