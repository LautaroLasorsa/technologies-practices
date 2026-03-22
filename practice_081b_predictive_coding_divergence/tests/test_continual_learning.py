"""Tests for Phase 4: Continual Learning."""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.predictive_coding_network import PredictiveCodingNetwork
from src.continual_learning import continual_learning_experiment, evaluate_accuracy, create_one_hot


@pytest.fixture
def pcn():
    """Create a small PCN for testing."""
    return PredictiveCodingNetwork(
        layer_dims=[784, 32, 10],
        num_inference_steps=5,
    )


def _make_fake_loader(n_samples=64, n_classes=10, input_dim=784, batch_size=32):
    """Create a fake DataLoader for testing (random images, random labels)."""
    images = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_continual_learning_returns_required_keys(pcn):
    """The result dict must contain all four required keys."""
    task1_train = _make_fake_loader(n_samples=32)
    task2_train = _make_fake_loader(n_samples=32)
    task1_test = _make_fake_loader(n_samples=16)
    task2_test = _make_fake_loader(n_samples=16)

    result = continual_learning_experiment(
        pcn, task1_train, task2_train, task1_test, task2_test,
        num_epochs_per_task=1,
    )

    required_keys = {"task1_baseline", "task1_after_task2", "task2_accuracy", "forgetting"}
    assert required_keys.issubset(result.keys()), \
        f"Missing keys: {required_keys - result.keys()}"


def test_continual_learning_values_are_valid(pcn):
    """All accuracy values should be between 0 and 1, forgetting between -1 and 1."""
    task1_train = _make_fake_loader(n_samples=32)
    task2_train = _make_fake_loader(n_samples=32)
    task1_test = _make_fake_loader(n_samples=16)
    task2_test = _make_fake_loader(n_samples=16)

    result = continual_learning_experiment(
        pcn, task1_train, task2_train, task1_test, task2_test,
        num_epochs_per_task=1,
    )

    for key in ("task1_baseline", "task1_after_task2", "task2_accuracy"):
        assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of range [0, 1]"

    # Forgetting can theoretically be negative (improvement) but usually positive
    assert -1.0 <= result["forgetting"] <= 1.5, \
        f"forgetting = {result['forgetting']} seems unreasonable"


def test_evaluate_accuracy_with_known_data(pcn):
    """evaluate_accuracy should work correctly with a simple loader."""
    loader = _make_fake_loader(n_samples=16, batch_size=8)
    acc = evaluate_accuracy(pcn, loader)
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range"


def test_create_one_hot():
    """create_one_hot should produce correct one-hot vectors."""
    labels = torch.tensor([0, 3, 7, 9])
    one_hot = create_one_hot(labels, num_classes=10)
    assert one_hot.shape == (4, 10)
    assert one_hot.sum().item() == 4.0  # Exactly one 1 per row
    assert one_hot[0, 0] == 1.0
    assert one_hot[1, 3] == 1.0
    assert one_hot[2, 7] == 1.0
    assert one_hot[3, 9] == 1.0
