"""Tests for the full training loop (train_step and Hebbian updates)."""

import torch
import pytest

from src.predictive_coding_network import PredictiveCodingNetwork


@pytest.fixture
def pcn():
    """Create a small PCN for testing."""
    torch.manual_seed(42)
    return PredictiveCodingNetwork(
        layer_dims=[8, 6, 4],
        num_inference_steps=20,
        inference_rate=0.1,
        learning_rate=0.01,  # Higher LR for visible learning in tests
    )


@pytest.fixture
def training_data():
    """Create a small fixed dataset (4 samples, 4 classes)."""
    torch.manual_seed(42)
    inputs = torch.randn(4, 8)
    targets = torch.eye(4)  # One-hot: sample i -> class i
    return inputs, targets


class TestTraining:
    """Tests for PredictiveCodingNetwork.train_step."""

    def test_train_step_returns_dict(self, pcn, training_data):
        """train_step should return a dict with expected keys."""
        inputs, targets = training_data
        result = pcn.train_step(inputs, targets)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "free_energy" in result, "Result should contain 'free_energy'"
        assert "energy_trace" in result, "Result should contain 'energy_trace'"
        assert "accuracy" in result, "Result should contain 'accuracy'"

    def test_energy_decreases_over_training(self, pcn, training_data):
        """Free energy should decrease over multiple training steps."""
        inputs, targets = training_data
        energies = []

        for _ in range(20):
            result = pcn.train_step(inputs, targets)
            energies.append(result["free_energy"])

        # Compare first few vs last few
        early_avg = sum(energies[:5]) / 5
        late_avg = sum(energies[-5:]) / 5

        assert late_avg < early_avg, (
            f"Energy should decrease over training: "
            f"early avg={early_avg:.4f}, late avg={late_avg:.4f}"
        )

    def test_accuracy_increases_over_training(self, pcn, training_data):
        """Accuracy should increase over multiple training steps on the same batch."""
        inputs, targets = training_data

        first_result = pcn.train_step(inputs, targets)

        # Train for several more steps
        for _ in range(30):
            last_result = pcn.train_step(inputs, targets)

        assert last_result["accuracy"] >= first_result["accuracy"], (
            f"Accuracy should increase: "
            f"first={first_result['accuracy']:.4f}, last={last_result['accuracy']:.4f}"
        )

    def test_weights_change_after_training(self, pcn, training_data):
        """Synaptic weights should be different after training."""
        inputs, targets = training_data

        # Save initial weights
        initial_weights = [
            layer.synaptic_weights.clone()
            for layer in pcn.cortical_layers
        ]

        # Train
        for _ in range(5):
            pcn.train_step(inputs, targets)

        # Check weights changed
        for idx, (init_w, layer) in enumerate(zip(initial_weights, pcn.cortical_layers)):
            assert not torch.allclose(init_w, layer.synaptic_weights, atol=1e-6), (
                f"Weights at layer {idx} should change after training"
            )

    def test_forward_output_changes_after_training(self, pcn, training_data):
        """Forward pass output should differ after training."""
        inputs, targets = training_data

        output_before = pcn.forward(inputs).clone()

        for _ in range(10):
            pcn.train_step(inputs, targets)

        output_after = pcn.forward(inputs)

        assert not torch.allclose(output_before, output_after, atol=1e-6), (
            "Forward pass output should change after training"
        )
