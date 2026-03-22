"""Tests for the inference phase (perceptual inference)."""

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
        learning_rate=0.001,
    )


@pytest.fixture
def batch_and_target():
    """Create a small random batch and one-hot target."""
    torch.manual_seed(42)
    batch = torch.randn(4, 8)
    target = torch.zeros(4, 4)
    target[0, 0] = 1.0
    target[1, 1] = 1.0
    target[2, 2] = 1.0
    target[3, 3] = 1.0
    return batch, target


class TestInferencePhase:
    """Tests for PredictiveCodingNetwork.run_inference_phase."""

    def test_energy_decreases(self, pcn, batch_and_target):
        """Free energy should generally decrease during inference."""
        batch, target = batch_and_target
        activities = pcn._initialize_activities(batch)
        _, energy_trace = pcn.run_inference_phase(batch, target, activities)

        assert len(energy_trace) == pcn.num_inference_steps, (
            f"Expected {pcn.num_inference_steps} energy entries, got {len(energy_trace)}"
        )
        # Energy should decrease from first to last step
        assert energy_trace[-1] < energy_trace[0], (
            f"Energy should decrease during inference: "
            f"first={energy_trace[0]:.4f}, last={energy_trace[-1]:.4f}"
        )

    def test_input_stays_clamped(self, pcn, batch_and_target):
        """The sensory input (a_0) should remain clamped to the original input."""
        batch, target = batch_and_target
        activities = pcn._initialize_activities(batch)
        updated, _ = pcn.run_inference_phase(batch, target, activities)

        assert torch.allclose(updated[0], batch, atol=1e-6), (
            "Input layer should remain clamped to sensory_input"
        )

    def test_output_stays_clamped(self, pcn, batch_and_target):
        """The output (a_L) should remain clamped to the target."""
        batch, target = batch_and_target
        activities = pcn._initialize_activities(batch)
        updated, _ = pcn.run_inference_phase(batch, target, activities)

        assert torch.allclose(updated[-1], target, atol=1e-6), (
            "Output layer should remain clamped to target"
        )

    def test_hidden_activities_change(self, pcn, batch_and_target):
        """Hidden layer activities should differ from their initial values after inference."""
        batch, target = batch_and_target
        activities = pcn._initialize_activities(batch)
        initial_hidden = [a.clone() for a in activities[1:-1]]

        updated, _ = pcn.run_inference_phase(batch, target, activities)
        updated_hidden = updated[1:-1]

        for idx, (init, upd) in enumerate(zip(initial_hidden, updated_hidden)):
            assert not torch.allclose(init, upd, atol=1e-6), (
                f"Hidden layer {idx + 1} should change during inference"
            )

    def test_energy_trace_length(self, pcn, batch_and_target):
        """Energy trace should have exactly num_inference_steps entries."""
        batch, target = batch_and_target
        activities = pcn._initialize_activities(batch)
        _, energy_trace = pcn.run_inference_phase(batch, target, activities)

        assert len(energy_trace) == pcn.num_inference_steps

    def test_activities_list_length(self, pcn, batch_and_target):
        """Returned activities should have one entry per layer."""
        batch, target = batch_and_target
        activities = pcn._initialize_activities(batch)
        updated, _ = pcn.run_inference_phase(batch, target, activities)

        assert len(updated) == pcn.num_layers, (
            f"Expected {pcn.num_layers} activity tensors, got {len(updated)}"
        )
