"""Tests for free energy computation."""

import torch
import pytest

from src.predictive_coding_network import PredictiveCodingNetwork


@pytest.fixture
def pcn():
    """Create a small PCN for testing."""
    torch.manual_seed(42)
    return PredictiveCodingNetwork(
        layer_dims=[8, 6, 4],
        num_inference_steps=10,
        inference_rate=0.1,
        learning_rate=0.001,
    )


@pytest.fixture
def batch():
    """Create a small random batch."""
    torch.manual_seed(42)
    return torch.randn(4, 8)


class TestFreeEnergy:
    """Tests for PredictiveCodingNetwork.compute_free_energy."""

    def test_returns_scalar(self, pcn, batch):
        """Free energy should be a 0-dimensional tensor (scalar)."""
        activities = pcn._initialize_activities(batch)
        energy = pcn.compute_free_energy(activities)
        assert energy.dim() == 0, f"Expected scalar, got shape {energy.shape}"

    def test_zero_when_activities_match_predictions(self, pcn, batch):
        """When activities are initialized via forward pass, the hidden layers
        perfectly match predictions (by construction), so energy should be
        very close to zero (only floating-point error)."""
        activities = pcn._initialize_activities(batch)
        energy = pcn.compute_free_energy(activities)
        assert energy.item() < 1e-5, (
            f"Energy should be ~0 for forward-pass-initialized activities, got {energy.item()}"
        )

    def test_positive_when_activities_perturbed(self, pcn, batch):
        """After perturbing hidden activities, energy should be strictly positive."""
        activities = pcn._initialize_activities(batch)
        # Perturb hidden layer activities
        for i in range(1, len(activities) - 1):
            activities[i] = activities[i] + torch.randn_like(activities[i]) * 0.5
        energy = pcn.compute_free_energy(activities)
        assert energy.item() > 0.0, f"Energy should be positive after perturbation, got {energy.item()}"

    def test_energy_decreases_toward_predictions(self, pcn, batch):
        """Energy should decrease as perturbed activities move back toward predictions."""
        activities = pcn._initialize_activities(batch)

        # Save the "correct" hidden activities
        correct_hidden = [a.clone() for a in activities]

        # Perturb hidden activities
        for i in range(1, len(activities) - 1):
            activities[i] = activities[i] + torch.randn_like(activities[i]) * 1.0

        energy_perturbed = pcn.compute_free_energy(activities)

        # Move halfway back toward correct values
        for i in range(1, len(activities) - 1):
            activities[i] = 0.5 * activities[i] + 0.5 * correct_hidden[i]

        energy_halfway = pcn.compute_free_energy(activities)

        assert energy_halfway.item() < energy_perturbed.item(), (
            f"Energy should decrease when moving toward predictions: "
            f"{energy_halfway.item()} should be < {energy_perturbed.item()}"
        )

    def test_energy_nonnegative(self, pcn, batch):
        """Free energy (sum of squared errors) should never be negative."""
        activities = pcn._initialize_activities(batch)
        # Random perturbation
        for i in range(1, len(activities)):
            activities[i] = torch.randn_like(activities[i])
        energy = pcn.compute_free_energy(activities)
        assert energy.item() >= 0.0, f"Energy should be non-negative, got {energy.item()}"
