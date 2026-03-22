"""Tests for Phase 3: Precision Weighting."""

import torch
import pytest

from src.predictive_coding_network import PredictiveCodingNetwork
from src.precision_weighting import precision_weighted_energy, precision_weighted_inference


@pytest.fixture
def pcn():
    """Create a small PCN for testing."""
    return PredictiveCodingNetwork(layer_dims=[784, 64, 32, 10])


def test_precision_weighted_energy_with_uniform_precision_equals_standard(pcn):
    """When all precisions are 1.0, precision-weighted energy should equal standard energy."""
    batch = 4
    activities = pcn._initialize_activities(torch.randn(batch, 784))

    # Uniform precisions (all ones)
    precisions = [torch.ones(dim) for dim in pcn.layer_dims[1:]]

    pw_energy = precision_weighted_energy(activities, pcn.cortical_layers, precisions)
    standard_energy = pcn.compute_free_energy(activities)

    assert torch.allclose(pw_energy, standard_energy, atol=1e-5), \
        f"Uniform precision energy ({pw_energy.item():.6f}) != standard energy ({standard_energy.item():.6f})"


def test_precision_weighted_energy_is_scalar(pcn):
    """Precision-weighted energy should return a scalar."""
    batch = 4
    activities = pcn._initialize_activities(torch.randn(batch, 784))
    precisions = [torch.ones(dim) for dim in pcn.layer_dims[1:]]

    energy = precision_weighted_energy(activities, pcn.cortical_layers, precisions)
    assert energy.dim() == 0, f"Expected scalar, got shape {energy.shape}"


def test_precision_weighted_energy_scales_with_precision(pcn):
    """Higher precision should produce higher energy for the same errors."""
    batch = 4
    activities = pcn._initialize_activities(torch.randn(batch, 784))

    low_prec = [torch.ones(dim) * 0.1 for dim in pcn.layer_dims[1:]]
    high_prec = [torch.ones(dim) * 10.0 for dim in pcn.layer_dims[1:]]

    low_energy = precision_weighted_energy(activities, pcn.cortical_layers, low_prec)
    high_energy = precision_weighted_energy(activities, pcn.cortical_layers, high_prec)

    assert high_energy > low_energy, \
        f"Higher precision ({high_energy.item():.4f}) should give higher energy than lower ({low_energy.item():.4f})"


def test_precision_weighted_inference_returns_correct_types(pcn):
    """Precision-weighted inference should return activities list and energy trace."""
    batch = 4
    sensory = torch.randn(batch, 784)
    target = torch.zeros(batch, 10)
    target[:, 3] = 1.0
    precisions = [torch.ones(dim) for dim in pcn.layer_dims[1:]]

    activities, energy_trace = precision_weighted_inference(
        pcn, sensory, target, precisions,
        num_inference_steps=5,
    )

    assert isinstance(activities, list)
    assert len(activities) == len(pcn.layer_dims)
    assert isinstance(energy_trace, list)
    assert len(energy_trace) == 5
    assert all(isinstance(e, float) for e in energy_trace)
