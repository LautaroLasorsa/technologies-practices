"""Tests for Phase 5: Retinotopic (Convolutional) Cortical Layers."""

import torch
import pytest

from src.retinotopic_layer import RetinotopicCorticalLayer


@pytest.fixture
def v1_layer():
    """V1-like layer: 3 input channels -> 32 output channels, preserving spatial dims."""
    return RetinotopicCorticalLayer(
        in_channels=3, out_channels=32,
        kernel_size=3, stride=1, padding=1,
    )


@pytest.fixture
def v4_layer():
    """V4-like layer: 32 -> 64 channels with stride 2 (spatial reduction)."""
    return RetinotopicCorticalLayer(
        in_channels=32, out_channels=64,
        kernel_size=3, stride=2, padding=1,
    )


def test_compute_prediction_preserves_spatial_dims(v1_layer):
    """With stride=1 and padding=1, spatial dimensions should be preserved."""
    batch = torch.randn(4, 3, 32, 32)
    output = v1_layer.compute_prediction(batch)
    assert output.shape == (4, 32, 32, 32), f"Expected (4, 32, 32, 32), got {output.shape}"


def test_compute_prediction_reduces_spatial_with_stride(v4_layer):
    """With stride=2, spatial dimensions should halve."""
    batch = torch.randn(4, 32, 16, 16)
    output = v4_layer.compute_prediction(batch)
    assert output.shape == (4, 64, 8, 8), f"Expected (4, 64, 8, 8), got {output.shape}"


def test_compute_prediction_values_bounded(v1_layer):
    """Output of tanh activation should be in [-1, 1]."""
    batch = torch.randn(4, 3, 16, 16) * 10  # Large values to test saturation
    output = v1_layer.compute_prediction(batch)
    assert output.min() >= -1.0, f"Min value {output.min()} below -1"
    assert output.max() <= 1.0, f"Max value {output.max()} above 1"


def test_compute_prediction_error_shape(v1_layer):
    """Prediction error should have same shape as prediction."""
    batch = torch.randn(4, 3, 32, 32)
    prediction = v1_layer.compute_prediction(batch)
    actual = torch.randn_like(prediction)
    error = v1_layer.compute_prediction_error(actual, prediction)
    assert error.shape == prediction.shape


def test_compute_prediction_error_zero_for_same_input(v1_layer):
    """Error should be zero when actual equals prediction."""
    batch = torch.randn(4, 3, 32, 32)
    prediction = v1_layer.compute_prediction(batch)
    error = v1_layer.compute_prediction_error(prediction, prediction)
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-7), \
        f"Error should be zero, got max abs {error.abs().max().item()}"


def test_hebbian_update_changes_weights(v1_layer):
    """Hebbian update with non-zero error should change weights."""
    original_weights = v1_layer.synaptic_weights.clone()
    activity = torch.randn(4, 3, 32, 32)
    error = torch.randn(4, 32, 32, 32) * 0.1

    v1_layer.hebbian_update(activity, error, learning_rate=0.01)

    assert not torch.allclose(v1_layer.synaptic_weights, original_weights, atol=1e-8), \
        "Weights should change after Hebbian update"


def test_transpose_weight_multiply_shape(v1_layer):
    """Transposed convolution should map back to input spatial dimensions."""
    batch = torch.randn(4, 3, 32, 32)
    pred = v1_layer.compute_prediction(batch)
    error = torch.randn_like(pred)
    transposed = v1_layer.transpose_weight_multiply(error)
    assert transposed.shape == batch.shape, \
        f"Expected {batch.shape}, got {transposed.shape}"
