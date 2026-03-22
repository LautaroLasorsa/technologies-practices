"""Tests for Phase 2: Missing Data Inference."""

import torch
import pytest

from src.predictive_coding_network import PredictiveCodingNetwork
from src.missing_data import infer_missing_data, create_masks


@pytest.fixture
def pcn():
    """Create a small PCN for testing."""
    return PredictiveCodingNetwork(layer_dims=[784, 64, 32, 10])


def test_infer_missing_data_returns_correct_shapes(pcn):
    """Reconstructed input and predicted labels should have correct shapes."""
    batch = 4
    partial_input = torch.randn(batch, 784)
    mask = create_masks(784, "top_half").unsqueeze(0).expand(batch, -1)

    reconstructed, pred_labels = infer_missing_data(
        pcn, partial_input * mask, mask,
        num_inference_steps=5,
    )
    assert reconstructed.shape == (batch, 784), \
        f"Expected reconstructed shape (4, 784), got {reconstructed.shape}"
    assert pred_labels.shape == (batch,), \
        f"Expected pred_labels shape (4,), got {pred_labels.shape}"


def test_infer_missing_data_observed_pixels_preserved(pcn):
    """Observed pixels should remain equal to the original input."""
    batch = 2
    input_data = torch.rand(batch, 784)
    mask = create_masks(784, "bottom_half").unsqueeze(0).expand(batch, -1)
    partial = input_data * mask

    reconstructed, _ = infer_missing_data(
        pcn, partial, mask,
        num_inference_steps=10,
    )
    # Observed pixels (mask == 1) should match the original partial input
    observed = mask.bool()
    assert torch.allclose(reconstructed[observed], partial[observed], atol=1e-5), \
        "Observed pixels were modified during inference"


def test_infer_missing_data_values_are_finite(pcn):
    """Reconstructed values should not contain NaN or Inf."""
    batch = 2
    partial_input = torch.randn(batch, 784) * 0.1
    mask = create_masks(784, "random_50").unsqueeze(0).expand(batch, -1)

    reconstructed, pred_labels = infer_missing_data(
        pcn, partial_input * mask, mask,
        num_inference_steps=5,
    )
    assert torch.isfinite(reconstructed).all(), "Reconstructed images contain NaN or Inf"
    assert (pred_labels >= 0).all() and (pred_labels < 10).all(), \
        "Predicted labels out of range"


def test_create_masks_shapes():
    """Mask creation utility should return correct shapes and values."""
    for mask_type in ["top_half", "bottom_half", "left_half", "random_50", "random_75"]:
        mask = create_masks(784, mask_type)
        assert mask.shape == (784,)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        # Should have both 0s and 1s
        assert mask.sum() > 0, f"{mask_type} mask is all zeros"
        assert mask.sum() < 784, f"{mask_type} mask is all ones"
