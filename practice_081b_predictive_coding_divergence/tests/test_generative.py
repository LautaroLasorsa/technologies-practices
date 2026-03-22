"""Tests for Phase 1: Generative Inference."""

import torch
import pytest

from src.predictive_coding_network import PredictiveCodingNetwork
from src.generative_inference import generate_from_label, generate_all_classes


@pytest.fixture
def trained_pcn():
    """Create a small PCN with random weights (not truly trained, but functional)."""
    pcn = PredictiveCodingNetwork(layer_dims=[784, 64, 32, 10])
    return pcn


def test_generate_from_label_returns_correct_shape(trained_pcn):
    """Generated images should have shape (batch_size, input_dim)."""
    result = generate_from_label(
        trained_pcn, label=3, num_classes=10,
        num_inference_steps=5, batch_size=4,
    )
    assert result.shape == (4, 784), f"Expected (4, 784), got {result.shape}"


def test_generate_from_label_values_are_finite(trained_pcn):
    """Generated pixel values should not contain NaN or Inf."""
    result = generate_from_label(
        trained_pcn, label=0, num_classes=10,
        num_inference_steps=10, batch_size=2,
    )
    assert torch.isfinite(result).all(), "Generated images contain NaN or Inf"


def test_generate_from_label_different_labels_differ(trained_pcn):
    """Different labels should produce different generated images."""
    torch.manual_seed(42)
    img_0 = generate_from_label(
        trained_pcn, label=0, num_classes=10,
        num_inference_steps=10, batch_size=1,
    )
    torch.manual_seed(42)
    img_5 = generate_from_label(
        trained_pcn, label=5, num_classes=10,
        num_inference_steps=10, batch_size=1,
    )
    # They should differ because different labels are clamped
    assert not torch.allclose(img_0, img_5, atol=1e-3), \
        "Different labels produced identical images"


def test_generate_all_classes_returns_all_labels(trained_pcn):
    """generate_all_classes should return entries for all 10 classes."""
    results = generate_all_classes(
        trained_pcn, num_classes=10, samples_per_class=2,
        num_inference_steps=3,
    )
    assert len(results) == 10
    for label in range(10):
        assert label in results
        assert results[label].shape == (2, 784)
