"""Latency prediction model module.

Contains the neural network architecture and training logic for predicting
query execution latency from plan feature vectors.

Used by: 02_cost_model.py, 03_hint_selection.py, 04_evaluation.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# TODO(human): Implement LatencyPredictor
# ---------------------------------------------------------------------------

# TODO(human): Implement class LatencyPredictor(nn.Module)
#
# A feedforward neural network (MLP) that takes a plan feature vector and
# predicts query execution latency.
#
# Architecture:
#   input (n_features) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)
#
# Key design decision -- predict log(latency) instead of raw latency:
#   Query latencies span orders of magnitude: a simple index lookup takes
#   ~0.1 ms while a complex multi-join scan can take 5000+ ms. If you train
#   with MSE on raw latency, the loss is dominated by the slowest queries
#   (a 100ms error on a 5000ms query matters less than a 100ms error on a
#   1ms query, but MSE treats them equally). Predicting log(latency) makes
#   errors proportional -- the model learns to get the ORDER OF MAGNITUDE
#   right, which is what matters for plan comparison.
#
#   In __init__: define the three linear layers and activation.
#   In forward: pass input through the network, return a single value.
#     The output is log(latency_ms), so at inference time you'll need
#     to exp() it to get actual milliseconds.
#
# Hint: nn.Sequential makes this very clean:
#   self.net = nn.Sequential(
#       nn.Linear(n_features, 64),
#       nn.ReLU(),
#       nn.Linear(64, 32),
#       nn.ReLU(),
#       nn.Linear(32, 1),
#   )
#
# Why MLP and not tree-CNN?
#   A tree-CNN (as used in Bao) would preserve the plan's tree structure,
#   letting the model learn that "Hash Join above Seq Scan" behaves
#   differently from "Hash Join above Index Scan." Our flat featurization
#   loses this structure, but an MLP on flat features is simpler to
#   implement and still effective for plan ranking (we only need to get
#   the relative ordering right, not the exact latency).
class LatencyPredictor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        raise NotImplementedError("Implement LatencyPredictor — see TODO above")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement forward pass — see TODO above")


# ---------------------------------------------------------------------------
# TODO(human): Implement train_model
# ---------------------------------------------------------------------------

# TODO(human): Implement train_model(
#     model: LatencyPredictor,
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     epochs: int = 200,
#     lr: float = 1e-3,
#     batch_size: int = 32,
# ) -> dict[str, list[float]]
#
# Train the latency prediction model using MSE loss and Adam optimizer.
#
# Steps:
#   1. Convert numpy arrays to torch tensors (float32).
#      IMPORTANT: if predicting log-latency, transform y values here:
#      y_train_log = np.log1p(y_train)  # log1p = log(1+x), handles 0 safely
#      y_val_log = np.log1p(y_val)
#
#   2. Normalize features (zero mean, unit variance) using TRAINING set
#      statistics. Compute mean and std from X_train, apply to both
#      X_train and X_val. Store the mean/std for inference later.
#      Hint: replace zero-std features with 1.0 to avoid division by zero.
#
#   3. Create DataLoader for training data with shuffling.
#
#   4. Training loop (epochs iterations):
#      - For each batch: forward pass, compute MSE loss, backward, step.
#      - After each epoch: compute full train loss and val loss (no_grad).
#      - Print progress every 20 epochs.
#      - Optional: early stopping if val loss hasn't improved for 30 epochs.
#
#   5. Return a dict with keys "train_loss" and "val_loss", each a list
#      of per-epoch loss values (for plotting).
#
# Why normalize features?
#   Plan features have wildly different scales: node type counts are 0-5,
#   but total_cost can be 0.01 to 1,000,000+. Without normalization, the
#   large-scale features dominate the gradient and the model effectively
#   ignores the small-scale features. Standard normalization (subtract mean,
#   divide by std) puts all features on equal footing.
#
# Why Adam optimizer?
#   Adam adapts learning rates per-parameter, handling the different
#   gradient magnitudes across the network. For small datasets like ours,
#   Adam converges faster and more reliably than plain SGD.
def train_model(
    model: LatencyPredictor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> dict[str, list[float]]:
    raise NotImplementedError("Implement train_model — see TODO above")
