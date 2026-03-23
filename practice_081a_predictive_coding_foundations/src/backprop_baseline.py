"""
Standard backprop MLP -- the baseline for comparison.

This is pre-built because the user already knows backprop cold.
It uses the SAME architecture as the PCN (same layer dims, same activation)
to ensure a fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BackpropMLP(nn.Module):
    """Standard feedforward MLP trained with backpropagation."""

    def __init__(self, layer_dims: list[int], activation_fn=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        self.activation_fn = activation_fn or torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output layer
                x = self.activation_fn(x)
        return x


def train_backprop_model(
    model: BackpropMLP,
    train_loader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device = None,
) -> dict:
    """Train a backprop MLP and return training history.

    Args:
        model: BackpropMLP instance.
        train_loader: DataLoader yielding (images, labels) batches.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        device: torch.device for computation.

    Returns:
        Dict with 'loss' and 'accuracy' lists (one entry per epoch).
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": []}

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  [BP] Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in pbar:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        print(f"  [Backprop] Epoch {epoch + 1}/{num_epochs} -- Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return history
