"""
CorticalLayer -- Completed implementation from Session A.

In neuroscience, each cortical area (V1, V2, V4, IT) processes information
at a specific level of abstraction. Higher areas generate top-down predictions
of what lower areas should be seeing. Lower areas compute prediction errors
(mismatches between predictions and actual input) and send them upward.

This class models one such cortical area:
- It holds synaptic_weights (W_l) connecting it to the layer above
- It computes top-down predictions: what this layer expects the layer above to look like
- It computes prediction errors: the mismatch between actual activity and the prediction

This is provided pre-built so Session B is standalone. The user implemented
these methods in Practice 081a.
"""

import torch


class CorticalLayer:
    """A single cortical area in the predictive coding hierarchy.

    Each layer holds:
    - synaptic_weights: W_l of shape (output_dim, input_dim), connecting this
      layer's activity to the prediction of the layer above
    - activation_fn: nonlinearity (default: tanh, NOT ReLU -- ReLU causes
      energy pathologies due to flat gradients at zero)

    Neuroscience mapping:
    - Deep pyramidal cells -> carry top-down predictions (compute_prediction)
    - Superficial pyramidal cells -> carry prediction errors (compute_prediction_error)
    """

    def __init__(self, input_dim: int, output_dim: int, activation_fn=None):
        """Initialize a cortical layer.

        Args:
            input_dim: Dimension of this layer's neural activity (a_l)
            output_dim: Dimension of the layer above (a_{l+1})
            activation_fn: Nonlinearity. Default: tanh.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn or torch.tanh

        # Synaptic weights: W_l connects this layer to the one above
        # Xavier initialization for stable initial predictions
        self.synaptic_weights = torch.randn(output_dim, input_dim) * (2.0 / (input_dim + output_dim)) ** 0.5
        self.synaptic_weights.requires_grad_(False)  # We do NOT use autograd -- updates are local/Hebbian

    def compute_prediction(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Top-down prediction: f(a_l @ W^T)

        Args:
            neural_activity: This layer's activity a_l, shape (batch_size, input_dim)

        Returns:
            Prediction of the layer above's activity, shape (batch_size, output_dim)
        """
        pre_act = neural_activity @ self.synaptic_weights.T
        return self.activation_fn(pre_act)

    def compute_prediction_error(
        self, neural_activity_above: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Prediction error: epsilon = actual - predicted

        Args:
            neural_activity_above: Actual activity at layer l+1, shape (batch_size, output_dim)
            prediction: Top-down prediction of layer l+1, shape (batch_size, output_dim)

        Returns:
            Prediction error epsilon_{l+1}, shape (batch_size, output_dim)
        """
        return neural_activity_above - prediction

    def activation_derivative(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """Compute f'(x) -- derivative of the activation function.

        For tanh: f'(x) = 1 - tanh(x)^2

        This is needed in both inference dynamics and weight updates.
        """
        if self.activation_fn == torch.tanh:
            return 1.0 - torch.tanh(pre_activation) ** 2
        else:
            x = pre_activation.detach().requires_grad_(True)
            y = self.activation_fn(x)
            return torch.autograd.grad(y.sum(), x)[0]

    def pre_activation(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Compute the linear pre-activation W . a_l (before applying f).

        Needed for computing f'(W . a_l) in inference dynamics.
        """
        return neural_activity @ self.synaptic_weights.T
