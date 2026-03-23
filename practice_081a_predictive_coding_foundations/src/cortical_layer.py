"""
CorticalLayer -- A single level in the predictive coding hierarchy.

In neuroscience, each cortical area (V1, V2, V4, IT) processes information
at a specific level of abstraction. Higher areas generate top-down predictions
of what lower areas should be seeing. Lower areas compute prediction errors
(mismatches between predictions and actual input) and send them upward.

NOTE on direction: The neuroscience version is top-down (abstract → sensory).
This implementation follows the Whittington & Bogacz (2017) supervised
classification formulation, where predictions go bottom-up (input → output):
layer l predicts layer l+1. The math (free energy, inference dynamics, Hebbian
updates, backprop equivalence) is identical either way.

This class models one such cortical area:
- It holds synaptic_weights (W_l) connecting it to the layer above
- It computes predictions: what this layer expects the layer above to look like
- It computes prediction errors: the mismatch between actual activity and the prediction
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
    - Deep pyramidal cells -> carry predictions (compute_prediction)
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
        """Compute prediction: what this layer predicts the layer above looks like.

        In the W&B (2017) formulation, layer l predicts layer l+1 (bottom-up):

            prediction = f(W_l . a_l)

        where:
        - a_l is this layer's neural_activity, shape (batch_size, input_dim)
        - W_l is synaptic_weights, shape (output_dim, input_dim)
        - f is the activation function (tanh by default)

        The result has shape (batch_size, output_dim) -- it predicts what the layer
        above's activity "should" look like according to this layer.

        Convention (Bogacz 2017, arXiv:2407.04117):
        - Layer l has activity a_l of dimension dim_l
        - W_l has shape (dim_{l+1}, dim_l) -- maps FROM layer l TO layer l+1
        - The prediction OF a_{l+1} made BY layer l is: f(W_l . a_l)

        (In neuroscience, predictions go top-down: abstract → sensory. This code
        uses the equivalent bottom-up formulation for supervised classification.)

        Args:
            neural_activity: This layer's activity a_l, shape (batch_size, input_dim)

        Returns:
            Prediction of the layer above's activity, shape (batch_size, output_dim)
        """
        # TODO(human): Compute the prediction (layer l → layer l+1).
        #
        # The prediction is: f(a_l @ W^T)
        # where:
        #   - a_l is neural_activity, shape (batch_size, input_dim)
        #   - W is self.synaptic_weights, shape (output_dim, input_dim)
        #   - f is self.activation_fn (torch.tanh by default)
        #
        # Steps:
        # 1. Compute the linear pre-activation: a_l @ W^T -> shape (batch_size, output_dim)
        # 2. Apply the activation function to get the prediction
        #
        # Neuroscience: This is what deep pyramidal cells compute -- the brain's
        # "best guess" of what the next level of processing should look like.
        # When this prediction matches reality, prediction error is zero and
        # no learning signal is generated (the brain is "unsurprised").
        return self.activation_fn(neural_activity @ self.synaptic_weights.T)

    def compute_prediction_error(
        self, neural_activity_above: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error: the mismatch between actual activity and prediction.

        In the brain, superficial pyramidal cells compute this error signal.
        It's the ONLY signal that propagates upward (feedforward) -- the brain
        communicates surprises, not raw data.

            epsilon_{l+1} = a_{l+1} - prediction

        where prediction = f(W_l . a_l) (computed by compute_prediction).

        Args:
            neural_activity_above: Actual activity at layer l+1, shape (batch_size, output_dim)
            prediction: Top-down prediction of layer l+1, shape (batch_size, output_dim)

        Returns:
            Prediction error epsilon_{l+1}, shape (batch_size, output_dim)
        """
        # TODO(human): Compute the prediction error.
        #
        # This is straightforward: error = actual - predicted
        #
        #   epsilon_{l+1} = a_{l+1} - f(W_l . a_l)
        #
        # The sign matters: positive error means "activity is HIGHER than predicted"
        # (the brain is under-predicting). Negative means over-predicting.
        #
        # This error drives BOTH:
        # 1. Inference: activities adjust to reduce this error
        # 2. Learning: weights update to make better predictions next time
        #
        # Neuroscience: These are the signals carried by superficial pyramidal
        # cells. Only prediction errors travel upward -- if a cortical area
        # perfectly predicts its input, it sends nothing (efficient coding).
        return neural_activity_above - prediction

    def activation_derivative(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """Compute f'(x) -- derivative of the activation function.

        For tanh: f'(x) = 1 - tanh(x)^2

        This is needed in both inference dynamics and weight updates.
        Pre-built because it's boilerplate, not a learning objective.
        """
        if self.activation_fn == torch.tanh:
            return 1.0 - torch.tanh(pre_activation) ** 2
        else:
            # For other activations, use autograd
            x = pre_activation.detach().requires_grad_(True)
            y = self.activation_fn(x)
            return torch.autograd.grad(y.sum(), x)[0]

    def pre_activation(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Compute the linear pre-activation W . a_l (before applying f).

        Needed for computing f'(W . a_l) in inference dynamics.
        """
        return neural_activity @ self.synaptic_weights.T
