"""
Retinotopic Cortical Layers -- Convolutional Predictive Coding.

In the visual cortex, neurons have LOCAL receptive fields -- each V1 neuron
responds to a small patch of the visual field. This is the biological basis
for convolutions. The receptive fields tile the visual field (retinotopic
mapping), and higher areas (V2, V4) have progressively larger receptive fields.

A RetinotopicCorticalLayer replaces the fully-connected CorticalLayer with
convolutions:
- Top-down prediction: conv2d (higher -> lower area)
- Bottom-up error: actual_activity - prediction (same as before)
- Inference: conv_transpose2d for the W^T term (transpose convolution)

This is more biologically faithful than the FC version: real cortical areas
ARE retinotopic, and local receptive fields ARE convolutions.
"""

import torch
import torch.nn.functional as F


class RetinotopicCorticalLayer:
    """Convolutional cortical layer with retinotopic (local) receptive fields.

    Instead of full matrix multiplication, uses convolutions:
    - prediction: Conv2d (maps activity to prediction of lower area)
    - W^T operation: ConvTranspose2d (maps errors back up through the same filter)

    The scaffold provides shape management and transposed convolution setup.
    The user implements the core prediction, error, and Hebbian update.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation_fn=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn or torch.tanh

        # Synaptic weights as conv filters
        # Shape: (out_channels, in_channels, kernel_size, kernel_size)
        scale = (2.0 / (in_channels * kernel_size * kernel_size + out_channels)) ** 0.5
        self.synaptic_weights = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.synaptic_weights.requires_grad_(False)

    def compute_prediction(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Top-down prediction using convolution.

        The conv analogue of the FC prediction f(a @ W^T):
        Apply conv filters to the input activity, then activation function.

        In the brain, this is what V2 sends down to V1: "I predict you
        should see these local features at these positions."

        Args:
            neural_activity: shape (batch, in_channels, H, W)

        Returns:
            Prediction of next layer, shape (batch, out_channels, H', W')
            where H', W' depend on stride and padding.
        """
        # TODO(human): Implement convolutional top-down prediction.
        #
        # This is the conv analogue of the FC prediction f(a @ W^T):
        #   pre_activation = F.conv2d(neural_activity, self.synaptic_weights,
        #                              stride=self.stride, padding=self.padding)
        #   prediction = self.activation_fn(pre_activation)
        #   return prediction
        #
        # The conv filter IS the local receptive field -- each output neuron
        # "sees" only a small patch of the input, just like V1 neurons.
        # The kernel_size controls the receptive field size.
        #
        # Note: F.conv2d applies each of the out_channels filters to the
        # entire input feature map. With padding=1 and stride=1, spatial
        # dimensions are preserved. With stride=2, spatial dims halve --
        # modeling how higher cortical areas have lower spatial resolution
        # but more channels (more abstract features).
        raise NotImplementedError("TODO(human): Implement convolutional prediction")

    def compute_prediction_error(
        self, activity_above: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Prediction error: actual - predicted. Same as FC version.

        The error is computed pointwise across the spatial dimensions.
        Each position in the feature map has its own prediction error.

        Args:
            activity_above: Actual activity, shape (batch, out_channels, H', W')
            prediction: Top-down prediction, shape (batch, out_channels, H', W')

        Returns:
            Prediction error, shape (batch, out_channels, H', W')
        """
        # TODO(human): Compute the prediction error for conv layers.
        #
        # Same as the FC version -- just subtract:
        #   return activity_above - prediction
        #
        # The error has the same spatial structure as the prediction.
        # Each spatial position independently computes its own error.
        # In neuroscience terms: each point in the retinotopic map has
        # its own error neuron that signals local mismatches.
        raise NotImplementedError("TODO(human): Implement conv prediction error")

    def transpose_weight_multiply(self, error_signal: torch.Tensor) -> torch.Tensor:
        """Compute W^T * error using transposed convolution.

        In FC layers: W^T * epsilon multiplies by the transpose of the weight matrix.
        In conv layers: this is a TRANSPOSED convolution (ConvTranspose2d).

        Pre-built because transposed conv setup is boilerplate.
        """
        return F.conv_transpose2d(
            error_signal, self.synaptic_weights,
            stride=self.stride, padding=self.padding,
        )

    def activation_derivative(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """f'(x) for the activation function. Pre-built."""
        if self.activation_fn == torch.tanh:
            return 1.0 - torch.tanh(pre_activation) ** 2
        else:
            x = pre_activation.detach().requires_grad_(True)
            y = self.activation_fn(x)
            return torch.autograd.grad(y.sum(), x)[0]

    def pre_activation(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Conv pre-activation (before applying f). Pre-built."""
        return F.conv2d(
            neural_activity, self.synaptic_weights,
            stride=self.stride, padding=self.padding,
        )

    def hebbian_update(
        self,
        neural_activity: torch.Tensor,
        error_above: torch.Tensor,
        learning_rate: float = 0.001,
    ) -> None:
        """Hebbian weight update for conv filters.

        The conv analogue of the FC update: instead of an outer product,
        we use the cross-correlation between input activity and error signal.

        Args:
            neural_activity: This layer's activity, (batch, in_ch, H, W)
            error_above: Modulated error from above, (batch, out_ch, H', W')
            learning_rate: alpha
        """
        # TODO(human): Implement Hebbian update for conv weights.
        #
        # The FC weight update was: delta_W = alpha * (modulated_error.T @ activity) / batch
        # The conv equivalent uses cross-correlation:
        #
        # For each filter, the update is the correlation between the input patch
        # and the error at that position. In PyTorch, this is done by treating
        # the problem as a convolution with rearranged dimensions.
        #
        # Simpler approach (works correctly for learning):
        #   batch_size = neural_activity.shape[0]
        #   delta_w = torch.zeros_like(self.synaptic_weights)
        #   for b in range(batch_size):
        #       # For each sample: correlate input activity with error signal
        #       # neural_activity[b]: (in_ch, H, W) -> (in_ch, 1, H, W) as "batch"
        #       # error_above[b]:     (out_ch, H', W') -> (out_ch, 1, H', W') as "filters"
        #       # conv2d with input as "batch dim" and error as "filter" computes correlation
        #       inp = neural_activity[b:b+1].transpose(0, 1)  # (in_ch, 1, H, W)
        #       err = error_above[b:b+1].transpose(0, 1)      # (out_ch, 1, H', W')
        #       corr = F.conv2d(inp, err, padding=self.padding)  # (in_ch, out_ch, kH, kW)
        #       delta_w += corr.transpose(0, 1)  # -> (out_ch, in_ch, kH, kW)
        #   delta_w /= batch_size
        #   self.synaptic_weights += learning_rate * delta_w
        #
        # Note: The loop-over-batch approach is slow but clear. For production,
        # you'd vectorize this -- but clarity matters more here.
        #
        # Think about: Why is this "Hebbian"? The update correlates what the
        # input neurons are doing (neural_activity) with what the output
        # neurons need (error_above). Synapses strengthen when pre-synaptic
        # activity aligns with the error signal -- "neurons that fire
        # together wire together," but modulated by the prediction error.
        raise NotImplementedError("TODO(human): Implement conv Hebbian update")
