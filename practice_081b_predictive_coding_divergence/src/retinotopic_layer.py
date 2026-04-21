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

The one non-biological aspect of standard CNNs is **weight sharing** --
all spatial positions use identical filters. Real V1 neurons at different
positions are similar but not identical. This is an acceptable approximation
that dramatically reduces parameter count.

You implement three small pieces (prediction, error, Hebbian update). Every
other op (transpose conv, activation derivative, pre-activation) is
pre-built because the shape bookkeeping is boilerplate.
"""

import torch
import torch.nn.functional as F


class RetinotopicCorticalLayer:
    """Convolutional cortical layer with retinotopic (local) receptive fields.

    Instead of full matrix multiplication, uses convolutions:
    - prediction: Conv2d (maps activity to prediction of lower area)
    - W^T operation: ConvTranspose2d (maps errors back up through the same filter)
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

    # -- TODO 1 ----------------------------------------------------------------

    def compute_prediction(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Conv analogue of f(a @ W^T): apply F.conv2d then the activation fn.

        The conv filter IS the local receptive field -- each output neuron
        "sees" only a small patch of the input, just like V1 neurons. The
        kernel_size controls the receptive field size. With stride=2 spatial
        dims halve, modelling how higher cortical areas have lower spatial
        resolution but more abstract features.

        `neural_activity` is (batch, in_channels, H, W); the prediction
        is (batch, out_channels, H', W') with H'/W' set by stride/padding.
        Use `F.conv2d(x, self.synaptic_weights, stride=self.stride, padding=self.padding)`.
        """
        # TODO(human): implement conv top-down prediction.
        raise NotImplementedError("Implement compute_prediction()")

    # -- TODO 2 ----------------------------------------------------------------

    def compute_prediction_error(
        self, activity_above: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Pointwise prediction error `actual - predicted`, same as the FC case.

        Both tensors are (batch, out_channels, H', W'); each spatial position
        independently computes its own error. In neuroscience terms, each
        point in the retinotopic map has its own error neuron signalling
        local mismatches.
        """
        # TODO(human): implement conv prediction error.
        raise NotImplementedError("Implement compute_prediction_error()")

    # -- Scaffolded: transposed conv and activation helpers -------------------

    def transpose_weight_multiply(self, error_signal: torch.Tensor) -> torch.Tensor:
        """Compute W^T * error using transposed convolution.

        In FC layers: W^T * epsilon multiplies by the transpose of the weight matrix.
        In conv layers: this is a TRANSPOSED convolution (ConvTranspose2d).
        """
        return F.conv_transpose2d(
            error_signal, self.synaptic_weights,
            stride=self.stride, padding=self.padding,
        )

    def activation_derivative(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """f'(x) for the activation function."""
        if self.activation_fn == torch.tanh:
            return 1.0 - torch.tanh(pre_activation) ** 2
        else:
            x = pre_activation.detach().requires_grad_(True)
            y = self.activation_fn(x)
            return torch.autograd.grad(y.sum(), x)[0]

    def pre_activation(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """Conv pre-activation (before applying f)."""
        return F.conv2d(
            neural_activity, self.synaptic_weights,
            stride=self.stride, padding=self.padding,
        )

    # -- TODO 3 ----------------------------------------------------------------

    def hebbian_update(
        self,
        neural_activity: torch.Tensor,
        error_above: torch.Tensor,
        learning_rate: float = 0.001,
    ) -> None:
        """Hebbian weight update for conv filters (cross-correlation).

        The FC update was `delta_W = alpha * modulated_error.T @ activity / batch`
        (an outer product).  The conv equivalent is the cross-correlation
        between input activity and the error signal at each spatial position.

        Why "Hebbian"? The update correlates what the pre-synaptic neurons
        are doing (`neural_activity`) with what the post-synaptic neurons
        need (`error_above`). Synapses strengthen when input and error signal
        align -- "neurons that fire together wire together," modulated by
        the prediction error.

        A clear (if slow) approach is to loop over the batch, treating each
        sample's input as a mini-batch and its error as filters so that
        `F.conv2d` computes the correlation:

            inp = neural_activity[b:b+1].transpose(0, 1)   # (in_ch, 1, H, W)
            err = error_above[b:b+1].transpose(0, 1)       # (out_ch, 1, H', W')
            corr = F.conv2d(inp, err, padding=self.padding)  # (in_ch, out_ch, kH, kW)
            delta_w += corr.transpose(0, 1)                # (out_ch, in_ch, kH, kW)

        Then divide by batch_size and do `self.synaptic_weights += learning_rate * delta_w`.
        """
        # TODO(human): implement conv Hebbian update.
        raise NotImplementedError("Implement hebbian_update()")
