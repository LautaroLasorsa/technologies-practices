"""
PredictiveCodingNetwork -- The hierarchical generative model.

In neuroscience, the cortex is organized as a hierarchy of cortical areas
(V1 -> V2 -> V4 -> IT -> PFC). Each area generates predictions of the area below
and receives prediction errors from below. The ENTIRE system minimizes a single
objective: free energy (total prediction error across all levels).

This class orchestrates:
1. Inference phase: iterate neural activities to minimize free energy (perception)
2. Weight updates: local Hebbian learning after inference converges (learning)
3. Forward pass: standard feedforward for test-time classification

The key insight: backpropagation computes weight gradients via a GLOBAL backward
pass using the chain rule. PCN achieves the SAME gradients (Whittington & Bogacz 2017)
using only LOCAL information -- each layer only needs its own activity, the prediction
error from above, and its synaptic weights. No weight transport, no backward pass.
"""

import torch

from src.cortical_layer import CorticalLayer


class PredictiveCodingNetwork:
    """Hierarchical predictive coding network.

    Architecture:
        sensory_input (a_0) -> CorticalLayer_0 -> a_1 -> CorticalLayer_1 -> ... -> a_L (output)

    During training:
        1. Clamp: a_0 = input, a_L = target (one-hot label)
        2. Initialize hidden activities via forward pass
        3. Run inference phase (T steps): adjust hidden activities to minimize free energy
        4. Hebbian weight update: adjust synaptic weights using converged activities

    During test:
        Single forward pass (no inference loop needed).

    Two learning rates:
        - inference_rate (gamma): how fast activities adjust (~0.1)
        - learning_rate (alpha): how fast weights change (~0.001)
    """

    def __init__(
        self,
        layer_dims: list[int],
        activation_fn=None,
        inference_rate: float = 0.1,
        learning_rate: float = 0.001,
        num_inference_steps: int = 20,
    ):
        """Initialize the predictive coding hierarchy.

        Args:
            layer_dims: Dimensions of each layer, e.g. [784, 256, 128, 10]
                        First is sensory input dim, last is output/label dim.
            activation_fn: Nonlinearity (default: tanh).
            inference_rate: gamma -- activity update step size.
            learning_rate: alpha -- weight update step size.
            num_inference_steps: T -- number of inference iterations.
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.activation_fn = activation_fn or torch.tanh
        self.inference_rate = inference_rate
        self.learning_rate = learning_rate
        self.num_inference_steps = num_inference_steps

        # Create cortical layers (one between each pair of adjacent areas)
        # CorticalLayer_l connects area l (dim_l) to area l+1 (dim_{l+1})
        self.cortical_layers: list[CorticalLayer] = []
        for ell in range(self.num_layers - 1):
            layer = CorticalLayer(
                input_dim=layer_dims[ell],
                output_dim=layer_dims[ell + 1],
                activation_fn=self.activation_fn,
            )
            self.cortical_layers.append(layer)

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Standard forward pass -- used for test-time classification.

        No inference loop, just propagate through layers like a regular MLP.
        This works because at test time we don't need to minimize free energy --
        we just want the network's best prediction.

        Args:
            sensory_input: Input batch, shape (batch_size, input_dim)

        Returns:
            Output activations, shape (batch_size, output_dim)
        """
        activity = sensory_input
        for layer in self.cortical_layers:
            activity = layer.compute_prediction(activity)
        return activity

    def _initialize_activities(self, sensory_input: torch.Tensor) -> list[torch.Tensor]:
        """Initialize neural activities via a forward pass.

        This sets the initial prediction error to zero at all hidden layers
        (since each activity matches the prediction from below). Only the
        output layer will have non-zero error (target vs. prediction).

        Returns a list of activities [a_0, a_1, ..., a_L] where a_0 = sensory_input.
        """
        activities = [sensory_input]
        current = sensory_input
        for layer in self.cortical_layers:
            current = layer.compute_prediction(current)
            activities.append(current.clone().detach())
        return activities

    def compute_free_energy(self, neural_activities: list[torch.Tensor]) -> torch.Tensor:
        """Compute the total free energy (prediction error) across all cortical layers.

        Free energy is THE objective of predictive coding. Everything the network does --
        inference (adjusting activities) and learning (adjusting weights) -- serves to
        minimize this single quantity.

            E = (1/2) * sum_{l=0}^{L-1} ||epsilon_{l+1}||^2

        where epsilon_{l+1} = a_{l+1} - f(W_l . a_l) is the prediction error at each level.

        Connection to neuroscience: This is the variational free energy from Friston's
        Free Energy Principle. Minimizing it is equivalent to approximate Bayesian
        inference -- the network finds the most probable explanation of its inputs.

        Connection to backprop: At convergence, the gradient of E with respect to
        the weights equals the backprop gradient (Whittington & Bogacz 2017). So
        minimizing this energy with local rules achieves the same result as backprop.

        Args:
            neural_activities: List [a_0, a_1, ..., a_L] of activities at each layer.
                               Each a_l has shape (batch_size, dim_l).

        Returns:
            Scalar free energy (averaged over the batch).
        """
        # TODO(human): Compute the total free energy.
        #
        # For each cortical layer l (0 to L-1):
        #   1. Get the prediction: f(W_l . a_l) using self.cortical_layers[l].compute_prediction(a_l)
        #   2. Get the prediction error: epsilon_{l+1} = a_{l+1} - prediction
        #      using self.cortical_layers[l].compute_prediction_error(a_{l+1}, prediction)
        #   3. Add (1/2) * ||epsilon_{l+1}||^2 to the total energy
        #      The norm is over the feature dimension, averaged over the batch
        #
        # The total energy is the sum across all layers.
        #
        # Implementation hint:
        #   batch_size = neural_activities[0].shape[0]
        #   total_energy = 0.0
        #   for l, layer in enumerate(self.cortical_layers):
        #       prediction = layer.compute_prediction(neural_activities[l])
        #       epsilon = layer.compute_prediction_error(neural_activities[l + 1], prediction)
        #       total_energy += 0.5 * torch.sum(epsilon ** 2) / batch_size
        #   return total_energy
        #
        # Think about: Why is this the right objective? What happens when free energy
        # is zero? (Every layer perfectly predicts the one above -- no surprise.)
        raise NotImplementedError("TODO(human): Implement free energy computation")

    def run_inference_phase(
        self,
        sensory_input: torch.Tensor,
        target: torch.Tensor,
        neural_activities: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Run the inference phase -- perceptual inference.

        This is the HEART of predictive coding. In the brain, this happens
        in ~100-200ms -- neural activities settle into a state that best explains
        the sensory input given the current model (weights).

        Mathematically, we minimize free energy with respect to activities
        (not weights -- that comes later). For each hidden layer l (not input, not output):

            delta_a_l = -gamma * epsilon_l + gamma * W_l^T @ (epsilon_{l+1} * f'(W_l . a_l))

        Two biological pathways:
            - First term (-gamma * epsilon_l): Bottom-up error correction.
              "My own activity differs from what the layer below predicted -> correct myself."
            - Second term (+gamma * W_l^T @ ...): Top-down error propagation.
              "The layer above has a prediction error -> adjust to help reduce it."

        Input (a_0) is clamped to sensory_input -- it doesn't change.
        Output (a_L) is clamped to the target -- we know the answer during training.
        Only hidden layers (a_1 to a_{L-1}) are updated.

        Args:
            sensory_input: Clamped input, shape (batch_size, input_dim)
            target: Clamped target (one-hot), shape (batch_size, output_dim)
            neural_activities: Initial activities [a_0, ..., a_L], modified in-place.

        Returns:
            (updated_activities, energy_trace) -- activities after convergence,
            plus free energy at each inference step (for visualization).
        """
        # TODO(human): Implement the inference loop.
        #
        # 1. Clamp boundaries:
        #    neural_activities[0] = sensory_input  (sensory layer -- fixed)
        #    neural_activities[-1] = target         (output layer -- clamped to label)
        #
        # 2. For T = self.num_inference_steps iterations:
        #    a. Compute ALL prediction errors first:
        #       For each layer l in range(len(self.cortical_layers)):
        #         prediction_l = cortical_layers[l].compute_prediction(neural_activities[l])
        #         epsilon_{l+1} = cortical_layers[l].compute_prediction_error(
        #                           neural_activities[l+1], prediction_l)
        #       Store these epsilons in a list indexed 1..L (epsilon[0] is unused).
        #
        #    b. Update each HIDDEN layer l (from 1 to L-1, NOT 0 or L):
        #       - Bottom-up term: -gamma * epsilon_l
        #         epsilon_l is the prediction error AT layer l, i.e., epsilons[l]
        #         (error between a_l and prediction from layer l-1)
        #
        #       - Top-down term:  +gamma * (epsilon_{l+1} * f'(W_l . a_l)) @ W_l
        #         where:
        #           epsilon_{l+1} = epsilons[l+1] (error at layer above)
        #           pre_act = cortical_layers[l].pre_activation(neural_activities[l])
        #           f_prime = cortical_layers[l].activation_derivative(pre_act)
        #           modulated = epsilons[l+1] * f_prime  (element-wise)
        #           top_down = modulated @ cortical_layers[l].synaptic_weights
        #           (Note: W_l^T @ modulated^T = (modulated @ W_l)^T, so for batch
        #            processing we use modulated @ W_l to get shape (batch, input_dim))
        #
        #       delta_a_l = -gamma * epsilons[l] + gamma * top_down
        #       neural_activities[l] = neural_activities[l] + delta_a_l
        #
        #    c. Track free energy for visualization:
        #       energy_trace.append(self.compute_free_energy(neural_activities).item())
        #
        # 3. Return (neural_activities, energy_trace)
        #
        # Key insight: The input and output are CLAMPED. Only the hidden layers
        # are free to move. The inference phase finds the hidden representation
        # that best explains how the input maps to the output -- this IS perception.
        #
        # Watch for: epsilon_l is the error AT layer l (between a_l and the prediction
        # from layer l-1). Make sure indices are correct -- off-by-one errors here
        # will make the network diverge instead of converge.
        raise NotImplementedError("TODO(human): Implement inference phase")

    def hebbian_weight_update(
        self,
        neural_activities: list[torch.Tensor],
    ) -> None:
        """Update synaptic weights using local Hebbian learning rules.

        After inference converges, each cortical layer updates its weights
        using ONLY local information -- the activity at its level and the
        prediction error from the level above. No global backward pass needed.

            delta_W_l = alpha * (epsilon_{l+1} * f'(W_l . a_l))^T @ a_l

        This is an outer product: result shape (dim_{l+1}, dim_l) matches W_l.

        Neuroscience: "Neurons that fire together wire together" (Hebb, 1949).
        The update is LOCAL -- each synapse adjusts based on the correlation
        between pre-synaptic activity (a_l) and post-synaptic error signal
        (epsilon_{l+1} * f'(...)). No neuron needs to know about distant layers.

        The remarkable result (Whittington & Bogacz 2017): this local rule
        produces the SAME weight gradients as backpropagation, but without
        the weight transport problem (backprop needs W^T, which isn't locally
        available in biological neurons).

        Args:
            neural_activities: Converged activities from inference phase.
                               List [a_0, ..., a_L], each shape (batch_size, dim_l).
        """
        # TODO(human): Implement Hebbian weight updates for all cortical layers.
        #
        # For each layer l in range(len(self.cortical_layers)):
        #   1. Get pre- and post-synaptic activities:
        #      a_l = neural_activities[l]          (pre-synaptic, shape: batch, dim_l)
        #      a_{l+1} = neural_activities[l+1]    (post-synaptic, shape: batch, dim_{l+1})
        #
        #   2. Compute the prediction and error:
        #      prediction = cortical_layers[l].compute_prediction(a_l)
        #      epsilon = cortical_layers[l].compute_prediction_error(a_{l+1}, prediction)
        #
        #   3. Compute the modulated error signal:
        #      pre_act = cortical_layers[l].pre_activation(a_l)
        #      f_prime = cortical_layers[l].activation_derivative(pre_act)
        #      modulated = epsilon * f_prime   (element-wise, shape: batch, dim_{l+1})
        #
        #   4. Compute the weight update (outer product, averaged over batch):
        #      batch_size = a_l.shape[0]
        #      delta_W = self.learning_rate * (modulated.T @ a_l) / batch_size
        #      This gives shape (dim_{l+1}, dim_l) -- matches W_l
        #
        #   5. Apply update:
        #      cortical_layers[l].synaptic_weights += delta_W
        #
        # Think about: Why outer product? It captures the correlation between
        # the error signal (what needs to change) and the input activity
        # (what caused the current prediction). This is pure Hebbian learning.
        raise NotImplementedError("TODO(human): Implement Hebbian weight updates")

    def train_step(
        self,
        sensory_input: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """Complete PCN training step: initialize -> infer -> learn.

        This combines all the pieces into one training iteration:
        1. Initialize activities via forward pass
        2. Run inference phase (clamp input + target, iterate hidden layers)
        3. Apply Hebbian weight updates using converged activities
        4. Return metrics for monitoring

        Args:
            sensory_input: Input batch, shape (batch_size, input_dim)
            target: Target labels as one-hot, shape (batch_size, num_classes)

        Returns:
            Dict with 'free_energy' (final), 'energy_trace' (list), 'accuracy' (float)
        """
        # TODO(human): Wire together the full training step.
        #
        # Steps:
        # 1. Initialize activities: self._initialize_activities(sensory_input)
        # 2. Run inference: self.run_inference_phase(sensory_input, target, activities)
        #    This returns (updated_activities, energy_trace)
        # 3. Update weights: self.hebbian_weight_update(updated_activities)
        # 4. Compute accuracy: compare forward pass prediction with target
        #    - predictions = self.forward(sensory_input)
        #    - predicted_labels = predictions.argmax(dim=1)
        #    - true_labels = target.argmax(dim=1)
        #    - accuracy = (predicted_labels == true_labels).float().mean().item()
        # 5. Return dict with:
        #    - 'free_energy': energy_trace[-1] (final energy after inference)
        #    - 'energy_trace': energy_trace (full list for visualization)
        #    - 'accuracy': accuracy from step 4
        #
        # Note: The forward pass for accuracy is separate from inference.
        # During inference, output is clamped to the target (teacher forcing).
        # The accuracy check uses an unclamped forward pass to see what the
        # network actually predicts -- this is how we measure learning.
        raise NotImplementedError("TODO(human): Implement full training step")

    def to(self, device: torch.device) -> "PredictiveCodingNetwork":
        """Move all weights to a device (CPU/GPU)."""
        for layer in self.cortical_layers:
            layer.synaptic_weights = layer.synaptic_weights.to(device)
        return self
