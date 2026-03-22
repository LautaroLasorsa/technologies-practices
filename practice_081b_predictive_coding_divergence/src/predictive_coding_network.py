"""
PredictiveCodingNetwork -- Completed implementation from Session A.

In neuroscience, the cortex is organized as a hierarchy of cortical areas
(V1 -> V2 -> V4 -> IT -> PFC). Each area generates predictions of the area below
and receives prediction errors from below. The ENTIRE system minimizes a single
objective: free energy (total prediction error across all levels).

This class orchestrates:
1. Inference phase: iterate neural activities to minimize free energy (perception)
2. Weight updates: local Hebbian learning after inference converges (learning)
3. Forward pass: standard feedforward for test-time classification

This is provided pre-built so Session B is standalone. The user implemented
these methods in Practice 081a.
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
    """

    def __init__(
        self,
        layer_dims: list[int],
        activation_fn=None,
        inference_rate: float = 0.1,
        learning_rate: float = 0.001,
        num_inference_steps: int = 20,
    ):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.activation_fn = activation_fn or torch.tanh
        self.inference_rate = inference_rate
        self.learning_rate = learning_rate
        self.num_inference_steps = num_inference_steps

        self.cortical_layers: list[CorticalLayer] = []
        for ell in range(self.num_layers - 1):
            layer = CorticalLayer(
                input_dim=layer_dims[ell],
                output_dim=layer_dims[ell + 1],
                activation_fn=self.activation_fn,
            )
            self.cortical_layers.append(layer)

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Standard forward pass -- used for test-time classification."""
        activity = sensory_input
        for layer in self.cortical_layers:
            activity = layer.compute_prediction(activity)
        return activity

    def _initialize_activities(self, sensory_input: torch.Tensor) -> list[torch.Tensor]:
        """Initialize neural activities via a forward pass."""
        activities = [sensory_input]
        current = sensory_input
        for layer in self.cortical_layers:
            current = layer.compute_prediction(current)
            activities.append(current.clone().detach())
        return activities

    def compute_free_energy(self, neural_activities: list[torch.Tensor]) -> torch.Tensor:
        """Compute total free energy: E = (1/2) * sum ||epsilon||^2 across layers."""
        total_energy = torch.tensor(0.0, device=neural_activities[0].device)
        batch_size = neural_activities[0].shape[0]
        for l in range(len(self.cortical_layers)):
            prediction = self.cortical_layers[l].compute_prediction(neural_activities[l])
            epsilon = self.cortical_layers[l].compute_prediction_error(neural_activities[l + 1], prediction)
            total_energy = total_energy + 0.5 * torch.sum(epsilon ** 2) / batch_size
        return total_energy

    def run_inference_phase(
        self,
        sensory_input: torch.Tensor,
        target: torch.Tensor,
        neural_activities: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Run the inference phase -- perceptual inference.

        Clamp a_0 = input, a_L = target, iterate hidden layers to minimize free energy.
        """
        neural_activities[0] = sensory_input
        neural_activities[-1] = target
        energy_trace = []

        for step in range(self.num_inference_steps):
            # Compute all prediction errors
            predictions = []
            epsilons = []
            for l in range(len(self.cortical_layers)):
                pred = self.cortical_layers[l].compute_prediction(neural_activities[l])
                eps = self.cortical_layers[l].compute_prediction_error(neural_activities[l + 1], pred)
                predictions.append(pred)
                epsilons.append(eps)

            # Update hidden layers only (not input or output)
            for l in range(1, self.num_layers - 1):
                # Bottom-up: -gamma * epsilon at this layer
                # epsilons[l-1] is the prediction error for neural_activities[l]
                # (error between a_l and the prediction from layer l-1)
                bottom_up = -self.inference_rate * epsilons[l - 1]

                # Top-down: +gamma * W_l^T @ (epsilon_{l+1} * f'(W_l * a_l))
                pre_act = self.cortical_layers[l].pre_activation(neural_activities[l])
                f_prime = self.cortical_layers[l].activation_derivative(pre_act)
                top_down = self.inference_rate * (epsilons[l] * f_prime) @ self.cortical_layers[l].synaptic_weights

                neural_activities[l] = neural_activities[l] + bottom_up + top_down

            energy_trace.append(self.compute_free_energy(neural_activities).item())

        return neural_activities, energy_trace

    def hebbian_weight_update(self, neural_activities: list[torch.Tensor]) -> None:
        """Update synaptic weights using local Hebbian learning rules."""
        batch_size = neural_activities[0].shape[0]
        for l in range(len(self.cortical_layers)):
            a_l = neural_activities[l]
            a_next = neural_activities[l + 1]
            prediction = self.cortical_layers[l].compute_prediction(a_l)
            epsilon = self.cortical_layers[l].compute_prediction_error(a_next, prediction)
            pre_act = self.cortical_layers[l].pre_activation(a_l)
            f_prime = self.cortical_layers[l].activation_derivative(pre_act)
            modulated = epsilon * f_prime
            delta_w = self.learning_rate * (modulated.T @ a_l) / batch_size
            self.cortical_layers[l].synaptic_weights += delta_w

    def train_step(
        self,
        sensory_input: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """Complete PCN training step: initialize -> infer -> learn."""
        activities = self._initialize_activities(sensory_input)
        activities, energy_trace = self.run_inference_phase(sensory_input, target, activities)
        self.hebbian_weight_update(activities)

        with torch.no_grad():
            predictions = self.forward(sensory_input)
            predicted_labels = predictions.argmax(dim=1)
            true_labels = target.argmax(dim=1)
            accuracy = (predicted_labels == true_labels).float().mean().item()

        return {
            "free_energy": energy_trace[-1] if energy_trace else 0.0,
            "energy_trace": energy_trace,
            "accuracy": accuracy,
        }

    def to(self, device: torch.device) -> "PredictiveCodingNetwork":
        """Move all weights to a device (CPU/GPU)."""
        for layer in self.cortical_layers:
            layer.synaptic_weights = layer.synaptic_weights.to(device)
        return self
