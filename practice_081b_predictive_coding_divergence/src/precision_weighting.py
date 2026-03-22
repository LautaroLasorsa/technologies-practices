"""
Precision Weighting -- The neuroscience of attention.

In Friston's framework, precision is the inverse variance (1/sigma^2) of the
prediction error. High precision means "this error signal is reliable,
pay attention to it." Low precision means "this is noisy, downweight it."

In the brain, precision weighting is the proposed mechanism for ATTENTION:
- Attending to something = increasing the precision of its prediction errors
- Ignoring noise = decreasing the precision of noisy error signals
- Anxiety disorders may involve dysfunctional precision (overweighting errors)

In our PCN, precision weighting modifies the energy function:

    E = (1/2) * sum_l (pi_l * epsilon_l^2)

where pi_l is the precision at layer l. High pi amplifies the error;
low pi suppresses it. This makes the network more robust to noise
because noisy layers can be downweighted.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork
from src.cortical_layer import CorticalLayer


def precision_weighted_energy(
    neural_activities: list[torch.Tensor],
    cortical_layers: list[CorticalLayer],
    precisions: list[torch.Tensor],
) -> torch.Tensor:
    """Compute precision-weighted free energy.

    Standard free energy treats all prediction errors equally.
    Precision weighting lets the network attend to reliable signals
    and ignore noisy ones.

        E = (1/2) * sum_l || sqrt(pi_l) * epsilon_l ||^2
          = (1/2) * sum_l (pi_l * epsilon_l^2).sum()

    Args:
        neural_activities: [a_0, ..., a_L]
        cortical_layers: List of CorticalLayer
        precisions: List of precision tensors, one per layer transition.
                    Each has shape (dim_{l+1},) -- per-neuron precision.
                    Higher = more attention to that neuron's error.

    Returns:
        Scalar precision-weighted free energy (averaged over batch).
    """
    # TODO(human): Compute precision-weighted free energy.
    #
    # Same as standard free energy, but multiply each squared error by precision:
    #
    # total_energy = 0.0
    # batch_size = neural_activities[0].shape[0]
    #
    # For each layer l in range(len(cortical_layers)):
    #   prediction = cortical_layers[l].compute_prediction(neural_activities[l])
    #   epsilon = cortical_layers[l].compute_prediction_error(
    #       neural_activities[l + 1], prediction
    #   )
    #   # Precision broadcasts: (batch, dim) * (dim,) -> (batch, dim)
    #   weighted_error = precisions[l] * epsilon ** 2
    #   total_energy += 0.5 * weighted_error.sum() / batch_size
    #
    # return total_energy
    #
    # Think about: What happens when precision is uniform (all 1s)?
    # You get standard free energy -- precision weighting is a strict
    # generalization. What if precision is 0 for a neuron?
    # Its error is completely ignored -- the network doesn't care what
    # that neuron does. This is how the brain "tunes out" noise.
    #
    # What if precision is very HIGH for certain neurons?
    # Those errors dominate the energy -- the network pays maximum
    # attention to those signals. This is selective attention.
    raise NotImplementedError("TODO(human): Implement precision-weighted energy")


def precision_weighted_inference(
    network: PredictiveCodingNetwork,
    sensory_input: torch.Tensor,
    target: torch.Tensor,
    precisions: list[torch.Tensor],
    num_inference_steps: int = 20,
    inference_rate: float = 0.1,
) -> tuple[list[torch.Tensor], list[float]]:
    """Run inference with precision weighting.

    The inference dynamics are modified to include precision:

        delta_a_l = -gamma * pi_{l-1} * epsilon_{l-1}
                    + gamma * W_l^T * (pi_l * epsilon_l * f'(W_l * a_l))

    Notice: precision scales the error signal BEFORE it's used.
    High precision on a layer = strong correction signal from that layer.
    Low precision = weak signal, almost ignored.

    Args:
        network: PredictiveCodingNetwork
        sensory_input: shape (batch, input_dim)
        target: one-hot, shape (batch, num_classes)
        precisions: per-layer precision tensors, each shape (dim_{l+1},)
        num_inference_steps: T
        inference_rate: gamma

    Returns:
        (activities, energy_trace) after precision-weighted inference
    """
    # TODO(human): Implement precision-weighted inference.
    #
    # Same structure as standard run_inference_phase, but:
    # 1. Multiply each epsilon by its precision before using it:
    #    weighted_eps[l] = precisions[l] * epsilon[l]
    # 2. Use weighted_eps in both bottom-up and top-down terms
    # 3. Track precision-weighted energy (not standard energy)
    #
    # Steps:
    # 1. Initialize activities, clamp input and target:
    #    activities = network._initialize_activities(sensory_input)
    #    activities[0] = sensory_input
    #    activities[-1] = target
    #    energy_trace = []
    #
    # 2. For each step in range(num_inference_steps):
    #    a. Compute predictions and raw errors:
    #       epsilons = []
    #       for l in range(len(network.cortical_layers)):
    #           pred = network.cortical_layers[l].compute_prediction(activities[l])
    #           eps = network.cortical_layers[l].compute_prediction_error(
    #               activities[l + 1], pred
    #           )
    #           epsilons.append(eps)
    #
    #    b. Weight errors by precision:
    #       weighted_eps = [precisions[l] * epsilons[l]
    #                       for l in range(len(epsilons))]
    #
    #    c. Update hidden layers using weighted errors:
    #       for l in range(1, network.num_layers - 1):
    #           bottom_up = -inference_rate * weighted_eps[l - 1]
    #           pre_act = network.cortical_layers[l].pre_activation(activities[l])
    #           f_prime = network.cortical_layers[l].activation_derivative(pre_act)
    #           top_down = inference_rate * (weighted_eps[l] * f_prime) @ network.cortical_layers[l].synaptic_weights
    #           activities[l] = activities[l] + bottom_up + top_down
    #
    #    d. Track precision_weighted_energy:
    #       energy = precision_weighted_energy(activities, network.cortical_layers, precisions)
    #       energy_trace.append(energy.item())
    #
    # 3. Return (activities, energy_trace)
    #
    # Neuroscience: This is literally how the brain handles unreliable
    # senses. In a dark room, visual precision drops and the brain
    # relies more on other modalities (auditory, proprioceptive).
    # In anxious brains, interoceptive (body sensation) precision is
    # cranked too high -- every heartbeat becomes alarming.
    raise NotImplementedError("TODO(human): Implement precision-weighted inference")


def learn_precisions(
    network: PredictiveCodingNetwork,
    neural_activities: list[torch.Tensor],
    precisions: list[torch.Tensor],
    learning_rate: float = 0.01,
) -> list[torch.Tensor]:
    """Update precisions based on observed prediction error variance.

    Precision should be high where errors are consistently small (reliable signal)
    and low where errors are large/variable (noisy signal).

    Simple rule: pi_l = 1 / (var(epsilon_l) + eps)

    Pre-built -- this is an optimization detail, not the core learning objective.
    """
    updated = []
    for l in range(len(network.cortical_layers)):
        pred = network.cortical_layers[l].compute_prediction(neural_activities[l])
        eps = network.cortical_layers[l].compute_prediction_error(neural_activities[l + 1], pred)
        variance = eps.var(dim=0) + 1e-6
        new_precision = 1.0 / variance
        new_precision = new_precision / new_precision.mean()  # Normalize
        updated.append(new_precision.detach())
    return updated


def noise_robustness_experiment(
    network: PredictiveCodingNetwork,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    noise_levels: list[float] = None,
    num_inference_steps: int = 30,
    device: torch.device = None,
) -> dict:
    """Pre-built experiment harness for testing precision weighting robustness.

    Adds Gaussian noise at various levels and compares:
    1. Standard PCN accuracy (forward pass only, no precision)
    2. Precision-weighted PCN accuracy (uses inference with learned precisions)

    Returns results dict for plotting.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    device = device or torch.device("cpu")

    results = {"noise_levels": noise_levels, "pcn_standard": [], "pcn_precision": []}

    # Learn precisions from clean data statistics
    clean_subset = test_images[:64].to(device)
    clean_acts = network._initialize_activities(clean_subset)
    init_precisions = [torch.ones(dim, device=device) for dim in network.layer_dims[1:]]
    precisions = learn_precisions(network, clean_acts, init_precisions)

    for noise_std in noise_levels:
        noisy = test_images + torch.randn_like(test_images) * noise_std
        noisy = noisy.to(device)
        labels_dev = test_labels.to(device)

        # Standard PCN (forward pass only)
        with torch.no_grad():
            preds = network.forward(noisy)
            acc_standard = (preds.argmax(1) == labels_dev).float().mean().item()
            results["pcn_standard"].append(acc_standard)

        # Precision-weighted (uses inference with precisions)
        # Create one-hot targets from forward-pass predictions as weak prior
        one_hot = torch.zeros(noisy.size(0), network.layer_dims[-1], device=device)
        one_hot.scatter_(1, preds.argmax(1).unsqueeze(1), 1.0)

        try:
            pw_acts, _ = precision_weighted_inference(
                network, noisy, one_hot, precisions,
                num_inference_steps=num_inference_steps,
                inference_rate=0.1,
            )
            pw_preds = pw_acts[-1].argmax(1)
            acc_precision = (pw_preds == labels_dev).float().mean().item()
            results["pcn_precision"].append(acc_precision)
        except NotImplementedError:
            results["pcn_precision"].append(acc_standard)

    return results
