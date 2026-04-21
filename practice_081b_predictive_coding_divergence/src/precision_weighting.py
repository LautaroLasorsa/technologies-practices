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

where pi_l is the precision at layer l. Uniform precision recovers standard
free energy (strict generalization). Zero precision completely ignores that
layer's error (the network "tunes out" noise). Very high precision makes
those errors dominate the energy (selective attention).

You implement two pieces: the weighted energy and the weighted inference
loop. Precision learning (`learn_precisions`) and the noise robustness
experiment harness are pre-built.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork
from src.cortical_layer import CorticalLayer


# -- TODO 1 ----------------------------------------------------------------


def precision_weighted_energy(
    neural_activities: list[torch.Tensor],
    cortical_layers: list[CorticalLayer],
    precisions: list[torch.Tensor],
) -> torch.Tensor:
    """Scalar (1/2) * sum_l (pi_l * epsilon_l^2), averaged over the batch.

    Same loop as standard free energy, but multiply each squared error by
    the per-neuron precision for that layer (`precisions[l]`, shape
    `(dim_{l+1},)`, broadcasts over the batch dim). Clean signal: loop layers,
    `pred = layer.compute_prediction(activities[l])`, compute epsilon, then
    accumulate `0.5 * (precisions[l] * eps**2).sum() / batch_size`.
    """
    # TODO(human): implement precision-weighted free energy.
    raise NotImplementedError("Implement precision_weighted_energy()")


# -- TODO 2 ----------------------------------------------------------------


def precision_weighted_inference(
    network: PredictiveCodingNetwork,
    sensory_input: torch.Tensor,
    target: torch.Tensor,
    precisions: list[torch.Tensor],
    num_inference_steps: int = 20,
    inference_rate: float = 0.1,
) -> tuple[list[torch.Tensor], list[float]]:
    """Run inference with precision-weighted error signals.

    Same structure as `PredictiveCodingNetwork.run_inference_phase`, but
    scale each epsilon by its precision *before* using it in the updates:

        weighted_eps[l] = precisions[l] * epsilon[l]
        delta_a_l = -gamma * weighted_eps[l-1]
                    + gamma * (weighted_eps[l] * f'(W_l a_l)) @ W_l

    Clamp `a_0 = sensory_input` and `a_L = target`; update hidden layers
    l = 1 .. L-2. Track `precision_weighted_energy` (NOT the standard one)
    at each step in `energy_trace`.

    Neuroscience: this is how the brain handles unreliable senses. In a dark
    room, visual precision drops and the brain leans on other modalities.
    In anxious brains, interoceptive precision is too high -- every heartbeat
    feels alarming.
    """
    # TODO(human): implement precision-weighted inference.
    raise NotImplementedError("Implement precision_weighted_inference()")


# -- Scaffolded: precision learning and experiment harness ----------------


def learn_precisions(
    network: PredictiveCodingNetwork,
    neural_activities: list[torch.Tensor],
    precisions: list[torch.Tensor],
    learning_rate: float = 0.01,
) -> list[torch.Tensor]:
    """Update precisions based on observed prediction error variance.

    Precision is high where errors are consistently small (reliable signal)
    and low where errors are large/variable (noisy signal).

    Simple rule: pi_l = 1 / (var(epsilon_l) + eps).  Pre-built -- this is
    an optimization detail, not the core learning objective.
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
