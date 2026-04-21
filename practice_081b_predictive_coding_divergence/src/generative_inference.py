"""
Generative Mode -- Running the PCN backward to generate images.

A standard backprop MLP is a one-way function: input -> output. It cannot
generate inputs from outputs because it has no generative model.

A PCN IS a generative model. During training, it learned how each cortical
layer predicts the one below. To generate: clamp a LABEL at the top layer,
initialize hidden/sensory layers randomly, and run inference. The network
will adjust all layers to minimize free energy -- producing an image that
the network "believes" corresponds to that label.

The key difference from training inference:
- Training:   a_0 clamped (sensory input), a_L clamped (target)
- Generation: a_0 FREE (to be generated),  a_L clamped (desired label)

Because a_0 has no "layer below," its update has no bottom-up error term --
only the top-down correction from layer 1.

Neuroscience: this is IMAGINATION. Top-down predictions flowing downward
without bottom-up sensory input to constrain them. Mental imagery activates
the same cortical areas as perception -- because it uses the same generative
model running in reverse.

You implement two small pieces: the sensory-layer update for a_0 (the new
dynamic) and the top-level `generate_from_label` orchestrator that drives
inference with a_0 free and a_L clamped.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork


# -- TODO 1 ----------------------------------------------------------------


def _sensory_layer_update(
    network: PredictiveCodingNetwork,
    activities: list[torch.Tensor],
    epsilons: list[torch.Tensor],
    inference_rate: float,
) -> torch.Tensor:
    """New value for `activities[0]` -- the free sensory layer.

    In training, a_0 is clamped (no update). In generation it is free, but
    there is no "layer below" to give it a bottom-up error. Only the
    top-down correction applies:

        delta_a_0 = +gamma * (epsilons[0] * f'(W_0 a_0)) @ W_0
        activities[0] = activities[0] + delta_a_0

    where `epsilons[0]` is the layer-0 prediction error and `W_0` is
    `network.cortical_layers[0].synaptic_weights`. Use the layer's
    `pre_activation` / `activation_derivative` helpers.
    """
    # TODO(human): compute the top-down-only update for activities[0] and
    # return the new tensor.
    raise NotImplementedError("Implement _sensory_layer_update()")


# -- TODO 2 ----------------------------------------------------------------


def generate_from_label(
    network: PredictiveCodingNetwork,
    label: int,
    num_classes: int = 10,
    num_inference_steps: int = 100,
    inference_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate images by clamping a label and running inference downward.

    Returns `(batch_size, input_dim)` -- `activities[0]` after inference,
    optionally clamped to [0, 1] for visualisation.

    Algorithm:
    1. One-hot `target` of shape `(batch_size, num_classes)` with `label` set.
    2. Initialise ALL activities randomly (including `activities[0]`).
    3. Clamp the output: `activities[-1] = target`.
    4. For `num_inference_steps` steps:
        a. Compute every prediction and prediction error
           (same loop as `PredictiveCodingNetwork.run_inference_phase`).
        b. Update hidden layers l = 1..L-2 with the standard two-term rule.
        c. Update `activities[0]` via `_sensory_layer_update(...)`.
        d. Leave `activities[-1]` clamped.
    5. Return `activities[0]` (optionally `.clamp(0.0, 1.0)`).

    Different random initialisations give different variations of the same
    class, because the generative model has many images consistent with
    each label.
    """
    # TODO(human): implement the generative inference loop.
    raise NotImplementedError("Implement generate_from_label()")


# -- Scaffolded: driver for all classes ------------------------------------


def generate_all_classes(
    network: PredictiveCodingNetwork,
    num_classes: int = 10,
    samples_per_class: int = 8,
    num_inference_steps: int = 100,
    device: torch.device = None,
) -> dict[int, torch.Tensor]:
    """Generate samples for all classes. Returns dict mapping label -> generated images."""
    results = {}
    for label in range(num_classes):
        results[label] = generate_from_label(
            network, label, num_classes, num_inference_steps,
            batch_size=samples_per_class, device=device,
        )
    return results
