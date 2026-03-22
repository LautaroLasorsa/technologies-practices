"""
Generative Mode -- Running the PCN backward to generate images.

A standard backprop MLP is a one-way function: input -> output. It cannot
generate inputs from outputs because it has no generative model.

A PCN IS a generative model. During training, it learned how each cortical
layer predicts the one below. To generate: clamp a LABEL at the top layer,
initialize hidden/sensory layers randomly, and run inference. The network
will adjust all layers to minimize free energy -- producing an image that
the network "believes" corresponds to that label.

This is the neuroscience analogue of IMAGINATION: top-down predictions
flowing downward without bottom-up sensory input to constrain them.
Mental imagery activates the same cortical areas as perception -- because
it uses the same generative model running in reverse.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork


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

    The process:
    1. Create one-hot target for the desired label
    2. Initialize sensory layer (a_0) with random noise or zeros
    3. Initialize hidden layers randomly
    4. Clamp a_L = one-hot label (we KNOW what class we want)
    5. Leave a_0 FREE (unlike training, where input is clamped)
    6. Run inference: all layers (including a_0) adjust to minimize free energy
    7. After convergence, a_0 IS the generated image

    The key difference from training inference:
    - Training: a_0 clamped (sensory input), a_L clamped (target)
    - Generation: a_0 FREE (to be generated), a_L clamped (desired label)

    Args:
        network: Trained PredictiveCodingNetwork
        label: Class index to generate (0-9)
        num_classes: Total number of classes (default: 10)
        num_inference_steps: More steps = clearer image (default: 100)
        inference_rate: gamma for generation inference
        batch_size: Number of images to generate (different random inits
                    produce different variations of the same class)
        device: CPU or CUDA

    Returns:
        Generated images, shape (batch_size, input_dim) -- e.g., (16, 784) for MNIST
    """
    # TODO(human): Implement generative inference.
    #
    # Step 1: Create the clamped target (one-hot label)
    #   target = torch.zeros(batch_size, num_classes, device=device)
    #   target[:, label] = 1.0
    #
    # Step 2: Initialize ALL activities (including sensory layer) randomly
    #   activities = []
    #   for dim in network.layer_dims:
    #       activities.append(torch.randn(batch_size, dim, device=device) * 0.1)
    #
    # Step 3: Clamp the OUTPUT layer to the target
    #   activities[-1] = target
    #
    # Step 4: Run inference loop (similar to training, but a_0 is FREE)
    #   For each step in range(num_inference_steps):
    #     a. Compute all prediction errors (same as training inference):
    #        for l in range(len(network.cortical_layers)):
    #            pred = network.cortical_layers[l].compute_prediction(activities[l])
    #            eps = network.cortical_layers[l].compute_prediction_error(activities[l+1], pred)
    #
    #     b. Update ALL layers EXCEPT the output (a_L stays clamped)
    #        This includes a_0 -- the sensory layer is free to change!
    #
    #     c. For layer 0 (sensory): only the top-down term applies
    #        (there's no "layer below layer 0" to produce a bottom-up error)
    #        pre_act_0 = network.cortical_layers[0].pre_activation(activities[0])
    #        f_prime_0 = network.cortical_layers[0].activation_derivative(pre_act_0)
    #        delta_a_0 = +inference_rate * (epsilons[0] * f_prime_0) @ network.cortical_layers[0].synaptic_weights
    #        activities[0] = activities[0] + delta_a_0
    #
    #     d. For hidden layers 1..L-2: same formula as training
    #        delta_a_l = -inference_rate * epsilons[l-1]
    #                    + inference_rate * (epsilons[l] * f'(W_l * a_l)) @ W_l
    #
    #     e. Keep a_L clamped (do NOT update the output layer)
    #
    # Step 5: Return activities[0] -- the generated sensory input
    #   Optionally clamp to [0, 1] range for visualization:
    #   return activities[0].clamp(0.0, 1.0)
    #
    # Think about: Why does this work? The network learned W matrices that
    # predict lower layers from higher ones. By clamping the label and letting
    # everything else adjust, the network "dreams" an image consistent with
    # its learned world model. Different random initializations produce
    # different variations of the same digit/clothing item.
    raise NotImplementedError("TODO(human): Implement generative inference")


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
