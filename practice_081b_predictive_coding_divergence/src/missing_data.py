"""
Missing Data Inference -- Filling in what the brain can't see.

When you see a partially occluded object, your brain fills in the missing
parts using its internal generative model. This is not guessing -- it's
inference: the brain finds the most probable completion given what it CAN see.

A PCN does this naturally: clamp the OBSERVED pixels, leave the MISSING
pixels free, and run inference. The network adjusts the free pixels to
minimize free energy -- finding the completion that best fits the model.

A backprop MLP cannot do this at all. It's a fixed function x -> y.
If half the input is missing, it can only produce a (likely bad) classification.
It has no mechanism to reconstruct the input.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork


def infer_missing_data(
    network: PredictiveCodingNetwork,
    partial_input: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps: int = 50,
    inference_rate: float = 0.1,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer missing pixels using the PCN's generative model.

    The mask indicates which pixels are OBSERVED (1) and which are MISSING (0).
    Observed pixels are clamped; missing pixels are free to be inferred.

    Unlike training:
    - a_0 is PARTIALLY clamped (observed pixels fixed, missing pixels free)
    - a_L is NOT clamped (we don't know the label -- we want to infer it too!)
    - Hidden layers are free as usual

    The network simultaneously:
    1. Reconstructs the missing pixels (adjusts free parts of a_0)
    2. Classifies the image (adjusts a_L toward a class)

    Args:
        network: Trained PredictiveCodingNetwork
        partial_input: Input with missing regions (e.g., zeros), shape (batch, input_dim)
        mask: Binary mask, 1 = observed, 0 = missing, shape (batch, input_dim)
        num_inference_steps: How many inference steps to run
        inference_rate: gamma for inference
        device: CPU or CUDA

    Returns:
        (reconstructed_input, predicted_labels):
        - reconstructed_input: Full image with missing pixels filled in, shape (batch, input_dim)
        - predicted_labels: Classification from a_L, shape (batch,)
    """
    # TODO(human): Implement missing data inference.
    #
    # Step 1: Initialize activities via forward pass using the partial input
    #   activities = network._initialize_activities(partial_input)
    #
    # Step 2: Run inference loop
    #   For each step in range(num_inference_steps):
    #     a. Compute all prediction errors (same as always):
    #        epsilons = []
    #        for l in range(len(network.cortical_layers)):
    #            pred = network.cortical_layers[l].compute_prediction(activities[l])
    #            eps = network.cortical_layers[l].compute_prediction_error(activities[l+1], pred)
    #            epsilons.append(eps)
    #
    #     b. Update hidden layers (1 to L-2) with the standard formula:
    #        delta_a_l = -inference_rate * epsilons[l-1]
    #                    + inference_rate * (epsilons[l] * f'(W_l * a_l)) @ W_l
    #
    #     c. Update the OUTPUT layer (a_L) -- it's FREE in this mode:
    #        delta_a_L = -inference_rate * epsilons[-1]
    #        activities[-1] = activities[-1] + delta_a_L
    #        (Only the bottom-up term -- there's no "layer above L")
    #
    #     d. Update the SENSORY layer (a_0) -- PARTIALLY:
    #        Compute the top-down correction for layer 0:
    #        pre_act_0 = network.cortical_layers[0].pre_activation(activities[0])
    #        f_prime_0 = network.cortical_layers[0].activation_derivative(pre_act_0)
    #        delta_a_0 = inference_rate * (epsilons[0] * f_prime_0) @ network.cortical_layers[0].synaptic_weights
    #        Apply ONLY to missing pixels:
    #        activities[0] = mask * partial_input + (1 - mask) * (activities[0] + delta_a_0)
    #        Observed pixels stay clamped to partial_input, missing pixels get updated.
    #
    # Step 3: Extract results
    #   reconstructed = activities[0]
    #   predicted_labels = activities[-1].argmax(dim=1)
    #   return (reconstructed, predicted_labels)
    #
    # Think about: This is simultaneous top-down and bottom-up inference.
    # The observed pixels constrain the hidden layers from below. The hidden
    # layers constrain the missing pixels from above. The network finds the
    # state that minimizes TOTAL free energy -- the best joint explanation.
    raise NotImplementedError("TODO(human): Implement missing data inference")


def create_masks(input_dim: int, mask_type: str = "top_half", image_size: int = 28) -> torch.Tensor:
    """Create standard occlusion masks for testing.

    Pre-built utility -- not a learning objective.

    Args:
        input_dim: Total number of pixels (e.g., 784 for 28x28)
        mask_type: One of "top_half", "bottom_half", "left_half", "random_50", "random_75"
        image_size: Side length of the square image

    Returns:
        Binary mask tensor of shape (input_dim,). 1 = observed, 0 = missing.
    """
    mask = torch.ones(input_dim)
    pixels = image_size * image_size

    if mask_type == "top_half":
        mask[:pixels // 2] = 0.0
    elif mask_type == "bottom_half":
        mask[pixels // 2:] = 0.0
    elif mask_type == "left_half":
        mask_2d = torch.ones(image_size, image_size)
        mask_2d[:, :image_size // 2] = 0.0
        mask = mask_2d.flatten()
    elif mask_type == "random_50":
        indices = torch.randperm(input_dim)[:input_dim // 2]
        mask[indices] = 0.0
    elif mask_type == "random_75":
        indices = torch.randperm(input_dim)[:3 * input_dim // 4]
        mask[indices] = 0.0

    return mask
