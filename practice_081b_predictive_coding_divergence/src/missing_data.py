"""
Missing Data Inference -- Filling in what the brain can't see.

When you see a partially occluded object, your brain fills in the missing
parts using its internal generative model. This is not guessing -- it's
inference: the brain finds the most probable completion given what it CAN see.

A PCN does this naturally: clamp the OBSERVED pixels, leave the MISSING
pixels free, and run inference. The network adjusts the free pixels to
minimize free energy -- finding the completion that best fits the model.

A backprop MLP cannot do this at all. It's a fixed function x -> y.  If
half the input is missing, it can only produce a (likely bad) classification
-- it has no mechanism to reconstruct the input.

Three-way modification of standard inference:
1. Sensory layer a_0 is PARTIALLY clamped via a binary mask
   (observed pixels fixed, missing pixels free)
2. Output layer a_L is FREE -- we want to classify at the same time
3. Hidden layers update normally

The observed pixels constrain the hidden layers from below; the hidden
layers constrain the missing pixels from above. The network finds the joint
state minimizing free energy -- the best joint explanation of what's there
and what class it belongs to.

You implement two small pieces: the masked a_0 update and the top-level
`infer_missing_data` that drives inference. Mask construction is pre-built.
"""

import torch

from src.predictive_coding_network import PredictiveCodingNetwork


# -- TODO 1 ----------------------------------------------------------------


def _masked_sensory_update(
    network: PredictiveCodingNetwork,
    activities: list[torch.Tensor],
    epsilons: list[torch.Tensor],
    partial_input: torch.Tensor,
    mask: torch.Tensor,
    inference_rate: float,
) -> torch.Tensor:
    """New value for `activities[0]` with observed pixels clamped.

    Same top-down-only update as in generative mode (no layer below), but
    only applied to *missing* pixels; observed pixels stay clamped to
    `partial_input`:

        delta_a_0 = +gamma * (epsilons[0] * f'(W_0 a_0)) @ W_0
        new_a_0   = mask * partial_input + (1 - mask) * (activities[0] + delta_a_0)

    `mask` is 1 where pixels are observed, 0 where missing; both tensors
    have shape `(batch, input_dim)`.
    """
    # TODO(human): compute the masked sensory update and return it.
    raise NotImplementedError("Implement _masked_sensory_update()")


# -- TODO 2 ----------------------------------------------------------------


def infer_missing_data(
    network: PredictiveCodingNetwork,
    partial_input: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps: int = 50,
    inference_rate: float = 0.1,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct missing pixels AND classify simultaneously.

    Returns `(reconstructed_input, predicted_labels)` where
    `reconstructed_input` is `activities[0]` after inference and
    `predicted_labels = activities[-1].argmax(dim=1)`.

    Algorithm per step:
    1. Compute all prediction errors (same loop as training inference).
    2. Update hidden layers l = 1..L-2 with the standard two-term rule.
    3. Update `activities[-1]` with ONLY the bottom-up term
       (no layer above it): `a_L += -gamma * epsilons[-1]`.
    4. Update `activities[0]` via `_masked_sensory_update(...)` --
       missing pixels get the top-down correction, observed pixels
       stay clamped to `partial_input`.

    Initialise activities via `network._initialize_activities(partial_input)`.
    """
    # TODO(human): implement the missing-data inference loop.
    raise NotImplementedError("Implement infer_missing_data()")


# -- Scaffolded: mask construction ----------------------------------------


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
