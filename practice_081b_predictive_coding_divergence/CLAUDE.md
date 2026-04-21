# Practice 081b: Predictive Coding Networks -- Where PCNs Diverge from Backprop

## Technologies

- **PyTorch** -- Pure tensor operations with manual inference dynamics (no autograd for PCN training)
- **torchvision** -- Fashion-MNIST (FC experiments), CIFAR-10 (convolutional experiments)
- **matplotlib** -- Experiment visualizations (generated images, reconstruction, robustness curves)

## Stack

- Python 3.12+ (uv)
- No Docker needed

## Theoretical Context

### Beyond Classification -- PCN as a Generative Model

A backprop MLP computes `f(x) -> y`. It is a one-way function: input in, prediction out. It has no internal model of the data distribution, no way to run backwards, and no mechanism to answer questions it was not explicitly trained on.

A PCN learns a **joint generative model** `p(x, y, z)` where `x` is sensory input, `y` is the label, and `z` are hidden representations. The key insight: you can **condition on any subset of variables and infer the rest** by clamping the known variables and running inference on the free ones.

This means the **same network, same weights** can:
- **Classify**: clamp `x` (input), free `y` (output) -- standard forward pass
- **Generate**: clamp `y` (label), free `x` (sensory layer) -- imagination
- **Complete**: partially clamp `x` (observed pixels), free missing `x` and `y` -- perception under occlusion
- **Attend**: weight prediction errors by precision -- selective processing

Backprop MLPs cannot do any of these except classification. This is the fundamental divergence explored in this session.

### Generative Mode -- Imagination as Inverse Inference

When you imagine a cat, your visual cortex activates in patterns similar to actually seeing a cat. This is not metaphorical -- fMRI studies show overlapping activation patterns for perception and imagery (Kosslyn, Thompson & Ganis, 2006). The predictive coding explanation: the brain's generative model runs **top-down** without bottom-up sensory constraints.

In a PCN: clamp the top layer to a one-hot label, initialize sensory and hidden layers randomly, and run inference. The network adjusts all free layers to minimize free energy -- producing an input that is **maximally consistent** with the clamped label according to the learned model. Different random initializations produce different variations (like imagining different cats).

A backprop MLP has no generative direction. The weights encode `x -> y`, not `y -> x`. You would need a separate generative model (VAE, GAN, diffusion) to produce images from labels.

### Missing Data -- Perception Under Occlusion

When you see a person partially hidden behind a tree, your brain fills in the occluded body parts. This is called **amodal completion** -- you perceive the complete object even though only parts are visible. The predictive coding explanation: top-down predictions fill in what bottom-up sensory input cannot provide.

In a PCN: clamp the **observed** pixels (mask = 1), leave **missing** pixels free, and leave the output layer free too. Run inference. The network simultaneously:
1. Reconstructs missing pixels using top-down predictions from higher layers
2. Classifies the image based on whatever partial information is available

A backprop MLP fed a partially masked image can only produce a (likely degraded) classification. It has no mechanism to reconstruct the input -- its computation flows strictly from input to output.

### Precision Weighting -- The Neuroscience of Attention

In Friston's framework, **precision** is the inverse variance (`1/sigma^2`) of prediction errors. It controls how much each error signal influences inference and learning:

- **High precision** = "this signal is reliable, pay attention" -- the error strongly drives activity updates
- **Low precision** = "this is noisy, ignore it" -- the error is downweighted

The modified energy function becomes:

    E = (1/2) * sum_l (pi_l * epsilon_l^2)

where `pi_l` is the precision at layer `l`. This is the proposed neural mechanism for **attention** (Feldman & Friston, 2010): attending to a stimulus means increasing the precision of its prediction errors. The gain modulation observed in cortical attention studies maps directly onto precision scaling.

Clinical relevance: **anxiety** may involve chronically high interoceptive precision (overweighting body-state prediction errors, making every heartbeat feel alarming). **Autism** may involve uniformly high precision (overweighting sensory details, making it hard to extract the "gist"). These are active research hypotheses in computational psychiatry.

### Continual Learning -- Why the Brain Does Not Forget

**Catastrophic forgetting** is a well-known failure of backprop networks: training on task B destroys the weights needed for task A. This is dramatically unlike biological learning -- humans learn new skills without forgetting old ones.

PCNs are hypothesized to forget less because:
1. **Local weight updates**: In backprop, the global backward pass adjusts ALL weights with non-zero gradients, even those far from the output. In PCN, weights only update if their local prediction error is non-zero. If task B does not activate certain cortical areas, those areas' weights are untouched.
2. **Prospective configuration** (Salvatori et al., 2023): During inference, PCN hidden layers settle into task-specific representations that minimize interference. The inference phase acts as a "buffer" between the input and the weight update, allowing the network to find representations that accommodate both old and new knowledge.
3. **Biological plausibility**: The brain uses predictive coding-like mechanisms AND does not catastrophically forget. While correlation is not causation, the locality of updates provides a mechanistic explanation.

### Retinotopic Organization -- Why Vision is Convolutional

In the primary visual cortex (V1), each neuron responds to a small patch of the visual field called its **receptive field**. These receptive fields tile the entire visual field in an orderly mapping called **retinotopy** -- neighboring neurons respond to neighboring visual locations.

This IS a convolution: a local filter (receptive field) applied at every spatial position (tiling). Higher cortical areas (V2, V4, IT) have progressively larger receptive fields, corresponding to deeper conv layers with larger effective receptive fields.

A **RetinotopicCorticalLayer** replaces the fully-connected CorticalLayer with convolutions:
- Top-down prediction: `Conv2d` (higher area predicts lower area's activity)
- Prediction error: pointwise subtraction (same as FC)
- Transpose weight multiply: `ConvTranspose2d` (the `W^T` operation in inference)
- Hebbian update: cross-correlation between input activity and error signal

The one non-biological aspect of standard CNNs is **weight sharing** -- all spatial positions use identical filters. Real V1 neurons at different positions are similar but not identical. This is an acceptable approximation that dramatically reduces parameter count.

### References

- Rao, R. P. N. & Ballard, D. H. (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." Nature Neuroscience, 2(1), 79-87.
- Bogacz, R. (2017). "A tutorial on the free-energy framework for modelling perception and learning." Journal of Mathematical Psychology, 76, 198-211.
- Whittington, J. C. R. & Bogacz, R. (2017). "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity." Neural Computation, 29(5), 1229-1262.
- Millidge, B., Tschantz, A. & Buckley, C. L. (2024). "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?" arXiv:2407.04117.
- Kosslyn, S. M., Thompson, W. L. & Ganis, G. (2006). "The Case for Mental Imagery." Oxford University Press.
- Feldman, H. & Friston, K. (2010). "Attention, Uncertainty, and Free-Energy." Frontiers in Human Neuroscience, 4, 215.
- Salvatori, T., Song, Y., Hong, Y., Sha, L., Frieder, S., Xu, Z., Bogacz, R. & Lukasiewicz, T. (2023). "Brain-Inspired Computational Intelligence via Predictive Coding." arXiv:2308.07870.
- Friston, K. (2005). "A theory of cortical responses." Philosophical Transactions of the Royal Society B, 360(1456), 815-836.

## Description

Explore what PCNs can do that backprop MLPs fundamentally cannot. While Session A showed that PCNs approximate backprop gradients using local rules, Session B reveals the **unique capabilities** that emerge from the PCN's generative architecture:

1. **Generative mode** -- Run the network backward to "imagine" images from labels
2. **Missing data inference** -- Fill in occluded pixels using top-down predictions
3. **Precision weighting** -- Implement the neuroscience of attention as error modulation
4. **Continual learning** -- Measure catastrophic forgetting and compare PCN vs backprop
5. **Convolutional PCN** -- Build retinotopic (conv) cortical layers for spatial data

Each phase uses the SAME trained PCN weights in a different inference mode -- demonstrating that a generative model is fundamentally more versatile than a discriminative one.

## Instructions

### Phase 1: Setup (~5 min)

1. Install Python dependencies: `uv sync`
2. Sanity check your environment: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
3. All exercises use the PCN/CorticalLayer from Session A (pre-built here in `src/predictive_coding_network.py` and `src/cortical_layer.py`) -- no need to re-implement them.

### Phase 2: Generative Mode (~20 min)

1. Open `src/generative_inference.py`.
2. **TODO 1 -- `_sensory_layer_update()`**: Compute the new value for `activities[0]` using only the top-down term (there is no layer below a_0 to supply a bottom-up error). This is the piece that distinguishes generative inference from training inference.
3. **TODO 2 -- `generate_from_label()`**: Orchestrate the loop. Initialise all activities randomly, clamp `activities[-1] = one_hot(label)`, then for each step compute errors, update hidden layers l = 1..L-2 with the standard rule, and update `activities[0]` via TODO 1. Keep `activities[-1]` clamped.
4. Run: `uv run python src/experiments.py --experiment generative`
5. Inspect `outputs/generated_images.png` -- different random inits should produce different variations of the same class. Key question: what would change in your loop if `activations[-1]` was also free?

### Phase 3: Missing Data Inference (~25 min)

1. Open `src/missing_data.py`.
2. **TODO 1 -- `_masked_sensory_update()`**: Same top-down-only update as generative mode for `activities[0]`, but apply it only to missing pixels. Observed pixels stay clamped to `partial_input` via `mask * partial_input + (1 - mask) * (...)`.
3. **TODO 2 -- `infer_missing_data()`**: Drive inference with three modifications vs training: hidden layers update as usual, `activities[-1]` is FREE (bottom-up only), and `activities[0]` uses TODO 1. Return `(reconstructed, predicted_labels)`.
4. Run: `uv run python src/experiments.py --experiment missing-data`
5. Inspect `outputs/missing_data.png`. Key question: why does the classification stay reasonable even under heavy occlusion?

### Phase 4: Precision Weighting (~25 min)

1. Open `src/precision_weighting.py`.
2. **TODO 1 -- `precision_weighted_energy()`**: Same loop as standard free energy, but multiply each squared error by the per-neuron precision before summing. Uniform precision must recover the standard energy.
3. **TODO 2 -- `precision_weighted_inference()`**: Copy the structure of `PredictiveCodingNetwork.run_inference_phase`, but multiply each epsilon by `precisions[l]` before plugging it into the bottom-up and top-down update terms. Track the precision-weighted energy (not the standard one) each step.
4. Run: `uv run python src/experiments.py --experiment precision`
5. Inspect `outputs/precision_robustness.png`. Key question: which noise level shows the biggest gap between precision-weighted and standard PCN, and why?

### Phase 5: Continual Learning (~20 min)

1. Open `src/continual_learning.py`.
2. **TODO -- `train_pcn_on_loader()`**: For each epoch iterate the loader, flatten images, build a one-hot target via `create_one_hot`, and call `pcn.train_step(...)`. This is the only "hot loop" in the experiment -- the orchestrator, evaluation, and forgetting metric are scaffolded.
3. Run: `uv run python src/experiments.py --experiment continual`
4. Inspect `outputs/continual_learning.png`. Key question: does your PCN really forget less than the backprop MLP, and can you explain why from the local-Hebbian perspective?

### Phase 6: Retinotopic Cortical Layers (~30 min)

1. Open `src/retinotopic_layer.py`.
2. **TODO 1 -- `compute_prediction()`**: Conv analogue of `f(a @ W^T)`. Use `F.conv2d(activity, self.synaptic_weights, stride=..., padding=...)` then apply `self.activation_fn`.
3. **TODO 2 -- `compute_prediction_error()`**: Pointwise `actual - predicted` on 4D tensors (identical to the FC case).
4. **TODO 3 -- `hebbian_update()`**: Conv analogue of the outer-product weight update -- cross-correlation between input activity and error signal. A loop-over-batch implementation using `F.conv2d` with rearranged dims is clear enough; vectorising is optional.
5. Run: `uv run python src/experiments.py --experiment retinotopic`
6. Inspect `outputs/retinotopic.png` and the printed output shapes. Key question: why does `stride=2` correspond to higher cortical areas having lower spatial resolution?

## Motivation

Session A established that PCNs match backprop's classification accuracy using biologically plausible local learning. Session B demonstrates **why that matters** -- the generative architecture unlocks capabilities that discriminative models simply do not have:

- **Generative AI context**: Understanding generative models from first principles (not just "prompt engineering")
- **Robustness**: Missing data and noise handling without architectural modifications
- **Continual learning**: A major open problem in ML that PCNs address structurally
- **Neuroscience grounding**: Mental imagery, attention, and amodal completion all emerge naturally from the same framework
- **Beyond classification**: Modern AI systems need models that can generate, complete, and reason -- not just classify

## Commands

All commands are run from `practice_081b_predictive_coding_divergence/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (torch, torchvision, matplotlib) |
| `uv run pytest tests/ -v` | Run all tests across every phase |

### Phase 2: Generative Mode

| Command | Description |
|---------|-------------|
| `uv run pytest tests/test_generative.py -v` | Verify `_sensory_layer_update` and `generate_from_label` |
| `uv run python src/experiments.py --experiment generative` | Train/load PCN and generate images for every class -> `outputs/generated_images.png` |

### Phase 3: Missing Data Inference

| Command | Description |
|---------|-------------|
| `uv run pytest tests/test_missing_data.py -v` | Verify `_masked_sensory_update` and `infer_missing_data` |
| `uv run python src/experiments.py --experiment missing-data` | Reconstruct masked Fashion-MNIST images -> `outputs/missing_data.png` |

### Phase 4: Precision Weighting

| Command | Description |
|---------|-------------|
| `uv run pytest tests/test_precision.py -v` | Verify `precision_weighted_energy` and `precision_weighted_inference` |
| `uv run python src/experiments.py --experiment precision` | Noise-robustness sweep (standard vs precision-weighted) -> `outputs/precision_robustness.png` |

### Phase 5: Continual Learning

| Command | Description |
|---------|-------------|
| `uv run pytest tests/test_continual_learning.py -v` | Verify `train_pcn_on_loader` via `continual_learning_experiment` |
| `uv run python src/experiments.py --experiment continual` | Task1 -> Task2 comparison of PCN vs backprop MLP -> `outputs/continual_learning.png` |

### Phase 6: Retinotopic Cortical Layers

| Command | Description |
|---------|-------------|
| `uv run pytest tests/test_retinotopic.py -v` | Verify `compute_prediction`, `compute_prediction_error`, `hebbian_update` |
| `uv run python src/experiments.py --experiment retinotopic` | Smoke-test conv PCN forward/error/Hebbian on CIFAR-10 shapes -> `outputs/retinotopic.png` |

### All Experiments

| Command | Description |
|---------|-------------|
| `uv run python src/experiments.py --experiment all` | Run every experiment end-to-end |

## State

`not-started`
