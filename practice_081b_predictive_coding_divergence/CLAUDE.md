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

### Phase 1: Generative Mode (~20 min)

**File:** `src/generative_inference.py`

**Concepts:** A PCN is a generative model. By clamping the output layer to a desired label and leaving the sensory layer free, inference "dreams" an image. This is impossible with a standard MLP -- it has no generative direction. You will implement the modified inference loop where layer 0 is free (only receives top-down corrections, no bottom-up errors since there is no layer below it).

**TODO(human) tasks:**
- Implement `generate_from_label()` -- Initialize activities randomly, clamp the output to a one-hot label, run inference with the sensory layer free. The key difference from training inference: layer 0 is updated by top-down signals only (no bottom-up term).

### Phase 2: Missing Data Inference (~25 min)

**File:** `src/missing_data.py`

**Concepts:** When pixels are partially occluded, the PCN can simultaneously reconstruct missing pixels AND classify the image -- using inference with partial clamping. Observed pixels are clamped in the sensory layer; missing pixels and the output layer are free. This is amodal completion -- the neural mechanism behind seeing "through" occlusion.

**TODO(human) tasks:**
- Implement `infer_missing_data()` -- Three-way modification of standard inference: (1) sensory layer is partially clamped via mask, (2) output layer is free for classification, (3) hidden layers update normally. The mask determines which pixels are observed (fixed) vs missing (free to be inferred).

### Phase 3: Precision Weighting (~25 min)

**File:** `src/precision_weighting.py`

**Concepts:** Precision (inverse variance) modulates how strongly each prediction error drives inference. This is the proposed neural mechanism for attention. High precision on a layer means its errors strongly influence updates; low precision means its errors are downweighted. The energy function becomes `E = (1/2) * sum(pi * epsilon^2)`.

**TODO(human) tasks:**
- Implement `precision_weighted_energy()` -- Standard energy with per-neuron precision scaling. Uniform precision recovers standard free energy.
- Implement `precision_weighted_inference()` -- Modified inference dynamics where each epsilon is scaled by its precision before being used in the update equations.

### Phase 4: Continual Learning (~20 min)

**File:** `src/continual_learning.py`

**Concepts:** Train on task 1 (Fashion-MNIST classes 0-4), then task 2 (classes 5-9), measure task 1 retention. PCNs are hypothesized to forget less because local Hebbian updates only affect synapses with non-zero prediction errors -- unused pathways are preserved. Compare against backprop MLP on the same protocol.

**TODO(human) tasks:**
- Implement `continual_learning_experiment()` -- Train on task 1, measure baseline, train on task 2 without resetting, measure both tasks, compute forgetting metric. The experiment runner handles the backprop comparison automatically.

### Phase 5: Retinotopic Cortical Layers (~30 min)

**File:** `src/retinotopic_layer.py`

**Concepts:** Replace fully-connected layers with convolutions, mirroring the retinotopic organization of the visual cortex. V1 neurons have local receptive fields (= conv filters). Top-down prediction uses `Conv2d`, the transpose weight operation uses `ConvTranspose2d`, and Hebbian updates use cross-correlation between input activity and error signal.

**TODO(human) tasks:**
- Implement `compute_prediction()` -- Conv2d followed by activation function (the conv analogue of `f(a @ W^T)`)
- Implement `compute_prediction_error()` -- Pointwise subtraction (same as FC, but on 4D tensors)
- Implement `hebbian_update()` -- Cross-correlation between input activity and modulated error signal (the conv analogue of the outer product weight update)

## Motivation

Session A established that PCNs match backprop's classification accuracy using biologically plausible local learning. Session B demonstrates **why that matters** -- the generative architecture unlocks capabilities that discriminative models simply do not have:

- **Generative AI context**: Understanding generative models from first principles (not just "prompt engineering")
- **Robustness**: Missing data and noise handling without architectural modifications
- **Continual learning**: A major open problem in ML that PCNs address structurally
- **Neuroscience grounding**: Mental imagery, attention, and amodal completion all emerge naturally from the same framework
- **Beyond classification**: Modern AI systems need models that can generate, complete, and reason -- not just classify

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies |
| All | `uv run pytest tests/ -v` | Run all tests |
| Phase 1 | `uv run pytest tests/test_generative.py -v` | Test generative mode |
| Phase 1 | `uv run python src/experiments.py --experiment generative` | Generate images from labels |
| Phase 2 | `uv run pytest tests/test_missing_data.py -v` | Test missing data inference |
| Phase 2 | `uv run python src/experiments.py --experiment missing-data` | Missing data reconstruction |
| Phase 3 | `uv run pytest tests/test_precision.py -v` | Test precision weighting |
| Phase 3 | `uv run python src/experiments.py --experiment precision` | Noise robustness experiment |
| Phase 4 | `uv run pytest tests/test_continual_learning.py -v` | Test continual learning |
| Phase 4 | `uv run python src/experiments.py --experiment continual` | Continual learning comparison |
| Phase 5 | `uv run pytest tests/test_retinotopic.py -v` | Test convolutional layers |
| Phase 5 | `uv run python src/experiments.py --experiment retinotopic` | Conv PCN on CIFAR-10 |

## State

`not-started`
