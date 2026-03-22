# Practice 081a: Predictive Coding Networks -- Foundations & Backprop Comparison

## Technologies

- **PyTorch** -- Pure tensor operations, NO autograd for PCN (manual local Hebbian weight updates)
- **torchvision** -- MNIST, Fashion-MNIST, KMNIST datasets
- **matplotlib** -- Comparison visualizations (accuracy curves, energy traces, benchmark bar charts)

## Stack

- Python 3.12+ (uv)
- No Docker needed

## Theoretical Context

### The Problem: How Does the Brain Learn?

Standard deep learning uses **backpropagation** -- a global backward pass through the entire network, computing gradients via the chain rule from output to input. Backprop is mathematically elegant and computationally efficient, but it has three fundamental **biological implausibility** problems:

1. **Weight transport problem**: Backprop requires the transpose of the forward weights (W^T) to propagate errors backward. But biological neurons have no mechanism to "know" the weights of synapses in other neurons. Each synapse is a separate physical structure -- there is no shared weight matrix that can be transposed.

2. **Non-local error signals**: In backprop, the error signal at each layer depends on ALL layers above it (via the chain rule). Biological neurons only have access to LOCAL information -- their own activity, their immediate inputs, and modulatory signals from nearby neurons.

3. **Sequential backward pass**: Backprop requires a distinct backward phase where computation flows in reverse through the network. The brain has no such phase -- feedforward and feedback connections are active simultaneously and continuously.

**Predictive Coding** offers an alternative: a biologically plausible learning algorithm that uses only **local** information at each layer, requires **no weight transport**, and achieves the **same weight gradients** as backpropagation.

### Neuroscience Foundation

#### Rao & Ballard (1999): Hierarchical Predictive Coding in Visual Cortex

The seminal paper that started computational predictive coding. Rao and Ballard proposed that the visual cortex operates as a hierarchical prediction machine:

- **Top-down connections** (from higher to lower cortical areas) carry **predictions** of what the lower area should be seeing
- **Bottom-up connections** (from lower to higher areas) carry **prediction errors** -- the mismatch between the prediction and actual activity
- Only the **surprise** (prediction error) propagates upward, not the raw sensory data

Their model spontaneously developed **Gabor-like receptive fields** (edge detectors matching V1 neurons) when trained on natural images -- without being told to learn edges. The prediction errors also showed **end-stopping** effects matching known V1 physiology. This was the first demonstration that predictive coding could explain actual neural response properties.

**Source**: Rao, R. P. N., & Ballard, D. H. (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." Nature Neuroscience, 2(1), 79-87.

#### Cortical Circuit Anatomy (Bastos et al., 2012)

Bastos et al. mapped the predictive coding framework onto the actual anatomy of cortical circuits:

- **Superficial pyramidal cells** (layers 2/3) = **prediction error neurons**. They project feedforward (upward) to the next cortical area. Their activity represents the mismatch between actual input and top-down prediction.
- **Deep pyramidal cells** (layers 5/6) = **prediction neurons**. They project feedback (downward) to the cortical area below. Their activity represents the brain's best guess of what that area should look like.
- **Precision** (inverse variance of prediction errors) acts as **gain control** on error signals. High precision = "trust this error signal, it's reliable." This maps directly to **attention** -- attending to a stimulus increases the precision (gain) of its prediction errors, giving them more influence on perception and learning.

**Source**: Bastos, A. M., et al. (2012). "Canonical microcircuits for predictive coding." Neuron, 76(4), 695-711.

#### Karl Friston's Free Energy Principle (2003-2010)

Karl Friston unified predictive coding into a grand theory: ALL living systems minimize **variational free energy**. Free energy is an upper bound on surprise (negative log-evidence). Minimizing it is equivalent to:

- **Perception**: Update internal beliefs (neural activities) to reduce prediction error -- "explain away the input"
- **Action**: Change the world to match predictions -- "make your predictions come true"
- **Learning**: Update model parameters (synaptic weights) to make better predictions in the future

This "Free Energy Principle" provides a single objective function that unifies perception, action, and learning. Predictive coding networks are the **algorithmic instantiation** of this principle for hierarchical neural systems.

**Source**: Friston, K. (2009). "The free-energy principle: a rough guide to the brain?" Trends in Cognitive Sciences, 13(7), 293-301. (PMC2666703)

#### The Bayesian Brain Hypothesis

The brain is a **hierarchical Bayesian inference machine**. Each cortical area maintains a probability distribution over what it thinks the world looks like. Top-down predictions are the current best estimate (the prior). Bottom-up prediction errors are the data (the likelihood). Inference (adjusting activities) performs **approximate Bayesian inference** -- finding the posterior distribution that best explains the sensory data given the model's prior beliefs.

PCN is the algorithmic realization of this hypothesis: the inference phase performs variational inference, and the free energy is the variational bound being optimized.

### The Predictive Coding Algorithm

#### The Energy Function

The network has L+1 layers with activities a_0, a_1, ..., a_L and L sets of weights W_0, W_1, ..., W_{L-1}. Each W_l maps from layer l to layer l+1.

The **free energy** (total prediction error) is:

    E(a, W) = (1/2) * sum_{l=0}^{L-1} || epsilon_{l+1} ||^2

where the **prediction error** at layer l+1 is:

    epsilon_{l+1} = a_{l+1} - f(W_l . a_l)

Here f is the activation function (tanh by default) and f(W_l . a_l) is the **top-down prediction** of layer l+1 made by layer l.

#### Inference Dynamics: Two Biological Pathways

During the inference phase (perception), hidden activities are updated to minimize free energy. The update rule for each hidden layer l is:

    delta_a_l = -gamma * epsilon_l + gamma * W_l^T @ (epsilon_{l+1} * f'(W_l . a_l))

This has two terms, each corresponding to a distinct biological pathway:

1. **Bottom-up error correction** (-gamma * epsilon_l): "My own activity differs from what the layer below predicted for me. Correct myself to reduce my own error."

2. **Top-down error propagation** (+gamma * W_l^T @ (epsilon_{l+1} * f'(...))): "The layer above has a prediction error that I caused. Adjust my activity to help reduce THEIR error."

The input layer (a_0 = sensory input) and output layer (a_L = target) are **clamped** during training. Only hidden layers (a_1 through a_{L-1}) are free to change.

#### Local Weight Update: Hebbian Plasticity

After inference converges, weights are updated using a **purely local** learning rule:

    delta_W_l = alpha * (epsilon_{l+1} * f'(W_l . a_l))^T @ a_l

This is an **outer product** between the modulated error signal at layer l+1 and the activity at layer l. Each weight W_l[i,j] adjusts based only on:
- The prediction error at the neuron above (epsilon_{l+1}[i])
- The activation derivative (f'(...)[i])
- The activity of the neuron below (a_l[j])

No neuron needs to know about distant layers. This is **pure Hebbian learning**: "neurons that fire together wire together."

### Key Differences from Backpropagation

| Property | Backpropagation | Predictive Coding |
|----------|----------------|-------------------|
| **Locality** | Global -- each gradient depends on all layers above via chain rule | Local -- each weight update uses only adjacent-layer information |
| **Information flow** | Unidirectional: forward pass, then separate backward pass | Bidirectional: top-down predictions and bottom-up errors simultaneously |
| **Learning phases** | Two phases: forward (compute activations), backward (compute gradients) | Two phases: inference (settle activities), learning (update weights) |
| **Parallelism** | Backward pass is sequential (layer by layer from output to input) | Weight updates at all layers can happen in parallel (all local) |
| **Weight transport** | Requires W^T in backward pass (biologically implausible) | No weight transport needed -- errors propagate via forward weights |

### The Whittington-Bogacz Equivalence

The remarkable result from Whittington & Bogacz (2017): under the **fixed-prediction assumption** (inference converges before weights update), the PC weight updates **converge to the exact backprop gradients**.

This means:
- PCN and backprop MLP should achieve the **same accuracy** on the same task
- The difference is **HOW** they compute gradients, not **WHERE** they converge
- PCN provides a biologically plausible mechanism that reproduces backprop's mathematical result

The proof works by showing that at inference convergence, the prediction errors at each layer equal the error signals that backprop would compute, and the local weight update rule is equivalent to the gradient descent step.

**Source**: Whittington, J. C. R., & Bogacz, R. (2017). "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity." Neural Computation, 29(5), 1229-1262.

### Implementation Notes

- **Activation function**: Use **tanh** (NOT ReLU). ReLU causes energy pathologies because its derivative is zero for negative inputs, creating dead zones where inference cannot make progress. tanh has smooth, non-zero derivatives everywhere.
- **Two learning rates**: gamma (~0.1) for inference dynamics, alpha (~0.001) for weight updates. Inference should be fast (large gamma), learning should be slow (small alpha).
- **Inference steps**: T = 20-50 is typically sufficient. Monitor the energy trace -- it should decrease monotonically. If it oscillates, reduce gamma.
- **Activity initialization**: Initialize via a forward pass. This makes the initial energy zero at hidden layers, so the only initial error is at the output (target vs. prediction).
- **Test time**: Single forward pass -- no inference loop needed. The weights encode the learned mapping, and inference is only needed during training to find the right hidden representations.
- **Batch processing**: All operations use mini-batches. Tensor shapes are (batch_size, dim) throughout.

### Neuroscience Naming Convention

| Code name | Neuroscience concept | Description |
|-----------|---------------------|-------------|
| `CorticalLayer` | A cortical area (V1, V2, etc.) | One level in the hierarchy: holds synaptic weights |
| `PredictiveCodingNetwork` | Hierarchical generative model | Full cortical hierarchy: inference + learning |
| `prediction_error` / `epsilon` | Mismatch signal | Actual activity minus top-down prediction |
| `neural_activity` | Neuronal firing rate | The activity state a_l |
| `top_down_prediction` | Descending prediction | What higher layer predicts lower should be |
| `synaptic_weights` | Synaptic connections | W_l -- learnable parameters |
| `inference_phase` | Perceptual inference | Iterating activities to minimize energy |
| `free_energy` | Variational free energy | Total prediction error |
| `hebbian_update` | Hebbian plasticity | Local weight update |

### References

- Rao, R. P. N., & Ballard, D. H. (1999). "Predictive coding in the visual cortex." Nature Neuroscience, 2(1), 79-87.
- Friston, K. (2009). "The free-energy principle: a rough guide to the brain?" Trends in Cognitive Sciences, 13(7), 293-301. (PMC2666703)
- Bastos, A. M., et al. (2012). "Canonical microcircuits for predictive coding." Neuron, 76(4), 695-711.
- Whittington, J. C. R., & Bogacz, R. (2017). "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity." Neural Computation, 29(5), 1229-1262.
- Bogacz, R. (2017). "A tutorial on the free-energy framework for modelling perception and learning." Journal of Mathematical Psychology, 76, 198-211.
- Millidge, B., Seth, A., & Buckley, C. L. (2022). "Predictive coding: a theoretical and experimental review." arXiv:2107.12979.
- Pinchetti, L., et al. (2024). "Predictive Coding Networks: A Tutorial and Survey." arXiv:2407.04117.

## Description

Build a Predictive Coding Network from scratch in pure PyTorch (no autograd for PCN learning) and compare it side-by-side with a standard backprop MLP across three MNIST variants (MNIST, Fashion-MNIST, KMNIST).

### What you'll build

1. **CorticalLayer** -- A single cortical area with top-down predictions and prediction error computation
2. **PredictiveCodingNetwork** -- Full hierarchy with inference dynamics and Hebbian weight updates
3. **BackpropMLP** (pre-built) -- Standard MLP baseline for comparison
4. **3-dataset benchmark** (pre-built) -- Automated comparison with visualization

### What you'll learn

1. **Free energy as an objective** -- How prediction error sums to form a single loss function
2. **Inference dynamics** -- How clamping input/output and iterating hidden layers performs perception
3. **Local Hebbian learning** -- How outer-product weight updates replace backprop gradients
4. **Backprop equivalence** -- Empirical verification that PCN matches backprop accuracy

## Instructions

### Phase 1: Cortical Layer Basics (~15 min)

**File:** `src/cortical_layer.py`

**Concepts:** The CorticalLayer is the building block -- one cortical area in the hierarchy. It holds synaptic weights W_l and computes two things: the top-down prediction f(W . a) and the prediction error (actual - predicted). These correspond to deep pyramidal cells (predictions) and superficial pyramidal cells (errors) in the actual cortex.

**TODO(human) tasks:**
- Implement `compute_prediction()` -- Linear transformation followed by activation: f(a @ W^T). This is what deep pyramidal cells compute: the brain's "best guess" of what the next layer should look like.
- Implement `compute_prediction_error()` -- Simple subtraction: actual - predicted. This is the signal carried by superficial pyramidal cells: only surprises propagate upward.

### Phase 2: Free Energy (~15 min)

**File:** `src/predictive_coding_network.py`

**Concepts:** Free energy is THE objective of predictive coding -- the sum of squared prediction errors across all layers. Minimizing it is equivalent to approximate Bayesian inference. When free energy is zero, every layer perfectly predicts the one above: the brain is completely "unsurprised."

**TODO(human) tasks:**
- Implement `compute_free_energy()` -- Loop over all cortical layers, compute each prediction error, sum up (1/2) * ||epsilon||^2 averaged over the batch. This is the variational free energy from Friston's Free Energy Principle.

### Phase 3: Inference Phase (~25 min)

**File:** `src/predictive_coding_network.py`

**Concepts:** The inference phase is the HEART of predictive coding. Input and output are clamped; hidden activities iterate to minimize free energy. Each hidden layer receives two signals: bottom-up error correction (fix my own error) and top-down error propagation (help the layer above fix its error). This IS perception -- finding the internal representation that best explains the sensory input.

**TODO(human) tasks:**
- Implement `run_inference_phase()` -- Clamp input/output, then for T steps: compute all prediction errors, update each hidden layer with the two-term dynamics. Track energy at each step. Watch the indices carefully -- off-by-one errors here cause divergence.

### Phase 4: Hebbian Learning & Training (~20 min)

**File:** `src/predictive_coding_network.py`

**Concepts:** After inference converges, weights update using a purely local Hebbian rule: the outer product of the modulated error signal and pre-synaptic activity. This is biologically plausible (each synapse only needs local information) yet mathematically equivalent to backprop gradients (Whittington & Bogacz 2017).

**TODO(human) tasks:**
- Implement `hebbian_weight_update()` -- For each layer: compute prediction error, modulate by activation derivative, take outer product with pre-synaptic activity, apply learning rate. All local, no weight transport.
- Implement `train_step()` -- Wire together: initialize activities, run inference, update weights, compute accuracy via unclamped forward pass.

### Phase 5: Run & Compare (~15 min)

**Concepts:** With the PCN fully implemented, train it on Fashion-MNIST and then run the 3-dataset benchmark to compare against backprop. The Whittington-Bogacz equivalence predicts that both should achieve similar accuracy. Observe the energy trace during inference -- it should decrease monotonically, showing that the inference dynamics are working correctly.

**Tasks:**
- Run `src/train_pcn.py` to train on Fashion-MNIST and verify the PCN learns
- Run `src/benchmark.py` to compare PCN vs backprop across all three datasets
- Examine the generated plots in `outputs/`

## Motivation

- **Biological plausibility**: Understanding how the brain might actually learn -- not just as a curiosity, but as a source of inspiration for next-generation AI architectures that go beyond backprop's limitations
- **Emerging ML paradigm**: Predictive coding networks are gaining traction in ML research (NeurIPS, ICML papers 2022-2024) as alternatives to backprop for continual learning, few-shot learning, and unsupervised representation learning
- **Complementary to backprop expertise**: As someone who knows backprop cold, understanding PCN provides a deeper perspective on WHY backprop works (gradient descent on prediction errors) and WHAT alternatives exist
- **Free Energy Principle**: Connects to one of the most influential frameworks in computational neuroscience, relevant to understanding intelligence broadly
- **Systems thinking**: PCN demonstrates how a global objective (classification accuracy) can be achieved through purely local computations -- a powerful architectural principle applicable to distributed systems

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies |
| Phase 1 | `uv run pytest tests/test_energy.py -v` | Verify CorticalLayer (energy tests exercise predictions) |
| Phase 2 | `uv run pytest tests/test_energy.py -v` | Verify free energy computation |
| Phase 3 | `uv run pytest tests/test_inference.py -v` | Verify inference dynamics |
| Phase 4 | `uv run pytest tests/test_training.py -v` | Verify Hebbian updates and full training |
| Phase 5 | `uv run python src/train_pcn.py` | Train PCN on Fashion-MNIST |
| Phase 5 | `uv run python src/benchmark.py` | Run 3-dataset benchmark (all datasets) |
| Phase 5 | `uv run python src/benchmark.py --dataset mnist` | Benchmark MNIST only |
| Phase 5 | `uv run python src/benchmark.py --dataset fashion_mnist` | Benchmark Fashion-MNIST only |
| Phase 5 | `uv run python src/benchmark.py --dataset kmnist` | Benchmark KMNIST only |
| All | `uv run pytest tests/ -v` | Run all tests |

## State

`not-started`
