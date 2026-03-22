# Design Spec: Predictive Coding Networks (PCN) Practice

**Practice numbers:** 081a, 081b
**Category:** ML Systems
**Stack:** Python (PyTorch)
**Date:** 2026-03-22
**Status:** Draft

---

## Summary

Two-session hands-on practice implementing Predictive Coding Networks (PCNs) — the neuroscience-inspired ML paradigm that replaces backpropagation with local Hebbian learning rules driven by hierarchical prediction error minimization. Pure PyTorch, no external PC libraries. Code uses neuroscience-inspired naming throughout.

---

## Motivation

- **Neuroscience-ML bridge:** PCN is the leading computational theory of how the brain learns — understanding it connects ML practice to neuroscience foundations (Rao & Ballard 1999, Friston's Free Energy Principle).
- **Alternative to backprop:** PCNs train via local learning rules (no weight transport problem, no global backward pass). This is an active NeurIPS/ICLR research frontier (μPC 2025, iPC ICLR 2024, PCX benchmarks 2024).
- **Unique capabilities:** PCNs are joint generative models — they can generate, infer missing data, and do continual learning natively. Standard backprop MLPs cannot.
- **Complementary to existing practices:** Extends the ML Systems category (017 series: RL world models, 025 series: ML compilers) with a fundamentally different learning paradigm.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Branch of PC | PCN / Inference Learning (not CPC) | User interest is in the neuroscience-inspired algorithm, not the self-supervised objective |
| Sessions | 2 (~90 min each) | Enough for foundations + divergence from backprop, without overextending |
| Stack | Pure PyTorch from scratch | Maximum learning depth — see every moving part of the algorithm |
| Neuroscience naming | Yes — classes, variables, and methods named after neuroscience concepts | Code becomes a direct map of the brain theory |
| Baseline approach | Backprop MLP pre-built as scaffold | User knows backprop cold — all hands-on time goes to PCN implementation |
| Session A datasets | MNIST + Fashion-MNIST + KMNIST (3-dataset benchmark) | Robustness study across pattern complexity levels |
| Session B primary dataset | Fashion-MNIST (generative/missing-data experiments) | Visually interesting for generation and inference |
| Session B stretch | CIFAR-10 with convolutional cortical layers | Tests understanding at scale, biologically grounded (retinotopic receptive fields) |

---

## Theoretical Context (to be expanded in CLAUDE.md)

### Neuroscience Foundation

The practice opens with rich neuroscience framing:

- **Rao & Ballard (1999):** Hierarchical predictive coding in visual cortex. Top-down connections carry predictions, bottom-up connections carry prediction errors. Trained on natural images, the model spontaneously develops Gabor-like simple-cell receptive fields.
- **Cortical circuit anatomy (Bastos et al. 2012):** Superficial pyramidal cells carry prediction errors (feedforward/ascending). Deep pyramidal cells carry predictions (feedback/descending). Precision (inverse variance) modulates error gain — proposed mechanism for attention.
- **Karl Friston's Free Energy Principle:** Every living system minimizes variational free energy (upper bound on surprise). Perception = update beliefs to reduce prediction error. Action = change the world to match predictions. Learning = update model parameters. Unifies perception, action, and learning under one objective.
- **Bayesian Brain Hypothesis:** The brain IS a hierarchical Bayesian inference machine. PCN is its algorithmic instantiation.

### Mathematical Foundations

- **Energy function:** E(a, W) = ½ Σ_ℓ ‖ε_ℓ‖² where ε_ℓ = a_ℓ - f(W_{ℓ-1} · a_{ℓ-1})
- **Inference dynamics (two biological pathways):**
  - Δa_ℓ = **-γ · ε_ℓ** + **γ · W_ℓᵀ · (ε_{ℓ+1} ⊙ f'(W_ℓ · a_ℓ))**
  - First term: bottom-up error correction (reduce this layer's own prediction error)
  - Second term: top-down error propagation (adjust to reduce the layer above's prediction error)
- **Local weight update:** ΔW_ℓ = α · (ε_{ℓ+1} ⊙ f'(W_ℓ · a_ℓ)) · a_ℓᵀ — outer product, result shape matches W_ℓ: (dim_{ℓ+1}, dim_ℓ)
- **Connection to variational inference:** The energy IS the variational free energy F = E_q[log q(z) - log p(x,z)]
- **Whittington & Bogacz (2017) result:** Under fixed-prediction assumption, PC weight updates converge to exact backprop gradients

### Key Differences from Backpropagation

1. **Locality:** PC weight updates use only pre/post-synaptic activity + local error. Backprop requires weight transport (non-local).
2. **Bidirectional flow:** Predictions top-down, errors bottom-up simultaneously. Backprop is purely backward sequential.
3. **Two-phase learning:** Inference phase (iterate to equilibrium) → weight update phase. Backprop is single forward+backward.
4. **Parallelism:** All PC layers can update in parallel once errors are local. Backprop requires strict sequential backward ordering.

---

## Session A: PCN Foundations & Backprop Comparison

**Practice folder:** `practice_081a_predictive_coding_foundations/`

### Neuroscience Naming Convention

| Code name | Neuroscience concept | Description |
|-----------|---------------------|-------------|
| `CorticalLayer` | A cortical area (V1, V2, etc.) | One level in the hierarchy: holds synaptic weights, computes predictions and errors |
| `PredictiveCodingNetwork` | Hierarchical generative model | Full cortical hierarchy: orchestrates inference + learning |
| `prediction_error` / `epsilon` | ε — mismatch signal | Bottom-up error: actual activity minus top-down prediction |
| `neural_activity` | Neuronal firing rate / activation | The activity state a_ℓ at each cortical layer |
| `top_down_prediction` | Descending prediction | What a higher layer predicts the lower layer should be |
| `synaptic_weights` | Synaptic connection strengths | W_ℓ — the learnable parameters |
| `inference_phase` | Perceptual inference | Iterating activities to minimize total energy |
| `free_energy` | Variational free energy | Total prediction error across all layers |
| `precision` | Inverse variance (1/σ²) | Confidence weighting on prediction errors |
| `hebbian_update` | Hebbian plasticity | "Neurons that fire together wire together" — local weight update |
| `RetinotopicCorticalLayer` | Retinotopic cortical area (V1, V2) | Convolutional variant — local receptive fields tiled across visual field (Session B) |

### Pre-built (Scaffold)

- Data loading for MNIST, Fashion-MNIST, KMNIST (torchvision)
- Standard backprop MLP with identical architecture (3 hidden layers, same widths)
- Backprop MLP training loop (already trained or trains automatically)
- 3-dataset benchmark harness: runs both models on all three datasets
- Comparison visualization: loss curves, accuracy curves, prediction error heatmaps
- Utility functions: plotting, metrics, device selection

### Time Budget

Phases total ~80 min of coding. The theoretical context should be read before the session starts (or the first ~10 min). Environment setup (`uv sync`, dataset downloads) happens once and is not counted.

### Function Ownership

All exercise functions are **methods** on their owning class, following Information Expert (GRASP):

- `CorticalLayer`: holds `synaptic_weights`, owns `compute_prediction()` (top-down) and `compute_prediction_error()` for its own level
- `PredictiveCodingNetwork`: orchestrates all layers, owns `compute_free_energy()`, `run_inference_phase()`, `hebbian_weight_update()`, `train_step()`

### Batch Processing

All operations work on **mini-batches** (not single samples). Tensor shapes follow `(batch_size, dim_ℓ)` for activities and errors. This is standard in PCN implementations (arXiv:2407.04117 tutorial uses batched operations) and necessary for practical training speed.

### Exercises (TODO(human))

**Phase 1: Free Energy (~15 min)**
Implement `PredictiveCodingNetwork.compute_free_energy()` — the total prediction error energy across all cortical layers. This is THE objective that drives everything in predictive coding.

**Phase 2: Top-Down Predictions (~15 min)**
Implement `CorticalLayer.compute_prediction()` — each cortical layer generates a prediction of what the layer below should look like. This is the descending (feedback) pathway.

**Phase 3: Inference Dynamics (~20 min)**
Implement `PredictiveCodingNetwork.run_inference_phase()` — the core iterative loop. Clamp input at bottom, target at top, iterate hidden activities to minimize free energy. This is the heart of the algorithm — perceptual inference.

**Phase 4: Hebbian Weight Updates (~15 min)**
Implement `PredictiveCodingNetwork.hebbian_weight_update()` — the local learning rule. After inference converges, update each layer's weights using only adjacent-layer information. No global backward pass.

**Phase 5: Full Training Step & 3-Dataset Benchmark (~15 min)**
Wire inference + weight updates into `PredictiveCodingNetwork.train_step()`. Run the 3-dataset benchmark against the backprop baseline. Analyze: does PCN match backprop accuracy? How do convergence dynamics differ? What do prediction error heatmaps reveal about hierarchical processing?

### Expected Outcomes

- PCN accuracy within ~1-2% of backprop MLP on all three datasets
- Visible difference in training dynamics: PCN converges differently (inference overhead per step, but different generalization trajectory)
- Prediction error heatmaps showing hierarchical decomposition (lower layers: pixel-level errors; higher layers: semantic errors)

---

## Session B: Where PCNs Diverge from Backprop

**Practice folder:** `practice_081b_predictive_coding_divergence/`

### Code Dependency on Session A

Session B's scaffold includes the **completed** `CorticalLayer` and `PredictiveCodingNetwork` classes as pre-built code. The user already implemented these in Session A, so providing them completed preserves the standalone session constraint while avoiding code duplication across practice folders.

### Pre-built (Scaffold)

- Completed `CorticalLayer` and `PredictiveCodingNetwork` from Session A (pre-built, not imported)
- Trained PCN model checkpoint (or re-trains quickly from the pre-built code)
- Trained backprop MLP (same architecture, for comparison)
- Visualization utilities for generated images, missing data reconstruction, continual learning curves
- Experiment harness for each exercise
- Data loading with corruption/masking utilities

### Exercises (TODO(human))

**Phase 1: Generative Mode (~20 min)**
Implement `generate_from_label(network, label, num_inference_steps)` — clamp a one-hot label at the top cortical layer, run inference *downward* to generate an image at the sensory layer. The PCN already learned a generative model implicitly during classification training — you just run it in reverse. The backprop MLP cannot do this.

**Phase 2: Missing Data Inference (~20 min)**
Implement `infer_missing_data(network, partial_input, mask, num_inference_steps)` — clamp observed pixels, let inference fill in the missing ones. Test with top-half/bottom-half masking, random pixel dropout, and heavy occlusion. Compare with backprop MLP's degradation on the same corrupted inputs (discriminative classification only — no reconstruction).

**Phase 3: Precision Weighting (~20 min)**
Implement precision (inverse variance) on prediction errors — `precision_weighted_error(prediction_error, precision)`. High precision = "trust this signal" (the neuroscience mechanism for attention). Train with precision, test on noisy inputs. Show that precision-weighted PCN is more robust to noise than both unweighted PCN and backprop MLP.

**Phase 4: Continual Learning (~15 min)**
Implement `continual_learning_experiment(network, dataset_sequence)` — train sequentially on Fashion-MNIST split into two task groups (e.g., tops/bottoms then accessories/footwear). Measure catastrophic forgetting. Compare PCN vs backprop MLP. PCNs are hypothesized to forget less because local weight updates disturb fewer unrelated weights.

**The session is complete and satisfying after Phase 4 (~75 min).**

**Phase 5 (Stretch Goal): Retinotopic Cortical Layers — Convolutional PCN on CIFAR-10 (~25 min)**
Implement `RetinotopicCorticalLayer` — a convolutional variant of `CorticalLayer` that mirrors V1→V2→V4 local receptive field structure. Adapt inference dynamics and Hebbian updates for conv layers (replace matrix multiply with convolution in both prediction and error computation). Train on CIFAR-10. This mirrors the actual retinotopic organization of visual cortex — local receptive fields tiled across the visual field. **Note:** The scaffold provides conv boilerplate (layer shape management, transposed convolutions for top-down path); the user implements only the inference/update adaptation.

### Expected Outcomes

- Generative mode produces recognizable (if blurry) Fashion-MNIST items from labels alone
- Missing data inference successfully reconstructs masked images; backprop MLP accuracy degrades sharply on same inputs
- Precision weighting measurably improves robustness to Gaussian noise
- PCN shows less catastrophic forgetting than backprop MLP on sequential tasks
- Convolutional PCN achieves reasonable CIFAR-10 accuracy, demonstrating the architecture scales

---

## Practical Implementation Notes

### Activation Functions
Use `tanh` or `LeakyReLU`, NOT `ReLU`. ReLU creates energy function pathologies at zero (flat gradient → dead inference dynamics).

### Two Learning Rates
- γ (inference rate): larger (~0.1), controls how fast activities converge during inference phase
- α (weight learning rate): smaller (~0.001), controls synaptic plasticity

### Inference Steps
T = 20-50 steps is typically sufficient for convergence on MNIST-scale problems.

### Initialization
Initialize hidden activities via a forward pass: a_ℓ = f(W_{ℓ-1} · a_{ℓ-1}). This sets initial prediction error to zero everywhere except the output layer — most efficient starting point.

### Test-Time Inference
At test time, no inference loop is needed for classification — a single forward pass suffices (same as backprop). The inference loop is only needed during training and for the Session B generative/missing-data experiments.

### Output Layer
The output (top) layer uses **cross-entropy energy** instead of MSE: E_L = -log softmax(a_L)[y]. This replaces the top-layer's squared error term in the total energy. The output prediction error becomes:

- **ε_L = a_L - one_hot(y)** (after softmax derivative simplification)

This is the error signal used in the inference dynamics for the layer below (a_{L-1}). All other layers use the standard MSE-based prediction error ε_ℓ = a_ℓ - f(W_{ℓ-1} · a_{ℓ-1}). The output layer is the only special case.

---

## File Structure

```
practice_081a_predictive_coding_foundations/
    CLAUDE.md
    .gitignore
    clean.py
    pyproject.toml
    src/
        __init__.py
        cortical_layer.py          # CorticalLayer class (TODO(human): compute_prediction, compute_prediction_error)
        predictive_coding_network.py  # PredictiveCodingNetwork (TODO(human): free_energy, inference, hebbian, train_step)
        backprop_baseline.py       # Pre-built standard MLP
        benchmark.py               # 3-dataset comparison harness
        train_pcn.py               # Training script
        visualization.py           # Plotting utilities
    tests/
        __init__.py
        test_energy.py             # Verify energy computation
        test_inference.py          # Verify inference converges
        test_training.py           # Verify PCN trains and improves

practice_081b_predictive_coding_divergence/
    CLAUDE.md
    .gitignore
    clean.py
    pyproject.toml
    src/
        __init__.py
        cortical_layer.py          # Pre-built: completed CorticalLayer from Session A
        predictive_coding_network.py  # Pre-built: completed PredictiveCodingNetwork from Session A
        generative_inference.py    # Generative mode (TODO(human): top-down generation)
        missing_data.py            # Missing data inference (TODO(human): partial input inference)
        precision_weighting.py     # Precision-weighted prediction errors (TODO(human))
        continual_learning.py      # Sequential task experiment (TODO(human))
        retinotopic_layer.py       # Convolutional cortical layer (TODO(human), stretch goal)
        experiments.py             # Experiment runner
        visualization.py           # Session B specific plots
    tests/
        __init__.py
        test_generative.py
        test_missing_data.py
        test_precision.py
        test_continual_learning.py
```

---

## Key References

| Paper | Year | Relevance |
|-------|------|-----------|
| Rao & Ballard — "Predictive coding in the visual cortex" | 1999 | Founding paper: hierarchical generative model of visual cortex |
| Friston — "Predictive coding under the free-energy principle" | 2009 | Variational inference connection, Free Energy Principle |
| Bastos et al. — "Canonical microcircuits for predictive coding" | 2012 | Cortical circuit anatomy: superficial (errors) vs deep (predictions) |
| Whittington & Bogacz — "An approximation of the error backpropagation algorithm" | 2017 | Proved PC ≈ backprop under local Hebbian rules |
| Bogacz — "A tutorial on the free-energy framework" | 2017 | Accessible mathematical tutorial |
| Millidge et al. — "Predictive coding approximates backprop along arbitrary computation graphs" | 2022 | Generalized the PC-backprop equivalence |
| Millidge et al. — "A theoretical framework for inference learning" | 2022 | Unified PC, Equilibrium Propagation, Contrastive Hebbian Learning |
| Salvatori et al. — "Incremental Predictive Coding" (iPC) | 2024 | Weight updates during inference, more stable training |
| Pinchetti et al. — "Benchmarking PCNs Made Simple" | 2024 | First CIFAR-100 results competitive with backprop |
| PCN Tutorial and Survey (arXiv:2407.04117) | 2024 | 47-page authoritative tutorial with pseudocode and benchmarks |
| Innocenti et al. — "μPC: Scaling to 100+ Layers" | 2025 | Depth-μP parameterization, 128-layer ResNets via inference learning |

---

## Commands (Draft)

### Session A

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies |
| Phase 1-4 | `uv run pytest tests/ -v` | Run tests to verify each phase's implementation |
| Phase 5 | `uv run python src/train_pcn.py` | Train PCN on Fashion-MNIST |
| Phase 5 | `uv run python src/benchmark.py` | Run 3-dataset benchmark (PCN vs backprop MLP) |
| Phase 5 | `uv run python src/benchmark.py --dataset mnist` | Benchmark on MNIST only |
| Phase 5 | `uv run python src/benchmark.py --dataset kmnist` | Benchmark on KMNIST only |

### Session B

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies |
| All | `uv run pytest tests/ -v` | Run tests for each phase |
| Phase 1 | `uv run python src/experiments.py --experiment generative` | Generate images from labels |
| Phase 2 | `uv run python src/experiments.py --experiment missing-data` | Missing data inference |
| Phase 3 | `uv run python src/experiments.py --experiment precision` | Precision weighting vs noise |
| Phase 4 | `uv run python src/experiments.py --experiment continual` | Continual learning comparison |
| Phase 5 | `uv run python src/experiments.py --experiment retinotopic` | Conv PCN on CIFAR-10 (stretch) |

---

## Open Questions / Risks

1. **Session B Phase 5 (stretch goal):** Implementing conv inference dynamics from scratch in ~25 min may be tight. Mitigation: scaffold provides conv boilerplate (shape management, transposed convolutions); user implements only the inference/update adaptation. Labeled as stretch goal — session is complete after Phase 4.
2. **Generative quality:** From-scratch PCN generation on Fashion-MNIST will be blurry. Set expectations clearly — this demonstrates the *capability*, not SOTA quality.
3. **Continual learning signal:** The forgetting difference between PCN and backprop may be subtle on a small benchmark. Mitigation: design the task split to maximize interference (similar-looking classes in different tasks).
