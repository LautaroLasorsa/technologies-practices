# Practice 075 -- Federated Learning

## Technologies

Flower (flwr), PyTorch, torchvision, NumPy, matplotlib

## Stack

Python 3.12 (uv). No Docker needed.

## Theoretical Context

### What Is Federated Learning?

Federated Learning (FL) is a machine learning paradigm where a model is trained across multiple decentralized data sources (called **clients** or **nodes**) without ever moving the raw data to a central server. Instead of centralizing data, FL centralizes the model: the server sends the current global model to selected clients, each client trains on its own local data, and only the **model updates** (weights or gradients) travel back to the server for aggregation.

The core insight: you can train a model equivalent to one trained on all data combined, without any single entity ever seeing all the data.

**Why does this matter?**

- **Privacy**: Regulations like GDPR and HIPAA restrict moving personal or medical data. FL keeps data on-device, sharing only model parameters that are much harder to reverse-engineer into raw data.
- **Data sovereignty**: Organizations (hospitals, banks, government agencies) often cannot legally transfer data across borders or to third parties. FL lets them collaborate on ML without data sharing agreements.
- **Bandwidth efficiency**: A model's weights (tens of MB) are orders of magnitude smaller than the training data (potentially GB-TB). Sending weights is cheaper than sending raw data.
- **Edge computing**: Mobile phones, IoT devices, and autonomous vehicles generate data locally. FL trains on that data without uploading it, reducing latency and server load.

Sources: [McMahan et al. 2017 -- Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629), [Kairouz et al. 2019 -- Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)

---

### The FedAvg Algorithm

**Federated Averaging** (FedAvg), introduced by McMahan et al. (2017), is the foundational FL algorithm. It is surprisingly simple and forms the basis for most FL systems in production.

**Process (one communication round):**

1. **Server broadcasts** the current global model weights `w_global` to a random subset of K clients (out of N total).
2. **Each selected client k** initializes its local model with `w_global`, then trains on its local dataset `D_k` for `E` local epochs using SGD (or any optimizer).
3. **Each client sends** its updated local weights `w_k` back to the server.
4. **Server aggregates** by computing a weighted average of the client weights.
5. Repeat for R communication rounds.

**Mathematical formulation:**

```
w_global^{t+1} = sum_{k=1}^{K} (n_k / n) * w_k^{t+1}
```

Where:
- `n_k` = number of training samples on client k
- `n = sum(n_k)` = total samples across all participating clients
- `w_k^{t+1}` = updated weights from client k after local training
- The weighting by dataset size ensures clients with more data have proportionally more influence

**Key hyperparameters:**
- `R` (communication rounds): How many server-client round trips. More rounds = better convergence but more communication.
- `K` (clients per round): How many clients participate per round. Often a fraction (e.g., 10%) of total clients.
- `E` (local epochs): How many passes over local data before sending updates. Higher E = fewer rounds needed but risk of client drift.
- `B` (local batch size): Standard SGD batch size for local training.

**Trade-off**: More local epochs (high E) reduce communication cost but increase divergence between clients, especially with non-IID data.

Source: [McMahan et al. 2017](https://arxiv.org/abs/1602.05629)

---

### The Non-IID Data Challenge

In real FL deployments, each client's data is **non-identically and non-independently distributed** (non-IID). This is the single biggest challenge in FL.

**What non-IID means in practice:**
- Hospital A specializes in pediatrics (mostly young patients), Hospital B in geriatrics (mostly elderly). Their data distributions are fundamentally different.
- A user's phone keyboard learns from *their* typing patterns, which differ drastically by language, age, profession.
- Financial institutions in different countries see different transaction patterns.

**Why non-IID is a problem for FedAvg:**

1. **Client drift**: Each client's local model drifts toward its own data distribution during local training. Client A's model gets good at classifying digit 0-2, Client B at 7-9. When averaged, neither is well-served.
2. **Slow convergence**: The global model oscillates as it tries to satisfy conflicting local optima, requiring many more rounds to converge.
3. **Poor global accuracy**: The averaged model may not perform well on *any* client's distribution, even though each local model was good for its own data.
4. **Weight divergence**: Client models move far apart in weight space, making simple averaging suboptimal.

**Data partitioning strategies for experiments:**

| Strategy | Description | Realism |
|----------|-------------|---------|
| **IID** | Random uniform split. Each client gets a representative sample of all classes. | Baseline (unrealistic) |
| **Non-IID by label** | Each client gets data from only 2-3 classes. | Extreme skew |
| **Dirichlet(alpha)** | For each class, sample a multinomial from `Dir(alpha)` to decide what fraction goes to each client. `alpha` controls heterogeneity. | Realistic, tunable |

**Dirichlet distribution for non-IID partitioning:**
- `alpha -> 0` (e.g., 0.1): Highly non-IID. Each client gets data mostly from 1-2 classes.
- `alpha = 1.0`: Moderate heterogeneity.
- `alpha -> infinity` (e.g., 100): Approaches IID. Each client gets roughly uniform class distribution.

Source: [Hsu et al. 2019 -- Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)

---

### FedProx: Handling Heterogeneity

**FedProx** (Li et al. 2020) addresses the client drift problem in non-IID settings by adding a **proximal term** to the local objective function:

```
min_w  F_k(w) + (mu/2) * ||w - w_global||^2
```

Where:
- `F_k(w)` = standard local loss (e.g., cross-entropy) on client k's data
- `w_global` = the global model weights received from the server at the start of the round
- `mu` = proximal coefficient controlling regularization strength
- `||w - w_global||^2` = L2 distance between local weights and global weights

**Intuition**: The proximal term acts as a "leash" that prevents local models from wandering too far from the global model during training. Higher `mu` = tighter leash = less client drift but potentially underfitting local data.

**Key properties:**
- When `mu = 0`, FedProx reduces to FedAvg.
- Typical values: `mu in {0.001, 0.01, 0.1, 1.0}`. Needs tuning per task.
- The proximal term is added to the loss during backpropagation on the client side. The server-side aggregation is identical to FedAvg.
- FedProx also handles **systems heterogeneity** (clients with different compute power) by allowing partial work -- clients can do fewer local epochs and still contribute.

**Implementation**: In the client's training loop, after computing the standard loss, add:

```python
proximal_term = sum((w - w_global)**2 for w, w_global in zip(model.parameters(), global_params))
loss = standard_loss + (mu / 2) * proximal_term
```

Source: [Li et al. 2020 -- Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)

---

### The Flower Framework

[Flower](https://flower.ai/) (flwr) is the leading open-source FL framework. It provides:

- **Client abstraction** (`fl.client.NumPyClient`): Implement `get_parameters`, `fit`, and `evaluate` methods. Flower handles serialization, communication, and orchestration.
- **Strategy pattern** (`fl.server.strategy`): Built-in implementations of FedAvg, FedProx, FedAvgM, FedAdam, and more. Strategies define how clients are sampled, how results are aggregated, and how the global model evolves.
- **Simulation mode**: Run FL experiments on a single machine with virtual clients. Uses Ray as the backend to parallelize client execution.
- **Production mode**: Deploy FL across real distributed nodes with gRPC communication.

**Key Flower abstractions:**

| Component | Role |
|-----------|------|
| `NumPyClient` | Client-side logic: local training and evaluation |
| `Strategy` | Server-side logic: client sampling, aggregation |
| `ServerConfig` | Round count, timeout settings |
| `start_simulation` | Legacy API to run virtual FL on one machine |
| `run_simulation` | Modern API using `ClientApp` + `ServerApp` |

**Note on simulation and Windows**: Flower's simulation engine relies on [Ray](https://ray.io/) as its backend, which has [limited Windows support](https://github.com/adap/flower/issues/5512). For this practice, we implement a **manual simulation loop** that directly instantiates Flower clients and uses Flower's strategy classes for aggregation. This approach: (a) works reliably on all platforms, (b) teaches the FL mechanics at a deeper level since you see every step of the protocol, and (c) uses the same Flower client/strategy APIs you would use in production.

**Flower API used in this practice:**

```python
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, EvaluateRes, Parameters
from flwr.server.strategy import FedAvg, FedProx

# Client: subclass NumPyClient
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config): ...   # -> list[np.ndarray]
    def fit(self, parameters, config): ...  # -> (list[np.ndarray], int, dict)
    def evaluate(self, parameters, config): ... # -> (float, int, dict)

# Strategy: aggregation logic
strategy = FedAvg(fraction_fit=0.5, min_fit_clients=2)
strategy = FedProx(proximal_mu=0.1, fraction_fit=0.5)
```

Sources: [Flower documentation](https://flower.ai/docs/framework/index.html), [Flower GitHub](https://github.com/adap/flower)

---

### Communication Efficiency

Sending full model weights each round can be expensive for large models. Techniques to reduce communication cost:

- **Gradient compression**: Send sparse gradients (top-k) or quantized gradients instead of full weights.
- **Federated distillation**: Clients share predictions or logits instead of weights.
- **Reducing rounds**: More local training (higher E) means fewer communication rounds, at the cost of client drift.
- **Model pruning**: Smaller models = smaller updates.

These are active research areas beyond the scope of this practice but important context for production FL.

---

### Security Considerations

FL is not automatically private. Known attack vectors:

- **Model inversion / gradient leakage**: An adversary observing gradients can reconstruct training samples. Mitigated by **differential privacy** (adding calibrated noise to updates) and **secure aggregation** (cryptographic protocols that let the server see only the aggregate, not individual updates).
- **Model poisoning**: A malicious client sends corrupted weights to degrade the global model. Mitigated by **robust aggregation** (e.g., trimmed mean, Krum) that detects and excludes outlier updates.
- **Data poisoning**: A client injects bad training data. Harder to detect since the server never sees raw data.

Source: [Kairouz et al. 2019](https://arxiv.org/abs/1912.04977)

---

### Real-World Applications

| Application | Company | Details |
|------------|---------|---------|
| Next-word prediction (Gboard) | Google | FL trains language models across millions of phones without collecting typing data |
| Siri improvements | Apple | On-device learning for voice recognition and suggestions |
| Cross-hospital diagnostics | Multiple | Train diagnostic models on medical images across hospitals without sharing patient data |
| Credit scoring | WeBank | Federated models across financial institutions |
| Autonomous driving | Multiple | Share driving model improvements across vehicle fleets |

---

## Description

This practice builds a federated learning system from scratch using PyTorch and Flower. Starting from a centralized baseline on MNIST, you progressively decompose the training process: partition data across simulated clients, implement the Flower client protocol, run FedAvg aggregation, and analyze how non-IID data affects convergence. The final exercise compares FedAvg vs FedProx under varying data heterogeneity.

The practice uses a **manual simulation loop** (not Flower's Ray-based simulation engine) for cross-platform compatibility and deeper understanding of the FL protocol.

## Instructions

### Prerequisites

```bash
uv sync
```

### Exercise 0: Centralized Baseline (scaffolded)

`src/00_centralized_baseline.py` is fully implemented. It trains a simple CNN on MNIST for 5 epochs and saves the accuracy as the upper bound for federated experiments. Run it to establish the baseline.

This file also defines the shared `MNISTNet` model and helper functions (`train_one_epoch`, `evaluate_model`) that all subsequent exercises import.

### Exercise 1: Data Partitioning

`src/01_data_partitioning.py` -- You implement three functions:

1. **`partition_iid`**: Split MNIST into `num_clients` equal shards with uniform class distribution. This is the "easy case" baseline for FL -- all clients see similar data.

2. **`partition_non_iid_dirichlet`**: Use the Dirichlet distribution to create heterogeneous partitions. This is the realistic case. Understanding how `alpha` controls heterogeneity is key: you will use `np.random.dirichlet` to sample per-class allocation vectors.

3. **`visualize_partitions`**: Create stacked bar charts showing class distribution per client. Visualization is essential for understanding non-IID effects -- you should be able to see at a glance which clients have which classes.

### Exercise 2: Flower Client Implementation

`src/02_flower_client.py` -- You implement `MNISTFlowerClient(fl.client.NumPyClient)`:

This is the core FL abstraction. You implement the three methods that define the client protocol: `get_parameters` (serialize model to numpy arrays), `fit` (train locally and return updated weights), `evaluate` (test model and return metrics). Understanding the numpy-to-torch conversion is important since FL frameworks operate on numpy arrays for serialization, but PyTorch uses tensors.

### Exercise 3: FedAvg Simulation

`src/03_fedavg_simulation.py` -- You implement two functions:

1. **`client_fn`**: Factory that creates a FlowerClient for a given client ID with the appropriate data partition.

2. **`run_federated_simulation`**: The FL simulation loop. You orchestrate the FedAvg protocol manually: for each round, create clients, send them global weights, call `fit`, collect results, aggregate using Flower's FedAvg strategy, then evaluate. This teaches you exactly what happens inside Flower's simulation engine.

### Exercise 4: Non-IID Analysis and FedProx

`src/04_non_iid_analysis.py` -- You implement three functions:

1. **`train_with_proximal_term`**: Implement the FedProx training loop that adds the proximal penalty `(mu/2) * ||w - w_global||^2` to the standard cross-entropy loss. This is the key difference from FedAvg -- everything else stays the same.

2. **`run_comparison_experiment`**: Run multiple FL simulations with different configurations (IID/non-IID, FedAvg/FedProx) and collect results.

3. **`plot_convergence_comparison`**: Create a multi-line convergence plot comparing all configurations against the centralized baseline.

## Motivation

- **Privacy-preserving ML is a growing industry requirement** -- GDPR, HIPAA, and data sovereignty laws make FL increasingly necessary for production ML systems.
- **Distributed systems + ML intersection** -- FL combines distributed systems knowledge (client-server protocols, aggregation, fault tolerance) with ML training, matching the user's interest in both domains.
- **Practical framework experience** -- Flower is the dominant open-source FL framework, used in production at Google, Samsung, and research labs worldwide.
- **Understanding non-IID challenges** -- The key insight of FL is that data heterogeneity fundamentally changes how training works, which has implications for any distributed ML system.

## Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (flwr, torch, torchvision, numpy, matplotlib) |
| `uv run python src/00_centralized_baseline.py` | Train centralized MNIST baseline (5 epochs), save accuracy to `data/centralized_accuracy.txt` |
| `uv run python src/01_data_partitioning.py` | Partition MNIST data into IID and non-IID splits, save visualizations to `plots/` |
| `uv run python src/02_flower_client.py` | Test Flower client locally with one fit/evaluate cycle |
| `uv run python src/03_fedavg_simulation.py` | Run FedAvg simulation with 10 clients for 20 rounds, plot accuracy per round |
| `uv run python src/04_non_iid_analysis.py` | Compare IID vs non-IID with FedAvg and FedProx, generate convergence plots |
| `python clean.py` | Remove all generated files (data/, plots/, models/, caches) |
