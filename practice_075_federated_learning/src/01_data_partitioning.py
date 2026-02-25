"""Exercise 1 -- IID and Non-IID Data Partitioning.

Partitioning data across clients is the foundation of any FL experiment.
How data is distributed determines whether FedAvg works well or struggles.
This exercise builds the partitioning functions used by all subsequent exercises.
"""

import sys
from pathlib import Path

# Ensure practice root is on sys.path for cross-module imports
_PRACTICE_DIR = Path(__file__).resolve().parent.parent
if str(_PRACTICE_DIR) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset

# Reuse MNIST loading from exercise 0
from src.00_centralized_baseline import load_mnist_train

PRACTICE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PRACTICE_DIR / "data"
PLOTS_DIR = PRACTICE_DIR / "plots"


# ---------------------------------------------------------------------------
# TODO(human): Implement IID partitioning
# ---------------------------------------------------------------------------

def partition_iid(dataset: Dataset, num_clients: int) -> list[Subset]:
    """Split *dataset* into *num_clients* IID (uniform) partitions.

    IID partitioning means each client receives a random, uniformly sampled
    subset of the data. Every client should end up with approximately
    len(dataset) // num_clients samples, and the class distribution within
    each partition should roughly mirror the global distribution (all 10
    digit classes represented proportionally).

    Algorithm:
      1. Generate a random permutation of all dataset indices (0..len-1).
      2. Split the permuted indices into num_clients equal-sized chunks.
         If the dataset size isn't perfectly divisible, distribute the
         remainder indices across the first few clients (one extra each).
      3. Wrap each chunk as a torch.utils.data.Subset of the original dataset.

    Args:
        dataset: The full MNIST training dataset (60,000 samples).
        num_clients: Number of clients to partition across (e.g., 10).

    Returns:
        A list of num_clients Subset objects, each containing ~6,000 samples
        for 10 clients, with approximately uniform class distribution.
    """
    # TODO(human): Implement IID partitioning as described above.
    # Hint: Use np.random.permutation(len(dataset)) to shuffle indices,
    # then np.array_split(shuffled_indices, num_clients) to divide evenly.
    raise NotImplementedError("Implement partition_iid")


# ---------------------------------------------------------------------------
# TODO(human): Implement non-IID Dirichlet partitioning
# ---------------------------------------------------------------------------

def partition_non_iid_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
) -> list[Subset]:
    """Split *dataset* into *num_clients* non-IID partitions using Dirichlet distribution.

    The Dirichlet distribution Dir(alpha) controls heterogeneity:
      - alpha = 0.1 -> highly non-IID (each client gets mostly 1-2 classes)
      - alpha = 1.0 -> moderate heterogeneity
      - alpha = 100  -> approximately IID (uniform distribution)

    Algorithm:
      1. Extract the label/target for every sample in the dataset.
         For MNIST: dataset.targets is a tensor of shape (60000,).
      2. Group sample indices by their class label (0-9). You'll have
         10 lists, one per digit class.
      3. For EACH class c (0 through 9):
         a. Sample a proportion vector from Dir(alpha): proportions = np.random.dirichlet([alpha]*num_clients)
            This gives a vector of length num_clients that sums to 1.0.
         b. Split the indices of class c across clients according to these proportions.
            Use np.split with cumulative sum of (proportions * num_samples_of_class_c).
      4. Concatenate each client's indices from all classes.
      5. Handle edge case: some clients may receive 0 samples if alpha is very low.
         Either skip empty clients or ensure a minimum allocation.
      6. Wrap each client's indices as a Subset.

    Args:
        dataset: The full MNIST training dataset.
        num_clients: Number of clients (e.g., 10).
        alpha: Dirichlet concentration parameter. Lower = more heterogeneous.

    Returns:
        A list of num_clients Subset objects with non-IID class distributions.
        Client i's partition may have very different class proportions than client j's.
    """
    # TODO(human): Implement Dirichlet-based non-IID partitioning.
    # Key numpy functions: np.random.dirichlet, np.cumsum, np.split, np.where
    # Hint: targets = np.array(dataset.targets) gives you all labels.
    # For each class, get indices with np.where(targets == c)[0], then
    # split those indices using the Dirichlet-sampled proportions.
    raise NotImplementedError("Implement partition_non_iid_dirichlet")


# ---------------------------------------------------------------------------
# TODO(human): Implement partition visualization
# ---------------------------------------------------------------------------

def visualize_partitions(
    partitions: list[Subset],
    dataset: Dataset,
    save_path: Path,
    title: str = "Data Distribution Across Clients",
) -> None:
    """Create a stacked bar chart showing class distribution per client.

    The chart should have:
      - X-axis: Client ID (0, 1, 2, ..., num_clients-1)
      - Y-axis: Number of samples
      - Each bar is stacked by digit class (0-9), with a different color per class
      - A legend mapping colors to digit classes
      - The title should identify the partitioning strategy (passed as *title*)

    This visualization is the key diagnostic tool for FL experiments:
    you can immediately see whether data is IID (uniform bars) or
    non-IID (uneven, dominated by 1-2 colors per client).

    Algorithm:
      1. For each partition (client), count how many samples belong to each
         class (0-9). Use the dataset's targets indexed by the partition's indices.
      2. Build a 2D array of shape (num_clients, 10) with these counts.
      3. Use matplotlib's bar() with bottom parameter for stacking:
         for each class c, plot bars at height = sum of all previous classes.
      4. Add axis labels, title, and legend.
      5. Save to save_path (create parent directories if needed).

    Args:
        partitions: List of Subset objects (one per client).
        dataset: The original dataset (needed to look up targets by index).
        save_path: Path to save the plot image (e.g., plots/iid_distribution.png).
        title: Title string for the plot.
    """
    # TODO(human): Implement the stacked bar chart visualization.
    # Hint: targets = np.array(dataset.targets) to get all labels.
    # For each partition, get its indices via partition.indices,
    # then count per-class: np.bincount(targets[indices], minlength=10).
    # Use plt.bar(..., bottom=...) to stack bars.
    raise NotImplementedError("Implement visualize_partitions")


# ---------------------------------------------------------------------------
# Scaffolded: orchestration
# ---------------------------------------------------------------------------

def save_partition_indices(partitions: list[Subset], name: str) -> None:
    """Save partition indices to disk for use by later exercises."""
    indices_dir = DATA_DIR / "partitions" / name
    indices_dir.mkdir(parents=True, exist_ok=True)
    for i, partition in enumerate(partitions):
        indices = np.array(partition.indices)
        np.save(indices_dir / f"client_{i:02d}.npy", indices)
    print(f"  Saved {len(partitions)} partition index files to {indices_dir}")


def load_partition_indices(name: str) -> list[np.ndarray]:
    """Load previously saved partition indices."""
    indices_dir = DATA_DIR / "partitions" / name
    files = sorted(indices_dir.glob("client_*.npy"))
    return [np.load(f) for f in files]


def main() -> None:
    print("=" * 60)
    print("Exercise 1: Data Partitioning")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    num_clients = 10
    dataset = load_mnist_train()
    print(f"\nMNIST training set: {len(dataset)} samples")

    # --- IID partitioning ---
    print("\n--- IID Partitioning ---")
    iid_partitions = partition_iid(dataset, num_clients)
    for i, p in enumerate(iid_partitions):
        print(f"  Client {i}: {len(p)} samples")
    visualize_partitions(iid_partitions, dataset, PLOTS_DIR / "iid_distribution.png", title="IID Distribution")
    save_partition_indices(iid_partitions, "iid")

    # --- Non-IID with various alpha values ---
    alphas = [0.1, 0.5, 1.0, 10.0]
    for alpha in alphas:
        print(f"\n--- Non-IID Dirichlet (alpha={alpha}) ---")
        partitions = partition_non_iid_dirichlet(dataset, num_clients, alpha)
        for i, p in enumerate(partitions):
            print(f"  Client {i}: {len(p)} samples")
        visualize_partitions(
            partitions,
            dataset,
            PLOTS_DIR / f"non_iid_alpha_{alpha}_distribution.png",
            title=f"Non-IID Dirichlet (alpha={alpha})",
        )
        save_partition_indices(partitions, f"non_iid_alpha_{alpha}")

    print("\n--- All partitions created and visualized ---")
    print(f"Plots saved to {PLOTS_DIR}")
    print(f"Partition indices saved to {DATA_DIR / 'partitions'}")


if __name__ == "__main__":
    main()
