"""Phase 4: Optuna Advanced — Multi-Objective & Conditional Search Spaces.

Covers multi-objective optimization with Pareto front visualization,
conditional parameter spaces (tree-structured), and sampler comparison.

Key Optuna features:
  - create_study(directions=["minimize", "minimize"]) — multi-objective
  - trial.suggest_* inside conditionals — conditional search spaces
  - study.best_trials — Pareto-optimal trials
  - optuna.visualization.matplotlib — built-in plotting
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler, NSGAIISampler, RandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Multi-objective synthetic objectives
# ============================================================================

def compute_accuracy_proxy(
    model_complexity: float,
    learning_rate: float,
    regularization: float,
    noise_seed: int,
) -> float:
    """Synthetic accuracy proxy (higher is better, we'll minimize negative accuracy).

    Models the tradeoff: more complexity → higher accuracy but diminishing returns.
    Too high learning rate or too low regularization → overfitting (lower accuracy).
    """
    rng = np.random.default_rng(noise_seed)

    # Accuracy increases with complexity but saturates
    base_accuracy = 1.0 - np.exp(-0.5 * model_complexity)

    # Learning rate penalty: optimal around 0.01
    lr_penalty = 0.1 * (np.log10(learning_rate) + 2) ** 2

    # Regularization: too much hurts accuracy, too little hurts on noisy data
    reg_penalty = 0.05 * (np.log10(regularization) + 3) ** 2

    accuracy = base_accuracy - lr_penalty - reg_penalty + rng.normal(0, 0.02)
    return float(np.clip(accuracy, 0.01, 0.99))


def compute_latency_proxy(
    model_complexity: float,
    batch_size: int,
    use_gpu: bool,
) -> float:
    """Synthetic latency proxy in milliseconds (lower is better).

    Models: more complexity → higher latency. GPU reduces latency but not to zero.
    Larger batch size → higher throughput but higher per-batch latency.
    """
    base_latency = 5.0 + 10.0 * model_complexity

    # Batch size effect: larger batches → more work per forward pass
    batch_factor = 1.0 + 0.3 * np.log2(batch_size / 16)

    # GPU reduces latency by ~3x
    gpu_factor = 0.35 if use_gpu else 1.0

    return float(base_latency * batch_factor * gpu_factor)


# ============================================================================
# TODO(human): Multi-objective Optuna objective
# ============================================================================

def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    """Multi-objective: minimize (1 - accuracy) AND minimize latency.

    These objectives conflict: higher accuracy requires more complex models
    (more layers, larger hidden size) which increase latency.

    Args:
        trial: Optuna Trial object.

    Returns:
        Tuple of (1 - accuracy, latency) — both to be minimized.
    """
    # TODO(human): Multi-Objective Optimization with Optuna
    #
    # In multi-objective optimization, the objective function returns MULTIPLE values
    # (as a tuple). Optuna finds the Pareto front: the set of solutions where no
    # objective can be improved without worsening another.
    #
    # Step 1: Suggest parameters that affect BOTH objectives.
    #   model_complexity = trial.suggest_float("model_complexity", 0.5, 10.0)
    #     - Higher complexity → better accuracy BUT higher latency.
    #     - This is the main tradeoff knob.
    #
    #   learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    #     - Affects accuracy only (not latency in our model).
    #
    #   regularization = trial.suggest_float("regularization", 1e-6, 1e-1, log=True)
    #     - Affects accuracy only (controls overfitting).
    #
    #   batch_size = trial.suggest_int("batch_size", 8, 128, log=True)
    #     - Affects latency (larger batch → higher per-batch latency).
    #
    #   use_gpu = trial.suggest_categorical("use_gpu", [True, False])
    #     - Affects latency (GPU gives ~3x speedup).
    #
    # Step 2: Compute both objectives.
    #   accuracy = compute_accuracy_proxy(model_complexity, learning_rate,
    #                                      regularization, trial.number)
    #   latency = compute_latency_proxy(model_complexity, batch_size, use_gpu)
    #
    # Step 3: Return BOTH objectives as a tuple.
    #   return (1.0 - accuracy, latency)
    #
    #   We return (1 - accuracy) because Optuna minimizes by default.
    #   Minimizing (1 - accuracy) = maximizing accuracy.
    #   Minimizing latency is already what we want.
    #
    # The Pareto front will show the tradeoff: low-complexity models have
    # low latency but low accuracy; high-complexity models have high accuracy
    # but high latency. The Pareto-optimal solutions are the "best compromises."
    raise NotImplementedError("TODO(human): implement multi-objective Optuna objective")


# ============================================================================
# TODO(human): Conditional search space objective
# ============================================================================

def conditional_objective(trial: optuna.Trial) -> float:
    """Objective with conditional parameter spaces.

    The optimizer type determines which additional hyperparameters exist.
    This demonstrates Optuna's tree-structured search space capability.

    Args:
        trial: Optuna Trial object.

    Returns:
        Synthetic loss (float) — lower is better.
    """
    # TODO(human): Conditional Search Spaces in Optuna
    #
    # Optuna naturally supports conditional parameters because search spaces
    # are defined by-run (inside the objective function). Parameters that
    # are only suggested when a condition is met form a TREE structure.
    #
    # This is something grid search and scipy.optimize CANNOT do — they
    # require a flat parameter grid. Optuna's TPE handles tree-structured
    # spaces natively by building independent KDE models per parameter.
    #
    # Step 1: Suggest the top-level "branching" parameter.
    #   optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
    #
    # Step 2: Suggest optimizer-specific parameters CONDITIONALLY.
    #   if optimizer == "sgd":
    #       lr = trial.suggest_float("sgd_lr", 1e-4, 1.0, log=True)
    #       momentum = trial.suggest_float("sgd_momentum", 0.0, 0.99)
    #       # SGD with momentum. Optimal config: lr~0.05, momentum~0.9
    #       base_loss = 2.0 * (np.log10(lr) + 1.3)**2 + 0.5 * (momentum - 0.9)**2
    #
    #   elif optimizer == "adam":
    #       lr = trial.suggest_float("adam_lr", 1e-5, 1e-1, log=True)
    #       beta1 = trial.suggest_float("adam_beta1", 0.8, 0.999)
    #       beta2 = trial.suggest_float("adam_beta2", 0.9, 0.9999)
    #       # Adam is more forgiving on lr. Optimal: lr~0.001, beta1~0.9, beta2~0.999
    #       base_loss = 1.5 * (np.log10(lr) + 3)**2 + (beta1 - 0.9)**2 + 2*(beta2 - 0.999)**2
    #
    #   elif optimizer == "rmsprop":
    #       lr = trial.suggest_float("rmsprop_lr", 1e-5, 1e-1, log=True)
    #       alpha = trial.suggest_float("rmsprop_alpha", 0.8, 0.999)
    #       # RMSProp. Optimal: lr~0.001, alpha~0.99
    #       base_loss = 1.8 * (np.log10(lr) + 3)**2 + 1.5 * (alpha - 0.99)**2
    #
    # Step 3: Suggest shared parameters (not conditional).
    #   n_layers = trial.suggest_int("n_layers", 1, 5)
    #   dropout = trial.suggest_float("dropout", 0.0, 0.5)
    #
    #   arch_penalty = 0.3 * (n_layers - 3)**2 + 2.0 * (dropout - 0.2)**2
    #
    # Step 4: Compute and return the total loss.
    #   noise = np.random.default_rng(trial.number).normal(0, 0.05)
    #   return base_loss + arch_penalty + noise
    #
    # NOTE: Parameter names MUST be unique across branches. Use prefixed names
    # like "sgd_lr", "adam_lr" instead of just "lr" — otherwise Optuna would
    # see them as the same parameter with inconsistent ranges.
    #
    # The TPE sampler handles this naturally: it only models the relationship
    # between parameters that co-occur in the same trial.
    raise NotImplementedError("TODO(human): implement conditional search space objective")


# ============================================================================
# TODO(human): Run multi-objective study
# ============================================================================

def run_multi_objective_study(n_trials: int, seed: int = 42) -> optuna.Study:
    """Create and run a multi-objective Optuna study.

    Args:
        n_trials: Number of trials to run.
        seed: Random seed for reproducibility.

    Returns:
        Completed multi-objective Study.
    """
    # TODO(human): Multi-Objective Study Creation
    #
    # For multi-objective optimization, create_study takes a `directions` parameter
    # (a list) instead of the single `direction` parameter used in Phase 3.
    #
    # Step 1: Create a multi-objective study.
    #   study = optuna.create_study(
    #       study_name="accuracy_vs_latency",
    #       directions=["minimize", "minimize"],
    #       sampler=TPESampler(seed=seed),
    #   )
    #
    #   directions=["minimize", "minimize"] means both objectives are minimized.
    #   We return (1-accuracy, latency), so minimizing both = maximizing accuracy
    #   while minimizing latency. You can mix directions: ["minimize", "maximize"].
    #
    #   Alternative sampler for multi-objective: NSGAIISampler(seed=seed)
    #   NSGA-II is a classic multi-objective evolutionary algorithm. TPE also
    #   supports multi-objective but uses a different selection strategy.
    #
    # Step 2: Optimize.
    #   study.optimize(multi_objective, n_trials=n_trials)
    #
    # Step 3: Return the study.
    #   return study
    #
    # After optimization, the Pareto-optimal trials are accessible via:
    #   study.best_trials  — list of trials on the Pareto front
    # Each trial has .values (list of objective values) and .params (dict).
    #
    # Note: study.best_trial (singular) raises an error for multi-objective
    # studies — there is no single "best" when objectives conflict.
    raise NotImplementedError("TODO(human): implement multi-objective study")


# ============================================================================
# Visualization
# ============================================================================

def plot_pareto_front(study: optuna.Study):
    """Plot the Pareto front from a multi-objective study."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # All trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    all_obj1 = [t.values[0] for t in completed]
    all_obj2 = [t.values[1] for t in completed]

    # Pareto front
    pareto_trials = study.best_trials
    pareto_obj1 = [t.values[0] for t in pareto_trials]
    pareto_obj2 = [t.values[1] for t in pareto_trials]

    # Sort Pareto front for line plot
    pareto_sorted = sorted(zip(pareto_obj1, pareto_obj2))
    pareto_obj1_s = [p[0] for p in pareto_sorted]
    pareto_obj2_s = [p[1] for p in pareto_sorted]

    ax.scatter(all_obj1, all_obj2, c="lightgray", alpha=0.5, s=20, label="All trials")
    ax.scatter(pareto_obj1, pareto_obj2, c="red", s=50, zorder=5, label="Pareto front")
    ax.plot(pareto_obj1_s, pareto_obj2_s, "r--", alpha=0.5, zorder=4)

    ax.set_xlabel("1 - Accuracy (minimize)")
    ax.set_ylabel("Latency [ms] (minimize)")
    ax.set_title(f"Pareto Front: Accuracy vs Latency ({len(pareto_trials)} Pareto-optimal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase4_pareto_front.png", dpi=120)
    print("\n  [Saved: phase4_pareto_front.png]")
    plt.close()


def plot_conditional_param_distribution(study: optuna.Study):
    """Plot parameter distributions grouped by optimizer type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    optimizer_types = ["sgd", "adam", "rmsprop"]
    colors = ["blue", "green", "orange"]

    # Plot 1: Loss distribution by optimizer
    for opt_type, color in zip(optimizer_types, colors):
        trials = [t for t in completed if t.params.get("optimizer") == opt_type]
        if trials:
            values = [t.value for t in trials]
            axes[0].hist(values, bins=15, alpha=0.5, color=color, label=opt_type)
    axes[0].set_xlabel("Loss")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Loss Distribution by Optimizer")
    axes[0].legend()

    # Plot 2: Learning rate distribution by optimizer
    for opt_type, color in zip(optimizer_types, colors):
        trials = [t for t in completed if t.params.get("optimizer") == opt_type]
        if trials:
            lr_key = f"{opt_type}_lr"
            lrs = [t.params[lr_key] for t in trials if lr_key in t.params]
            if lrs:
                axes[1].hist(np.log10(lrs), bins=15, alpha=0.5, color=color, label=opt_type)
    axes[1].set_xlabel("log10(learning_rate)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Learning Rate Distribution by Optimizer")
    axes[1].legend()

    # Plot 3: n_layers vs loss (shared parameter)
    for opt_type, color in zip(optimizer_types, colors):
        trials = [t for t in completed if t.params.get("optimizer") == opt_type]
        if trials:
            layers = [t.params["n_layers"] for t in trials]
            values = [t.value for t in trials]
            axes[2].scatter(layers, values, alpha=0.4, color=color, label=opt_type, s=15)
    axes[2].set_xlabel("n_layers")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Architecture vs Loss by Optimizer")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("phase4_conditional_params.png", dpi=120)
    print("  [Saved: phase4_conditional_params.png]")
    plt.close()


# ============================================================================
# Display helpers
# ============================================================================

def print_pareto_summary(study: optuna.Study) -> None:
    """Print the Pareto front from a multi-objective study."""
    pareto = study.best_trials
    print(f"\n  Pareto front: {len(pareto)} trials")
    print(f"  {'Trial':<8s} {'1-Accuracy':>12s} {'Latency':>10s} {'Complexity':>12s} {'GPU':>6s}")
    print(f"  {'-'*50}")

    # Sort by first objective for display
    pareto_sorted = sorted(pareto, key=lambda t: t.values[0])
    for t in pareto_sorted[:15]:
        complexity = t.params.get("model_complexity", "?")
        gpu = t.params.get("use_gpu", "?")
        print(f"  #{t.number:<7d} {t.values[0]:>12.4f} {t.values[1]:>10.2f} "
              f"{complexity:>12.2f} {str(gpu):>6s}")

    if len(pareto) > 15:
        print(f"  ... and {len(pareto) - 15} more Pareto-optimal trials")


def print_conditional_summary(study: optuna.Study) -> None:
    """Print summary grouped by optimizer type."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"\n  Conditional Search Space Results:")
    print(f"  {'Optimizer':<12s} {'Count':>6s} {'Best':>10s} {'Mean':>10s} {'Std':>8s}")
    print(f"  {'-'*50}")

    for opt_type in ["sgd", "adam", "rmsprop"]:
        trials = [t for t in completed if t.params.get("optimizer") == opt_type]
        if trials:
            values = [t.value for t in trials]
            print(f"  {opt_type:<12s} {len(trials):>6d} {min(values):>10.4f} "
                  f"{np.mean(values):>10.4f} {np.std(values):>8.4f}")

    best = min(completed, key=lambda t: t.value)
    print(f"\n  Overall best:")
    print(f"    Loss: {best.value:.6f}")
    print(f"    Params: {best.params}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 4: Optuna Advanced — Multi-Objective & Conditional Search Spaces")
    print("=" * 80)

    # --- Part A: Multi-Objective Optimization ---
    print("\n" + "=" * 80)
    print("Part A: Multi-Objective Optimization (Accuracy vs Latency)")
    print("=" * 80)

    n_trials_mo = 120
    print(f"\nRunning {n_trials_mo} trials with TPE sampler...")
    study_mo = run_multi_objective_study(n_trials_mo)

    completed = [t for t in study_mo.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Completed trials: {len(completed)}")
    print_pareto_summary(study_mo)
    plot_pareto_front(study_mo)

    # Show extreme points on the Pareto front
    pareto = study_mo.best_trials
    if pareto:
        best_acc = min(pareto, key=lambda t: t.values[0])
        best_lat = min(pareto, key=lambda t: t.values[1])
        print(f"\n  Extreme Pareto points:")
        print(f"    Best accuracy: 1-acc={best_acc.values[0]:.4f}, latency={best_acc.values[1]:.2f}ms")
        print(f"      Params: complexity={best_acc.params['model_complexity']:.2f}, "
              f"gpu={best_acc.params['use_gpu']}")
        print(f"    Best latency:  1-acc={best_lat.values[0]:.4f}, latency={best_lat.values[1]:.2f}ms")
        print(f"      Params: complexity={best_lat.params['model_complexity']:.2f}, "
              f"gpu={best_lat.params['use_gpu']}")

    # --- Part B: Conditional Search Spaces ---
    print("\n" + "=" * 80)
    print("Part B: Conditional Search Spaces (Optimizer Selection)")
    print("=" * 80)

    n_trials_cond = 150
    print(f"\nRunning {n_trials_cond} trials with TPE sampler...")

    study_cond = optuna.create_study(
        study_name="conditional_optimizer",
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study_cond.optimize(conditional_objective, n_trials=n_trials_cond)
    print_conditional_summary(study_cond)
    plot_conditional_param_distribution(study_cond)

    # --- Part C: Sampler Comparison on Multi-Objective ---
    print("\n" + "=" * 80)
    print("Part C: Sampler Comparison (TPE vs NSGA-II vs Random)")
    print("=" * 80)

    n_trials_comp = 80
    samplers = {
        "TPE": TPESampler(seed=42),
        "NSGA-II": NSGAIISampler(seed=42),
        "Random": RandomSampler(seed=42),
    }

    print(f"\n  {'Sampler':<12s} {'Pareto size':>12s} {'Best 1-acc':>12s} {'Best latency':>13s}")
    print(f"  {'-'*52}")

    for sampler_name, sampler in samplers.items():
        study = optuna.create_study(
            study_name=f"compare_{sampler_name}",
            directions=["minimize", "minimize"],
            sampler=sampler,
        )
        study.optimize(multi_objective, n_trials=n_trials_comp)

        pareto = study.best_trials
        best_acc = min(pareto, key=lambda t: t.values[0]) if pareto else None
        best_lat = min(pareto, key=lambda t: t.values[1]) if pareto else None

        if best_acc and best_lat:
            print(f"  {sampler_name:<12s} {len(pareto):>12d} "
                  f"{best_acc.values[0]:>12.4f} {best_lat.values[1]:>13.2f}")

    print("\n" + "=" * 80)
    print("Phase 4 Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
