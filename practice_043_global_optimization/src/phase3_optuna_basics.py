"""Phase 3: Bayesian Hyperparameter Optimization with Optuna.

Introduces Optuna's define-by-run API for black-box optimization.
Creates studies with TPE sampler, mixed parameter types, and pruning.
Compares TPE against random sampling and CMA-ES.

Key Optuna concepts:
  - Study: optimization session (create_study)
  - Trial: single evaluation within a study
  - Sampler: algorithm for suggesting parameters (TPE, Random, CMA-ES)
  - Pruner: early stopping of unpromising trials (MedianPruner)
  - suggest_float / suggest_int / suggest_categorical: parameter definition
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Suppress Optuna's verbose logging (set to WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Synthetic objective (simulates an ML training loop)
# ============================================================================

def synthetic_ml_objective_raw(
    learning_rate: float,
    n_layers: int,
    dropout: float,
    activation: str,
    batch_size: int,
    n_epochs: int = 20,
    seed: int = 42,
) -> list[float]:
    """Simulates an ML training loop returning per-epoch validation losses.

    This is a synthetic function designed to have:
    - A clear optimal region (lr ~0.01, layers ~3, dropout ~0.3)
    - Noisy evaluation (simulated training variance)
    - Multiple parameter types (float, int, categorical)
    - Intermediate values (per-epoch losses) for pruning

    Returns:
        List of validation losses for each epoch (lower is better).
    """
    rng = np.random.default_rng(seed)

    # Activation function penalty (ReLU is best, then tanh, then sigmoid)
    activation_penalty = {"relu": 0.0, "tanh": 0.15, "sigmoid": 0.35}[activation]

    # Learning rate: too high diverges, too low underfits
    lr_penalty = 2.0 * (np.log10(learning_rate) + 2) ** 2  # optimal around lr=0.01

    # Architecture: sweet spot around 3 layers
    arch_penalty = 0.3 * (n_layers - 3) ** 2

    # Dropout: sweet spot around 0.3
    dropout_penalty = 2.0 * (dropout - 0.3) ** 2

    # Batch size: moderate is best
    bs_penalty = 0.1 * (np.log2(batch_size) - 5) ** 2  # optimal around batch_size=32

    # Base loss that decreases with epochs
    base_loss = lr_penalty + arch_penalty + dropout_penalty + bs_penalty + activation_penalty

    losses = []
    for epoch in range(n_epochs):
        # Exponential decay toward the base loss + noise
        progress = 1 - np.exp(-0.3 * (epoch + 1))
        noise = rng.normal(0, 0.05)
        loss = (1 - progress) * 2.0 + progress * base_loss + noise
        losses.append(max(0.01, loss))

    return losses


# ============================================================================
# TODO(human): Optuna objective function with pruning
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """Optuna objective: define search space and evaluate.

    Args:
        trial: Optuna Trial object used to suggest parameter values.

    Returns:
        Final validation loss (float) — lower is better.
    """
    # TODO(human): Define an Optuna Objective with Mixed Parameter Types and Pruning
    #
    # Optuna's "define-by-run" API means you define the search space INSIDE the
    # objective function, not before creating the study. This allows conditional
    # parameters (Phase 4) and dynamic search spaces.
    #
    # Step 1: Suggest parameters using the trial object.
    #   Each suggest_* method registers a parameter in the trial and returns a value
    #   sampled by the study's sampler (TPE, Random, or CMA-ES).
    #
    #   learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    #     - "learning_rate" is the parameter name (must be unique per trial)
    #     - 1e-5 to 1.0 is the range
    #     - log=True means sample in log-space (uniform in log10(lr))
    #       This is critical for learning rates because 0.001 and 0.01 should
    #       be as likely as 0.1 and 1.0 — linear sampling would oversample large values.
    #
    #   n_layers = trial.suggest_int("n_layers", 1, 6)
    #     - Integer parameter: 1, 2, 3, 4, 5, or 6
    #
    #   dropout = trial.suggest_float("dropout", 0.0, 0.7)
    #     - Float in [0, 0.7], linear scale (no log=True)
    #
    #   activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    #     - Categorical: one of the three strings
    #
    #   batch_size = trial.suggest_int("batch_size", 8, 128, log=True)
    #     - Integer sampled in log-space: 8, 16, 32, 64, 128 are roughly equi-probable
    #
    # Step 2: Evaluate the synthetic objective to get per-epoch losses.
    #   losses = synthetic_ml_objective_raw(
    #       learning_rate, n_layers, dropout, activation, batch_size,
    #       seed=trial.number  # different seed per trial for noise
    #   )
    #
    # Step 3: Report intermediate values and check for pruning.
    #   For each epoch's loss, call:
    #     trial.report(loss, step=epoch)
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()
    #
    #   trial.report(value, step) records an intermediate value at the given step.
    #   trial.should_prune() asks the pruner (e.g., MedianPruner) whether this
    #   trial is performing worse than the median of completed trials at this step.
    #   If so, raise TrialPruned to stop the trial early — saving compute.
    #
    # Step 4: Return the final loss (last epoch's value).
    #   return losses[-1]
    #
    # The TPE sampler will use the reported final values to build l(x) and g(x)
    # distributions and suggest better parameters in future trials.
    raise NotImplementedError("TODO(human): implement Optuna objective with pruning")


# ============================================================================
# TODO(human): Run an Optuna study
# ============================================================================

def run_study(
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner | None,
    n_trials: int,
    study_name: str,
) -> optuna.Study:
    """Create and run an Optuna study.

    Args:
        sampler: Sampling algorithm (TPE, Random, CMA-ES)
        pruner: Pruning strategy (MedianPruner or None)
        n_trials: Number of trials to run
        study_name: Human-readable name for the study

    Returns:
        The completed Optuna Study object.
    """
    # TODO(human): Create and Optimize an Optuna Study
    #
    # Step 1: Create the study.
    #   study = optuna.create_study(
    #       study_name=study_name,
    #       direction="minimize",   # we want to minimize validation loss
    #       sampler=sampler,        # TPESampler(), RandomSampler(), etc.
    #       pruner=pruner,          # MedianPruner() or None
    #   )
    #
    #   direction="minimize" tells Optuna that lower objective values are better.
    #   For maximization problems, use direction="maximize".
    #
    # Step 2: Run the optimization.
    #   study.optimize(objective, n_trials=n_trials)
    #
    #   This calls objective(trial) n_trials times, using the sampler to
    #   suggest parameters and the pruner to stop unpromising trials.
    #   After all trials, study.best_trial has the best parameters found.
    #
    # Step 3: Return the study.
    #   return study
    #
    # The study object contains all results:
    #   study.best_trial.params  — best parameter values found
    #   study.best_trial.value   — best objective value
    #   study.trials             — list of all Trial objects
    #   len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    #     → count of pruned trials
    raise NotImplementedError("TODO(human): implement Optuna study creation and optimization")


# ============================================================================
# Visualization
# ============================================================================

def plot_optimization_history(studies: dict[str, optuna.Study]):
    """Plot optimization history for each study (best value vs trial number)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"TPE + Pruning": "blue", "Random + Pruning": "red", "TPE (no pruning)": "green"}

    for name, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            continue

        values = [t.value for t in trials]
        best_so_far = np.minimum.accumulate(values)
        ax.plot(range(len(best_so_far)), best_so_far,
                label=f"{name} (best: {min(values):.4f})",
                color=colors.get(name, "gray"), linewidth=2)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective Value")
    ax.set_title("Optimization History: Best Value vs Trial")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase3_optimization_history.png", dpi=120)
    print("\n  [Saved: phase3_optimization_history.png]")
    plt.close()


def plot_parameter_importances(study: optuna.Study):
    """Plot parameter importance using fANOVA (requires completed trials)."""
    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)

        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(importances.keys())
        values = list(importances.values())
        ax.barh(names, values, color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title("Parameter Importances (fANOVA)")
        plt.tight_layout()
        plt.savefig("phase3_param_importances.png", dpi=120)
        print("  [Saved: phase3_param_importances.png]")
        plt.close()
    except Exception as e:
        print(f"  [Skipped parameter importance plot: {e}]")


# ============================================================================
# Display helpers
# ============================================================================

def print_study_summary(name: str, study: optuna.Study) -> None:
    """Print summary of a completed study."""
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n  {name}:")
    print(f"    Total trials:    {len(trials)}")
    print(f"    Completed:       {len(completed)}")
    print(f"    Pruned:          {len(pruned)} ({100*len(pruned)/max(1,len(trials)):.0f}%)")

    if completed:
        values = [t.value for t in completed]
        print(f"    Best value:      {min(values):.6f}")
        print(f"    Worst value:     {max(values):.6f}")
        print(f"    Mean value:      {np.mean(values):.6f}")

        best = study.best_trial
        print(f"    Best params:")
        for k, v in best.params.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.6f}")
            else:
                print(f"      {k}: {v}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Phase 3: Optuna — Bayesian Hyperparameter Optimization")
    print("=" * 80)

    n_trials = 80
    studies = {}

    # --- Study 1: TPE with Median Pruning ---
    print("\n" + "-" * 80)
    print(f"Study 1: TPE Sampler + MedianPruner ({n_trials} trials)")
    print("-" * 80)

    tpe_sampler = TPESampler(seed=42, n_startup_trials=10)
    median_pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study_tpe = run_study(tpe_sampler, median_pruner, n_trials, "TPE + Pruning")
    print_study_summary("TPE + Pruning", study_tpe)
    studies["TPE + Pruning"] = study_tpe

    # --- Study 2: Random with Median Pruning ---
    print("\n" + "-" * 80)
    print(f"Study 2: Random Sampler + MedianPruner ({n_trials} trials)")
    print("-" * 80)

    random_sampler = RandomSampler(seed=42)
    study_random = run_study(random_sampler, median_pruner, n_trials, "Random + Pruning")
    print_study_summary("Random + Pruning", study_random)
    studies["Random + Pruning"] = study_random

    # --- Study 3: TPE without Pruning ---
    print("\n" + "-" * 80)
    print(f"Study 3: TPE Sampler, no pruning ({n_trials} trials)")
    print("-" * 80)

    tpe_sampler_2 = TPESampler(seed=42, n_startup_trials=10)
    study_tpe_no_prune = run_study(tpe_sampler_2, None, n_trials, "TPE (no pruning)")
    print_study_summary("TPE (no pruning)", study_tpe_no_prune)
    studies["TPE (no pruning)"] = study_tpe_no_prune

    # --- Comparison ---
    print("\n" + "=" * 80)
    print("COMPARISON: Sampler & Pruner Strategies")
    print("=" * 80)
    print(f"\n  {'Study':<25s} {'Best':>10s} {'Mean':>10s} {'Pruned':>8s} {'Total evals':>12s}")
    print(f"  {'-'*67}")

    for name, study in studies.items():
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        values = [t.value for t in completed] if completed else [float("inf")]
        print(f"  {name:<25s} {min(values):>10.4f} {np.mean(values):>10.4f} "
              f"{len(pruned):>8d} {len(completed):>12d}")

    # --- Visualization ---
    plot_optimization_history(studies)
    plot_parameter_importances(study_tpe)

    # --- Print best configuration ---
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION FOUND (TPE + Pruning)")
    print("=" * 80)
    best = study_tpe.best_trial
    print(f"  Objective value: {best.value:.6f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
