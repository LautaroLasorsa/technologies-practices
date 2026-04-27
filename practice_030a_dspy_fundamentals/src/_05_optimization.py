"""Phase 5 — BootstrapFewShot: compile, evaluate, inspect demos.

Brings everything together. ``BootstrapFewShot``:

1. Runs the program (as "teacher") on each training example.
2. Filters traces with your metric (keeping only correct ones).
3. Injects the surviving traces into the student program as few-shot demos.

The compiled student then performs better because its prompts now contain
curated, task-specific examples.

Run: uv run python -m src._05_optimization

NOTE: the SentimentAnalyzer / sentiment_metric / create_datasets stubs
below are intentionally separate from phases 3 and 4 so this file is
runnable on its own. After you finish phases 3–4, paste your
implementations into the matching stubs here.
"""

import dspy
from dspy.teleprompt import BootstrapFewShot

from .llm_config import configure_lm


# -- Setup: configure LM ----------------------------------------------------

def configure_dspy() -> None:
    configure_lm()


# -- Reuse SentimentAnalyzer and data from previous phases -------------------

RAW_DATA: list[dict[str, str]] = [
    {"review": "Absolutely love this product! Works perfectly and arrived early.", "sentiment": "positive"},
    {"review": "Complete waste of money. Broke after two days of normal use.", "sentiment": "negative"},
    {"review": "It's okay. Does what it says but nothing special.", "sentiment": "neutral"},
    {"review": "The quality is outstanding. Best purchase I've made this year.", "sentiment": "positive"},
    {"review": "Terrible experience. The item was damaged and customer service was unhelpful.", "sentiment": "negative"},
    {"review": "Decent for the price. Some minor issues but overall acceptable.", "sentiment": "neutral"},
    {"review": "Exceeded all my expectations! Five stars without hesitation.", "sentiment": "positive"},
    {"review": "Do not buy this. It looks nothing like the pictures.", "sentiment": "negative"},
    {"review": "Works as described. Not amazing, not terrible. Just fine.", "sentiment": "neutral"},
    {"review": "Incredible build quality and the performance is top-notch.", "sentiment": "positive"},
    {"review": "The worst purchase I've ever made. Completely non-functional.", "sentiment": "negative"},
    {"review": "Average product. Gets the job done but I expected more polish.", "sentiment": "neutral"},
    {"review": "Surprisingly good! The features are well thought out and intuitive.", "sentiment": "positive"},
    {"review": "Cheap materials, poor construction. Fell apart in a week.", "sentiment": "negative"},
    {"review": "It's functional but unremarkable. Middle of the road.", "sentiment": "neutral"},
    {"review": "A masterpiece of engineering. Worth every penny and more.", "sentiment": "positive"},
    {"review": "Arrived broken. Replacement was also defective. Never again.", "sentiment": "negative"},
    {"review": "Standard quality. Nothing to complain about, nothing to praise.", "sentiment": "neutral"},
    {"review": "Perfect gift! The recipient absolutely loved it.", "sentiment": "positive"},
    {"review": "Flimsy and poorly designed. Save your money for something better.", "sentiment": "negative"},
    {"review": "Does the job. Not the best but not the worst either.", "sentiment": "neutral"},
    {"review": "Game changer! This has completely transformed my workflow.", "sentiment": "positive"},
    {"review": "Misleading description. The actual product is nothing like advertised.", "sentiment": "negative"},
    {"review": "Meets basic expectations. Could use some improvements.", "sentiment": "neutral"},
    {"review": "Premium feel and excellent attention to detail. Highly recommend!", "sentiment": "positive"},
]

TRAIN_SIZE = 5


class SentimentAnalyzer(dspy.Module):
    """Stub — paste your Phase 3 implementation here."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError(
            "Copy your SentimentAnalyzer.__init__ from _03_custom_module.py"
        )

    def forward(self, review: str) -> dspy.Prediction:
        raise NotImplementedError(
            "Copy your SentimentAnalyzer.forward from _03_custom_module.py"
        )


def sentiment_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Stub — paste your Phase 4 metric here."""
    raise NotImplementedError(
        "Copy your sentiment_metric from _04_metrics_datasets.py"
    )


def create_datasets() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Stub — paste your Phase 4 dataset builder here."""
    raise NotImplementedError(
        "Copy your create_datasets from _04_metrics_datasets.py"
    )


# ---------------------------------------------------------------------------
# TODO(human) #1 — Compile the SentimentAnalyzer with BootstrapFewShot
# ---------------------------------------------------------------------------
# BootstrapFewShot is DSPy's simplest optimizer. It works by:
#   1. Running the program (as "teacher") on each training example
#   2. Evaluating each run with your metric function
#   3. Keeping traces that score above the threshold as demonstrations
#   4. Injecting those demos into the program's prompts
#
# Key parameters:
#   - metric: your evaluation function (sentiment_metric)
#   - max_bootstrapped_demos: max number of auto-generated demos per module
#     (these come from successful teacher traces). Start with 4.
#   - max_labeled_demos: max number of labeled examples to include directly
#     from the training set (without bootstrapping). Start with 8.
#   - max_rounds: how many optimization rounds to run (default 1 is fine).
#
# What to do:
#   1. Build the optimizer:
#        optimizer = BootstrapFewShot(
#            metric=sentiment_metric,
#            max_bootstrapped_demos=4,
#            max_labeled_demos=8,
#        )
#   2. Build a baseline (unoptimized) SentimentAnalyzer instance.
#   3. Compile a fresh student:
#        optimized = optimizer.compile(
#            student=SentimentAnalyzer(),
#            trainset=train_set,
#        )
#   4. Return (baseline, optimized).
#
# Compilation will take a minute — the teacher runs on every training
# example and the metric filters the results. Watch the output for traces
# being accepted/rejected.
# ---------------------------------------------------------------------------
def compile_program(
    train_set: list[dspy.Example],
) -> tuple[SentimentAnalyzer, SentimentAnalyzer]:
    raise NotImplementedError("TODO(human): Build optimizer, return (baseline, optimized)")


# ---------------------------------------------------------------------------
# TODO(human) #2 — Score baseline vs optimized with dspy.Evaluate
# ---------------------------------------------------------------------------
# dspy.Evaluate runs your program on a dataset and computes the average
# metric score. By comparing baseline vs optimized scores, you see the
# concrete impact of BootstrapFewShot optimization.
#
# What to do:
#   1. Build an evaluator:
#        evaluator = dspy.Evaluate(
#            devset=val_set,
#            metric=sentiment_metric,
#            num_threads=1,          # single thread for local Ollama
#            display_progress=True,
#        )
#   2. Score both programs:
#        baseline_score  = evaluator(baseline)
#        optimized_score = evaluator(optimized)
#   3. Return (baseline_score, optimized_score). The wrapper below prints
#      both numbers and the delta.
# ---------------------------------------------------------------------------
def score_programs(
    baseline: SentimentAnalyzer,
    optimized: SentimentAnalyzer,
    val_set: list[dspy.Example],
) -> tuple[float, float]:
    raise NotImplementedError("TODO(human): Run dspy.Evaluate on baseline and optimized")


# ---------------------------------------------------------------------------
# TODO(human) #3 — Inspect what the optimizer learned
# ---------------------------------------------------------------------------
# Every compiled DSPy program exposes its sub-modules via
# `named_predictors()`. Each predictor stores the demonstrations the
# optimizer chose for it in `predictor.demos`. Reading these demos is
# how you see WHICH examples BootstrapFewShot decided would best teach
# the model your task — the auto-curated demos are what makes the
# optimized program better.
#
# What to do:
#   - Iterate `optimized.named_predictors()` and print, for each one:
#       * the predictor name
#       * len(predictor.demos)
#       * each demo (a dspy.Example) on its own line
# ---------------------------------------------------------------------------
def print_optimized_demos(optimized: SentimentAnalyzer) -> None:
    raise NotImplementedError("TODO(human): Walk named_predictors() and print demos")


# -- Reporting glue (scaffolded) --------------------------------------------

def evaluate_programs(
    baseline: SentimentAnalyzer,
    optimized: SentimentAnalyzer,
    val_set: list[dspy.Example],
) -> None:
    baseline_score, optimized_score = score_programs(baseline, optimized, val_set)
    print(f"\n  Baseline score:  {baseline_score:.3f}")
    print(f"  Optimized score: {optimized_score:.3f}")
    print(f"  Improvement:     {optimized_score - baseline_score:+.3f}")

    print("\n--- Optimized program demos ---")
    print_optimized_demos(optimized)


def main() -> None:
    configure_dspy()

    print("=" * 70)
    print("Phase 5: BootstrapFewShot Optimization")
    print("=" * 70)

    print("\n--- Creating datasets ---")
    train_set, val_set = create_datasets()
    print(f"  Train: {len(train_set)} examples")
    print(f"  Val:   {len(val_set)} examples")

    print("\n--- Compiling with BootstrapFewShot ---")
    baseline, optimized = compile_program(train_set)

    print("\n--- Evaluating baseline vs optimized ---")
    evaluate_programs(baseline, optimized, val_set)


if __name__ == "__main__":
    main()
