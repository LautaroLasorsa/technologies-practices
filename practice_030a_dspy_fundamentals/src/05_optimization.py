"""
Phase 5 — BootstrapFewShot Optimization: Compile & Evaluate
=============================================================
This script brings everything together: compile the SentimentAnalyzer
with BootstrapFewShot, then evaluate baseline vs optimized performance.

BootstrapFewShot works by:
1. Running a teacher LM on training examples
2. Filtering traces with your metric (keeping only correct ones)
3. Injecting the best traces as few-shot demonstrations into the student

The result is a compiled program that performs better because it has
curated, task-specific examples in its prompt.

Run: uv run python src/05_optimization.py
"""

import dspy
from dspy.teleprompt import BootstrapFewShot


# -- Setup: configure LM ----------------------------------------------------

def configure_dspy() -> None:
    lm = dspy.LM(
        "ollama_chat/qwen2.5:7b",
        api_base="http://localhost:11434",
        api_key="",
    )
    dspy.configure(lm=lm)


# -- Reuse SentimentAnalyzer and data from previous phases -------------------
# NOTE: After you complete phases 3 and 4, copy your implementations here
# or import them. For now, we provide minimal stubs that you'll replace.

RAW_DATA = [
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
    """Stub — replace with your implementation from Phase 3."""

    def __init__(self) -> None:
        super().__init__()
        # TODO: Copy your __init__ from 03_custom_module.py after completing it
        raise NotImplementedError(
            "Copy your SentimentAnalyzer implementation from 03_custom_module.py"
        )

    def forward(self, review: str) -> dspy.Prediction:
        # TODO: Copy your forward() from 03_custom_module.py after completing it
        raise NotImplementedError(
            "Copy your SentimentAnalyzer implementation from 03_custom_module.py"
        )


def sentiment_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Stub — replace with your implementation from Phase 4."""
    # TODO: Copy your metric from 04_metrics_datasets.py after completing it
    raise NotImplementedError(
        "Copy your sentiment_metric implementation from 04_metrics_datasets.py"
    )


def create_datasets() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Stub — replace with your implementation from Phase 4."""
    # TODO: Copy your dataset creation from 04_metrics_datasets.py after completing it
    raise NotImplementedError(
        "Copy your create_datasets implementation from 04_metrics_datasets.py"
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
#   1. Create the optimizer:
#        optimizer = BootstrapFewShot(
#            metric=sentiment_metric,
#            max_bootstrapped_demos=4,
#            max_labeled_demos=8,
#        )
#   2. Create a baseline (unoptimized) SentimentAnalyzer instance.
#   3. Compile the program:
#        optimized = optimizer.compile(
#            student=SentimentAnalyzer(),
#            trainset=train_set,
#        )
#   4. Return both baseline and optimized programs.
#
# Note: compilation will take a minute — the teacher runs on every training
# example and the metric filters the results. Watch the output for traces
# being accepted/rejected.
# ---------------------------------------------------------------------------
def compile_program(
    train_set: list[dspy.Example],
) -> tuple[SentimentAnalyzer, SentimentAnalyzer]:
    raise NotImplementedError("TODO(human): Compile with BootstrapFewShot")


# ---------------------------------------------------------------------------
# TODO(human) #2 — Evaluate baseline vs optimized program
# ---------------------------------------------------------------------------
# dspy.Evaluate runs your program on a dataset and computes the average
# metric score. By comparing baseline vs optimized scores, you see the
# concrete impact of BootstrapFewShot optimization.
#
# What to do:
#   1. Create an evaluator:
#        evaluator = dspy.Evaluate(
#            devset=val_set,
#            metric=sentiment_metric,
#            num_threads=1,          # single thread for local Ollama
#            display_progress=True,
#        )
#
#   2. Evaluate the baseline (unoptimized) program:
#        baseline_score = evaluator(baseline)
#
#   3. Evaluate the optimized program:
#        optimized_score = evaluator(optimized)
#
#   4. Print both scores and the improvement delta.
#
#   5. Inspect what the optimizer learned — look at the optimized program's
#      demos. For each predictor in the optimized program:
#        for name, predictor in optimized.named_predictors():
#            print(f"\nPredictor: {name}")
#            print(f"  Number of demos: {len(predictor.demos)}")
#            for i, demo in enumerate(predictor.demos):
#                print(f"  Demo {i}: {demo}")
#
#      This reveals WHICH examples the optimizer selected as most useful.
#      These auto-curated demos are what makes the optimized program better.
# ---------------------------------------------------------------------------
def evaluate_programs(
    baseline: SentimentAnalyzer,
    optimized: SentimentAnalyzer,
    val_set: list[dspy.Example],
) -> None:
    raise NotImplementedError("TODO(human): Evaluate and compare programs")


def main() -> None:
    configure_dspy()

    print("=" * 70)
    print("Phase 5: BootstrapFewShot Optimization")
    print("=" * 70)

    # Step 1: Create datasets (reuse from Phase 4)
    print("\n--- Creating datasets ---")
    train_set, val_set = create_datasets()
    print(f"  Train: {len(train_set)} examples")
    print(f"  Val:   {len(val_set)} examples")

    # Step 2: Compile
    print("\n--- Compiling with BootstrapFewShot ---")
    baseline, optimized = compile_program(train_set)

    # Step 3: Evaluate
    print("\n--- Evaluating baseline vs optimized ---")
    evaluate_programs(baseline, optimized, val_set)


if __name__ == "__main__":
    main()
