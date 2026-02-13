"""
Phase 4 — Metrics & Datasets: Preparing for Optimization
=========================================================
This script teaches two prerequisites for DSPy optimization:
1. Metric functions — how to measure if a program's output is correct
2. dspy.Example datasets — the standard data format for train/val splits

Metrics guide the optimizer: without a good metric, BootstrapFewShot cannot
distinguish useful demonstrations from useless ones.

Run: uv run python src/04_metrics_datasets.py
"""

import dspy


# -- Setup: configure LM ----------------------------------------------------

def configure_dspy() -> None:
    lm = dspy.LM(
        "ollama_chat/qwen2.5:7b",
        api_base="http://localhost:11434",
        api_key="",
    )
    dspy.configure(lm=lm)


# -- Raw labeled data for sentiment analysis ---------------------------------
# Each entry has a review (input) and a gold sentiment label (expected output).
# This simulates a small labeled dataset you'd have in a real project.

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

# Train/val split sizes
TRAIN_SIZE = 5
# The rest goes to validation


# ---------------------------------------------------------------------------
# TODO(human) #1 — Define a metric function for sentiment analysis
# ---------------------------------------------------------------------------
# A metric function is the compass that guides DSPy's optimization. It
# evaluates a single prediction and returns a score. The optimizer calls
# this function on every training example to decide which demonstrations
# are worth keeping.
#
# The metric function signature is:
#   metric(example, prediction, trace=None) -> float | bool
#
# Parameters:
#   - example: a dspy.Example with gold-standard fields (e.g., .sentiment)
#   - prediction: the program's output (a dspy.Prediction with .sentiment)
#   - trace: optional — when not None, the optimizer is in "bootstrapping"
#     mode. You can use this to apply stricter filtering during bootstrap.
#
# What to do:
#   1. Define sentiment_metric(example, pred, trace=None) -> float
#   2. Compare pred.sentiment (lowercased, stripped) with example.sentiment
#      (lowercased, stripped). Case-insensitive matching is important because
#      LMs often capitalize or add whitespace unpredictably.
#   3. Return 1.0 if they match, 0.0 if they don't.
#
# Design consideration: This is an exact-match metric — the simplest kind.
# For production, you might use fuzzy matching, semantic similarity, or
# multi-dimensional scoring. But exact match is the right starting point
# for learning how metrics work in DSPy.
# ---------------------------------------------------------------------------
def sentiment_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    raise NotImplementedError("TODO(human): Implement sentiment metric")


# ---------------------------------------------------------------------------
# TODO(human) #2 — Create train/val datasets as dspy.Example lists
# ---------------------------------------------------------------------------
# dspy.Example is DSPy's standard data container. Each example holds named
# fields (like a dict, but with attribute access). The critical method is
# .with_inputs() — it tells DSPy which fields are inputs (available at
# inference time) vs labels (used only for evaluation/optimization).
#
# Why .with_inputs() matters:
#   Without it, the optimizer might include label fields (like "sentiment")
#   in the prompt as inputs, essentially giving the model the answer. The
#   .with_inputs("review") call says: "only 'review' is an input field;
#   'sentiment' is a label for evaluation only."
#
# What to do:
#   1. Convert RAW_DATA into a list of dspy.Example objects:
#        example = dspy.Example(
#            review=item["review"],
#            sentiment=item["sentiment"]
#        ).with_inputs("review")
#
#   2. Split into train_set (first TRAIN_SIZE examples) and val_set (rest).
#      We use a small train set (5) because BootstrapFewShot works well
#      with few examples — it's generating demonstrations, not fine-tuning.
#
#   3. Return (train_set, val_set) from a function called create_datasets().
#
#   4. Print the sizes and a sample example from each set to verify.
# ---------------------------------------------------------------------------
def create_datasets() -> tuple[list[dspy.Example], list[dspy.Example]]:
    raise NotImplementedError("TODO(human): Create train/val datasets")


def test_metric(val_set: list[dspy.Example]) -> None:
    """Quick sanity check: run the metric on a few mock predictions."""
    print("\n--- Metric Sanity Check ---")

    # Simulate a correct prediction
    correct_pred = dspy.Prediction(sentiment=val_set[0].sentiment)
    score = sentiment_metric(val_set[0], correct_pred)
    print(f"  Correct prediction score: {score} (expected 1.0)")

    # Simulate an incorrect prediction
    wrong_pred = dspy.Prediction(sentiment="wrong_label")
    score = sentiment_metric(val_set[0], wrong_pred)
    print(f"  Incorrect prediction score: {score} (expected 0.0)")


def main() -> None:
    configure_dspy()

    print("=" * 70)
    print("PART 1: Create Datasets")
    print("=" * 70)
    train_set, val_set = create_datasets()
    print(f"  Train size: {len(train_set)}")
    print(f"  Val size:   {len(val_set)}")
    print(f"\n  Sample train example:")
    print(f"    review:    {train_set[0].review[:60]}...")
    print(f"    sentiment: {train_set[0].sentiment}")

    print("\n" + "=" * 70)
    print("PART 2: Test Metric")
    print("=" * 70)
    test_metric(val_set)

    print("\nDatasets and metric ready for Phase 5 (optimization)!")


if __name__ == "__main__":
    main()
