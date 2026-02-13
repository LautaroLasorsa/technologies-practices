"""
Phase 3 — Custom Modules: Composing Multi-Step Programs
========================================================
This script teaches DSPy's most powerful pattern: building custom modules
by subclassing dspy.Module. Custom modules compose simple modules into
multi-step programs where each step is independently optimizable.

The pattern mirrors PyTorch's nn.Module:
  - __init__: declare sub-modules (each wrapping a signature)
  - forward(): define the logic that chains sub-modules together

Run: uv run python src/03_custom_module.py
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


# -- Sample product reviews for testing -------------------------------------

SAMPLE_REVIEWS = [
    "The laptop's display is absolutely gorgeous with vivid colors, "
    "but the battery barely lasts 3 hours. The keyboard feels great "
    "for typing, though the trackpad is a bit mushy.",

    "Terrible customer service experience. I waited 45 minutes on hold "
    "and the representative couldn't solve my issue. The product itself "
    "is fine but I'll never buy from this company again.",

    "Best headphones I've ever owned! The noise cancellation is incredible, "
    "sound quality is studio-grade, and they're comfortable enough to wear "
    "all day. Battery life is impressive at 30+ hours.",

    "The recipe book has some interesting ideas but the instructions are "
    "confusing and several recipes had wrong measurements. The photography "
    "is beautiful though.",
]


# ---------------------------------------------------------------------------
# TODO(human) — Build a SentimentAnalyzer custom module
# ---------------------------------------------------------------------------
# Custom modules are DSPy's core composition pattern. By subclassing
# dspy.Module, you create a multi-step program where each step is a
# separate DSPy module with its own signature. This is powerful because:
#
# 1. Each sub-module can be optimized independently — the optimizer can
#    find different few-shot examples for aspect extraction vs sentiment
#    analysis.
# 2. The forward() method defines how data flows between steps, giving
#    you full Python control over the pipeline logic.
# 3. The module is reusable, testable, and can be nested inside other
#    modules.
#
# What to do:
#   1. Define class SentimentAnalyzer(dspy.Module) with:
#
#      __init__(self):
#        - self.extract = dspy.ChainOfThought("review -> aspects: list[str]")
#          ^ Extracts key aspects mentioned in the review (e.g., "display",
#            "battery", "keyboard"). Uses CoT so the model reasons about
#            what constitutes an "aspect" before listing them.
#
#        - self.analyze = dspy.ChainOfThought(
#              "review, aspects -> sentiment, confidence"
#          )
#          ^ Given the original review AND the extracted aspects, determines
#            overall sentiment and confidence. Having aspects as input helps
#            the model weigh positive vs negative mentions systematically.
#
#      forward(self, review: str):
#        - Call self.extract(review=review) to get aspects
#        - Call self.analyze(review=review, aspects=extraction.aspects)
#          to get sentiment + confidence
#        - Return a dspy.Prediction with all fields:
#            return dspy.Prediction(
#                aspects=extraction.aspects,
#                sentiment=analysis.sentiment,
#                confidence=analysis.confidence,
#                reasoning=analysis.reasoning,
#            )
#
#   2. Instantiate the module: analyzer = SentimentAnalyzer()
#
#   3. Run it on each review in SAMPLE_REVIEWS. Print:
#      - The review (first 80 chars + "...")
#      - Extracted aspects
#      - Sentiment and confidence
#      - The model's reasoning
#
# Observe: the two-step pipeline (extract then analyze) often produces
# better results than a single "review -> sentiment" call because the
# model considers specific aspects rather than forming a vague impression.
# ---------------------------------------------------------------------------
class SentimentAnalyzer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("TODO(human): Define sub-modules in __init__")

    def forward(self, review: str) -> dspy.Prediction:
        raise NotImplementedError("TODO(human): Chain sub-modules in forward()")


def test_sentiment_analyzer() -> None:
    analyzer = SentimentAnalyzer()

    for i, review in enumerate(SAMPLE_REVIEWS, 1):
        print(f"\n--- Review {i} ---")
        print(f"  Text: {review[:80]}...")
        result = analyzer(review=review)
        print(f"  Aspects:    {result.aspects}")
        print(f"  Sentiment:  {result.sentiment}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning:  {result.reasoning}")


def main() -> None:
    configure_dspy()
    print("=" * 70)
    print("Custom Module: SentimentAnalyzer (extract aspects -> analyze sentiment)")
    print("=" * 70)
    test_sentiment_analyzer()


if __name__ == "__main__":
    main()
