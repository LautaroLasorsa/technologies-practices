"""Phase 3 — Custom modules via ``dspy.Module`` subclassing.

DSPy's most powerful composition pattern mirrors PyTorch's ``nn.Module``:

- ``__init__`` declares sub-modules (each wrapping a signature).
- ``forward`` defines the logic that chains them together.

Each sub-module is independently optimizable, so the optimizer can find
*different* few-shot examples for each step of the pipeline.

Run: uv run python -m src._03_custom_module
"""

import dspy

from .llm_config import configure_lm


# -- Setup: configure LM ----------------------------------------------------

def configure_dspy() -> None:
    configure_lm()


# -- Sample product reviews for testing -------------------------------------

SAMPLE_REVIEWS: list[str] = [
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


class SentimentAnalyzer(dspy.Module):
    """Two-step sentiment pipeline: extract aspects, then analyze sentiment."""

    # -----------------------------------------------------------------------
    # TODO(human) #1 — Declare sub-modules in __init__
    # -----------------------------------------------------------------------
    # Each sub-module is a separate DSPy module wrapping its own signature.
    # Storing them as attributes (self.extract, self.analyze) is what lets
    # the optimizer discover and optimize them independently — it walks
    # `named_predictors()` to find every sub-module.
    #
    # What to do:
    #   - self.extract = dspy.ChainOfThought("review -> aspects: list[str]")
    #       ^ Extracts key aspects mentioned in the review (e.g., "display",
    #         "battery", "keyboard"). Uses CoT so the model reasons about
    #         what constitutes an "aspect" before listing them.
    #
    #   - self.analyze = dspy.ChainOfThought(
    #         "review, aspects -> sentiment, confidence"
    #     )
    #       ^ Given the original review AND the extracted aspects, determines
    #         overall sentiment and confidence. Having aspects as input helps
    #         the model weigh positive vs negative mentions systematically.
    # -----------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("TODO(human): Declare self.extract and self.analyze")

    # -----------------------------------------------------------------------
    # TODO(human) #2 — Chain the sub-modules in forward()
    # -----------------------------------------------------------------------
    # forward() is where you define how data flows between steps. Full
    # Python control: call sub-modules, transform their outputs, branch,
    # loop — whatever you need. Return a dspy.Prediction so callers get
    # attribute access to every interesting field.
    #
    # What to do:
    #   - Call self.extract(review=review) to get an extraction with
    #     `.aspects`.
    #   - Call self.analyze(review=review, aspects=extraction.aspects) to
    #     get an analysis with `.sentiment`, `.confidence`, `.reasoning`.
    #   - Return a dspy.Prediction bundling the four fields:
    #       return dspy.Prediction(
    #           aspects=extraction.aspects,
    #           sentiment=analysis.sentiment,
    #           confidence=analysis.confidence,
    #           reasoning=analysis.reasoning,
    #       )
    #
    # Why two steps beats one: a single "review -> sentiment" call asks the
    # LM to form a vague impression. Extracting aspects first forces it to
    # weigh positive vs negative mentions systematically.
    # -----------------------------------------------------------------------
    def forward(self, review: str) -> dspy.Prediction:
        raise NotImplementedError("TODO(human): Chain extract -> analyze and bundle the fields")


# -- Demo loop (scaffolded) --------------------------------------------------

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
