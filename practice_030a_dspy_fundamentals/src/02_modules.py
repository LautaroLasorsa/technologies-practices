"""
Phase 2 — Modules: Predict vs ChainOfThought vs ProgramOfThought
=================================================================
This script explores DSPy's three built-in module strategies and introduces
class-based signatures for typed, production-grade task definitions.

Each module wraps a signature with a different prompting strategy:
- Predict: direct LM call, no reasoning trace
- ChainOfThought: adds a "reasoning" field before the output
- ProgramOfThought: generates Python code to compute the answer

Run: uv run python src/02_modules.py
"""

from typing import Literal

import dspy


# -- Setup: configure LM (same as Phase 1) ----------------------------------

def configure_dspy() -> None:
    lm = dspy.LM(
        "ollama_chat/qwen2.5:7b",
        api_base="http://localhost:11434",
        api_key="",
    )
    dspy.configure(lm=lm)


# -- Pre-defined class signature for math word problems ----------------------

class MathProblem(dspy.Signature):
    """Solve the given math word problem step by step."""

    problem = dspy.InputField(desc="A math word problem in natural language")
    answer = dspy.OutputField(desc="The numerical answer to the problem")


# -- Test problems -----------------------------------------------------------

MATH_PROBLEMS = [
    "A store sells apples for $2 each and oranges for $3 each. "
    "If Alice buys 4 apples and 5 oranges, how much does she spend in total?",
    "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
    "What is the total distance traveled?",
    "A rectangle has a perimeter of 30 cm and a width of 5 cm. "
    "What is its area?",
]


# ---------------------------------------------------------------------------
# TODO(human) #1 — Compare Predict, ChainOfThought, and ProgramOfThought
# ---------------------------------------------------------------------------
# Each DSPy module wraps the same signature but prompts the LM differently.
# Running the same problem through all three reveals these differences:
#
# - dspy.Predict(MathProblem): Asks the LM to produce `answer` directly.
#   Fast but may make arithmetic errors on complex problems.
#
# - dspy.ChainOfThought(MathProblem): Adds an intermediate `reasoning`
#   field. The LM must "think out loud" before producing `answer`. This
#   often improves accuracy because the model can self-correct during
#   reasoning. Access the trace via result.reasoning.
#
# - dspy.ProgramOfThought(MathProblem): Asks the LM to generate Python
#   code that computes the answer, then executes it. Eliminates LM
#   arithmetic errors entirely. The generated code is in result.program.
#
# What to do:
#   1. Create one instance of each module: Predict, ChainOfThought,
#      ProgramOfThought — all using the MathProblem signature.
#   2. For each problem in MATH_PROBLEMS, run it through all three modules.
#   3. Print the problem, then for each module print:
#      - Module name
#      - The answer (result.answer)
#      - For CoT: also print result.reasoning
#      - For PoT: also print result.program (if available)
#   4. Compare: which module gives correct answers most consistently?
# ---------------------------------------------------------------------------
def compare_modules() -> None:
    raise NotImplementedError("TODO(human): Compare Predict vs CoT vs PoT")


# ---------------------------------------------------------------------------
# TODO(human) #2 — Define a class-based signature for text classification
# ---------------------------------------------------------------------------
# Class-based signatures are DSPy's production-grade way to define tasks.
# Unlike string signatures, they support:
#   - Typed fields (constrain output format)
#   - Field descriptions (guide the LM's behavior)
#   - A docstring that becomes the task instruction in the prompt
#
# What to do:
#   1. Define a class TextClassification(dspy.Signature) with:
#      - Docstring: "Classify the text into one of the given categories."
#      - text = dspy.InputField(desc="The text to classify")
#      - categories = dspy.InputField(desc="Comma-separated list of valid categories")
#      - category = dspy.OutputField(desc="The single best matching category from the list")
#      - confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
#
#   2. Create a dspy.ChainOfThought(TextClassification) module.
#
#   3. Test it with these examples:
#      - text="The new GPU delivers 50% faster ray tracing", categories="tech, sports, politics, entertainment"
#      - text="The team scored a last-minute goal to win the championship", categories="tech, sports, politics, entertainment"
#      - text="The senator proposed a new tax reform bill", categories="tech, sports, politics, entertainment"
#
#   4. Print: text, predicted category, confidence, and reasoning.
#
# Observe how the field descriptions and docstring shape the LM's behavior.
# The `categories` input field constrains the output — the LM knows it must
# pick from the provided list, not invent new categories.
# ---------------------------------------------------------------------------
def test_class_signature() -> None:
    raise NotImplementedError("TODO(human): Define class-based signature and test")


def main() -> None:
    configure_dspy()

    print("=" * 70)
    print("PART 1: Comparing Module Strategies on Math Problems")
    print("=" * 70)
    compare_modules()

    print("\n" + "=" * 70)
    print("PART 2: Class-Based Signature for Text Classification")
    print("=" * 70)
    test_class_signature()


if __name__ == "__main__":
    main()
