"""Phase 2 — Predict vs ChainOfThought vs ProgramOfThought, plus class signatures.

Each built-in DSPy module wraps the *same* signature with a different
prompting strategy:

- ``Predict`` — direct LM call, no reasoning trace.
- ``ChainOfThought`` — adds an intermediate ``reasoning`` field.
- ``ProgramOfThought`` — generates Python code and executes it.

Part 2 introduces class-based signatures: typed I/O fields with
descriptions and a docstring that becomes the task instruction.

Run: uv run python -m src._02_modules
"""

import dspy

from .llm_config import configure_lm


# -- Setup: configure LM (same as Phase 1) ----------------------------------

def configure_dspy() -> None:
    configure_lm()


# -- Pre-defined class signature for math word problems ----------------------

class MathProblem(dspy.Signature):
    """Solve the given math word problem step by step."""

    problem = dspy.InputField(desc="A math word problem in natural language")
    answer = dspy.OutputField(desc="The numerical answer to the problem")


MATH_PROBLEMS: list[str] = [
    "A store sells apples for $2 each and oranges for $3 each. "
    "If Alice buys 4 apples and 5 oranges, how much does she spend in total?",
    "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
    "What is the total distance traveled?",
    "A rectangle has a perimeter of 30 cm and a width of 5 cm. "
    "What is its area?",
]

CLASSIFICATION_EXAMPLES: list[tuple[str, str]] = [
    ("The new GPU delivers 50% faster ray tracing", "tech, sports, politics, entertainment"),
    ("The team scored a last-minute goal to win the championship", "tech, sports, politics, entertainment"),
    ("The senator proposed a new tax reform bill", "tech, sports, politics, entertainment"),
]


# ---------------------------------------------------------------------------
# TODO(human) #1 — Build a dspy.Predict over MathProblem
# ---------------------------------------------------------------------------
# dspy.Predict(MathProblem) asks the LM to produce `answer` directly from
# `problem`. Fast but may make arithmetic errors on complex problems — no
# intermediate reasoning trace is exposed.
#
# What to do:
#   - Return dspy.Predict(MathProblem).
# ---------------------------------------------------------------------------
def build_predict_module() -> dspy.Predict:
    raise NotImplementedError("TODO(human): Return dspy.Predict(MathProblem)")


# ---------------------------------------------------------------------------
# TODO(human) #2 — Build a dspy.ChainOfThought over MathProblem
# ---------------------------------------------------------------------------
# dspy.ChainOfThought(MathProblem) adds an intermediate `reasoning` field
# before producing `answer`. The LM "thinks out loud" first, which often
# improves accuracy because the model can self-correct mid-reasoning.
# Access the trace via result.reasoning.
#
# What to do:
#   - Return dspy.ChainOfThought(MathProblem).
# ---------------------------------------------------------------------------
def build_cot_module() -> dspy.ChainOfThought:
    raise NotImplementedError("TODO(human): Return dspy.ChainOfThought(MathProblem)")


# ---------------------------------------------------------------------------
# TODO(human) #3 — Build a dspy.ProgramOfThought over MathProblem
# ---------------------------------------------------------------------------
# dspy.ProgramOfThought(MathProblem) asks the LM to *write Python code* that
# computes the answer, then executes the code in a sandbox. This eliminates
# LM arithmetic errors entirely. The generated code may be available on the
# result (e.g., result.program) depending on the DSPy version.
#
# What to do:
#   - Return dspy.ProgramOfThought(MathProblem).
# ---------------------------------------------------------------------------
def build_pot_module() -> dspy.ProgramOfThought:
    raise NotImplementedError("TODO(human): Return dspy.ProgramOfThought(MathProblem)")


# ---------------------------------------------------------------------------
# TODO(human) #4 — Define a class-based signature for text classification
# ---------------------------------------------------------------------------
# Class-based signatures are DSPy's production-grade way to define tasks.
# Unlike string signatures, they support:
#   - Typed fields (constrain output format)
#   - Field descriptions (guide the LM's behavior)
#   - A docstring that becomes the task instruction in the prompt
#
# What to do:
#   - Replace the placeholder below with a real class:
#       class TextClassification(dspy.Signature):
#           """Classify the text into one of the given categories."""
#           text       = dspy.InputField(desc="The text to classify")
#           categories = dspy.InputField(desc="Comma-separated list of valid categories")
#           category   = dspy.OutputField(desc="The single best matching category from the list")
#           confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
#
# The `categories` input field constrains the output — the LM knows it must
# pick from the provided list, not invent new categories.
# ---------------------------------------------------------------------------
# TODO(human): Define the `TextClassification` class signature here, then
# delete this placeholder assignment.
TextClassification: type[dspy.Signature] | None = None


# ---------------------------------------------------------------------------
# TODO(human) #5 — Wrap TextClassification in a ChainOfThought module
# ---------------------------------------------------------------------------
# Pairing a class signature with ChainOfThought lets the LM reason about
# which category fits the text *before* committing to one. The signature's
# docstring + field descriptions become part of the prompt.
#
# What to do:
#   - Return dspy.ChainOfThought(TextClassification).
# ---------------------------------------------------------------------------
def build_classifier() -> dspy.ChainOfThought:
    raise NotImplementedError("TODO(human): Return dspy.ChainOfThought(TextClassification)")


# -- Demo loops (scaffolded) -------------------------------------------------

def _print_module_result(name: str, result: dspy.Prediction) -> None:
    print(f"  [{name}] answer = {result.answer}")
    if hasattr(result, "reasoning") and result.reasoning:
        print(f"    reasoning: {result.reasoning}")
    if hasattr(result, "program") and result.program:
        print(f"    program:   {result.program}")


def compare_modules() -> None:
    predict = build_predict_module()
    cot = build_cot_module()
    pot = build_pot_module()

    for i, problem in enumerate(MATH_PROBLEMS, 1):
        print(f"\n--- Problem {i} ---")
        print(f"  {problem}")
        _print_module_result("Predict", predict(problem=problem))
        _print_module_result("CoT", cot(problem=problem))
        _print_module_result("PoT", pot(problem=problem))


def test_class_signature() -> None:
    classifier = build_classifier()
    for text, categories in CLASSIFICATION_EXAMPLES:
        result = classifier(text=text, categories=categories)
        print(f"\n  Text:       {text}")
        print(f"  Category:   {result.category}")
        print(f"  Confidence: {result.confidence}")
        if hasattr(result, "reasoning") and result.reasoning:
            print(f"  Reasoning:  {result.reasoning}")


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
