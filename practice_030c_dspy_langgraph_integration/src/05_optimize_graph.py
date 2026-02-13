"""
Practice 030c — Phase 5: Optimization Within Graph

DSPy's key differentiator is automatic prompt optimization. In this phase,
you'll optimize individual DSPy modules with BootstrapFewShot, then plug
them into LangGraph nodes and evaluate the full pipeline.

The workflow:
  1. Define training examples for each module
  2. Optimize each module independently
  3. Plug optimized modules into LangGraph
  4. Compare pipeline accuracy: baseline vs. optimized

Run: uv run python src/05_optimize_graph.py
"""

from typing import TypedDict

import dspy
from dspy.teleprompt import BootstrapFewShot
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"


def configure_dspy() -> None:
    """Configure DSPy to use the local Ollama model."""
    lm = dspy.LM(
        model=f"ollama_chat/{OLLAMA_MODEL}",
        api_base=f"{OLLAMA_BASE_URL}/v1",
        api_key="",
    )
    dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class QueryState(TypedDict):
    question: str
    category: str
    answer: str


# ---------------------------------------------------------------------------
# Training data for optimization
# ---------------------------------------------------------------------------

CLASSIFIER_TRAINSET = [
    dspy.Example(question="What is 15 * 23?", category="math").with_inputs("question"),
    dspy.Example(question="Calculate the square root of 144", category="math").with_inputs("question"),
    dspy.Example(question="What is 2^8 - 56?", category="math").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", category="factual").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", category="factual").with_inputs("question"),
    dspy.Example(question="When was the Eiffel Tower built?", category="factual").with_inputs("question"),
    dspy.Example(question="Write a haiku about the ocean", category="creative").with_inputs("question"),
    dspy.Example(question="Invent a name for a fantasy city", category="creative").with_inputs("question"),
    dspy.Example(question="Describe a sunset in three words", category="creative").with_inputs("question"),
]

MATH_TRAINSET = [
    dspy.Example(question="What is 7 * 8?", answer="56").with_inputs("question"),
    dspy.Example(question="What is 100 / 4?", answer="25").with_inputs("question"),
    dspy.Example(question="What is 2^5?", answer="32").with_inputs("question"),
]

FACT_TRAINSET = [
    dspy.Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What year did World War II end?", answer="1945").with_inputs("question"),
]

CREATIVE_TRAINSET = [
    dspy.Example(
        question="Write a one-sentence story about a robot",
        answer="The last robot on Earth spent its days painting sunsets it had never been programmed to appreciate.",
    ).with_inputs("question"),
    dspy.Example(
        question="Describe rain in a single metaphor",
        answer="Rain is the sky's way of weeping with joy.",
    ).with_inputs("question"),
]

# ---------------------------------------------------------------------------
# Evaluation data (separate from training)
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    {"question": "What is 99 + 1?", "expected_category": "math", "expected_answer_contains": "100"},
    {"question": "Who discovered gravity?", "expected_category": "factual", "expected_answer_contains": "Newton"},
    {"question": "Make up a funny word", "expected_category": "creative", "expected_answer_contains": None},
    {"question": "What is 12 * 12?", "expected_category": "math", "expected_answer_contains": "144"},
    {"question": "What is the speed of light?", "expected_category": "factual", "expected_answer_contains": None},
]


# ---------------------------------------------------------------------------
# TODO(human) #9: Optimize individual DSPy modules
# ---------------------------------------------------------------------------
# DSPy's BootstrapFewShot optimizer selects the best few-shot examples for
# a module by trying different combinations and keeping the ones that produce
# correct outputs (as measured by a metric function).
#
# Steps:
#   1. Create the four unoptimized modules:
#      - classifier = dspy.ChainOfThought("question -> category")
#      - math_qa = dspy.ChainOfThought("question -> answer")
#      - fact_qa = dspy.ChainOfThought("question -> answer")
#      - creative_qa = dspy.ChainOfThought("question -> answer")
#
#   2. Define metric functions for each module. For the classifier:
#      def classifier_metric(example, pred, trace=None):
#          return pred.category.strip().lower() == example.category.strip().lower()
#      For math/fact QA:
#      def qa_metric(example, pred, trace=None):
#          return example.answer.lower() in pred.answer.lower()
#      For creative (more lenient — just check it's non-empty):
#      def creative_metric(example, pred, trace=None):
#          return len(pred.answer.strip()) > 10
#
#   3. Optimize each module with BootstrapFewShot:
#      optimizer = BootstrapFewShot(metric=classifier_metric, max_bootstrapped_demos=3)
#      optimized_classifier = optimizer.compile(classifier, trainset=CLASSIFIER_TRAINSET)
#      (Repeat for math_qa, fact_qa, creative_qa with their respective metrics and trainsets)
#
#   4. Return all four optimized modules.
#
# Why optimize independently: Each module has a different task and different
# training data. Optimizing independently lets each module get the best
# few-shot examples for its specific task. Later, we compose them in the graph.

def optimize_modules() -> tuple:
    """Returns (optimized_classifier, optimized_math, optimized_fact, optimized_creative)."""
    raise NotImplementedError("TODO(human): optimize each DSPy module with BootstrapFewShot")


# ---------------------------------------------------------------------------
# TODO(human) #10: Build graph with optimized modules and evaluate
# ---------------------------------------------------------------------------
# Now plug the optimized modules into a LangGraph and evaluate the full
# pipeline against the baseline (unoptimized modules).
#
# Steps:
#   1. Create a function `build_graph(classifier_fn, math_fn, fact_fn, creative_fn)`
#      that builds a LangGraph with the provided node functions.
#      Structure: START -> classify -> conditional_edge -> specialist -> END
#      (Same structure as Phase 3, but parameterized so you can swap modules.)
#
#   2. Build two graphs:
#      a. Baseline graph: using unoptimized DSPy modules
#      b. Optimized graph: using the optimized modules from TODO #9
#
#   3. Define an evaluation function that runs both graphs on EVAL_QUESTIONS:
#      For each question:
#        - Run the graph: result = app.invoke({"question": q, "category": "", "answer": ""})
#        - Check if category matches expected_category
#        - Check if expected_answer_contains is in the answer (if not None)
#        - Score: fraction of checks that pass
#
#   4. Print a comparison table:
#      Question | Baseline Category | Optimized Category | Expected | Baseline Answer | Optimized Answer
#
#   5. Print overall accuracy for baseline vs. optimized.
#
# This demonstrates the full value proposition: LangGraph provides the
# routing structure, DSPy provides optimized LLM calls within each node,
# and the combination can be evaluated end-to-end.

def evaluate_pipeline() -> None:
    raise NotImplementedError("TODO(human): build graphs, run evaluation, compare results")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 5: Optimization Within Graph")
    print("=" * 60)

    print("\n--- Step 1: Optimizing individual modules ---\n")
    optimized = optimize_modules()

    print("\n--- Step 2: Evaluating full pipeline ---\n")
    evaluate_pipeline()


if __name__ == "__main__":
    main()
