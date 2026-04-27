"""Phase 1 — Configure DSPy and try a string signature.

Two foundational concepts wired together:

1. Configure a language-model backend so every module in the process
   uses the same LM (a global default, set once via ``dspy.configure``).
2. Declare a task with a string signature like ``"question -> answer"``
   — DSPy turns the field names into a prompt template and parses the
   reply back into a ``Prediction`` whose attributes match the output
   field names.

Run: uv run python -m src._01_first_signature
"""

import dspy


# Three sample questions exercise the same predictor below.
QUESTIONS: list[str] = [
    "What is the capital of France?",
    "Explain recursion in one sentence.",
    "What are the three primary colors?",
]


# ---------------------------------------------------------------------------
# TODO(human) #1 — Configure DSPy to use local Ollama
# ---------------------------------------------------------------------------
# DSPy needs a configured language model before any module can run. The LM
# configuration is a global setting that affects ALL subsequent dspy.Predict,
# dspy.ChainOfThought, etc. calls in the process.
#
# What to do:
#   1. Create an LM instance using dspy.LM() with the model identifier
#      "ollama_chat/qwen2.5:7b" — the "ollama_chat/" prefix tells DSPy's
#      LiteLLM backend to route the request to Ollama's chat API.
#   2. Set api_base="http://localhost:11434" to point at your local Ollama.
#   3. Set api_key="" (empty string) — Ollama doesn't require authentication,
#      but LiteLLM expects the parameter to exist.
#   4. Call dspy.configure(lm=your_lm) to set this as the global default.
#
# After this, every dspy module in this script will use your local Ollama
# automatically — no need to pass the LM explicitly to each module.
# ---------------------------------------------------------------------------
def configure_dspy() -> None:
    raise NotImplementedError("TODO(human): Configure DSPy with local Ollama")


# ---------------------------------------------------------------------------
# TODO(human) #2 — Build a Predict module from a string signature
# ---------------------------------------------------------------------------
# A string signature like "question -> answer" is DSPy's simplest abstraction.
# It declares: "given a 'question' input, produce an 'answer' output." DSPy
# converts this into a prompt template, sends it to the configured LM, and
# parses the response back into named fields.
#
# What to do:
#   1. Create a dspy.Predict module with the string signature
#      "question -> answer".
#   2. Return that module — the demo loop below will call it for each
#      question in QUESTIONS.
#
# Observe (when you run the demo): DSPy returns a Prediction object. Access
# output fields as attributes (result.answer), not as dict keys. The field
# name "answer" comes directly from your signature string.
# ---------------------------------------------------------------------------
def build_qa_predictor() -> dspy.Predict:
    raise NotImplementedError("TODO(human): Build dspy.Predict from string signature")


# -- Demo loop (scaffolded) --------------------------------------------------

def run_demo() -> None:
    qa = build_qa_predictor()
    for question in QUESTIONS:
        result = qa(question=question)
        print(f"\n  Q: {question}")
        print(f"  A: {result.answer}")


def main() -> None:
    configure_dspy()
    print("=" * 70)
    print("String signature: question -> answer")
    print("=" * 70)
    run_demo()


if __name__ == "__main__":
    main()
