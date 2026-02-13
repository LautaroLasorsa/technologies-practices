"""
Phase 1 — First Signature: Configure DSPy & String Signatures
==============================================================
This script introduces the two foundational DSPy concepts:
1. Configuring a language model backend (pointing DSPy at Ollama)
2. Defining string signatures — the simplest way to declare what an LM should do

String signatures use the format "input_field1, input_field2 -> output_field1, output_field2".
DSPy converts this into a prompt template automatically. The field names become
both the prompt's structure and the keys you use to access results.

Run: uv run python src/01_first_signature.py
"""

import dspy


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
# TODO(human) #2 — Define a string signature and test with dspy.Predict
# ---------------------------------------------------------------------------
# A string signature like "question -> answer" is DSPy's simplest abstraction.
# It declares: "given a 'question' input, produce an 'answer' output." DSPy
# converts this into a prompt template, sends it to the configured LM, and
# parses the response back into named fields.
#
# What to do:
#   1. Create a dspy.Predict module with the string signature "question -> answer".
#      Example: qa = dspy.Predict("question -> answer")
#   2. Call it with three different questions, e.g.:
#      - result = qa(question="What is the capital of France?")
#      - result = qa(question="Explain recursion in one sentence.")
#      - result = qa(question="What are the three primary colors?")
#   3. For each result, print both the question and result.answer.
#
# Observe: DSPy returns a Prediction object. Access output fields as
# attributes (result.answer), not as dict keys. The field name "answer"
# comes directly from your signature string.
# ---------------------------------------------------------------------------
def test_string_signature() -> None:
    raise NotImplementedError("TODO(human): Define string signature and test with Predict")


def main() -> None:
    configure_dspy()
    test_string_signature()


if __name__ == "__main__":
    main()
