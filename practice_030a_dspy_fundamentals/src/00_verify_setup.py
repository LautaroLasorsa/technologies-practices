"""
Phase 0 — Verify Setup: DSPy + Ollama Connection
==================================================
This script verifies that DSPy can communicate with the local Ollama instance.
It configures the LM backend, sends a simple test query, and prints the result.

Run: uv run python src/00_verify_setup.py
Prereq: docker compose up -d && docker exec ollama ollama pull qwen2.5:7b
"""

import dspy

from llm_config import configure_lm


def test_basic_call(lm: dspy.LM) -> None:
    """Send a simple test query to verify the connection works."""
    print("Testing DSPy + Ollama connection...")
    print(f"  Model: {lm.model}")
    print(f"  API base: {lm.api_base}")

    # Simple predict call to verify end-to-end
    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What is 2 + 2?")

    print(f"\n  Question: What is 2 + 2?")
    print(f"  Answer:   {result.answer}")
    print("\nSetup verified successfully!")


def main() -> None:
    lm = configure_lm()
    test_basic_call(lm)


if __name__ == "__main__":
    main()
