"""Phase 0 — Verify DSPy + Ollama connection.

Sends one tiny `dspy.Predict` call through the configured LM so you know the
backend is reachable before starting the real exercises.

Run: uv run python -m src._00_verify_setup
Prereq: docker compose up -d && docker exec ollama ollama pull qwen2.5:7b
"""

import dspy

from .llm_config import configure_lm


def test_basic_call(lm: dspy.LM) -> None:
    """Send a simple test query to verify the connection works."""
    print("Testing DSPy + Ollama connection...")
    print(f"  Model: {lm.model}")
    print(f"  API base: {lm.api_base}")

    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What is 2 + 2?")

    print("\n  Question: What is 2 + 2?")
    print(f"  Answer:   {result.answer}")
    print("\nSetup verified successfully!")


def main() -> None:
    lm = configure_lm()
    test_basic_call(lm)


if __name__ == "__main__":
    main()
