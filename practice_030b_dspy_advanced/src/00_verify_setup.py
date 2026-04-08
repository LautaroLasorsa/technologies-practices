"""
Phase 0 -- Verify Setup: DSPy + Ollama + Qdrant Connection
============================================================
This script verifies that DSPy can communicate with the local Ollama instance
and that the Qdrant vector database is reachable.

Run: uv run python src/00_verify_setup.py
Prereq: docker compose up -d && docker exec ollama ollama pull qwen2.5:7b
"""

import dspy
from qdrant_client import QdrantClient

from llm_config import LLM_MODEL, LLM_PROVIDER, configure_lm


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def verify_dspy_connection() -> None:
    """Configure DSPy with local Ollama and run a test query."""
    print("Testing DSPy + Ollama connection...")
    lm = configure_lm()

    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What is the third planet from the Sun?")

    print(f"  Provider: {LLM_PROVIDER}")
    print(f"  Model:    {LLM_MODEL}")
    print(f"  Question: What is the third planet from the Sun?")
    print(f"  Answer:   {result.answer}")
    print("  DSPy connection OK\n")


def verify_qdrant_connection() -> None:
    """Check that the Qdrant instance is reachable and healthy."""
    print("Testing Qdrant connection...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    collections = client.get_collections().collections
    print(f"  Host:        {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"  Collections: {len(collections)}")
    for col in collections:
        print(f"    - {col.name}")
    print("  Qdrant connection OK\n")


def main() -> None:
    verify_dspy_connection()
    verify_qdrant_connection()
    print("All checks passed. Ready for practice!")


if __name__ == "__main__":
    main()
