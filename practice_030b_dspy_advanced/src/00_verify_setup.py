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


OLLAMA_BASE = "http://localhost:11434"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MODEL_ID = "ollama_chat/qwen2.5:7b"


def verify_dspy_connection() -> None:
    """Configure DSPy with local Ollama and run a test query."""
    print("Testing DSPy + Ollama connection...")
    lm = dspy.LM(MODEL_ID, api_base=OLLAMA_BASE, api_key="")
    dspy.configure(lm=lm)

    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What is the third planet from the Sun?")

    print(f"  Model:    {lm.model}")
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
