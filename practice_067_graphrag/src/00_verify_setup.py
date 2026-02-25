"""Verify that Ollama is running and required models are available."""

import urllib.request
import json
import sys


def check_ollama_connection() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception as e:
        print(f"[ERROR] Cannot connect to Ollama: {e}")
        return []


def main() -> None:
    print("=" * 60)
    print("GraphRAG Practice — Setup Verification")
    print("=" * 60)

    # 1. Check Ollama connection
    print("\n[1/3] Checking Ollama connection...")
    models = check_ollama_connection()
    if not models:
        print("  FAIL: Ollama is not running or not reachable.")
        print("  Run: docker compose up -d")
        sys.exit(1)
    print(f"  OK: Ollama is running. Available models: {models}")

    # 2. Check chat model
    chat_model = "qwen2.5:7b"
    print(f"\n[2/3] Checking chat model ({chat_model})...")
    if any(chat_model in m for m in models):
        print(f"  OK: {chat_model} is available.")
    else:
        print(f"  MISSING: {chat_model} not found.")
        print(f"  Run: docker exec graphrag_ollama ollama pull {chat_model}")
        sys.exit(1)

    # 3. Check embedding model
    embed_model = "nomic-embed-text"
    print(f"\n[3/3] Checking embedding model ({embed_model})...")
    if any(embed_model in m for m in models):
        print(f"  OK: {embed_model} is available.")
    else:
        print(f"  MISSING: {embed_model} not found.")
        print(f"  Run: docker exec graphrag_ollama ollama pull {embed_model}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All checks passed! Ready for GraphRAG indexing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
