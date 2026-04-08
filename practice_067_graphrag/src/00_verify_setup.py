"""Verify that the configured LLM provider is running and required models are available."""

import urllib.request
import json
import sys

from llm_config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)


def check_ollama_connection(base_url: str) -> list[str]:
    """Check if an Ollama-compatible server is reachable and return its model list."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"[ERROR] Cannot connect to {base_url}: {e}")
        return []


def main() -> None:
    print("=" * 60)
    print("GraphRAG Practice — Setup Verification")
    print("=" * 60)
    print(f"\nProvider : {LLM_PROVIDER}")
    print(f"Chat model      : {LLM_MODEL}")
    print(f"Embedding model : {EMBEDDING_MODEL}")

    if LLM_PROVIDER not in ("ollama", "lmstudio"):
        print(f"\n[INFO] Provider '{LLM_PROVIDER}' is not Ollama/LM Studio.")
        print("  Skipping local connectivity check — ensure the provider is reachable.")
        print("  Set LLM_PROVIDER=ollama for the standard local setup.")
        return

    # 1. Check Ollama connection
    print("\n[1/3] Checking Ollama connection...")
    models = check_ollama_connection(LLM_BASE_URL)
    if not models:
        print("  FAIL: Ollama is not running or not reachable.")
        print("  Run: docker compose up -d")
        sys.exit(1)
    print(f"  OK: Ollama is running. Available models: {models}")

    # 2. Check chat model
    print(f"\n[2/3] Checking chat model ({LLM_MODEL})...")
    if any(LLM_MODEL in m for m in models):
        print(f"  OK: {LLM_MODEL} is available.")
    else:
        print(f"  MISSING: {LLM_MODEL} not found.")
        print(f"  Run: docker exec graphrag_ollama ollama pull {LLM_MODEL}")
        sys.exit(1)

    # 3. Check embedding model
    embed_base = EMBEDDING_BASE_URL or LLM_BASE_URL
    print(f"\n[3/3] Checking embedding model ({EMBEDDING_MODEL})...")
    embed_models = check_ollama_connection(embed_base) if embed_base != LLM_BASE_URL else models
    if any(EMBEDDING_MODEL in m for m in embed_models):
        print(f"  OK: {EMBEDDING_MODEL} is available.")
    else:
        print(f"  MISSING: {EMBEDDING_MODEL} not found.")
        print(f"  Run: docker exec graphrag_ollama ollama pull {EMBEDDING_MODEL}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All checks passed! Ready for GraphRAG indexing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
