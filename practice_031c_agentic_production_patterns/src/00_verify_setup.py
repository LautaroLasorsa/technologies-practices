"""Verify that all services (Ollama, Qdrant, Langfuse) are reachable.

Run after `docker compose up -d` and `docker exec ollama ollama pull qwen2.5:7b`:
    uv run python src/00_verify_setup.py
"""

from __future__ import annotations

import sys
import urllib.request
import json


OLLAMA_URL = "http://localhost:11434/api/tags"
QDRANT_URL = "http://localhost:6333/healthz"
LANGFUSE_URL = "http://localhost:3000/api/public/health"


def check_service(name: str, url: str, timeout: int = 5) -> bool:
    """Check if a service is reachable via HTTP GET."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            print(f"  [OK]   {name:12s} — {url} (HTTP {status})")
            return True
    except Exception as e:
        print(f"  [FAIL] {name:12s} — {url} ({e})")
        return False


def check_ollama_model(model: str = "qwen2.5:7b") -> bool:
    """Check if the required model is pulled in Ollama."""
    try:
        req = urllib.request.Request(OLLAMA_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            if any(model in m for m in models):
                print(f"  [OK]   {'Model':12s} — {model} is available")
                return True
            else:
                print(f"  [FAIL] {'Model':12s} — {model} not found. Available: {models}")
                print(f"         Run: docker exec ollama ollama pull {model}")
                return False
    except Exception as e:
        print(f"  [FAIL] {'Model':12s} — Could not check models ({e})")
        return False


def main() -> None:
    print("=" * 60)
    print("Practice 031c — Service Verification")
    print("=" * 60)
    print()

    results = [
        check_service("Ollama", OLLAMA_URL),
        check_service("Qdrant", QDRANT_URL),
        check_service("Langfuse", LANGFUSE_URL),
    ]
    print()

    if results[0]:
        results.append(check_ollama_model())
        print()

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"All {total} checks passed. Ready to start!")
    else:
        print(f"{passed}/{total} checks passed. Fix failing services before continuing.")
        print()
        print("Troubleshooting:")
        print("  1. Run: docker compose up -d")
        print("  2. Wait 10-15 seconds for services to initialize")
        print("  3. For Langfuse: first visit http://localhost:3000 to trigger DB migration")
        print("  4. For Ollama model: docker exec ollama ollama pull qwen2.5:7b")
        sys.exit(1)


if __name__ == "__main__":
    main()
