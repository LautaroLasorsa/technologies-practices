"""Verify Ollama connectivity and model availability.

Run:
    uv run python src/00_verify_setup.py
"""

import sys
import urllib.request
import json


OLLAMA_BASE_URL = "http://localhost:11434"
REQUIRED_MODEL = "qwen2.5:7b"


def check_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[FAIL] Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
        print("       Run: docker compose up -d")
        return False


def check_model_available() -> bool:
    """Check if the required model is downloaded."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            model_names = [m["name"] for m in data.get("models", [])]
            # Check both exact match and prefix match (e.g. "qwen2.5:7b" in "qwen2.5:7b")
            for name in model_names:
                if name == REQUIRED_MODEL or name.startswith(REQUIRED_MODEL.split(":")[0]):
                    print(f"[OK]   Model found: {name}")
                    return True
            print(f"[FAIL] Model '{REQUIRED_MODEL}' not found. Available: {model_names}")
            print(f"       Run: docker exec ollama ollama pull {REQUIRED_MODEL}")
            return False
    except Exception as e:
        print(f"[FAIL] Could not list models: {e}")
        return False


def check_model_responds() -> bool:
    """Send a simple prompt to verify the model generates output."""
    try:
        payload = json.dumps({
            "model": REQUIRED_MODEL,
            "prompt": "Say 'hello' and nothing else.",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            response_text = data.get("response", "").strip()
            if response_text:
                print(f"[OK]   Model responds: \"{response_text[:80]}\"")
                return True
            print("[FAIL] Model returned empty response")
            return False
    except Exception as e:
        print(f"[FAIL] Model inference failed: {e}")
        return False


def check_langchain_imports() -> bool:
    """Verify all required Python packages are importable."""
    packages = [
        ("langgraph", "langgraph.graph"),
        ("langchain-core", "langchain_core.messages"),
        ("langchain-ollama", "langchain_ollama"),
        ("pydantic", "pydantic"),
    ]
    all_ok = True
    for name, module in packages:
        try:
            __import__(module)
            print(f"[OK]   {name} importable")
        except ImportError as e:
            print(f"[FAIL] {name} not importable: {e}")
            print("       Run: uv sync")
            all_ok = False
    return all_ok


def main() -> None:
    print("=" * 60)
    print("Practice 031a — Setup Verification")
    print("=" * 60)

    checks = [
        ("Ollama server", check_ollama_running),
        ("Model available", check_model_available),
        ("Python packages", check_langchain_imports),
        ("Model inference", check_model_responds),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        results.append(check_fn())

    print("\n" + "=" * 60)
    if all(results):
        print("All checks passed. Ready to start practicing!")
    else:
        print("Some checks failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
