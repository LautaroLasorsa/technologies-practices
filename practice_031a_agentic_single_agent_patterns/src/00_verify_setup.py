"""Verify LLM connectivity and model availability.

Adapts to the configured LLM_PROVIDER (default: ollama).

Run:
    uv run python src/00_verify_setup.py
"""

import sys
import urllib.request
import json

from llm_config import LLM_PROVIDER, LLM_MODEL, get_chat_model

import os

_OLLAMA_BASE_URL = os.getenv("LLM_BASE_URL", "") or "http://localhost:11434"


def check_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[FAIL] Ollama not reachable at {_OLLAMA_BASE_URL}: {e}")
        print("       Run: docker compose up -d")
        return False


def check_model_available() -> bool:
    """Check if the required model is downloaded in Ollama."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            model_names = [m["name"] for m in data.get("models", [])]
            # Check both exact match and prefix match (e.g. "qwen2.5:7b" in "qwen2.5:7b")
            for name in model_names:
                if name == LLM_MODEL or name.startswith(LLM_MODEL.split(":")[0]):
                    print(f"[OK]   Model found: {name}")
                    return True
            print(f"[FAIL] Model '{LLM_MODEL}' not found. Available: {model_names}")
            print(f"       Run: docker exec ollama ollama pull {LLM_MODEL}")
            return False
    except Exception as e:
        print(f"[FAIL] Could not list models: {e}")
        return False


def check_model_responds_ollama() -> bool:
    """Send a simple prompt via Ollama HTTP API to verify the model generates output."""
    try:
        payload = json.dumps({
            "model": LLM_MODEL,
            "prompt": "Say 'hello' and nothing else.",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_BASE_URL}/api/generate",
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


def check_model_responds_langchain() -> bool:
    """Send a simple prompt via LangChain to verify model connectivity."""
    try:
        llm = get_chat_model(temperature=0)
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Say 'hello' and nothing else.")])
        text = response.content.strip() if hasattr(response, "content") else str(response)
        if text:
            print(f"[OK]   Model responds: \"{text[:80]}\"")
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


def build_checks_for_provider() -> list:
    """Return the ordered list of (name, check_fn) pairs for the active provider."""
    if LLM_PROVIDER == "ollama":
        return [
            ("Ollama server", check_ollama_running),
            ("Model available", check_model_available),
            ("Python packages", check_langchain_imports),
            ("Model inference", check_model_responds_ollama),
        ]
    else:
        return [
            ("Python packages", check_langchain_imports),
            ("Model inference", check_model_responds_langchain),
        ]


def main() -> None:
    print("=" * 60)
    print("Practice 031a — Setup Verification")
    print(f"Provider : {LLM_PROVIDER}")
    print(f"Model    : {LLM_MODEL}")
    print("=" * 60)

    checks = build_checks_for_provider()

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
