"""
LLM Configuration — Multi-Provider Support
==========================================
Reads provider/model settings from environment variables and returns a
configured dspy.LM instance. Falls back to local Ollama (qwen2.5:7b)
when no env vars are set, so the default behaviour is unchanged.

Supported providers (set LLM_PROVIDER):
  ollama    — local Ollama  (default, http://localhost:11434)
  lmstudio  — local LM Studio (http://localhost:1234/v1)
  openai    — OpenAI API
  anthropic — Anthropic API
  google    — Google Gemini API

Environment variables:
  LLM_PROVIDER   Provider name (default: ollama)
  LLM_MODEL      Model name without prefix (default: qwen2.5:7b)
  LLM_BASE_URL   Override the provider base URL (optional)
  LLM_API_KEY    API key (required for cloud providers)
"""

import os

import dspy


# Expose resolved values so callers can inspect them without re-reading env.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


def get_lm() -> dspy.LM:
    """Build and return a dspy.LM for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    if provider in ("ollama", "lmstudio"):
        return _build_local_lm(provider, model, base_url, api_key)
    elif provider == "openai":
        return dspy.LM(f"openai/{model}", api_key=api_key, api_base=base_url or None)
    elif provider == "anthropic":
        return dspy.LM(f"anthropic/{model}", api_key=api_key)
    elif provider == "google":
        return dspy.LM(f"gemini/{model}", api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def configure_lm() -> dspy.LM:
    """Build an LM, set it as the DSPy global default, and return it."""
    lm = get_lm()
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_local_lm(
    provider: str, model: str, base_url: str, api_key: str
) -> dspy.LM:
    _DEFAULT_URLS = {
        "ollama": "http://localhost:11434",
        "lmstudio": "http://localhost:1234/v1",
    }
    url = base_url or _DEFAULT_URLS[provider]
    return dspy.LM(f"ollama_chat/{model}", api_base=url, api_key=api_key or "")
