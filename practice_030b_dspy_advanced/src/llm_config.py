"""
LLM Configuration — Multi-Provider Support
===========================================
Centralizes DSPy LM configuration so any provider can be selected via
environment variables without touching practice source files.

Usage:
    from llm_config import configure_lm, LLM_PROVIDER, LLM_MODEL

    configure_lm()  # configures dspy.configure(lm=...) globally

Supported providers (LLM_PROVIDER env var):
    ollama    — local Ollama server (default)
    lmstudio  — local LM Studio OpenAI-compatible server
    openai    — OpenAI API
    anthropic — Anthropic API
    google    — Google Gemini API

See .env.example for all configurable variables.
"""

import os

import dspy


# -- Exported constants for modules that need the raw values -----------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# Default base URLs per provider
_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
}


# -- Factory -----------------------------------------------------------------

def get_lm() -> dspy.LM:
    """Build and return a dspy.LM instance from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    if provider in ("ollama", "lmstudio"):
        url = base_url or _DEFAULT_BASE_URLS[provider]
        return dspy.LM(f"ollama_chat/{model}", api_base=url, api_key=api_key or "")

    if provider == "openai":
        return dspy.LM(f"openai/{model}", api_key=api_key, api_base=base_url or None)

    if provider == "anthropic":
        return dspy.LM(f"anthropic/{model}", api_key=api_key)

    if provider == "google":
        return dspy.LM(f"gemini/{model}", api_key=api_key)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. "
                     "Valid values: ollama, lmstudio, openai, anthropic, google")


def configure_lm() -> dspy.LM:
    """Build the LM and set it as the global DSPy default. Returns the LM."""
    lm = get_lm()
    dspy.configure(lm=lm)
    return lm
