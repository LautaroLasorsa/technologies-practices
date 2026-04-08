"""LLM provider factory for multi-provider support.

Reads configuration from environment variables so the practice works with
Ollama (default), LM Studio, OpenAI, Anthropic, or Google Gemini — without
touching agent code.

Environment variables:
    LLM_PROVIDER  — Provider name. One of: ollama (default), lmstudio, openai,
                    anthropic, google.
    LLM_MODEL     — Model name without provider prefix (default: qwen2.5:7b).
    LLM_BASE_URL  — Override base URL for the provider. Optional; each
                    provider has a sensible default.
    LLM_API_KEY   — API key for cloud providers. Optional for local providers.

Examples:
    # Default — Ollama with qwen2.5:7b (no env vars needed)
    uv run python main.py

    # OpenAI GPT-4o
    LLM_PROVIDER=openai LLM_MODEL=gpt-4o LLM_API_KEY=sk-... uv run python main.py

    # Anthropic Claude 3 Haiku
    LLM_PROVIDER=anthropic LLM_MODEL=claude-3-haiku-20240307 LLM_API_KEY=... uv run python main.py

    # LM Studio (OpenAI-compatible)
    LLM_PROVIDER=lmstudio LLM_MODEL=lmstudio-community/qwen2.5-7b uv run python main.py
"""

import os

from google.adk.models.lite_llm import LiteLlm

# Expose resolved values so callers can print/log what's active.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


def get_model() -> LiteLlm:
    """Return a LiteLlm instance configured from environment variables.

    Reads LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, and LLM_API_KEY to
    configure the right LiteLLM model string and the corresponding
    provider API key / base-URL environment variables that LiteLLM expects.

    Returns:
        A LiteLlm instance ready to pass to an ADK Agent.

    Raises:
        ValueError: If LLM_PROVIDER is set to an unknown value.
    """
    provider = LLM_PROVIDER
    model = LLM_MODEL
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    if provider in ("ollama", "lmstudio"):
        return _build_ollama_model(provider, model, base_url)
    elif provider == "openai":
        return _build_openai_model(model, api_key)
    elif provider == "anthropic":
        return _build_anthropic_model(model, api_key)
    elif provider == "google":
        return _build_google_model(model, api_key)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            "Valid values: ollama, lmstudio, openai, anthropic, google."
        )


def _build_ollama_model(provider: str, model: str, base_url: str) -> LiteLlm:
    """Configure LiteLLM for Ollama or LM Studio (both use ollama_chat/)."""
    default_url = (
        "http://localhost:11434"
        if provider == "ollama"
        else "http://localhost:1234/v1"
    )
    url = base_url or default_url
    os.environ.setdefault("OLLAMA_API_BASE", url)
    return LiteLlm(model=f"ollama_chat/{model}")


def _build_openai_model(model: str, api_key: str) -> LiteLlm:
    """Configure LiteLLM for OpenAI."""
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    return LiteLlm(model=f"openai/{model}")


def _build_anthropic_model(model: str, api_key: str) -> LiteLlm:
    """Configure LiteLLM for Anthropic."""
    if api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
    return LiteLlm(model=f"anthropic/{model}")


def _build_google_model(model: str, api_key: str) -> LiteLlm:
    """Configure LiteLLM for Google Gemini."""
    if api_key:
        os.environ.setdefault("GEMINI_API_KEY", api_key)
    return LiteLlm(model=f"gemini/{model}")
