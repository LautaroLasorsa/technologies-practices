"""Multi-provider LLM client factory.

Reads provider configuration from environment variables:

    LLM_PROVIDER   - "ollama" (default) | "lmstudio" | "openai" | "anthropic" | "google"
    LLM_MODEL      - model name (default: "qwen2.5:3b")
    LLM_BASE_URL   - override base URL (optional; provider default used when unset)
    LLM_API_KEY    - API key (required for cloud providers; "ollama" used as default)

Supported providers
-------------------
ollama      Local Ollama via OpenAI-compatible /v1 endpoint (default)
lmstudio    Local LM Studio via OpenAI-compatible /v1 endpoint
openai      OpenAI cloud (requires LLM_API_KEY)
anthropic   Anthropic Claude (requires LLM_API_KEY; uses instructor.from_anthropic)
google      Google Gemini (OpenAI-compatible endpoint; requires LLM_API_KEY + LLM_BASE_URL)

Usage
-----
    from src.llm_config import get_openai_client, get_instructor_client, LLM_MODEL

    # Plain OpenAI client (for non-structured calls)
    client = get_openai_client()

    # Instructor-patched client (for structured output via Pydantic)
    client = get_instructor_client()

Default behaviour (no env vars set) is identical to the original hard-coded
Ollama configuration: base_url="http://localhost:11434/v1", api_key="ollama".
"""

from __future__ import annotations

import os

from openai import OpenAI


# ---------------------------------------------------------------------------
# Module-level constants resolved at import time
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:3b")

_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _resolve_base_url(provider: str, override: str) -> str:
    if override:
        return override
    env_url = os.getenv("LLM_BASE_URL", "")
    if env_url:
        return env_url
    return _DEFAULT_BASE_URLS.get(provider, "")


def _resolve_api_key(override: str) -> str:
    if override:
        return override
    return os.getenv("LLM_API_KEY", "")


def get_openai_client(
    base_url_override: str = "",
    api_key_override: str = "",
) -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider.

    Works for ollama, lmstudio, openai, and any other OpenAI-compatible endpoint.
    For Anthropic use get_instructor_client() which wraps the Anthropic SDK.
    """
    provider = LLM_PROVIDER
    base_url = _resolve_base_url(provider, base_url_override)
    api_key = _resolve_api_key(api_key_override)

    if provider in ("ollama", "lmstudio"):
        return OpenAI(base_url=base_url, api_key=api_key or "ollama")

    # openai / google / other OpenAI-compatible providers
    return OpenAI(
        api_key=api_key or None,
        base_url=base_url or None,
    )


def get_instructor_client(
    base_url_override: str = "",
    api_key_override: str = "",
):
    """Return an instructor-patched client for the configured provider.

    Supports structured output (Pydantic response_model) via instructor.
    For Anthropic, uses instructor.from_anthropic instead of the OpenAI wrapper.
    """
    import instructor

    provider = LLM_PROVIDER

    if provider == "anthropic":
        return _build_anthropic_instructor_client(api_key_override)

    openai_client = get_openai_client(base_url_override, api_key_override)
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


def _build_anthropic_instructor_client(api_key_override: str):
    """Build an instructor client backed by the Anthropic SDK."""
    import instructor

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "LLM_PROVIDER='anthropic' requires the anthropic package. "
            "Install it with: uv add anthropic"
        ) from exc

    api_key = _resolve_api_key(api_key_override)
    return instructor.from_anthropic(Anthropic(api_key=api_key or None))
