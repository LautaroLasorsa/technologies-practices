"""LLM client factory — creates OpenAI-compatible or Anthropic clients.

Centralises all provider-specific wiring so that the rest of the codebase
(introspection, memory, etc.) only ever sees an `instructor.Instructor` and a
model name string.

Supported providers
-------------------
* ollama    — local Ollama server (default, http://localhost:11434/v1)
* lmstudio  — local LM Studio server (default, http://localhost:1234/v1)
* openai    — OpenAI API (requires LLM_API_KEY)
* anthropic — Anthropic API (requires LLM_API_KEY; needs `anthropic` package)
* <other>   — treated as an OpenAI-compatible endpoint; set LLM_BASE_URL + LLM_API_KEY
"""

from __future__ import annotations

import instructor
from openai import OpenAI

from src.models import AgentConfig

# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_instructor_client(config: AgentConfig) -> instructor.Instructor:
    """Return an instructor-patched client for the configured provider."""
    provider = config.provider
    if provider == "anthropic":
        return _make_anthropic_client(config)
    return _make_openai_compatible_client(config)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_openai_compatible_client(config: AgentConfig) -> instructor.Instructor:
    """Build an instructor client over any OpenAI-compatible endpoint."""
    base_url, api_key = _resolve_openai_params(config)
    raw_client = OpenAI(base_url=base_url or None, api_key=api_key)
    # JSON mode works with Ollama and LM Studio; OpenAI itself supports TOOLS mode
    # but JSON is broadly compatible across providers, so we use it everywhere.
    return instructor.from_openai(raw_client, mode=instructor.Mode.JSON)


def _resolve_openai_params(config: AgentConfig) -> tuple[str, str]:
    """Return (base_url, api_key) for OpenAI-compatible providers."""
    base_url = config.base_url
    api_key = config.api_key

    if config.provider == "ollama":
        base_url = base_url or "http://localhost:11434/v1"
        api_key = api_key or "ollama"
    elif config.provider == "lmstudio":
        base_url = base_url or "http://localhost:1234/v1"
        api_key = api_key or "lmstudio"
    elif config.provider == "openai":
        # base_url stays empty → OpenAI SDK uses its own default
        if not api_key:
            raise ValueError("LLM_API_KEY must be set for provider='openai'")
    else:
        # Generic OpenAI-compatible: both base_url and api_key come from config
        api_key = api_key or "api_key"

    return base_url, api_key


def _make_anthropic_client(config: AgentConfig) -> instructor.Instructor:
    """Build an instructor client for the Anthropic API."""
    try:
        import anthropic  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for provider='anthropic'. "
            "Install it with: uv add anthropic"
        ) from exc

    if not config.api_key:
        raise ValueError("LLM_API_KEY must be set for provider='anthropic'")

    raw_client = anthropic.Anthropic(api_key=config.api_key)
    return instructor.from_anthropic(raw_client)


# ---------------------------------------------------------------------------
# Raw OpenAI client (used by verify_setup for model listing)
# ---------------------------------------------------------------------------

def create_raw_openai_client(config: AgentConfig) -> OpenAI:
    """Return a plain (non-instructor-patched) OpenAI client for health checks.

    Only meaningful for OpenAI-compatible providers (ollama, lmstudio, openai).
    Anthropic does not expose a /models endpoint via the OpenAI SDK.
    """
    base_url, api_key = _resolve_openai_params(config)
    return OpenAI(base_url=base_url or None, api_key=api_key)
