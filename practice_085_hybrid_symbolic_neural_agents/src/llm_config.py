"""LLM Configuration — Multi-Provider Support
==========================================
Reads provider/model settings from environment variables and returns an
``LMConfig`` that ``chat()`` knows how to dispatch.  Falls back to local
Ollama (qwen2.5:7b) when no env vars are set, so the default behaviour
is unchanged.

We route every call through **LiteLLM** so a single ``chat()`` function
transparently speaks Ollama, LM Studio, OpenAI, Anthropic, Google
Gemini, Mistral, and Groq.  Each provider has its own native auth + URL
conventions; LiteLLM hides them behind a uniform ``{provider}/{model}``
string.

This config is shared by:
- ``_01_constraint_extraction.py``   (LLM as structured-extractor)
- ``_04_explainer.py``               (LLM as natural-language explainer)
- ``_05_orchestrator.py``            (LLM-driven control flow)

The CP-SAT modules (``_02_cpsat_solver.py``, ``_03_infeasibility_analyzer.py``)
are LLM-free — that's the whole point of the hybrid pattern.

Environment variables:
  LLM_PROVIDER   ollama (default) | lmstudio | openai | anthropic | google | mistral | groq
  LLM_MODEL      Model name (default: qwen2.5:7b)
  LLM_BASE_URL   Override the provider base URL (optional)
  LLM_API_KEY    API key (required for cloud providers; ignored for local providers)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import litellm
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError

# Load .env if present so callers don't need to export by hand.
load_dotenv()

# LiteLLM expects ``{prefix}/{model}``.  ``ollama_chat`` is the chat-
# completions transport (vs. the raw ``ollama`` /api/generate one).
_PROVIDER_PREFIX = {
    "ollama": "ollama_chat",
    "lmstudio": "openai",       # LM Studio exposes an OpenAI-compatible server
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "mistral": "mistral",
    "groq": "groq",
}

_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
    # Cloud providers: leave URL empty so LiteLLM uses its built-in default.
}


@dataclass
class LMConfig:
    """Resolved provider settings for one LM."""
    provider: str
    model: str
    base_url: str
    api_key: str

    @property
    def litellm_model(self) -> str:
        prefix = _PROVIDER_PREFIX[self.provider]
        return f"{prefix}/{self.model}"


def _resolve(provider: str, model: str) -> LMConfig:
    if provider not in _PROVIDER_PREFIX:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            f"Supported: {sorted(_PROVIDER_PREFIX)}"
        )
    base_url = os.getenv("LLM_BASE_URL", "") or _DEFAULT_URLS.get(provider, "")
    api_key = os.getenv("LLM_API_KEY", "")
    if provider in ("ollama", "lmstudio") and not api_key:
        api_key = "ollama"  # LiteLLM rejects an empty key even for local servers
    return LMConfig(provider=provider, model=model, base_url=base_url, api_key=api_key)


def get_lm() -> LMConfig:
    """Single LM config used everywhere in this practice."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    return _resolve(provider, model)


def chat(
    cfg: LMConfig,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """One-shot chat completion. Returns the assistant string content."""
    kwargs: dict = {
        "model": cfg.litellm_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": 120,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["api_base"] = cfg.base_url
    for attempt in range(6):
        try:
            resp = litellm.completion(**kwargs)
            return resp.choices[0].message.content or ""
        except RateLimitError:
            wait = 30 * (attempt + 2)
            print(f"  [rate-limited] sleep {wait}s (attempt {attempt + 1}/6)")
            time.sleep(wait)
    raise RuntimeError("rate-limit retries exhausted")


def instructor_client(cfg: LMConfig):
    """Return an ``instructor``-patched LiteLLM client for structured output.

    ``instructor`` wraps the LiteLLM ``completion`` callable so the
    ``response_model=PydanticModel`` keyword causes the SDK to coerce the
    LLM reply into that model (with retries on validation errors).
    Used by ``_01_constraint_extraction.py``.
    """
    import instructor

    return instructor.from_litellm(litellm.completion)
