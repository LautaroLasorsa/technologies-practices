"""LLM Configuration — Multi-Provider Support
==========================================
Reads provider/model settings from environment variables and returns an
``LMConfig`` that ``chat()`` knows how to dispatch.  Falls back to local
Ollama (qwen2.5:7b) when no env vars are set, so the default behaviour
is unchanged.

We route every call through **LiteLLM** (the same library DSPy uses under
the hood) so a single ``chat()`` function transparently speaks Ollama,
LM Studio, OpenAI, Anthropic, Google Gemini, Mistral, and Groq.  Each
provider has its own native auth + URL conventions; LiteLLM hides them
behind a uniform ``{provider}/{model}`` string.

Why LiteLLM rather than the raw ``openai`` client?  Anthropic and Gemini
do not accept OpenAI-style chat-completions on their own SDKs.  Going
through LiteLLM is the same shortcut the other AI/LLM practices in this
repo use, and it keeps the recursive-query plumbing in
``_02_recursive_query.py`` provider-agnostic.

Supported providers (set ``LLM_PROVIDER``):
  ollama     — local Ollama  (default, http://localhost:11434)
  lmstudio   — local LM Studio (http://localhost:1234/v1)
  openai     — OpenAI API
  anthropic  — Anthropic Claude
  google     — Google Gemini
  mistral    — Mistral La Plateforme
  groq       — Groq (OpenAI-compatible)

Environment variables:
  LLM_PROVIDER   Provider name (default: ollama)
  LLM_MODEL      Root model name (default: qwen2.5:7b)
  LLM_SUB_MODEL  Sub-LM model used inside recursive query() calls.
                 Defaults to LLM_MODEL — set to a smaller/cheaper
                 model to mirror the GPT-5 / GPT-5-mini split from
                 the original RLM paper.
  LLM_BASE_URL   Override the provider base URL (optional)
  LLM_API_KEY    API key (required for cloud providers; ignored for
                 local providers)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import litellm
from litellm.exceptions import RateLimitError
import time

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
    """Resolved provider settings for one LM (root or sub)."""
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


def get_root_lm() -> LMConfig:
    """Config for the *root* model — the one writing REPL code."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    return _resolve(provider, model)


def get_sub_lm() -> LMConfig:
    """Config for the *sub* model invoked inside recursive query() calls.

    Defaults to the root model. Override with LLM_SUB_MODEL to mirror
    the original RLM experiments, where the root model was a frontier
    model (GPT-5) and the sub model a cheaper one (GPT-5-mini).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_SUB_MODEL") or os.getenv("LLM_MODEL", "qwen2.5:7b")
    return _resolve(provider, model)


_last_call = time.time()
SECONDS_BETWEEN_CALLS = 3

def chat(cfg: LMConfig, messages: list[dict], temperature: float = 0.0,
         max_tokens: int | None = None) -> str:
    """One-shot chat completion. Returns the assistant string content."""
    global _last_call
    # time.sleep(max(0,  _last_call + SECONDS_BETWEEN_CALLS - time.time()))
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
            _last_call = time.time()
            answer = resp.choices[0].message.content or ""
        #    print(f"{messages}\n\n{answer}")
            return answer
        except RateLimitError:
            wait = 30 * (attempt + 2)
            print(f"  [rate-limited] sleep {wait}s (attempt {attempt+1}/6)")
            time.sleep(wait)
    raise RuntimeError("rate-limit retries exhausted")
