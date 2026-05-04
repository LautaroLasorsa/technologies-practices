"""LLM Configuration — Multi-Provider Support (LangChain)
========================================================
Reads provider/model settings from environment variables and returns an
``LMConfig`` plus a factory that builds a LangChain ``BaseChatModel``
appropriate for the configured backend.  Falls back to local Ollama
(``qwen2.5:7b``) when no env vars are set.

We use **LangChain** (not LiteLLM) here because every CoVe stage that
produces a structured object — the verification *plan* and the per-question
*verdict* — relies on LangChain's ``model.with_structured_output(Schema)``
helper to coerce the LLM's reply into a Pydantic model.  That replaces the
``instructor`` dependency you may have seen in earlier practices.

Provider routing:
- ``ollama``    -> ``ChatOllama``
- ``lmstudio``  -> ``ChatOpenAI`` against the LM Studio OpenAI-compatible
                   server (``http://localhost:1234/v1``)
- ``openai``    -> ``ChatOpenAI``
- ``anthropic`` -> ``ChatAnthropic``
- ``google``    -> ``ChatGoogleGenerativeAI``
- ``mistral``   -> ``ChatMistralAI``
- ``groq``      -> ``ChatGroq``

Each provider package is imported lazily inside its branch, so ``uv sync``
works without installing every SDK.  Install only what you need:

    uv sync --extra openai          # adds langchain-openai
    uv sync --extra all             # adds every provider

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
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

# Load .env if present so callers don't need to export by hand.
load_dotenv()

_SUPPORTED = {"ollama", "lmstudio", "openai", "anthropic", "google", "mistral", "groq"}

_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
    # Cloud providers: leave URL empty so each LangChain integration uses
    # its own default endpoint.
}


@dataclass
class LMConfig:
    """Resolved provider settings for one LM."""
    provider: str
    model: str
    base_url: str
    api_key: str


def _resolve(provider: str, model: str) -> LMConfig:
    if provider not in _SUPPORTED:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Supported: {sorted(_SUPPORTED)}"
        )
    base_url = os.getenv("LLM_BASE_URL", "") or _DEFAULT_URLS.get(provider, "")
    api_key = os.getenv("LLM_API_KEY", "")
    if provider == "lmstudio" and not api_key:
        api_key = "lm-studio"  # ChatOpenAI rejects an empty key even for local servers
    return LMConfig(provider=provider, model=model, base_url=base_url, api_key=api_key)


def get_lm() -> LMConfig:
    """Single LM config used everywhere in this practice."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    return _resolve(provider, model)


def build_chat_model(cfg: LMConfig, *, temperature: float = 0.0) -> "BaseChatModel":
    """Construct a LangChain ``BaseChatModel`` for ``cfg``.

    The provider package is imported lazily so callers without a given
    SDK installed can still import this module.  Override ``temperature``
    for the *baseline* stage if you want CoVe to stress its hallucination
    floor; verification + refinement should stay near 0.0.
    """
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:11434",
            temperature=temperature,
        )

    if cfg.provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:1234/v1",
            api_key=cfg.api_key or "lm-studio",
            temperature=temperature,
        )

    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=temperature,
        )

    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=temperature,
        )

    if cfg.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.model,
            google_api_key=cfg.api_key,
            temperature=temperature,
        )

    if cfg.provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=temperature,
        )

    if cfg.provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=temperature,
        )

    raise ValueError(f"Unhandled provider {cfg.provider!r}")  # pragma: no cover


def chat(
    model: "BaseChatModel",
    messages: list["BaseMessage"] | list[dict],
    *,
    retries: int = 6,
) -> str:
    """One-shot chat completion through a pre-built LangChain model.

    Returns the assistant string content.  Retries on rate-limit errors
    with a simple linear backoff — enough for local development; tune
    ``retries`` per provider for production.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            result = model.invoke(messages)
            content = result.content
            return content if isinstance(content, str) else str(content)
        except Exception as exc:  # noqa: BLE001 — narrow at call site if needed
            msg = str(exc).lower()
            if "rate" not in msg and "429" not in msg:
                raise
            last_exc = exc
            wait = 30 * (attempt + 2)
            print(f"  [rate-limited] sleep {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("rate-limit retries exhausted") from last_exc
