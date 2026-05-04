"""LLM Configuration — Multi-Provider via LangChain
====================================================
Reads provider/model settings from environment variables and returns a
ready-to-call LangChain ``BaseChatModel``.  Falls back to local Ollama
(qwen2.5:7b) when no env vars are set, so the default behaviour is
unchanged.

Why LangChain (and not, e.g., LiteLLM)?  Two reasons specific to this
practice:

1. **Native structured output**.  Every recent ``langchain-*`` package
   ships ``model.with_structured_output(PydanticModel)`` which routes
   through the provider's tool-calling/JSON-mode endpoint when available
   and falls back to a JSON-schema prompt otherwise.  We use this for
   the simulated-user replies and judge verdicts — drop-in replacement
   for what people used to do with ``instructor``.
2. **Tool calling** is the same shape across providers.  The agent loop
   in ``_02_agent_loop.py`` binds a list of LangChain tools once and
   then calls ``.invoke(messages)`` regardless of provider.

This config is shared by:
- ``_01_tools_and_simulated_user.py``  (LLM as simulated user)
- ``_02_agent_loop.py``                (LLM as the agent under test)
- ``_03_judge.py``                     (LLM as a judge)

Environment variables:
  LLM_PROVIDER     ollama (default) | lmstudio | openai | anthropic | google | mistral | groq
  LLM_MODEL        Model name (default: qwen2.5:7b)
  LLM_BASE_URL     Override the provider base URL (optional)
  LLM_API_KEY      API key (required for cloud providers; ignored for local providers)

Optional second LM for the judge (decoupled from the agent's LM):
  JUDGE_PROVIDER   defaults to LLM_PROVIDER
  JUDGE_MODEL      defaults to LLM_MODEL
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

# Load .env if present so callers don't need to export by hand.
load_dotenv()


_SUPPORTED = {"ollama", "lmstudio", "openai", "anthropic", "google", "mistral", "groq"}

_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
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
            f"Unknown LLM_PROVIDER={provider!r}. "
            f"Supported: {sorted(_SUPPORTED)}"
        )
    base_url = os.getenv("LLM_BASE_URL", "") or _DEFAULT_URLS.get(provider, "")
    api_key = os.getenv("LLM_API_KEY", "")
    if provider in ("ollama", "lmstudio") and not api_key:
        api_key = "ollama"  # placeholder; local servers ignore it
    return LMConfig(provider=provider, model=model, base_url=base_url, api_key=api_key)


def get_lm() -> LMConfig:
    """LM config for the agent under test (and the simulated user)."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    return _resolve(provider, model)


def get_judge_lm() -> LMConfig:
    """LM config for the judge.  Falls back to the main LM when unset."""
    provider = os.getenv("JUDGE_PROVIDER") or os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "qwen2.5:7b")
    return _resolve(provider, model)


# ---------------------------------------------------------------------------
# Chat-model factory
# ---------------------------------------------------------------------------


def build_chat_model(cfg: LMConfig, *, temperature: float = 0.0) -> Any:
    """Return a LangChain ``BaseChatModel`` for ``cfg``.

    Ready to ``.invoke(messages)``, ``.bind_tools(tools)``, or
    ``.with_structured_output(SomePydanticModel)``.  Provider imports
    are lazy so ``uv sync`` works without every SDK installed.
    """
    p = cfg.provider
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:11434",
            temperature=temperature,
        )
    if p == "lmstudio":
        # LM Studio exposes an OpenAI-compatible endpoint — route via ChatOpenAI.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:1234/v1",
            api_key=cfg.api_key or "lm-studio",
            temperature=temperature,
        )
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.model, api_key=cfg.api_key, temperature=temperature)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, api_key=cfg.api_key, temperature=temperature)
    if p == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.model, google_api_key=cfg.api_key, temperature=temperature
        )
    if p == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=cfg.model, api_key=cfg.api_key, temperature=temperature)
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=cfg.model, api_key=cfg.api_key, temperature=temperature)
    raise ValueError(f"Unhandled provider: {p!r}")


def chat(model: Any, messages: list, *, retries: int = 6) -> str:
    """One-shot chat completion. Returns ``message.content`` as a string.

    Catches transient rate-limit-ish exceptions (anything whose message
    contains the substring ``rate``) and backs off exponentially.  This
    is intentionally simple — production code should use a proper
    backoff library, but for local-Ollama defaults we almost never hit
    this path.
    """
    for attempt in range(retries):
        try:
            result = model.invoke(messages)
            content = result.content if hasattr(result, "content") else str(result)
            return content if isinstance(content, str) else str(content)
        except Exception as e:  # noqa: BLE001 — bare except is intentional here
            msg = str(e).lower()
            if "rate" not in msg and "429" not in msg:
                raise
            wait = 30 * (attempt + 2)
            print(f"  [rate-limited] sleep {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("rate-limit retries exhausted")
