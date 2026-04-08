"""
Practice 030c — LLM Configuration

Central factory for both DSPy (dspy.LM) and LangChain (ChatXxx) clients.
All settings are read from environment variables; defaults match the original
Ollama setup so the practice works out-of-the-box without any .env file.

Environment variables
---------------------
LLM_PROVIDER  : "ollama" (default) | "lmstudio" | "openai" | "anthropic" | "google"
LLM_MODEL     : model name, default "qwen2.5:7b"
LLM_BASE_URL  : base URL override (optional; defaults are set per provider)
LLM_API_KEY   : API key (required for cloud providers; ignored for ollama/lmstudio)
"""

import os

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "dspy_prefix": "ollama_chat/",
        "api_key": "",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234",
        "model": "qwen2.5:7b",
        "dspy_prefix": "openai/",
        "api_key": "lm-studio",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "dspy_prefix": "openai/",
        "api_key": None,  # required
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-3-5-haiku-latest",
        "dspy_prefix": "anthropic/",
        "api_key": None,  # required
    },
    "google": {
        "base_url": None,
        "model": "gemini/gemini-2.0-flash-lite",
        "dspy_prefix": "gemini/",
        "api_key": None,  # required
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").lower()


def _resolve_settings() -> dict:
    """Return the resolved {provider, model, base_url, api_key, dspy_prefix}."""
    provider = _get_provider()
    defaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["ollama"])

    model = os.environ.get("LLM_MODEL") or defaults["model"]
    base_url = os.environ.get("LLM_BASE_URL") or defaults["base_url"]
    api_key = os.environ.get("LLM_API_KEY") or defaults["api_key"]
    dspy_prefix = defaults["dspy_prefix"]

    if api_key is None:
        raise ValueError(
            f"LLM_API_KEY is required for provider '{provider}'. "
            "Set it in your .env file or as an environment variable."
        )

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "dspy_prefix": dspy_prefix,
    }


# ---------------------------------------------------------------------------
# DSPy factory
# ---------------------------------------------------------------------------

def configure_lm() -> None:
    """Configure DSPy's global LM from environment variables.

    Uses dspy.configure(lm=...) so all DSPy modules in the process share it.
    Call this once at program startup.
    """
    import dspy

    s = _resolve_settings()
    dspy_model = f"{s['dspy_prefix']}{s['model']}"

    kwargs: dict = {"model": dspy_model, "api_key": s["api_key"]}
    if s["base_url"] and s["provider"] in ("ollama", "lmstudio"):
        kwargs["api_base"] = f"{s['base_url']}/v1"
    elif s["base_url"] and s["provider"] == "openai":
        kwargs["api_base"] = s["base_url"]

    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# LangChain factory
# ---------------------------------------------------------------------------

def get_chat_model():
    """Return a LangChain chat model instance configured from environment variables.

    Returns an instance of the appropriate ChatXxx class for the active provider.
    Optional provider packages must be installed separately (e.g. langchain-anthropic).
    """
    s = _resolve_settings()
    provider = s["provider"]

    if provider in ("ollama", "lmstudio"):
        return _make_ollama_or_lmstudio(s)
    if provider == "openai":
        return _make_openai(s)
    if provider == "anthropic":
        return _make_anthropic(s)
    if provider == "google":
        return _make_google(s)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Valid values: ollama, lmstudio, openai, anthropic, google"
    )


def _make_ollama_or_lmstudio(s: dict):
    from langchain_ollama import ChatOllama

    kwargs: dict = {"model": s["model"]}
    if s["base_url"]:
        kwargs["base_url"] = s["base_url"]
    return ChatOllama(**kwargs)


def _make_openai(s: dict):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for provider 'openai'. "
            "Install it with: uv add langchain-openai"
        ) from exc

    kwargs: dict = {"model": s["model"], "api_key": s["api_key"]}
    if s["base_url"]:
        kwargs["base_url"] = s["base_url"]
    return ChatOpenAI(**kwargs)


def _make_anthropic(s: dict):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is required for provider 'anthropic'. "
            "Install it with: uv add langchain-anthropic"
        ) from exc

    return ChatAnthropic(model=s["model"], api_key=s["api_key"])


def _make_google(s: dict):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "langchain-google-genai is required for provider 'google'. "
            "Install it with: uv add langchain-google-genai"
        ) from exc

    # Strip the "gemini/" prefix that DSPy uses — LangChain uses bare model names
    model = s["model"].removeprefix("gemini/")
    return ChatGoogleGenerativeAI(model=model, google_api_key=s["api_key"])
