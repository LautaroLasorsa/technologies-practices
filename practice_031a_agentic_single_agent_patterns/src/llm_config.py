"""LLM provider configuration — multi-provider factory.

Supports: ollama (default), lmstudio, openai, anthropic, google.

Environment variables:
    LLM_PROVIDER  — one of: ollama, lmstudio, openai, anthropic, google (default: ollama)
    LLM_MODEL     — model name (default: qwen2.5:7b)
    LLM_BASE_URL  — base URL override (optional; provider defaults used if unset)
    LLM_API_KEY   — API key for cloud providers (optional for local providers)
"""

import os


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


def get_chat_model(**kwargs):
    """Return a LangChain chat model configured from environment variables.

    Kwargs are forwarded to the underlying model constructor (e.g. temperature=0).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    if provider in ("ollama", "lmstudio"):
        from langchain_ollama import ChatOllama
        url = base_url or ("http://localhost:11434" if provider == "ollama" else "http://localhost:1234/v1")
        return ChatOllama(model=model, base_url=url, **kwargs)

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "LLM_PROVIDER='openai' requires langchain-openai. Install: uv add langchain-openai"
            )
        return ChatOpenAI(model=model, api_key=api_key or None, base_url=base_url or None, **kwargs)

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "LLM_PROVIDER='anthropic' requires langchain-anthropic. Install: uv add langchain-anthropic"
            )
        return ChatAnthropic(model=model, api_key=api_key or None, **kwargs)

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "LLM_PROVIDER='google' requires langchain-google-genai. Install: uv add langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key or None, **kwargs)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Choose from: ollama, lmstudio, openai, anthropic, google")
