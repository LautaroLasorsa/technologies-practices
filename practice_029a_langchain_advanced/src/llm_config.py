"""
Practice 029a — LLM Configuration
Multi-provider chat model factory. Set env vars to switch between providers.

Supported providers (LLM_PROVIDER):
  ollama    - Local Ollama (default). Uses LLM_BASE_URL or http://localhost:11434
  lmstudio  - Local LM Studio. Uses LLM_BASE_URL or http://localhost:1234/v1
  openai    - OpenAI or compatible API. Requires LLM_API_KEY.
  anthropic - Anthropic Claude. Requires LLM_API_KEY.
  google    - Google Gemini. Requires LLM_API_KEY.

Environment variables:
  LLM_PROVIDER  - Provider name (default: ollama)
  LLM_MODEL     - Model name (default: qwen2.5:3b)
  LLM_BASE_URL  - Override base URL (optional)
  LLM_API_KEY   - API key for cloud providers (optional for ollama/lmstudio)
"""

import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")


def get_chat_model(**kwargs):
    """Return a LangChain chat model based on environment configuration.

    Passes any additional keyword arguments (e.g., temperature) to the
    underlying model constructor.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:3b")
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    if provider in ("ollama", "lmstudio"):
        from langchain_ollama import ChatOllama

        url = base_url or (
            "http://localhost:11434" if provider == "ollama" else "http://localhost:1234/v1"
        )
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
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. Use: ollama, lmstudio, openai, anthropic, google"
        )
