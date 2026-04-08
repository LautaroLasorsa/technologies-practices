"""LLM and embedding configuration for exercise scripts.

Reads from environment variables so any OpenAI-compatible provider works.
Defaults to Ollama running locally — no .env file needed for the standard setup.

Supported providers via LLM_PROVIDER / EMBEDDING_PROVIDER:
  - "ollama"    — Local Ollama (default, http://localhost:11434)
  - "lmstudio"  — LM Studio OpenAI-compat server (http://localhost:1234/v1)
  - "openai"    — OpenAI API (https://api.openai.com/v1)
  - "groq"      — Groq cloud (https://api.groq.com/openai/v1)
  - Any other OpenAI-compatible endpoint via LLM_BASE_URL

Usage in exercise scripts:
    from llm_config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY
    from llm_config import EMBEDDING_MODEL, EMBEDDING_BASE_URL, EMBEDDING_API_KEY
"""

import os

# ---------------------------------------------------------------------------
# Chat / completion model
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "") or (
    "http://localhost:11434" if LLM_PROVIDER in ("ollama", "lmstudio") else ""
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or (
    "ollama" if LLM_PROVIDER in ("ollama", "lmstudio") else ""
)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "") or (
    "http://localhost:11434" if EMBEDDING_PROVIDER in ("ollama", "lmstudio") else ""
)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or (
    "ollama" if EMBEDDING_PROVIDER in ("ollama", "lmstudio") else ""
)
