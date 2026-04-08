"""Shared configuration for LangExtract practice.

Reads provider settings from environment variables with sensible defaults
so the practice works out-of-the-box with a local Ollama container and
can be switched to any other LLM provider without code changes.

Environment variables (all optional):
    LLM_PROVIDER  — "ollama" | "lmstudio" | "openai" | "anthropic" | "google"
                    Default: "ollama"
    LLM_MODEL     — Model name. Default: "gemma3:4b" (for ollama)
    LLM_BASE_URL  — Base URL override. Defaults per provider:
                      ollama   → http://localhost:11434
                      lmstudio → http://localhost:1234/v1
                      cloud    → provider SDK default
    LLM_API_KEY   — API key. Required for cloud providers, ignored for local.
    LLM_TIMEOUT   — Inference timeout in seconds. Default: 300
"""

import os

# ── Provider defaults ────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama":    {"base_url": "http://localhost:11434",  "model": "gemma3:4b"},
    "lmstudio":  {"base_url": "http://localhost:1234/v1", "model": "gemma3:4b"},
    "openai":    {"base_url": "",                         "model": "gpt-4o-mini"},
    "anthropic": {"base_url": "",                         "model": "claude-3-haiku-20240307"},
    "google":    {"base_url": "",                         "model": "gemini-1.5-flash"},
}

# ── Read from environment ────────────────────────────────────────────

PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()

_defaults = _PROVIDER_DEFAULTS.get(PROVIDER, _PROVIDER_DEFAULTS["ollama"])

MODEL_ID: str = os.getenv("LLM_MODEL", _defaults["model"])
BASE_URL: str = os.getenv("LLM_BASE_URL", _defaults["base_url"])
API_KEY: str  = os.getenv("LLM_API_KEY", "")
TIMEOUT: int  = int(os.getenv("LLM_TIMEOUT", "300"))

# ── Backward-compatible alias ────────────────────────────────────────
# Source files reference config.OLLAMA_URL — keep it pointing at BASE_URL
# so no existing call sites need to change.
OLLAMA_URL: str = BASE_URL
