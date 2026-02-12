"""Shared configuration for LangExtract practice.

Points all extraction calls at the local Ollama container.
"""

# Ollama server running in Docker
OLLAMA_URL = "http://localhost:11434"

# Default model — Gemma 3 4B is a good balance of quality and speed.
# For better quality, switch to "gemma3:12b" (needs ~8GB RAM).
MODEL_ID = "gemma3:4b"

# Timeout in seconds for LLM inference.
# Increase for larger models or longer documents.
TIMEOUT = 300
