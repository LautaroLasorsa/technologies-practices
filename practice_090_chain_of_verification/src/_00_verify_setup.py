"""Phase 0 — Verify the LangChain + LLM connection.

Sends a tiny chat request through the configured backend and runs one
``with_structured_output`` call against a 2-field Pydantic model so you
know both halves of the CoVe stack are reachable before starting the
real exercises.

Run: uv run python -m src._00_verify_setup
Prereq: docker compose up -d && docker exec cove_ollama ollama pull qwen2.5:7b
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .llm_config import build_chat_model, chat, get_lm


class _PingSchema(BaseModel):
    """Tiny Pydantic schema used to verify structured output works."""
    word: str = Field(description="A single short word.")
    length: int = Field(description="The length of `word`.")


def _ping_chat() -> None:
    cfg = get_lm()
    print(f"  [chat]   {cfg.provider} / {cfg.model} @ {cfg.base_url or '(default)'}")
    model = build_chat_model(cfg)
    answer = chat(model, [HumanMessage(content="Reply with the single word: pong.")])
    print(f"    -> {answer.strip()!r}")


def _ping_structured() -> None:
    cfg = get_lm()
    print(f"  [struct] with_structured_output({_PingSchema.__name__})")
    model = build_chat_model(cfg).with_structured_output(_PingSchema)
    result = model.invoke(
        "Pick a single short English word, then report the word and its length."
    )
    print(f"    -> {result!r}")


def main() -> None:
    print("Verifying CoVe stack (LangChain + LLM)...")
    _ping_chat()
    _ping_structured()
    print("\nSetup verified successfully!")


if __name__ == "__main__":
    main()
