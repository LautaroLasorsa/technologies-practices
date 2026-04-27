"""Phase 0 — Verify Ollama connection + the LLM helper.

Sends one tiny chat-completion request through both the root and sub
LM configs so you know the backend is reachable and the model is
loaded before starting the real exercises.

Run: uv run python -m src._00_verify_setup
Prereq: docker compose up -d && docker exec rlm_ollama ollama pull qwen2.5:7b
"""

from __future__ import annotations

from .llm_config import LMConfig, chat, get_root_lm, get_sub_lm


def _ping(cfg: LMConfig, label: str) -> None:
    print(f"  [{label}] {cfg.provider} / {cfg.model} @ {cfg.base_url}")
    answer = chat(
        cfg,
        messages=[{"role": "user", "content": "Reply with the single word: pong."}],
        max_tokens=8,
    )
    print(f"    -> {answer.strip()!r}")


def main() -> None:
    print("Verifying RLM stack...")
    _ping(get_root_lm(), "root")
    _ping(get_sub_lm(), "sub ")
    print("\nSetup verified successfully!")


if __name__ == "__main__":
    main()
