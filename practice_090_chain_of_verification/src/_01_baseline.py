"""Phase 1 — Baseline draft (the answer that often hallucinates).

The first stage of CoVe is intentionally naive: ask the LLM the user's
question with **no** verification scaffolding, no "be careful" hints,
no chain-of-thought.  The whole point of the rest of the pipeline is
to catch the mistakes this stage makes; if you let the model be
careful here, you'd hide them.

This is also the only stage you'd use if CoVe didn't exist — so
treating it as the control condition matters for the comparison
``demo.py`` produces.

Run on its own:
    uv run python -m src._01_baseline
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_config import LMConfig, build_chat_model, chat, get_lm
from .models import BaselineDraft

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly and "
    "concisely. When asked for a list, give a plain numbered list."
)


# ---------------------------------------------------------------------------
# TODO(human) — Generate a baseline answer
# ---------------------------------------------------------------------------
# Goal: ask the LLM the user's prompt and return a `BaselineDraft`.
#
# Constraints:
#   - DO NOT add any "double-check yourself" / "list only what you're sure
#     of" / "verify each item" instruction.  The whole CoVe paper rests
#     on the baseline being a typical (often-hallucinatory) answer; if
#     you tell the model to be careful here, the rest of the pipeline
#     has nothing to fix.
#   - Use BASELINE_SYSTEM_PROMPT as-is.
#   - Use the `chat()` helper from `llm_config` so retries are handled.
#
# What to do:
#   1. Build a `model` with `build_chat_model(cfg)`.
#   2. Build a 2-message list: [SystemMessage(BASELINE_SYSTEM_PROMPT),
#                               HumanMessage(user_prompt)].
#   3. Call `chat(model, messages)` and wrap the reply in BaselineDraft.
# ---------------------------------------------------------------------------
def generate_baseline(user_prompt: str, cfg: LMConfig | None = None) -> BaselineDraft:
    """Produce the model's first, unverified answer to ``user_prompt``."""
    cfg = cfg or get_lm()
    raise NotImplementedError("TODO(human): generate the baseline draft")


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    prompt = "Name 5 actors who were born in Brooklyn, New York."
    print(f"Prompt: {prompt}\n")
    draft = generate_baseline(prompt)
    print("BASELINE DRAFT")
    print("-" * 60)
    print(draft.text)


if __name__ == "__main__":
    main()
