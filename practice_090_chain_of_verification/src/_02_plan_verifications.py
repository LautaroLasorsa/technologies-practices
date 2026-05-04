"""Phase 2 — Plan verifications.

Given the original user prompt and the (possibly hallucinated) baseline
draft, the model proposes a list of independent factual questions whose
answers, taken together, are necessary for the baseline to be correct.

This stage is *load-bearing*: bad questions = no value from the rest of
the pipeline.  The CoVe paper's results are highly sensitive to the
quality and independence of these questions — that's why structured
output (a Pydantic schema) is essential here.  The schema forces the
model to commit to a list of atomic, comparable items instead of a
prose blob.

We use LangChain's ``model.with_structured_output(VerificationPlan)``,
which under the hood:
  1. Sends the schema to the model (function-calling for OpenAI/Anthropic;
     a JSON-mode prompt for Ollama).
  2. Parses the reply and validates it against ``VerificationPlan``.
  3. Returns a ``VerificationPlan`` instance directly.

See LangChain docs:
  https://python.langchain.com/docs/how_to/structured_output/

Run on its own:
    uv run python -m src._02_plan_verifications
"""

from __future__ import annotations

import textwrap

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_config import LMConfig, build_chat_model, get_lm
from .models import BaselineDraft, VerificationPlan, VerificationQuestion

PLANNER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a verification planner.

    Given the user's original question and a draft answer, propose a list
    of SHORT, INDEPENDENT, FACTUAL questions whose answers — taken
    together — verify the draft.

    Rules for questions:
      - Each question must be answerable WITHOUT seeing the draft or the
        original prompt; it must be self-contained.
      - One atomic fact per question.  Prefer "Was X born in Y?" over
        "Is the following list of birthplaces correct?".
      - Questions must be INDEPENDENT — the answer to one must not depend
        on the answer to another.
      - 3 to 7 questions total; pick the ones that, if any one is wrong,
        would change the draft.
      - Do NOT propose meta questions ("is the draft accurate?"); only
        atomic factual ones.
    """
)


# ---------------------------------------------------------------------------
# TODO(human) — Plan verifications via structured output
# ---------------------------------------------------------------------------
# Goal: turn (user_prompt, baseline) into a `VerificationPlan` using
# LangChain's `.with_structured_output(...)`.
#
# What to do:
#   1. Build the chat model with `build_chat_model(cfg)`.
#   2. Wrap it: `planner = model.with_structured_output(VerificationPlan)`.
#      This is the canonical way — do NOT hand-parse JSON or call
#      `.invoke()` on the unwrapped model and json.loads the reply.
#   3. Compose a 2-message list:
#        - SystemMessage(PLANNER_SYSTEM_PROMPT)
#        - HumanMessage with both the original prompt and the draft
#          (clearly labeled, e.g. "ORIGINAL QUESTION:\n...\n\nDRAFT ANSWER:\n...").
#   4. Call `planner.invoke(messages)`.  It returns a VerificationPlan.
#   5. Return that plan unchanged.
#
# Why structured output here: free-text question lists are messy to
# parse and worse to chunk into per-question prompts in the next stage.
# Pydantic + LangChain gives you a typed list you can iterate over.
# ---------------------------------------------------------------------------
def plan_verifications(
    user_prompt: str,
    baseline: BaselineDraft,
    cfg: LMConfig | None = None,
) -> VerificationPlan:
    """Propose verification questions that, if answered, would validate the draft."""
    cfg = cfg or get_lm()
    raise NotImplementedError("TODO(human): build a VerificationPlan via structured output")


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    prompt = "Name 5 actors who were born in Brooklyn, New York."
    # Hand-written stand-in for a baseline so this script runs without
    # depending on _01_baseline being implemented yet.
    fake_draft = BaselineDraft(text=textwrap.dedent("""\
        1. Mae West
        2. Adam Sandler
        3. Mel Brooks
        4. Barbra Streisand
        5. Eddie Murphy
    """))
    print(f"Prompt: {prompt}\n")
    print("DRAFT (stub):")
    print(fake_draft.text)
    plan = plan_verifications(prompt, fake_draft)
    print("\nVERIFICATION PLAN")
    print("-" * 60)
    for i, q in enumerate(plan.questions, 1):
        print(f"  {i}. {q.question}")
        if q.rationale:
            print(f"     ({q.rationale})")


if __name__ == "__main__":
    main()
