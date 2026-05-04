"""Phase 5 — Pipeline: factored vs joint CoVe.

Two end-to-end runners over the four stages:

  cove_factored(prompt)
      Stage 1: baseline (own context)
      Stage 2: plan verifications (own context)
      Stage 3: execute each verification in ITS OWN context  ◄ key step
      Stage 4: refine (sees baseline + verifications)

  cove_joint(prompt)
      One transcript, all four stages, no context isolation.  This is
      the comparison baseline — the form of CoVe that the paper shows
      gives the *least* improvement, because the verifications inherit
      the baseline's wrong assumptions.

Run both side by side with ``demo.py`` to see the difference on hard
prompts.

Run on its own:
    uv run python -m src._05_pipeline
"""

from __future__ import annotations

import textwrap

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_config import LMConfig, build_chat_model, chat, get_lm
from .models import (
    BaselineDraft,
    RefinedAnswer,
    RunRecord,
    VerificationAnswer,
    VerificationPlan,
    VerificationQuestion,
)


def cove_factored(prompt: str, cfg: LMConfig | None = None) -> RunRecord:
    """Run the factored (per-question isolated) CoVe pipeline."""
    cfg = cfg or get_lm()

    from ._01_baseline import generate_baseline
    from ._02_plan_verifications import plan_verifications
    from ._03_execute_factored import execute_factored
    from ._04_refine import refine

    baseline = generate_baseline(prompt, cfg)
    plan = plan_verifications(prompt, baseline, cfg)
    answers = execute_factored(plan, cfg)
    refined = refine(prompt, baseline, answers, cfg)

    return RunRecord(
        prompt=prompt,
        mode="factored",
        baseline=baseline,
        plan=plan,
        answers=answers,
        refined=refined,
    )


JOINT_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a careful assistant.  Answer the user's question using the
    following four-stage process IN A SINGLE REPLY:

      STAGE 1 — DRAFT: write your initial answer.
      STAGE 2 — VERIFICATIONS: list 3-7 short factual questions that, if
                answered correctly, would let you check the draft.
      STAGE 3 — ANSWERS: answer each verification question.
      STAGE 4 — FINAL: produce a final answer, correcting any items the
                verifications showed to be wrong, then list the
                corrections.

    Use clearly labelled section headers exactly as named above.
    """
)


# ---------------------------------------------------------------------------
# TODO(human) — Joint CoVe (the contaminated variant)
# ---------------------------------------------------------------------------
# Goal: implement CoVe as a SINGLE chained chat call so all four stages
# share one transcript.  This is the comparison baseline; you'll see in
# the demo that it underperforms factored CoVe on hallucination-prone
# prompts, because by the time the model writes its verifications it
# has already committed to the (wrong) draft above them in the context.
#
# What to do:
#   1. Build the model with `build_chat_model(cfg)`.
#   2. Send a 2-message conversation:
#        [SystemMessage(JOINT_SYSTEM_PROMPT), HumanMessage(prompt)]
#      Use `chat(model, messages)` so retries are handled.
#   3. Wrap the entire single reply in a RunRecord:
#        - baseline = BaselineDraft(text=reply)            # whole thing
#        - plan     = VerificationPlan(questions=[])       # we don't parse it
#        - answers  = []                                   # same
#        - refined  = RefinedAnswer(text=reply, corrections=[])
#      The point is the single reply IS the joint answer; we don't try
#      to parse the four sections out — the demo prints them as-is.
#   4. Return RunRecord(mode="joint", ...).
#
# Keep this short — under 15 lines.  The pedagogical purpose is to feel
# in your fingers WHY contamination happens by writing the contaminated
# version yourself.
# ---------------------------------------------------------------------------
def cove_joint(prompt: str, cfg: LMConfig | None = None) -> RunRecord:
    """Run a joint (single-context, contaminated) CoVe pipeline."""
    cfg = cfg or get_lm()
    raise NotImplementedError("TODO(human): implement joint CoVe in a single call")


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    prompt = "Name 5 actors who were born in Brooklyn, New York."
    print(f"Prompt: {prompt}\n")
    print("FACTORED")
    print("-" * 60)
    fr = cove_factored(prompt)
    print(fr.refined.text)
    print("\nJOINT")
    print("-" * 60)
    jr = cove_joint(prompt)
    print(jr.refined.text)


if __name__ == "__main__":
    main()
