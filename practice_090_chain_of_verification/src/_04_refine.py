"""Phase 4 — Refine the baseline using the verification results.

The model is shown:
  - the original user prompt,
  - its own baseline draft,
  - and the (Q, A, verdict) triples produced in Phase 3.

It then writes a final answer that either confirms the baseline or
corrects the items that failed verification, listing the changes in
``corrections``.

Note that *unlike* Phase 3, contamination is fine here — the whole
point of the refinement step is to combine sources.  The verifications
have already been computed cleanly.

Run on its own:
    uv run python -m src._04_refine
"""

from __future__ import annotations

import textwrap

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_config import LMConfig, build_chat_model, get_lm
from .models import BaselineDraft, RefinedAnswer, VerificationAnswer

REFINER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a careful editor.

    You will receive: the user's original question, a draft answer, and
    a list of independent verification (question, answer, verdict)
    triples that fact-check the draft.

    Produce a final answer that:
      - keeps every claim from the draft that is SUPPORTED,
      - removes or replaces every claim that is CONTRADICTED,
      - flags items where the verifications were UNCERTAIN if you cannot
        decide.

    Then list the specific changes you made in `corrections` (one short
    bullet each).  If you did not change anything, return an empty list.
    """
)


def _format_verifications(answers: list[VerificationAnswer]) -> str:
    """Render the verification triples into a single block for the prompt."""
    lines = []
    for i, a in enumerate(answers, 1):
        lines.append(f"  Q{i}: {a.question}")
        lines.append(f"  A{i}: {a.answer}")
        lines.append(f"  verdict: {a.verdict}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# TODO(human) — Refine the draft using the verifications
# ---------------------------------------------------------------------------
# Goal: produce a `RefinedAnswer` from the user prompt + baseline +
# verification results.
#
# What to do:
#   1. Build the model with structured output:
#        model = build_chat_model(cfg).with_structured_output(RefinedAnswer)
#   2. Build the human message body that contains, clearly labelled:
#        - the original user prompt
#        - the baseline draft (`baseline.text`)
#        - the formatted verifications (use `_format_verifications(answers)`)
#   3. Wrap with [SystemMessage(REFINER_SYSTEM_PROMPT), HumanMessage(body)]
#      and call `model.invoke(messages)`.
#   4. Return the RefinedAnswer.
#
# Why structured output here too: `corrections` is the part the demo
# uses to highlight what CoVe caught — keeping it as a typed list (not
# a regex over prose) makes the comparison robust.
# ---------------------------------------------------------------------------
def refine(
    user_prompt: str,
    baseline: BaselineDraft,
    answers: list[VerificationAnswer],
    cfg: LMConfig | None = None,
) -> RefinedAnswer:
    """Combine baseline + verifications into a final, corrected answer."""
    cfg = cfg or get_lm()
    raise NotImplementedError("TODO(human): refine the baseline using the verifications")


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    prompt = "Name 5 actors who were born in Brooklyn, New York."
    baseline = BaselineDraft(text=textwrap.dedent("""\
        1. Mae West
        2. Adam Sandler
        3. Mel Brooks
        4. Barbra Streisand
        5. Eddie Murphy
    """))
    answers = [
        VerificationAnswer(question="Was Mae West born in Brooklyn?",
                           answer="Yes, born in Brooklyn in 1893.", verdict="supports"),
        VerificationAnswer(question="Was Adam Sandler born in Brooklyn?",
                           answer="Yes, born in Brooklyn in 1966.", verdict="supports"),
        VerificationAnswer(question="Was Mel Brooks born in Brooklyn?",
                           answer="Yes, born in Brooklyn in 1926.", verdict="supports"),
        VerificationAnswer(question="Was Barbra Streisand born in Brooklyn?",
                           answer="Yes, born in Brooklyn in 1942.", verdict="supports"),
        VerificationAnswer(question="Was Eddie Murphy born in Brooklyn?",
                           answer="No, born in Brooklyn — actually born in Brooklyn? "
                                  "Eddie Murphy was born in Brooklyn, NY in 1961.",
                           verdict="contradicts"),  # contrived contradiction for the demo
    ]
    refined = refine(prompt, baseline, answers)
    print("REFINED ANSWER")
    print("-" * 60)
    print(refined.text)
    print("\nCORRECTIONS")
    for c in refined.corrections:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
