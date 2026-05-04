"""Phase 3 — Execute verifications (factored).

This is the *algorithmic insight* of CoVe.  Each verification question
gets its own, fresh LLM call — and that call sees ONLY the question.
It does NOT see the original user prompt and it does NOT see the
baseline draft.

Why?  Because if the baseline hallucinated "Eddie Murphy was born in
Brooklyn", and you ask "Was Eddie Murphy born in Brooklyn?" *in a
context that already contains the wrong claim*, the model is heavily
biased to confirm the prior.  Stripping the context breaks the
contamination chain.  This is the difference between *factored* CoVe
(strongest single setting in the paper) and *joint* CoVe (the
contaminated baseline you'll implement in `_05_pipeline.cove_joint`).

Each call returns a `VerificationAnswer` via structured output, so the
refinement stage can read the verdicts mechanically.

Run on its own:
    uv run python -m src._03_execute_factored
"""

from __future__ import annotations

import textwrap

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_config import LMConfig, build_chat_model, get_lm
from .models import VerificationAnswer, VerificationPlan, VerificationQuestion

VERIFIER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a fact-checker.

    Answer the user's question concisely and factually.  Then assign one
    of three verdicts about whether your answer SUPPORTS, CONTRADICTS,
    or is UNCERTAIN about a claim implicit in the question.

    Use 'supports' when the natural reading of the question presupposes
    a fact and your answer confirms it; 'contradicts' when your answer
    refutes it; 'uncertain' when you genuinely don't know.
    """
)


# ---------------------------------------------------------------------------
# TODO(human) — Answer ONE question in an isolated context
# ---------------------------------------------------------------------------
# Goal: given a single VerificationQuestion, return a VerificationAnswer
# computed in isolation.
#
# CRITICAL — what NOT to pass:
#   - The original user prompt.  Not in the system message, not in the
#     human message, not anywhere.
#   - The baseline draft.  Same rule.
#   - Other verification questions.  Each call is independent.
#
# Why: the entire point of "factored" CoVe is breaking contamination.
# If you smuggle the prompt or draft in, this stage degrades to
# "joint" CoVe and you've defeated the algorithm.
#
# What TO do:
#   1. `model = build_chat_model(cfg).with_structured_output(VerificationAnswer)`
#   2. Build messages:
#        - SystemMessage(VERIFIER_SYSTEM_PROMPT)
#        - HumanMessage(question.question)        # <- the question, ALONE
#   3. Call `model.invoke(messages)` and return the result.
# ---------------------------------------------------------------------------
def answer_question_isolated(
    question: VerificationQuestion,
    cfg: LMConfig | None = None,
) -> VerificationAnswer:
    """Answer one verification question in a fresh, contamination-free context."""
    cfg = cfg or get_lm()
    raise NotImplementedError("TODO(human): answer one verification question in isolation")


def execute_factored(
    plan: VerificationPlan,
    cfg: LMConfig | None = None,
) -> list[VerificationAnswer]:
    """Run every question in ``plan`` through ``answer_question_isolated``.

    Sequential by default — parallelising is left as an exercise.  The
    independence guarantee is what would *let* you parallelise safely.
    """
    cfg = cfg or get_lm()
    return [answer_question_isolated(q, cfg) for q in plan.questions]


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    plan = VerificationPlan(questions=[
        VerificationQuestion(question="Was Eddie Murphy born in Brooklyn, New York?"),
        VerificationQuestion(question="Was Mae West born in Brooklyn, New York?"),
        VerificationQuestion(question="Was Mel Brooks born in Brooklyn, New York?"),
    ])
    print(f"Executing {len(plan.questions)} verifications in isolation...\n")
    answers = execute_factored(plan)
    for i, a in enumerate(answers, 1):
        print(f"{i}. Q: {a.question}")
        print(f"   A: {a.answer}")
        print(f"   verdict: {a.verdict}\n")


if __name__ == "__main__":
    main()
