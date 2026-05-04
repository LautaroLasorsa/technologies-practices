"""Shared Pydantic data models for the CoVe pipeline.

These types are the *interface* between the four CoVe stages.  Keeping
them small, typed, and JSON-serialisable is what lets each stage be
tested in isolation:

    user prompt
        │
        ▼  _01_baseline
    BaselineDraft
        │
        ▼  _02_plan_verifications
    list[VerificationQuestion]
        │
        ▼  _03_execute_factored  (each Q in its OWN context — no draft)
    list[VerificationAnswer]
        │
        ▼  _04_refine
    RefinedAnswer

The whole CoVe run (joint or factored) is bundled into a ``RunRecord``
for side-by-side comparison in ``demo.py``.

Don't add a TODO here — this file is pure schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BaselineDraft(BaseModel):
    """The model's first answer, written without any verification scaffolding.

    This is the answer the model would have given a user with no CoVe
    pipeline.  It is intentionally allowed to be wrong — the rest of the
    pipeline is built to catch its mistakes.
    """
    text: str = Field(description="The raw baseline reply, exactly as written.")


class VerificationQuestion(BaseModel):
    """One short, independent factual question proposed by the planner.

    Quality matters: each question must be answerable on its own, not
    require the original user prompt as context, and target one atomic
    fact.  ``rationale`` is for debugging — it tells you *why* the
    planner thought this question mattered.
    """
    question: str = Field(description="Short, self-contained factual question.")
    rationale: str = Field(default="", description="Why answering this question matters.")


class VerificationPlan(BaseModel):
    """Ordered list of verification questions for one baseline draft.

    Wrapping the list in a top-level model lets us pass a single
    Pydantic schema to ``with_structured_output`` (LangChain doesn't
    support naked ``list[Model]`` for every provider).
    """
    questions: list[VerificationQuestion] = Field(default_factory=list)


class VerificationAnswer(BaseModel):
    """Answer to a single verification question, computed in isolation.

    ``verdict`` is a coarse 3-way label that the refinement stage can
    read mechanically.  ``answer`` is the natural-language answer for
    the LLM to consume.
    """
    question: str = Field(description="Echo of the verification question.")
    answer: str = Field(description="Concise natural-language answer.")
    verdict: Literal["supports", "contradicts", "uncertain"] = Field(
        description=(
            "Coarse machine-readable verdict the refinement stage can read.  "
            "'supports' = the answer backs up the baseline; "
            "'contradicts' = the answer disproves part of the baseline; "
            "'uncertain' = not enough information to tell."
        )
    )


class RefinedAnswer(BaseModel):
    """Final answer produced by the refinement stage.

    ``corrections`` is a short bulleted list of changes from the baseline
    so the user (and the practice's diff view) can see exactly what
    CoVe caught.
    """
    text: str = Field(description="Final natural-language answer.")
    corrections: list[str] = Field(
        default_factory=list,
        description="Short bulleted list of changes the model made to the baseline.",
    )


class RunRecord(BaseModel):
    """End-to-end record of one CoVe run on one prompt.

    Used by ``demo.py`` to print joint vs factored runs side-by-side and
    optionally serialise them to ``runs/*.jsonl``.
    """
    prompt: str
    mode: Literal["joint", "factored"]
    baseline: BaselineDraft
    plan: VerificationPlan
    answers: list[VerificationAnswer] = Field(default_factory=list)
    refined: RefinedAnswer
