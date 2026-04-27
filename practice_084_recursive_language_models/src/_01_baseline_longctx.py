"""Phase 1 — Baseline: vanilla long-context QA.

Before recursing, you need a baseline to beat. We build a synthetic
"haystack" — many short paragraphs of irrelevant filler with a single
sentence containing the answer dropped somewhere in the middle — and
ask the LM to answer the question by stuffing the *entire* haystack
into one prompt.

This is the "throw it all at the LM" strategy that RLMs argue against.
The pain points it surfaces (lost-in-the-middle, context rot, token
cost) are exactly the motivation for Phase 2 onward.

Run: uv run python -m src._01_baseline_longctx
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .llm_config import LMConfig, chat, get_root_lm


# A library of filler "facts" that have nothing to do with the answer.
# The haystack is built by sampling these N times and inserting the
# real needle at a random index.
FILLER_FACTS: list[str] = [
    "The library opens at 9 AM on weekdays.",
    "The cafeteria served pasta on Tuesday.",
    "The bus schedule changes every season.",
    "The conference room can fit twelve people.",
    "The printer on the second floor is out of toner.",
    "The new intern's name is hard to pronounce.",
    "The fire drill was rescheduled to next month.",
    "The coffee machine takes about forty seconds.",
    "The whiteboard markers are running out of ink.",
    "The window blinds were finally repaired.",
    "The mail room closes early on Fridays.",
    "The plant in the lobby needs more sunlight.",
    "The vending machine no longer accepts coins.",
    "The badge reader at gate 3 has been flaky.",
    "The break room microwave hums at 60 Hz.",
    "The supply closet is stocked with notebooks.",
]


@dataclass(frozen=True)
class HaystackProblem:
    question: str
    answer: str          # gold answer for grading
    needle: str          # the single sentence containing the answer
    haystack: str        # the full long context (filler + needle)
    needle_index: int    # paragraph position of the needle inside the haystack


# -- Haystack construction (scaffolded) --------------------------------------


def build_haystack(needle: str, n_filler: int, seed: int) -> tuple[str, int]:
    """Sample n_filler filler sentences and insert the needle at a random index.

    Returns (haystack_text, needle_paragraph_index). Each "paragraph" is
    one sentence on its own line so the position is easy to inspect.
    """
    rng = random.Random(seed)
    paragraphs = [rng.choice(FILLER_FACTS) for _ in range(n_filler)]
    idx = rng.randint(0, len(paragraphs))
    paragraphs.insert(idx, needle)
    return "\n\n".join(paragraphs), idx


# Three classic needle-in-a-haystack problems with known answers.
PROBLEMS: list[HaystackProblem] = [
    HaystackProblem(
        question="What is the secret access code for room 7?",
        answer="ALPHA-9831",
        needle="The secret access code for room 7 is ALPHA-9831.",
        haystack="",
        needle_index=-1,
    ),
    HaystackProblem(
        question="When was the building's emergency drill rescheduled to?",
        answer="October 14th",
        needle="The official emergency drill has been rescheduled to October 14th.",
        haystack="",
        needle_index=-1,
    ),
    HaystackProblem(
        question="What is the network admin's pager extension?",
        answer="x4471",
        needle="In case of outage, contact the network admin at pager extension x4471.",
        haystack="",
        needle_index=-1,
    ),
]


def materialize_problems(n_filler: int = 60, seed: int = 7) -> list[HaystackProblem]:
    """Fill in haystack + needle_index for each PROBLEMS entry."""
    out = []
    for i, p in enumerate(PROBLEMS):
        haystack, idx = build_haystack(p.needle, n_filler, seed + i)
        out.append(HaystackProblem(
            question=p.question, answer=p.answer, needle=p.needle,
            haystack=haystack, needle_index=idx,
        ))
    return out


# ---------------------------------------------------------------------------
# TODO(human) #1 — Build the long-context prompt
# ---------------------------------------------------------------------------
# The vanilla long-context strategy is "stuff the entire context into the
# system/user message and ask the question". This is exactly what an RLM is
# meant to *replace* — but you can't appreciate the replacement without
# feeling how this one struggles first.
#
# What to do:
#   1. Return a `messages` list of two dicts (system, user) suitable for
#      OpenAI-style chat-completions.
#   2. The system message should set a focused QA persona — something like
#      "You answer questions strictly using the provided document. Reply
#      with only the answer, no preamble."
#   3. The user message must contain BOTH the document (`haystack`) AND
#      the question. The exact ordering and delimiters matter — long
#      contexts work better when the question is *repeated* near the
#      start AND end of the prompt (a known long-context trick), but for
#      this baseline a single placement is fine. Pick one.
#   4. Return the messages list.
#
# Why this matters: this prompt is the experimental control. Every
# improvement an RLM shows in Phase 4 is measured against the score this
# prompt produces here.
# ---------------------------------------------------------------------------
def build_baseline_messages(haystack: str, question: str) -> list[dict]:
    raise NotImplementedError(
        "TODO(human): assemble the system + user chat messages for vanilla long-context QA"
    )


# ---------------------------------------------------------------------------
# TODO(human) #2 — Score one prediction against the gold answer
# ---------------------------------------------------------------------------
# We need a *cheap, deterministic* grader so the comparison harness in
# Phase 4 can score dozens of runs without an extra LM call. A simple
# substring check is good enough for the toy haystack problems we use:
# the gold answer is short and unambiguous (e.g. "ALPHA-9831"), so any
# correct response will contain it verbatim.
#
# What to do:
#   1. Lowercase + strip both `prediction` and `gold` (the model often
#      adds trailing whitespace, periods, or quotes).
#   2. Return True iff `gold` appears as a substring of `prediction`.
#   3. Return False otherwise.
#
# Why "contains" instead of "equals": the LM frequently wraps the answer
# in a sentence ("The code is ALPHA-9831.") even when told not to.
# Substring containment captures correctness without fighting formatting.
# ---------------------------------------------------------------------------
def is_correct(prediction: str, gold: str) -> bool:
    raise NotImplementedError("TODO(human): substring-match prediction against gold")


# -- Demo runner (scaffolded) -----------------------------------------------


def run_baseline(cfg: LMConfig, problems: list[HaystackProblem]) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    for i, p in enumerate(problems, 1):
        messages = build_baseline_messages(p.haystack, p.question)
        prediction = chat(cfg, messages=messages, max_tokens=64).strip()
        ok = is_correct(prediction, p.answer)
        results.append((ok, prediction))
        verdict = "OK  " if ok else "MISS"
        print(f"  [{verdict}] q{i}: {p.question}")
        print(f"           gold='{p.answer}'  needle@para#{p.needle_index}")
        print(f"           pred='{prediction}'")
    return results


def _print_summary(results: list[tuple[bool, str]]) -> None:
    correct = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n  Baseline accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")


def main() -> None:
    cfg = get_root_lm()
    problems = materialize_problems(n_filler=60, seed=7)
    print(f"Baseline long-context QA — {cfg.model}")
    print(f"  haystack size: ~{len(problems[0].haystack)} chars per problem\n")
    results = run_baseline(cfg, problems)
    _print_summary(results)


if __name__ == "__main__":
    main()
