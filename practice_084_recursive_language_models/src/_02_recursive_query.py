"""Phase 2 — The recursive query() tool.

This is the heart of an RLM: a function the *root* LM can call to
delegate a question over a *chunk* of context to a *sub* LM. From the
root model's point of view it looks like an ordinary tool — pass a
text blob and a question, get a string answer back.

Conceptually:
    query(text, question)  ==  ask a fresh LM "answer `question` using only `text`"

The interesting bits aren't the call itself — they're the *guard
rails*: depth bounds (so the sub-LM cannot recurse forever) and cost
tracking (so a runaway script doesn't burn 10M tokens silently).

Original RLM blog post limits depth to 1 in the published experiments.
We expose `max_depth` so you can experiment, but defaulting to 1
matches the paper.

Run on its own to sanity-check the tool:
    uv run python -m src._02_recursive_query
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .llm_config import LMConfig, chat, get_sub_lm


# Hard ceiling on tokens per sub-call so a buggy partition strategy
# can't blow up your laptop.
DEFAULT_MAX_TOKENS_PER_CALL = 256


@dataclass
class CostTracker:
    """Bookkeeping for a single root-LM session.

    The original paper notes 'Each recursive LM call is both blocking and
    does not take advantage of any kind of prefix caching!' — so the cost
    you actually pay grows roughly linearly with the number of sub-calls.
    Tracking it here makes that visible.
    """
    n_sub_calls: int = 0
    n_input_chars: int = 0
    n_output_chars: int = 0
    elapsed_s: float = 0.0
    depth_histogram: dict[int, int] = field(default_factory=dict)

    def record(self, depth: int, in_chars: int, out_chars: int, dt: float) -> None:
        self.n_sub_calls += 1
        self.n_input_chars += in_chars
        self.n_output_chars += out_chars
        self.elapsed_s += dt
        self.depth_histogram[depth] = self.depth_histogram.get(depth, 0) + 1


# ---------------------------------------------------------------------------
# TODO(human) #1 — The sub-LM call
# ---------------------------------------------------------------------------
# The sub-LM is a *fresh* invocation with NO memory of the root context.
# It only sees the text snippet the root LM hands it, plus a focused
# question. This is what makes the recursion useful: each sub-call has
# a tiny prompt, so attention isn't diluted across millions of tokens.
#
# What to do:
#   1. Build the sub-LM `messages` list:
#      - A system message that pins down the contract: "You are a
#        focused reading assistant. Answer the question using only the
#        provided text. If the text doesn't contain the answer, reply
#        with the literal string 'NOT_FOUND'."
#      - A user message of the form
#            f"TEXT:\n{text}\n\nQUESTION: {question}"
#   2. Call `chat(cfg, messages=..., max_tokens=DEFAULT_MAX_TOKENS_PER_CALL)`
#      and return the stripped string.
#
# Why a 'NOT_FOUND' sentinel: when the root LM partitions a long context
# into chunks, most chunks won't contain the answer. The root needs to
# tell "this chunk has nothing" apart from "the sub-LM hallucinated".
# A literal sentinel is the cheapest way to do that.
# ---------------------------------------------------------------------------
def _sub_lm_answer(cfg: LMConfig, text: str, question: str) -> str:
    raise NotImplementedError(
        "TODO(human): build sub-LM messages and call chat(); return stripped string"
    )


class DepthExceeded(RuntimeError):
    """Raised when a sub-LM tries to recurse past the configured depth bound."""


# ---------------------------------------------------------------------------
# TODO(human) #2 — The depth guard
# ---------------------------------------------------------------------------
# RLMs are bounded recursion. Without a depth limit the root LM can
# (and sometimes will) write code that recurses indefinitely — every
# sub-LM in turn could call query() again, blowing up the call tree.
#
# The original paper sets max_depth=1 (the root may recurse, but a
# sub-LM cannot recurse further). We expose `max_depth` as a parameter
# so you can experiment with depth=2 later.
#
# What to do:
#   - If `current_depth >= max_depth`, raise `DepthExceeded(...)` with a
#     short message that mentions both numbers. Otherwise return None.
#
# This function is called from `query()` *before* the sub-LM is invoked,
# so a depth violation costs zero tokens.
# ---------------------------------------------------------------------------
def _check_depth(current_depth: int, max_depth: int) -> None:
    raise NotImplementedError(
        "TODO(human): raise DepthExceeded if current_depth >= max_depth"
    )


# ---------------------------------------------------------------------------
# TODO(human) #3 — Wire query() together
# ---------------------------------------------------------------------------
# `query()` is the *public* tool the root LM calls. Internally it has to:
#   1. Enforce the depth guard.
#   2. Time the sub-LM call (the cost tracker wants wall-clock seconds).
#   3. Invoke the sub-LM.
#   4. Record what just happened on the cost tracker.
#   5. Return the answer string.
#
# What to do:
#   1. Call `_check_depth(current_depth, max_depth)`.
#   2. Record `t0 = time.perf_counter()`.
#   3. Call `_sub_lm_answer(cfg, text, question)` and store the result.
#   4. Compute `dt = time.perf_counter() - t0`.
#   5. Call `tracker.record(depth=current_depth + 1, in_chars=len(text),
#      out_chars=len(answer), dt=dt)`.
#   6. Return `answer`.
#
# The "+1" on depth: tracker records the depth of the call that *just
# completed*, which is one deeper than the caller's current_depth.
# ---------------------------------------------------------------------------
def query(
    text: str,
    question: str,
    *,
    cfg: LMConfig,
    tracker: CostTracker,
    current_depth: int = 0,
    max_depth: int = 1,
) -> str:
    """Run `question` against `text` using a fresh sub-LM. Bounded recursion."""
    raise NotImplementedError(
        "TODO(human): orchestrate depth check + timing + sub-LM call + tracker.record"
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


_DEMO_TEXT = """\
The library has 12 reading rooms. Room 1 is silent. Room 2 allows quiet
conversation. The secret access code for room 7 is ALPHA-9831. Room 8
is reserved for staff. Room 12 has natural lighting all day.
"""


def _demo_one(question: str, expected_substring: str | None) -> None:
    cfg = get_sub_lm()
    tracker = CostTracker()
    answer = query(_DEMO_TEXT, question, cfg=cfg, tracker=tracker)
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    if expected_substring is not None:
        ok = expected_substring.lower() in answer.lower()
        print(f"     -> contains {expected_substring!r}? {ok}")
    print(f"     stats: {tracker.n_sub_calls} call(s), {tracker.elapsed_s:.2f}s")


def main() -> None:
    print("Sanity-checking the recursive query() tool...\n")
    _demo_one("What is the secret access code for room 7?", "ALPHA-9831")
    print()
    _demo_one("Who is the building janitor?", "NOT_FOUND")
    print("\nTool works.")


if __name__ == "__main__":
    main()
