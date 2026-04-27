"""Phase 3 — REPL-style root agent.

This is the RLM proper. The root LM is given:
  - a Python REPL whose namespace ALREADY contains the long context as
    a variable named `context`, plus the `query(text, question)` tool
    from Phase 2 (and a few helpers like `len`, `re`),
  - one job: answer the user's question by writing Python code that
    inspects `context` and recurses via `query` as needed,
  - a strict output protocol: every reply is *either* a ```python ...```
    code block (the REPL executes it and returns the stdout) *or* a
    final answer wrapped as `FINAL(answer)` (the loop stops).

Conceptually this is ReAct, but the "tool" is a Python interpreter
and the recursion is structural: the model itself decides when to
`query()` a chunk vs. answer outright.

The original RLM blog post calls this design "context-centric
decomposition" and contrasts it with agents that decompose the
*problem* upfront. Here the LM decomposes the *context* on the fly.

Run: uv run python -m src._03_rlm_root_agent
"""

from __future__ import annotations

import io
import re
import textwrap
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass

from .llm_config import LMConfig, chat, get_root_lm, get_sub_lm
from ._01_baseline_longctx import HaystackProblem, materialize_problems
from ._02_recursive_query import CostTracker, query


# Hard caps so a runaway loop doesn't burn the day.
MAX_ROUNDS = 12
REPL_OUTPUT_TRUNC = 2000           # truncate REPL stdout fed back to the LM
FINAL_RE = re.compile(r"FINAL\((.*?)\)\s*$", re.DOTALL)
CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# TODO(human) #1 — The root system prompt
# ---------------------------------------------------------------------------
# Without a precise system prompt the root LM will either (a) try to
# answer from memory ignoring `context`, or (b) print the whole context
# back, defeating the point. The prompt has to spell out the protocol.
#
# What to do — return a string explaining, in order:
#   1. The model's role: "You answer questions about a long document
#      that is too large to read at once."
#   2. The environment: "You have a Python REPL. The variable `context`
#      is a string holding the document. The function
#      `query(text, question)` invokes a fresh sub-LM on a *snippet* of
#      text and returns its answer (or 'NOT_FOUND')."
#   3. The output protocol — exactly one of these per turn:
#         a) a ```python``` block. Whatever it `print(...)`s comes back
#            in the next turn.
#         b) `FINAL(answer)` on its own line, ending the session.
#   4. Concrete advice for using `context`:
#         - inspect `len(context)` and slice `context[:N]` to peek;
#         - use `re.findall(...)` to grep for keywords;
#         - call `query(chunk, sub_question)` for any slice that *might*
#           contain the answer — the sub-LM has fresh attention and
#           won't be lost in 100k tokens.
#   5. A tiny example showing one peek-then-final round.
#
# Keep it short — a long system prompt eats into the model's working
# context. ~25 lines is plenty.
# ---------------------------------------------------------------------------
def build_root_system_prompt() -> str:
    return (
        "You answer questions about a long document that is too large to read at once."
        "You have a Python REPL. The variable 'context' is a string holding the document."
        "The function `query(text, question)` invokes a fresh sub-LM on a *snipppet* of text and returns its answer (or 'NOT_FOUND')."
        "You must pass a subset of the context to the sub-agent. Never pass more than 200 characters at a sub-agent query."
        "You must return exactly one of these per turn:\n"
        "a) a ```python``` block. Whatever in 'print(...)'s comes back in the next turn.\n"
        "b) `FINAL(answer)` on its own line, ending the session.  It must not be inside a ```python``` block. You must produce this FINAL block as soon as you know the answer\n"
        """
                 - inspect `len(context)` and slice `context[:N]` to peek;
                 - use `re.findall(...)` to grep for keywords;
                 - call `query(chunk, sub_question)` for any slice that *might*"contain the answer — the sub-LM has fresh attention and won't be lost in 100k tokens.

        """
        """
        Example
        [TURN 1]
        ```python
            apps = re.find(input,pattern)
            for app in apps:
                print(query(apps, question))
        ```
        [TURN 1 Output]: NOT_FOUND, NOT_FOUND, ... , <ANSWER>, NOT_FOUND ...
        [TURN 2]
        FINAL(<ANSWER>)

        Note that [TURN 2] isn't between ``` ``` symbols.
        """
    )


@dataclass
class ReplState:
    """Persistent namespace for the root LM's REPL across turns."""
    namespace: dict


def make_repl_state(context: str, cfg_sub: LMConfig, tracker: CostTracker,
                    max_depth: int) -> ReplState:
    """Pre-populate the REPL with `context`, `query`, and a few helpers."""
    def _query_for_root(text: str, question: str) -> str:
        # Bind the configs so the root LM's `query(...)` calls the right sub-LM
        # and feeds the same tracker.
        return query(
            text, question,
            cfg=cfg_sub, tracker=tracker, current_depth=0, max_depth=max_depth,
        )
    ns: dict = {
        "__builtins__": __builtins__,
        "context": context,
        "query": _query_for_root,
        "re": re,
    }
    return ReplState(namespace=ns)


# ---------------------------------------------------------------------------
# TODO(human) #2 — Execute one Python block in the REPL namespace
# ---------------------------------------------------------------------------
# The root LM emits a ```python``` block; we have to run it and feed
# whatever it `print(...)`s back into the next turn. Two requirements:
#
#   1. The execution MUST share state across turns. The root LM may
#      assign a variable in turn 1 (e.g. `chunks = ...`) and reference
#      it in turn 3. So we exec into the SAME `state.namespace` every
#      time — never a fresh dict.
#   2. Errors must not crash the harness. If the model writes broken
#      Python, capture the traceback as a string and feed *that* back
#      to the LM. The model will usually recover on the next turn.
#
# What to do:
#   1. Create `buf = io.StringIO()`.
#   2. With `redirect_stdout(buf)`, call
#         exec(code, state.namespace)
#      inside a try/except. On exception, write
#         "ERROR: " + traceback.format_exc()
#      to `buf` (NOT to real stdout). On success do nothing extra.
#   3. Return `buf.getvalue()` — possibly empty if the code didn't print.
#
# Note: textwrap.dedent the code first so the model can indent the
# block however it likes inside the markdown fence.
# ---------------------------------------------------------------------------
def execute_python(code: str, state: ReplState) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = textwrap.dedent(code)
            exec(code, state.namespace)
        except Exception as e:
            buf.write("\n ERROR : " + traceback.format_exc())
    return buf.getvalue()


def _extract_code(reply: str) -> str | None:
    m = CODE_RE.search(reply)
    return textwrap.dedent(m.group(1)) if m else None


def _extract_final(reply: str) -> str | None:
    m = FINAL_RE.search(reply.strip())
    return m.group(1).strip() if m else None


def _truncate(s: str, limit: int = REPL_OUTPUT_TRUNC) -> str:
    if len(s) <= limit:
        return s
    return s[: limit // 2] + f"\n... [truncated {len(s) - limit} chars] ...\n" + s[-limit // 2 :]


# ---------------------------------------------------------------------------
# TODO(human) #3 — The root agent loop
# ---------------------------------------------------------------------------
# Standard ReAct-shaped loop, but "thought + action" is replaced by
# "code or FINAL". You drive the conversation until the LM emits
# `FINAL(...)` or you hit `MAX_ROUNDS`.
#
# What to do — for each round (1..MAX_ROUNDS):
#
#   1. Call `chat(cfg_root, messages=messages)` and append the assistant
#      reply to `messages` (role="assistant").
#
#   2. Try `final = _extract_final(reply)`:
#        - If not None, return `final` as the answer. Done.
#
#   3. Else try `code = _extract_code(reply)`:
#        - If None: the model produced neither code nor a final answer.
#          Append a `role="user"` nudge: "Reply with either a ```python```
#          block or `FINAL(answer)`." and continue to the next round.
#        - If present: call `execute_python(code, state)`, truncate the
#          stdout via `_truncate(...)`, and append a `role="user"`
#          message of the form
#              f"REPL OUTPUT:\n{stdout or '(no output)'}"
#          Continue.
#
#   4. If the loop exits without a FINAL, return a fallback string like
#      `"<no FINAL after MAX_ROUNDS rounds>"`.
#
# This is exactly the inference-time scaling axis the paper calls out:
# more rounds = more compute = (potentially) better answer.
# ---------------------------------------------------------------------------
def run_root_agent(
    question: str,
    context: str,
    *,
    cfg_root: LMConfig,
    cfg_sub: LMConfig,
    max_depth: int = 1,
) -> tuple[str, CostTracker]:
    tracker = CostTracker()
    state = make_repl_state(context, cfg_sub, tracker, max_depth)
    messages: list[dict] = [
        {"role": "system", "content": build_root_system_prompt()},
        {"role": "user", "content": (
            f"len(context) == {len(context)} characters.\n"
            f"QUESTION: {question}\n"
            f"Begin by writing Python that helps you locate the answer."
        )},
    ]

    for _ in range(MAX_ROUNDS):
        answer = chat(cfg_root, messages=messages)
        messages.append({"role":"assistant","content":answer})
        final = _extract_final(answer)
        if final: return final, tracker
        code = _extract_code(answer)
        if not code:
            messages.append({"role":"user","content":"Reply with either a ```python``` block or `FINAL(answer)`."})
            continue

        code_result = execute_python(code,state)
        messages.append({"role":"user", "content":f"REPL OUTPUT: \n {_truncate(code_result) or '(no output)'}"})


    return "<no FINAL after MAX_ROUNDS rounds>", tracker

# -- Demo (scaffolded) ------------------------------------------------------


def _demo_one(p: HaystackProblem, cfg_root: LMConfig, cfg_sub: LMConfig) -> None:
    print(f"\n  Q: {p.question}")
    print(f"     gold='{p.answer}'  needle@para#{p.needle_index}  haystack={len(p.haystack)} chars")
    answer, tracker = run_root_agent(p.question, p.haystack, cfg_root=cfg_root, cfg_sub=cfg_sub)
    print(f"     A: {answer}")
    print(f"     sub-calls: {tracker.n_sub_calls}  total {tracker.elapsed_s:.1f}s"
          f"  depth_hist={tracker.depth_histogram}")


def main() -> None:
    cfg_root = get_root_lm()
    cfg_sub = get_sub_lm()
    print(f"Root LM:  {cfg_root.model}")
    print(f"Sub  LM:  {cfg_sub.model}")
    problems = materialize_problems(n_filler=60, seed=7)
    for p in problems:
        _demo_one(p, cfg_root, cfg_sub)


if __name__ == "__main__":
    main()
