"""Phase 4 — Compare RLM vs vanilla long-context baseline.

The whole point of this practice. We run BOTH approaches over the
same haystack problems and tabulate accuracy, sub-LM call count,
total characters fed to LMs, and wall-clock time.

Expected outcome (from the original RLM paper, scaled down for a 7B
local model): the baseline drops to roughly 0% as the haystack grows;
the RLM hangs on much longer because each `query()` sub-call sees
only a small slice of context.

Run: uv run python -m src._04_compare_rlm_vs_baseline
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .llm_config import LMConfig, chat, get_root_lm, get_sub_lm
from ._01_baseline_longctx import (
    HaystackProblem,
    build_baseline_messages,
    is_correct,
    materialize_problems,
)
from ._02_recursive_query import CostTracker
from ._03_rlm_root_agent import run_root_agent


HAYSTACK_SIZES: list[int] = [20, 60, 200, 5000]   # n_filler paragraphs


@dataclass
class Row:
    n_filler: int
    method: str
    correct: int
    total: int
    sub_calls: int
    chars_to_lm: int
    seconds: float


# ---------------------------------------------------------------------------
# TODO(human) — Extract the final answer from a raw RLM reply
# ---------------------------------------------------------------------------
# `run_root_agent` already returns the inside of `FINAL(...)` — but the
# root LM sometimes wraps that with extra prose ("The answer is X.")
# or quotes. Before grading, normalize the string so `is_correct()`
# can do its substring match.
#
# What to do:
#   1. Strip surrounding whitespace.
#   2. Remove a leading/trailing pair of single or double quotes.
#   3. Drop common prefixes like "The answer is ", "Answer: ", "Final: "
#      (case-insensitive). One pass is enough — don't loop.
#   4. Strip a trailing "." or ",".
#   5. Return the cleaned string.
#
# Why a TODO and not a fixed scaffold: the exact noise depends on the
# local model you use. Inspect a few raw `run_root_agent` outputs from
# Phase 3 and tune this until the grader stops giving false negatives.
# ---------------------------------------------------------------------------
def normalize_final_answer(raw: str) -> str:
    cured = raw.strip().strip("'\"").rstrip(".,")
    # print("normalization "+raw+" -> "+cured)
    return cured

# -- Baseline runner (scaffolded) -------------------------------------------


def _run_baseline(cfg: LMConfig, problems: list[HaystackProblem]) -> Row:
    correct = 0
    chars = 0
    t0 = time.perf_counter()
    for p in problems:
        messages = build_baseline_messages(p.haystack, p.question)
        chars += sum(len(m["content"]) for m in messages)
        prediction = chat(cfg, messages=messages, max_tokens=64).strip()
        if is_correct(prediction, p.answer):
            correct += 1
    dt = time.perf_counter() - t0
    return Row(
        n_filler=len(problems[0].haystack.split("\n\n")) - 1,
        method="baseline",
        correct=correct, total=len(problems),
        sub_calls=0, chars_to_lm=chars, seconds=dt,
    )


# -- RLM runner (scaffolded) ------------------------------------------------


def _run_rlm(cfg_root: LMConfig, cfg_sub: LMConfig,
             problems: list[HaystackProblem]) -> Row:
    correct = 0
    sub_calls = 0
    chars = 0
    t0 = time.perf_counter()
    for p in problems:
        # chars += len(p.haystack)
        raw, tracker = run_root_agent(p.question, p.haystack,
                                      cfg_root=cfg_root, cfg_sub=cfg_sub)
        sub_calls += tracker.n_sub_calls
        chars += tracker.n_input_chars
        prediction = normalize_final_answer(raw)
        if is_correct(prediction, p.answer):
            correct += 1
    dt = time.perf_counter() - t0
    return Row(
        n_filler=len(problems[0].haystack.split("\n\n")) - 1,
        method="rlm",
        correct=correct, total=len(problems),
        sub_calls=sub_calls, chars_to_lm=chars, seconds=dt,
    )


# -- Reporting (scaffolded) -------------------------------------------------


def _print_table(rows: list[Row]) -> None:
    print()
    header = f"{'haystack':>10} {'method':>10} {'acc':>8} {'sub-calls':>10} {'chars':>10} {'time':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        acc = f"{r.correct}/{r.total}"
        print(f"{r.n_filler:>10} {r.method:>10} {acc:>8} "
              f"{r.sub_calls:>10} {r.chars_to_lm:>10} {r.seconds:>7.1f}s")


def main() -> None:
    cfg_root = get_root_lm()
    cfg_sub = get_sub_lm()
    print(f"Comparison harness — root={cfg_root.model}  sub={cfg_sub.model}")
    rows: list[Row] = []
    for n in HAYSTACK_SIZES:
        problems = materialize_problems(n_filler=n, seed=7)
        print(f"\n=== haystack n_filler={n} ({len(problems[0].haystack)} chars) ===")
        rows.append(_run_baseline(cfg_root, problems))
        rows.append(_run_rlm(cfg_root, cfg_sub, problems))
    _print_table(rows)


if __name__ == "__main__":
    main()
