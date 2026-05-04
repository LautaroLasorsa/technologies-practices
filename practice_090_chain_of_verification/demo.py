"""End-to-end demo: factored vs joint CoVe over the golden case set.

For each prompt in ``src.golden_cases.GOLDEN_CASES`` this script runs
both pipelines, prints them side by side, and writes a JSONL run record
under ``runs/``.  Use it to *see* the contamination effect — watch how
joint runs more often parrot the baseline back, while factored runs
catch and correct the bad items.

Until you implement the TODO(human) functions in ``src/_NN_*.py`` this
script will raise ``NotImplementedError`` from the first stage that
isn't done yet.  That's expected — work through the TODOs in order.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src._05_pipeline import cove_factored, cove_joint
from src.golden_cases import GOLDEN_CASES, GoldenCase
from src.models import RunRecord

RUNS_DIR = Path(__file__).parent / "runs"


def _print_block(title: str, body: str) -> None:
    print(f"  [{title}]")
    for line in body.splitlines():
        print(f"    {line}")
    print()


def _print_corrections(record: RunRecord) -> None:
    if not record.refined.corrections:
        print("    (no corrections)")
        return
    for c in record.refined.corrections:
        print(f"    - {c}")


def _run_one(case: GoldenCase) -> tuple[RunRecord, RunRecord]:
    print("=" * 70)
    print(f"CASE: {case.name}")
    print(f"PROMPT: {case.prompt}")
    if case.hint:
        print(f"HINT (private, not shown to LLM): {case.hint}")
    print("=" * 70)

    print("\n--- JOINT ---")
    joint = cove_joint(case.prompt)
    _print_block("final", joint.refined.text)
    print("  [corrections]")
    _print_corrections(joint)

    print("\n--- FACTORED ---")
    factored = cove_factored(case.prompt)
    _print_block("baseline", factored.baseline.text)
    print(f"  [verifications: {len(factored.plan.questions)} questions, "
          f"{sum(1 for a in factored.answers if a.verdict == 'contradicts')} contradicted]")
    _print_block("final", factored.refined.text)
    print("  [corrections]")
    _print_corrections(factored)
    print()

    return joint, factored


def _persist(stem: str, records: list[RunRecord]) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{stem}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} run records to {out.relative_to(Path(__file__).parent)}")


def main() -> None:
    print("Practice 090 — Chain-of-Verification (CoVe)")
    print("Comparing JOINT vs FACTORED on the golden case set\n")

    all_records: list[RunRecord] = []
    for case in GOLDEN_CASES:
        joint, factored = _run_one(case)
        all_records.extend([joint, factored])

    stem = datetime.now().strftime("%Y%m%d-%H%M%S") + "-cove"
    _persist(stem, all_records)


if __name__ == "__main__":
    main()
