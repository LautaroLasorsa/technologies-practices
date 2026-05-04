"""Phase 6 — End-to-end eval driver (no TODO; ties everything together).

Usage from a script:

    from src._06_run_eval import run_eval
    report = run_eval(tasks, k=3, seed=0)
    print(report.model_dump_json(indent=2))

Pipeline:
    for each (task, system) pair:
        for each of k rollouts:
            run_agent(task, system_prompt) -> Trajectory
            judge(trajectory, task)        -> JudgeVerdict (0/1)
    aggregate per-task scores
    compute pass^k for each system
    compute paired-bootstrap CI on (pass@1_A - pass@1_B)
    return EvalReport (and optionally write JSON)

Trajectories are persisted under ``runs/<timestamp>/`` so you can audit
why something passed or failed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ._02_agent_loop import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, run_agent
from ._03_judge import judge
from ._04_pass_at_k import build_task_scores, pass_at_k
from ._05_bootstrap import paired_bootstrap_ci
from .models import EvalReport, Task, Trajectory

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def _persist_trajectory(run_dir: Path, traj: Trajectory, rollout_idx: int) -> None:
    out = run_dir / f"{traj.system_id}_{traj.task_id}_r{rollout_idx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(traj.model_dump_json(indent=2), encoding="utf-8")


def _eval_one_system(
    tasks: list[Task],
    system_prompt: str,
    system_id: str,
    k: int,
    run_dir: Path,
) -> dict[str, list[int]]:
    """Run k rollouts per task, judge each, return {task_id: [scores]}."""
    rollout_scores: dict[str, list[int]] = {}
    for task in tasks:
        rollout_scores[task.id] = []
        for r in range(k):
            print(f"  [{system_id}] {task.id} rollout {r + 1}/{k} ...", end=" ", flush=True)
            traj = run_agent(task, system_prompt, system_id=system_id)
            _persist_trajectory(run_dir, traj, r)
            verdict = judge(traj, task)
            rollout_scores[task.id].append(int(verdict.score))
            print(f"score={verdict.score}  reason={verdict.reason[:80]!r}")
    return rollout_scores


def run_eval(tasks: list[Task], *, k: int = 3, seed: int = 0) -> EvalReport:
    """Run the full A-vs-B evaluation and return an ``EvalReport``."""
    ts_start = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / ts_start
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eval run dir: {run_dir.relative_to(ROOT)}")

    print("\n--- System A ---")
    raw_A = _eval_one_system(tasks, SYSTEM_PROMPT_A, "A", k, run_dir)
    print("\n--- System B ---")
    raw_B = _eval_one_system(tasks, SYSTEM_PROMPT_B, "B", k, run_dir)

    scores_A = build_task_scores("A", raw_A)
    scores_B = build_task_scores("B", raw_B)

    pak_A = pass_at_k(scores_A)
    pak_B = pass_at_k(scores_B)
    mean_p1_A = sum(ts.pass_at_1 for ts in scores_A) / len(scores_A)
    mean_p1_B = sum(ts.pass_at_1 for ts in scores_B) / len(scores_B)
    ci = paired_bootstrap_ci(scores_A, scores_B, n_boot=10_000, confidence=0.95, seed=seed)

    report = EvalReport(
        n_tasks=len(tasks),
        k_rollouts=k,
        pass_at_k_A=pak_A,
        pass_at_k_B=pak_B,
        mean_pass_at_1_A=mean_p1_A,
        mean_pass_at_1_B=mean_p1_B,
        paired_bootstrap_ci=ci,
        per_task_A=scores_A,
        per_task_B=scores_B,
    )
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nWrote report to {(run_dir / 'report.json').relative_to(ROOT)}")
    return report


def print_report(report: EvalReport) -> None:
    """Compact terminal-friendly summary table."""
    ci = report.paired_bootstrap_ci
    print("\n" + "=" * 60)
    print(f"EVAL REPORT  ({report.n_tasks} tasks, k={report.k_rollouts} rollouts)")
    print("=" * 60)
    print(f"{'metric':<20} {'A':>10} {'B':>10}")
    print(f"{'pass@1 (mean)':<20} {report.mean_pass_at_1_A:>10.3f} {report.mean_pass_at_1_B:>10.3f}")
    print(f"{'pass^k':<20} {report.pass_at_k_A:>10.3f} {report.pass_at_k_B:>10.3f}")
    print("-" * 60)
    print(
        f"paired bootstrap (A - B): delta={ci.delta:+.3f}   "
        f"{int(ci.confidence * 100)}% CI=[{ci.lo:+.3f}, {ci.hi:+.3f}]   "
        f"significant={ci.significant}"
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    from .golden_cases import GOLDEN_TASKS

    print("This driver runs the full A-vs-B eval. For a quick smoke,")
    print("invoke with k=1 on the first 2 tasks. For real runs use demo.py.")
    try:
        report = run_eval(GOLDEN_TASKS[:2], k=1, seed=0)
        print_report(report)
    except NotImplementedError as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
