"""Phase 4 — pass^k aggregation.

The headline metric of tau-bench (Yao et al., 2024).  Definition:

    pass^k(system) = P[ all k independent rollouts of the same task pass ]

Aggregated across N tasks:

    pass^k = (#tasks where every one of k rollouts passed) / N

This is *much* harsher than pass@1.  A model that succeeds 80% of the
time per rollout has an *expected* pass^3 of 0.8^3 = 0.512 *if rollouts
are independent*.  In practice the true pass^k is often even lower
because tasks have systematically harder modes the agent only hits some
of the time.  pass^k is the metric that exposes those modes.

This module:
  1. computes per-task ``TaskScore`` objects from a list of per-rollout
     verdict scores;
  2. aggregates them into a scalar pass^k for one system;
  3. provides an "expected pass^k under independence" comparison helper
     so the user can eyeball the gap.

Run on its own to compute pass^k on a hand-built fake score matrix:
    uv run python -m src._04_pass_at_k
"""

from __future__ import annotations

from .models import TaskScore


# ---------------------------------------------------------------------------
# TODO(human) — pass^k formula + expected-under-independence helper
# ---------------------------------------------------------------------------
# Goal: implement the two metrics that drive tau-bench-style reporting.
#
# (1) ``pass_at_k(scores)`` — the empirical pass^k.
#     Input: a list of TaskScore objects (one per task), each carrying
#     ``rollout_scores`` of length k.
#     Output: a float in [0, 1].
#     Definition: fraction of tasks for which every rollout_score is 1.
#     Equivalent to: mean of ``ts.all_pass`` across tasks.
#     Two lines max — don't over-engineer.
#
# (2) ``expected_pass_at_k_under_independence(scores)`` —
#     If the per-rollout success of *each task* were i.i.d. with
#     probability ``p_i = mean(rollout_scores_i)``, then under
#     independence we would expect pass^k for that task to be p_i^k
#     (where k = len(rollout_scores_i)).  Return the mean of p_i^k
#     across tasks.
#     This is the right baseline to compare against the empirical
#     pass^k — when the empirical value is significantly lower than
#     the expected one, your rollouts are *correlated* (the same task
#     fails for the same systematic reason every time), which is the
#     interesting failure mode tau-bench highlights.
#
# Hints:
#   - Each TaskScore guarantees ``len(rollout_scores) >= 1``.
#   - Use plain Python; ``statistics.mean`` is fine but a comprehension
#     plus ``sum() / len()`` is what most tau-bench impls do.
#   - Total: ~6–10 lines for both functions combined.
# ---------------------------------------------------------------------------
def pass_at_k(scores: list[TaskScore]) -> float:
    """Empirical pass^k across tasks: fraction whose every rollout passed."""
    raise NotImplementedError("TODO(human): implement pass_at_k")


def expected_pass_at_k_under_independence(scores: list[TaskScore]) -> float:
    """Mean over tasks of (per-task mean rollout score)^k under independence."""
    raise NotImplementedError("TODO(human): implement expected_pass_at_k_under_independence")


# ---------------------------------------------------------------------------
# TaskScore builder (fully scaffolded)
# ---------------------------------------------------------------------------


def build_task_scores(
    system_id: str,
    rollout_scores_by_task: dict[str, list[int]],
) -> list[TaskScore]:
    """Bundle the raw 0/1 score matrix into a list of TaskScore."""
    out: list[TaskScore] = []
    for task_id, rs in rollout_scores_by_task.items():
        assert rs, f"Task {task_id} has no rollouts"
        assert all(s in (0, 1) for s in rs), f"Non-binary score in {task_id}: {rs}"
        out.append(
            TaskScore(
                task_id=task_id,
                system_id=system_id,
                rollout_scores=list(rs),
                pass_at_1=sum(rs) / len(rs),
                all_pass=int(all(s == 1 for s in rs)),
            )
        )
    return out


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    fake = build_task_scores(
        "A",
        {
            "T01": [1, 1, 1],
            "T02": [1, 1, 0],
            "T03": [0, 0, 0],
            "T04": [1, 1, 1],
            "T05": [1, 0, 1],
        },
    )
    print("Per-task scores:")
    for ts in fake:
        print(f"  {ts.task_id}: rollouts={ts.rollout_scores} pass@1={ts.pass_at_1:.2f} all_pass={ts.all_pass}")
    try:
        emp = pass_at_k(fake)
        exp = expected_pass_at_k_under_independence(fake)
        print(f"\nempirical pass^k         = {emp:.3f}")
        print(f"expected pass^k (indep.) = {exp:.3f}")
        print(f"gap                      = {exp - emp:+.3f}")
    except NotImplementedError as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
