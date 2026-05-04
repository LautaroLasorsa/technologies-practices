"""Shared Pydantic data models for the agent-evaluation pipeline.

These types are the *interface* between every stage of the evaluation
harness:

    Task ──► [Agent (system A or B)] ──► Trajectory ──► [Judge] ──► JudgeVerdict
                                                                         │
                                                                         ▼
                                                         pass^k + paired-bootstrap CI
                                                                         │
                                                                         ▼
                                                                    EvalReport

Conventions:
- Tasks have a stable string ``id`` so per-task scores can be paired
  across systems for the bootstrap.
- A trajectory captures the *full* tool-call sequence + final agent
  message.  The judge sees this as a transcript.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """One evaluation task = one (user-goal, gold-state) pair.

    ``user_goal``     a natural-language description fed to the simulated user
    ``gold_answer``   what a correct final agent response should convey
    ``initial_state`` optional dict used to pre-seed the mini-airline DB
    """

    id: str
    user_goal: str
    gold_answer: str
    initial_state: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trajectories (what the agent produces)
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """One tool invocation inside a trajectory."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class Trajectory(BaseModel):
    """The full agent rollout for one task."""

    task_id: str
    system_id: str = Field(description="'A' or 'B' — which prompt variant produced this rollout.")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_message: str = ""
    submitted: bool = Field(
        default=False,
        description="True iff the agent invoked the submit() tool.",
    )
    turns: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Judge verdicts (LLM-as-Judge structured output)
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Single-answer judge verdict.

    ``score`` is the binary pass/fail metric used downstream by pass^k
    and the paired bootstrap.  ``reason`` is for human auditing.
    """

    score: Literal[0, 1] = Field(description="1 = task solved, 0 = not solved.")
    reason: str = Field(description="Concise justification, ideally citing tool calls or text.")


class PairwiseVerdict(BaseModel):
    """Pairwise judge verdict (used as an alternative to single-answer).

    The judge sees both trajectories side-by-side and picks a winner —
    or a tie.  Position bias is mitigated by random ordering at the
    call site (see ``_03_judge.py``).
    """

    winner: Literal["A", "B", "tie"] = Field(description="Which side did better, or 'tie'.")
    reason: str


# ---------------------------------------------------------------------------
# Aggregate eval outputs
# ---------------------------------------------------------------------------


class TaskScore(BaseModel):
    """Per-task aggregated score for one system across k rollouts."""

    task_id: str
    system_id: str
    rollout_scores: list[int] = Field(description="One 0/1 entry per rollout.")
    pass_at_1: float = Field(description="Mean of rollout_scores (= classical pass@1).")
    all_pass: int = Field(
        description="1 iff every rollout passed (this task contributes to pass^k).",
    )


class BootstrapCI(BaseModel):
    """Paired-bootstrap confidence interval for mean(score_A - score_B)."""

    delta: float = Field(description="Observed mean per-task delta (A - B).")
    lo: float
    hi: float
    confidence: float = Field(default=0.95, description="e.g. 0.95 for a 95% CI.")
    n_bootstrap: int = Field(description="Number of bootstrap resamples.")
    significant: bool = Field(
        description="True iff the CI excludes 0 (informal significance).",
    )


class EvalReport(BaseModel):
    """End-to-end evaluation report comparing two systems."""

    n_tasks: int
    k_rollouts: int
    pass_at_k_A: float
    pass_at_k_B: float
    mean_pass_at_1_A: float
    mean_pass_at_1_B: float
    paired_bootstrap_ci: BootstrapCI
    per_task_A: list[TaskScore]
    per_task_B: list[TaskScore]
