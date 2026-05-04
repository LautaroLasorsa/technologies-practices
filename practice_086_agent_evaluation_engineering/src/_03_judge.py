"""Phase 3 — LLM-as-a-Judge with structured output.

Given a ``Trajectory`` and the task's gold answer, ask a judge LM to
emit a typed ``JudgeVerdict`` (score 0/1 + reason).  We use LangChain's
``model.with_structured_output(JudgeVerdict)`` which routes through the
provider's tool-calling/JSON-mode endpoint and falls back to a JSON
schema prompt when the provider doesn't support either — drop-in
replacement for what people used to hand-roll with ``instructor``.

Bias mitigation choices live in the user-written prompt: see the TODO
below.

Run on its own to see one judge verdict on a hand-built fake trajectory:
    uv run python -m src._03_judge
"""

from __future__ import annotations

import textwrap

from .llm_config import LMConfig, build_chat_model, get_judge_lm
from .models import JudgeVerdict, Task, Trajectory


# ---------------------------------------------------------------------------
# TODO(human) — Judge prompt
# ---------------------------------------------------------------------------
# Goal: design the judge prompt that turns (task, trajectory) into a
# faithful JudgeVerdict.
#
# Read MT-Bench / "LLM-as-a-Judge" (https://arxiv.org/abs/2306.05685)
# section 3 for the empirical biases this prompt must mitigate:
#
#   - **Position bias**: in pairwise judging, the first-presented answer
#     wins more often.  We're doing *single-answer* grading here, so it
#     doesn't apply directly — but if you switch to pairwise (see
#     PairwiseVerdict in models.py) you must randomise order at the
#     call site.
#   - **Verbosity bias**: judges over-rate longer answers.  Counter by
#     instructing the judge to ignore length and grade ONLY whether the
#     gold goal was met.
#   - **Self-enhancement bias**: judges over-rate outputs from their own
#     model family.  Counter by using a *different* judge model when you
#     can (we expose JUDGE_PROVIDER / JUDGE_MODEL in .env.example for
#     exactly this).
#
# Write a system-prompt + user-prompt pair (returned as a tuple) where:
#
#   1. SYSTEM piece names the role ("a strict, terse evaluator"), states
#      the rubric, and the bias-mitigation instructions.
#   2. USER piece is an f-string-ready template with placeholders:
#        - {task_goal}      — the user_goal
#        - {gold_answer}    — the expected outcome
#        - {transcript}     — the trajectory rendered to text (use the
#                             helper ``render_trajectory()`` below)
#        - {final_message}  — the agent's last message
#      and ends with an explicit instruction asking for the JSON
#      conforming to JudgeVerdict (with_structured_output handles the
#      schema injection — your prompt just needs to be unambiguous about
#      what counts as score=1 vs score=0).
#
# Keep the SYSTEM string under ~25 lines, USER template under ~15.
# ---------------------------------------------------------------------------
def judge_prompt() -> tuple[str, str]:
    """Return (system_prompt, user_template) for the judge.

    The user_template must use these placeholders verbatim:
    ``{task_goal}``, ``{gold_answer}``, ``{transcript}``,
    ``{final_message}``.
    """
    raise NotImplementedError("TODO(human): write the judge prompt pair")


# ---------------------------------------------------------------------------
# Trajectory rendering (fully scaffolded)
# ---------------------------------------------------------------------------


def render_trajectory(traj: Trajectory) -> str:
    """Render a Trajectory as a compact transcript for the judge."""
    lines: list[str] = []
    for tc in traj.tool_calls:
        lines.append(f"TOOL {tc.name}({tc.args}) -> {tc.result}")
    if traj.final_message:
        lines.append(f"AGENT_FINAL: {traj.final_message}")
    if traj.error:
        lines.append(f"ERROR: {traj.error}")
    return "\n".join(lines) if lines else "(empty trajectory)"


# ---------------------------------------------------------------------------
# Judge call (fully scaffolded)
# ---------------------------------------------------------------------------


def judge(traj: Trajectory, task: Task, cfg: LMConfig | None = None) -> JudgeVerdict:
    """Score one trajectory against its task's gold answer."""
    cfg = cfg or get_judge_lm()
    sys_prompt, user_template = judge_prompt()
    user_msg = user_template.format(
        task_goal=task.user_goal,
        gold_answer=task.gold_answer,
        transcript=render_trajectory(traj),
        final_message=traj.final_message or "(none)",
    )
    model = build_chat_model(cfg, temperature=0.0).with_structured_output(JudgeVerdict)
    result = model.invoke(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
    )
    # with_structured_output returns either a Pydantic instance or a dict
    # depending on provider — normalise.
    if isinstance(result, dict):
        result = JudgeVerdict.model_validate(result)
    return result


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    from .models import ToolCall

    fake_task = Task(
        id="T_demo",
        user_goal="Cancel reservation R777.",
        gold_answer="Reservation R777 has been cancelled.",
    )
    fake_traj = Trajectory(
        task_id=fake_task.id,
        system_id="A",
        tool_calls=[
            ToolCall(name="cancel", args={"reservation_id": "R777"}, result={"ok": True}),
            ToolCall(
                name="submit",
                args={"message": "Done — R777 cancelled."},
                result={"ok": True},
            ),
        ],
        final_message="Done — R777 cancelled.",
        submitted=True,
        turns=2,
    )
    print("Judging a hand-built trajectory ...")
    try:
        verdict = judge(fake_traj, fake_task)
        print(f"  score  = {verdict.score}")
        print(f"  reason = {verdict.reason!r}")
    except NotImplementedError as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
