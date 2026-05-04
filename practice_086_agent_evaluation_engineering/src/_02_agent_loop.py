"""Phase 2 — ReAct-style agent loop using LangChain tool calling.

This module runs *one* agent rollout against *one* task.  The harness
in ``_06_run_eval.py`` calls it k times per task per system to compute
pass^k.

Loop shape (paraphrased from tau-bench ``run.py``):

    init state from task.initial_state
    transcript = []
    for turn in range(MAX_TURNS):
        sim_user_msg = simulated_user_reply(...)
        if sim_user_msg contains '###STOP###': break
        transcript.append({user})
        agent_response = model_with_tools.invoke(transcript)
        if agent_response.tool_calls:
            execute each tool call, append ToolMessage
        transcript.append({assistant})
        if termination_check(state, agent_response): break
    return Trajectory(...)

Two prompt variants live here as constants — ``SYSTEM_PROMPT_A`` and
``SYSTEM_PROMPT_B``.  They are what the bootstrap CI compares.  Tweak
them later to make the comparison interesting.

Run on its own to drive one rollout against one task:
    uv run python -m src._02_agent_loop
"""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ._01_tools_and_simulated_user import (
    AirlineState,
    make_tools,
    simulated_user_reply,
)
from .llm_config import LMConfig, build_chat_model, get_lm
from .models import Task, ToolCall, Trajectory

# Two prompt variants the bootstrap CI compares.
SYSTEM_PROMPT_A = textwrap.dedent(
    """\
    You are a customer-service agent for a small airline. You have tools
    to search flights, book and cancel reservations, and look up users.

    Always use a tool to obtain information instead of guessing. When
    you have answered the user's request, call the submit() tool with
    your final reply.
    """
).strip()

SYSTEM_PROMPT_B = textwrap.dedent(
    """\
    You are a customer-service agent for a small airline. You have tools
    to search flights, book and cancel reservations, and look up users.

    Procedure:
      1. Restate the user's request in one sentence.
      2. Call exactly the tools you need — never invent data.
      3. If a booking is impossible (e.g. sold out) explain why and
         suggest the closest alternative you can find with another tool.
      4. When done, call submit() with a 1–2 sentence reply.
    """
).strip()


MAX_TURNS = 8


# ---------------------------------------------------------------------------
# TODO(human) — Termination check
# ---------------------------------------------------------------------------
# Goal: decide when to break out of the agent loop.
#
# The loop calls this after every agent step.  Returning True ends the
# rollout cleanly (the trajectory is recorded and judged); returning
# False lets the loop ask the simulated user for another turn.
#
# A robust termination predicate combines THREE conditions (all are
# legitimate stopping cases — read tau-bench ``run.py`` for the
# canonical version):
#
#   1. The agent called ``submit(...)``.  We detect this by checking
#      ``state.submitted_message is not None``.  This is the *intended*
#      successful exit.
#   2. The simulated user said ``###STOP###`` on the *previous* turn.
#      The caller stores the latest user reply in ``last_user_message``.
#   3. The agent returned a non-empty ``final_message`` (i.e. plain
#      text, no tool call, no submit).  Treat this as an implicit
#      termination — many models forget to call submit(); we still
#      want a Trajectory to judge.
#
# Return True if ANY of those holds; otherwise False.
#
# Keep it ~5 lines.  No exceptions, no logging — the caller handles
# both.
# ---------------------------------------------------------------------------
def is_terminated(
    state: AirlineState,
    last_user_message: str,
    last_agent_text: str,
) -> bool:
    """Return True iff the loop should stop after the latest agent step."""
    raise NotImplementedError("TODO(human): write the termination predicate")


# ---------------------------------------------------------------------------
# Agent rollout (fully scaffolded)
# ---------------------------------------------------------------------------


def run_agent(
    task: Task,
    system_prompt: str,
    system_id: str,
    cfg: LMConfig | None = None,
) -> Trajectory:
    """Run one agent rollout on ``task`` with the given prompt variant.

    Returns a ``Trajectory`` regardless of outcome — failures are
    captured in ``trajectory.error`` so the eval harness keeps moving.
    """
    cfg = cfg or get_lm()
    state = AirlineState.from_dict(task.initial_state)
    tools = make_tools(state)
    model = build_chat_model(cfg, temperature=0.0).bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    transcript: list[Any] = [SystemMessage(content=system_prompt)]
    sim_user_history: list[dict] = []  # what the simulated user "sees"

    tool_calls_log: list[ToolCall] = []
    last_user_message = ""
    last_agent_text = ""
    error: str | None = None

    try:
        # The simulated user opens with their goal-flavoured intro.
        sim_user_history.append({"role": "user", "content": "Hello, how can you help me?"})
        opening = simulated_user_reply(task.user_goal, sim_user_history, cfg=cfg)
        sim_user_history.append({"role": "assistant", "content": opening})
        transcript.append(HumanMessage(content=opening))
        last_user_message = opening

        turn = 0
        for turn in range(1, MAX_TURNS + 1):
            ai_msg: AIMessage = model.invoke(transcript)
            transcript.append(ai_msg)
            last_agent_text = ai_msg.content if isinstance(ai_msg.content, str) else ""

            # Execute every tool call the model issued this turn.
            for tc in ai_msg.tool_calls or []:
                name = tc["name"]
                args = tc.get("args", {})
                tool = tools_by_name.get(name)
                if tool is None:
                    result = {"ok": False, "error": f"Unknown tool {name}"}
                else:
                    try:
                        result = tool.invoke(args)
                    except Exception as e:  # noqa: BLE001
                        result = {"ok": False, "error": str(e)}
                tool_calls_log.append(ToolCall(name=name, args=args, result=result))
                transcript.append(
                    ToolMessage(content=str(result), tool_call_id=tc.get("id", name))
                )

            if is_terminated(state, last_user_message, last_agent_text):
                break

            # Otherwise, ask the simulated user for the next reply.
            # Mirror the agent text into the simulated user's history.
            agent_for_user = last_agent_text or "(the agent is working — please wait or clarify)"
            sim_user_history.append({"role": "user", "content": agent_for_user})
            user_reply = simulated_user_reply(task.user_goal, sim_user_history, cfg=cfg)
            sim_user_history.append({"role": "assistant", "content": user_reply})
            last_user_message = user_reply

            if "###STOP###" in user_reply:
                break
            transcript.append(HumanMessage(content=user_reply))

    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    final = state.submitted_message or last_agent_text
    return Trajectory(
        task_id=task.id,
        system_id=system_id,
        tool_calls=tool_calls_log,
        final_message=final,
        submitted=state.submitted_message is not None,
        turns=turn,
        error=error,
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    from .golden_cases import GOLDEN_TASKS

    task = GOLDEN_TASKS[0]
    print(f"Running one agent rollout on {task.id} ...\n")
    try:
        traj = run_agent(task, SYSTEM_PROMPT_A, system_id="A")
        print(f"  turns         = {traj.turns}")
        print(f"  submitted     = {traj.submitted}")
        print(f"  tool_calls    = {len(traj.tool_calls)}")
        for tc in traj.tool_calls[:5]:
            print(f"    {tc.name}({tc.args}) -> {tc.result}")
        print(f"  final_message = {traj.final_message[:200]!r}")
        if traj.error:
            print(f"  ERROR         = {traj.error}")
    except NotImplementedError as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
