"""Phase 1 — Mini-airline tools + the simulated user.

Two pieces in one module because they're tightly coupled:

  1. **Tools** — five LangChain ``@tool``-decorated callables that mutate
     a per-rollout in-memory ``AirlineState``.  This is the *environment*
     the agent acts on.  Fully scaffolded — no TODO here.

  2. **Simulated user** — the part of tau-bench that makes evaluation
     interesting.  Instead of feeding the agent a single-shot prompt and
     hoping for the best, we simulate a *human user* with goals, who
     *replies* to the agent's clarifying questions.  This catches whole
     classes of failures (looping, asking endless clarifications, going
     off-script) that single-turn evals miss.

The simulated user is itself an LLM with a system prompt that defines:
  - the user's underlying goal (from ``Task.user_goal``);
  - what information the user knows vs. doesn't know;
  - that the user terminates the conversation when satisfied or fed up.

Run on its own to see one simulated user reply:
    uv run python -m src._01_tools_and_simulated_user
"""

from __future__ import annotations

import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .llm_config import LMConfig, build_chat_model, chat, get_lm


# ---------------------------------------------------------------------------
# Per-rollout mini-airline state
# ---------------------------------------------------------------------------


@dataclass
class AirlineState:
    """In-memory mini-airline DB for one rollout.

    Built fresh from ``Task.initial_state`` for every (task, rollout)
    pair so rollouts don't pollute each other.
    """

    users: dict[str, dict] = field(default_factory=dict)
    flights: list[dict] = field(default_factory=list)
    reservations: dict[str, dict] = field(default_factory=dict)
    submitted_message: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "AirlineState":
        return cls(
            users=dict(d.get("users", {})),
            flights=list(d.get("flights", [])),
            reservations=dict(d.get("reservations", {})),
        )


# ---------------------------------------------------------------------------
# Tool argument schemas
# ---------------------------------------------------------------------------


class SearchFlightsArgs(BaseModel):
    origin: str = Field(description="Origin airport code, e.g. 'JFK'.")
    destination: str = Field(description="Destination airport code, e.g. 'LAX'.")
    date: str = Field(description="ISO date, e.g. '2026-06-01'.")


class BookArgs(BaseModel):
    flight_id: str
    passenger_name: str


class CancelArgs(BaseModel):
    reservation_id: str


class LookupReservationArgs(BaseModel):
    reservation_id: str


class GetUserInfoArgs(BaseModel):
    user_id: str


class SubmitArgs(BaseModel):
    message: str = Field(description="Final natural-language reply to the user.")


# ---------------------------------------------------------------------------
# Tool factories — close over the per-rollout state
# ---------------------------------------------------------------------------


def make_tools(state: AirlineState) -> list[StructuredTool]:
    """Build the LangChain ``StructuredTool`` list for one rollout.

    Each tool closes over ``state``, so calls mutate the per-rollout DB
    in-place.  The returned list is what we hand to
    ``model.bind_tools(...)`` in ``_02_agent_loop.py``.
    """

    def search_flights(origin: str, destination: str, date: str) -> list[dict]:
        return [
            f
            for f in state.flights
            if f["origin"] == origin and f["destination"] == destination and f["date"] == date
        ]

    def book(flight_id: str, passenger_name: str) -> dict:
        flight = next((f for f in state.flights if f["id"] == flight_id), None)
        if flight is None:
            return {"ok": False, "error": f"Unknown flight_id {flight_id}"}
        if flight["seats"] <= 0:
            return {"ok": False, "error": f"Flight {flight_id} is sold out"}
        flight["seats"] -= 1
        rid = f"R{uuid.uuid4().hex[:6].upper()}"
        state.reservations[rid] = {"flight_id": flight_id, "passenger_name": passenger_name}
        return {"ok": True, "reservation_id": rid}

    def cancel(reservation_id: str) -> dict:
        if reservation_id not in state.reservations:
            return {"ok": False, "error": f"Unknown reservation_id {reservation_id}"}
        res = state.reservations.pop(reservation_id)
        flight = next((f for f in state.flights if f["id"] == res["flight_id"]), None)
        if flight is not None:
            flight["seats"] += 1
        return {"ok": True}

    def lookup_reservation(reservation_id: str) -> dict:
        res = state.reservations.get(reservation_id)
        if res is None:
            return {"ok": False, "error": f"Unknown reservation_id {reservation_id}"}
        return {"ok": True, **res}

    def get_user_info(user_id: str) -> dict:
        info = state.users.get(user_id)
        if info is None:
            return {"ok": False, "error": f"Unknown user_id {user_id}"}
        return {"ok": True, **info}

    def submit(message: str) -> dict:
        """Signal that the agent is done and return the final message to the user."""
        state.submitted_message = message
        return {"ok": True}

    specs: list[tuple[Callable, type[BaseModel], str]] = [
        (search_flights, SearchFlightsArgs, "Search for flights matching origin/destination/date."),
        (book, BookArgs, "Book a passenger on a specific flight; returns a reservation_id."),
        (cancel, CancelArgs, "Cancel a reservation by id."),
        (lookup_reservation, LookupReservationArgs, "Look up a reservation by id."),
        (get_user_info, GetUserInfoArgs, "Look up user metadata by user_id."),
        (
            submit,
            SubmitArgs,
            "Submit the final natural-language reply to the user. Call this exactly once when done.",
        ),
    ]
    return [
        StructuredTool.from_function(func=fn, args_schema=schema, description=desc)
        for fn, schema, desc in specs
    ]


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------


# TODO(human) — Simulated-user system prompt
# ---------------------------------------------------------------------------
# Goal: write a system prompt that turns a generic chat model into a
# *simulated user* for tau-bench-style evaluation.
#
# Why this matters: the simulated user IS the eval.  If the prompt lets
# the LM volunteer the gold answer up front, every rollout passes
# trivially.  If it makes the user too adversarial, every rollout fails.
# Striking the right balance is one of the core design skills tau-bench
# teaches.
#
# What the prompt MUST encode (read https://arxiv.org/abs/2406.12045
# section 3 — "User Simulation" — for the canonical version):
#   1. The persona: a customer with a goal, talking to a human agent.
#   2. The goal is given by ``{user_goal}`` (already inserted via
#      ``.format(user_goal=...)`` below — your prompt should reference it).
#   3. The user knows their goal but does NOT know the airline's internal
#      schema (no flight IDs unless the agent surfaces them, no DB
#      details).  Reply only with what a real user would plausibly say.
#   4. Reply concisely (1–3 sentences). Don't write the agent's lines.
#   5. When the agent has clearly satisfied the goal — or refuses with a
#      reason — the user terminates by replying with exactly the token
#      ``###STOP###`` and nothing else.  The harness watches for this.
#   6. Do NOT roleplay any tool calls or system messages.
#
# Style: terse, behavioural, ~10–20 lines.  Triple-quoted f-string-like
# template using ``{user_goal}`` as the single placeholder.  Do NOT
# .format() it here — return the *template string* and let the caller
# substitute.
# ---------------------------------------------------------------------------
def simulated_user_system_prompt() -> str:
    """Return the system-prompt *template* for the simulated user.

    Must contain the literal placeholder ``{user_goal}`` exactly once.
    The harness fills it in per-task at call time.
    """
    raise NotImplementedError("TODO(human): write the simulated-user system prompt template")


def simulated_user_reply(
    user_goal: str,
    transcript: list[dict],
    cfg: LMConfig | None = None,
) -> str:
    """Get the simulated user's next reply given the conversation so far.

    ``transcript`` is the list of {role, content} dicts as the user has
    seen them — i.e. the *agent* messages are role='user' from the
    simulated user's perspective, and the simulated user's prior replies
    are role='assistant'.  The harness flips perspectives in
    ``_02_agent_loop.py``.
    """
    cfg = cfg or get_lm()
    model = build_chat_model(cfg, temperature=0.7)  # a bit of variability is healthy
    sys_prompt = simulated_user_system_prompt().format(user_goal=user_goal)
    messages = [{"role": "system", "content": sys_prompt}, *transcript]
    return chat(model, messages).strip()


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    print("Mini-airline tools + simulated user — sanity check\n")

    # 1. Show the tool list as the agent will see it.
    state = AirlineState(
        flights=[{"id": "F100", "origin": "JFK", "destination": "LAX", "date": "2026-06-01", "seats": 5}],
        users={},
        reservations={},
    )
    tools = make_tools(state)
    print("Tools available to the agent:")
    for t in tools:
        print(f"  - {t.name}({list(t.args.keys())})  // {t.description}")

    # 2. Try invoking one tool directly.
    res = next(t for t in tools if t.name == "search_flights").invoke(
        {"origin": "JFK", "destination": "LAX", "date": "2026-06-01"}
    )
    print(f"\nsearch_flights -> {res}")

    # 3. Get one simulated-user reply (requires the LM to be reachable).
    print("\nSimulated user reply (agent just opened the chat with 'How can I help?'):")
    try:
        reply = simulated_user_reply(
            user_goal="I want to book a JFK -> LAX flight on 2026-06-01. My name is Alice.",
            transcript=[{"role": "user", "content": "How can I help you today?"}],
        )
        print(f"  USER: {reply}")
    except NotImplementedError as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
