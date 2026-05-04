"""Hand-built ``Task`` fixtures for the mini-airline domain.

These are the eval set: a small, hand-curated list of (user goal,
expected outcome) pairs that the agent must satisfy.  They live here
(not in JSON) so the user can inspect them, tweak them, and add new
ones during the practice without touching loader code.

The mini-airline tools (see ``_01_tools_and_simulated_user.py``) are:

  * search_flights(origin, destination, date)
  * book(flight_id, passenger_name)
  * cancel(reservation_id)
  * lookup_reservation(reservation_id)
  * get_user_info(user_id)

Each Task carries an ``initial_state`` dict that pre-seeds the
in-memory mini-airline DB before the rollout starts.  This is what
makes individual rollouts reproducible.

Don't add a TODO here — these are reference fixtures, not exercises.
"""

from __future__ import annotations

from .models import Task

# Shared snapshot of the mini-airline DB used as the initial state for
# every task.  Each rollout makes its own copy.
_BASE_STATE: dict = {
    "users": {
        "u_alice": {"name": "Alice", "membership": "gold"},
        "u_bob": {"name": "Bob", "membership": "basic"},
    },
    "flights": [
        {"id": "F100", "origin": "JFK", "destination": "LAX", "date": "2026-06-01", "seats": 5},
        {"id": "F101", "origin": "JFK", "destination": "LAX", "date": "2026-06-01", "seats": 0},
        {"id": "F200", "origin": "SFO", "destination": "ORD", "date": "2026-06-02", "seats": 3},
        {"id": "F300", "origin": "LAX", "destination": "JFK", "date": "2026-06-05", "seats": 2},
    ],
    "reservations": {
        "R777": {"flight_id": "F300", "passenger_name": "Bob"},
    },
}


def _state() -> dict:
    """Deep-ish copy of the base state for a single task."""
    import copy

    return copy.deepcopy(_BASE_STATE)


GOLDEN_TASKS: list[Task] = [
    Task(
        id="T01_search_only",
        user_goal="I want to know if there are any flights from JFK to LAX on 2026-06-01.",
        gold_answer="Flight F100 is available (5 seats). F101 is sold out.",
        initial_state=_state(),
    ),
    Task(
        id="T02_book_simple",
        user_goal=(
            "Please book me on a JFK-to-LAX flight on 2026-06-01. "
            "My name is Alice."
        ),
        gold_answer="Booked Alice on flight F100; a reservation ID was returned.",
        initial_state=_state(),
    ),
    Task(
        id="T03_book_no_seats",
        user_goal=(
            "Book me on flight F101 from JFK to LAX on 2026-06-01. "
            "My name is Alice."
        ),
        gold_answer=(
            "F101 is sold out. The agent should refuse the booking and "
            "ideally suggest F100 as an alternative."
        ),
        initial_state=_state(),
    ),
    Task(
        id="T04_cancel",
        user_goal="Please cancel reservation R777.",
        gold_answer="Reservation R777 has been cancelled.",
        initial_state=_state(),
    ),
    Task(
        id="T05_lookup_then_cancel",
        user_goal=(
            "Look up reservation R777 first, tell me what it's for, "
            "and then cancel it."
        ),
        gold_answer=(
            "R777 is for Bob on flight F300 (LAX -> JFK on 2026-06-05). "
            "Then it should be cancelled."
        ),
        initial_state=_state(),
    ),
    Task(
        id="T06_user_info",
        user_goal="What's the membership tier for user u_alice?",
        gold_answer="Alice is a gold-tier member.",
        initial_state=_state(),
    ),
    Task(
        id="T07_search_no_match",
        user_goal="Are there any flights from BOS to MIA on 2026-06-01?",
        gold_answer=(
            "No flights match. The agent should say so explicitly, not "
            "hallucinate one."
        ),
        initial_state=_state(),
    ),
    Task(
        id="T08_book_then_lookup",
        user_goal=(
            "Book Alice on a JFK-to-LAX flight on 2026-06-01, then read "
            "back the reservation details to confirm."
        ),
        gold_answer=(
            "F100 booked for Alice; reservation lookup returns matching "
            "flight_id and passenger_name."
        ),
        initial_state=_state(),
    ),
]
