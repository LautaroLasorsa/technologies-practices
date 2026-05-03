"""Phase 1 — Natural Language → ScheduleRequest

The first stage of the hybrid pipeline: take a user's free-form
scheduling description ("I need to staff Mon morning, Mon evening and
Tue morning. Alice can't work mornings, Bob is a manager and must be on
Tue morning, ...") and turn it into a *typed*, *validated*
``ScheduleRequest`` that the symbolic solver can consume.

We do this with **structured output** rather than free-form generation:

  - Pydantic ``ScheduleRequest`` is the contract.
  - ``instructor`` (https://python.useinstructor.com/) wraps the LLM call
    and re-prompts on validation failures, so by the time the function
    returns we already know the JSON parses and matches the schema.

This is the LLM playing to its strengths — language understanding — and
NOT trying to also solve the combinatorial problem.  The solver does
that next.

Run on its own to sanity-check extraction:
    uv run python -m src._01_constraint_extraction
"""

from __future__ import annotations

import textwrap

from .llm_config import LMConfig, get_lm, instructor_client
from .models import ScheduleRequest

EXTRACTION_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a scheduling-problem extractor.

    Given a natural-language description of a staffing problem, emit a
    JSON object that matches the ScheduleRequest schema you have been
    given.  Rules:

    - Employee and shift IDs are short snake_case strings derived from
      names/labels (e.g. 'alice', 'mon_morning').
    - Always include a CoverageConstraint and a MaxShiftsPerEmployee
      constraint unless the user explicitly opts out.
    - If a shift mentions a required qualification, include a
      QualificationConstraint and set Shift.required_qualification.
    - 'Cannot work' / 'unavailable' → ForbiddenAssignment.
    - 'Must work' / 'fixed' / 'pre-assigned' → RequiredAssignment.
    - Preferences ('would prefer', 'likes') → SoftPreference with a
      positive weight (1–5); aversions → negative weight.
    - If a number is missing, prefer demand=1 and max_shifts=5.

    Do not invent employees or shifts that aren't mentioned.
    """
)


# ---------------------------------------------------------------------------
# TODO(human) — Structured extraction
# ---------------------------------------------------------------------------
# Goal: turn `user_request` (free text) into a `ScheduleRequest`.
#
# We are doing this through `instructor`, which patches LiteLLM so that
# passing `response_model=ScheduleRequest` makes the SDK:
#   1. Inject the JSON schema of ScheduleRequest into the request,
#   2. Parse the model's reply,
#   3. Validate it against ScheduleRequest,
#   4. Re-prompt up to `max_retries` times on validation failure.
#
# This is the canonical way to use instructor — see
# https://python.useinstructor.com/concepts/models/ — do NOT hand-roll
# JSON parsing or regex extraction here.
#
# What to do:
#   1. Build a `messages` list:
#        [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
#         {"role": "user",   "content": user_request}]
#   2. Call `client.chat.completions.create(...)` with:
#        - model=cfg.litellm_model
#        - messages=messages
#        - response_model=ScheduleRequest
#        - max_retries=2
#        - temperature=0.0
#        - api_key / api_base from `cfg` if non-empty (LiteLLM forwards
#          them to the underlying provider — same pattern as `chat()`
#          in llm_config.py).
#   3. Return the result (instructor returns a ScheduleRequest already).
#
# Why we lean on instructor: the validator is the LLM's "tests".  If the
# extraction is wrong (missing field, bad enum value), Pydantic raises
# a ValidationError, instructor catches it, appends the error message to
# the conversation, and retries.  No manual parsing logic needed.
# ---------------------------------------------------------------------------
def extract_schedule_request(user_request: str, cfg: LMConfig | None = None) -> ScheduleRequest:
    """Extract a ScheduleRequest from a natural-language description."""
    cfg = cfg or get_lm()
    client = instructor_client(cfg)
    return client.chat.completions.create(
        response_model = ScheduleRequest,
        messages = [{"role":"system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role":"user", "content": user_request}],
        model = cfg.litellm_model,
        max_retries=2,
        temperature = 0.0,
        api_key = cfg.api_key,
        api_base = cfg.base_url
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


_DEMO_REQUEST = textwrap.dedent(
    """\
    Please schedule the coffee shop for Monday and Tuesday.

    Shifts:
      - Monday morning (needs 1 barista)
      - Monday evening (needs 1 barista)
      - Tuesday morning (needs 1 manager)

    Employees:
      - Alice, barista, can do up to 3 shifts. Cannot work mornings.
      - Bob, barista and manager, can do up to 3 shifts. Prefers Tuesday morning.
      - Carol, barista, can do up to 2 shifts.
    """
)


def main() -> None:
    print("Extracting structured ScheduleRequest from natural language...\n")
    request = extract_schedule_request(_DEMO_REQUEST)
    print(request.model_dump_json(indent=2))
    print(
        f"\nParsed {len(request.employees)} employees, "
        f"{len(request.shifts)} shifts, "
        f"{len(request.hard_constraints)} hard constraints, "
        f"{len(request.soft_preferences)} soft preferences."
    )


if __name__ == "__main__":
    main()
