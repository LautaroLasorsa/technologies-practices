"""Exercise 01: Circuit Breaker Pattern

Implement a circuit breaker as a finite state machine with three states:
  CLOSED   -- Requests pass through. Failures are counted.
  OPEN     -- Requests are rejected immediately (fail fast).
  HALF_OPEN -- A limited number of probe requests are allowed through.

The breaker wraps an async callable (e.g., an HTTP request to a downstream
service) and decides whether to let the call through or reject it based on
the current state and recent failure history.

Run (with 00_unreliable_service.py running in another terminal):
    uv run python src/01_circuit_breaker.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

import aiohttp


# ---------------------------------------------------------------------------
# Circuit Breaker State
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    """The three states of a circuit breaker FSM."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit breaker is OPEN."""

    def __init__(self, name: str, remaining_secs: float) -> None:
        self.name = name
        self.remaining_secs = remaining_secs
        super().__init__(
            f"Circuit '{name}' is OPEN. "
            f"Recovery in {remaining_secs:.1f}s."
        )


class CircuitBreakerCallError(Exception):
    """Wraps the original exception that caused a failure through the breaker."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreakerMetrics:
    """Counters exposed for monitoring and the demo dashboard."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: list[tuple[float, CircuitState, CircuitState]] = field(
        default_factory=list
    )

    @property
    def failure_rate(self) -> float:
        completed = self.successful_calls + self.failed_calls
        if completed == 0:
            return 0.0
        return self.failed_calls / completed


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Async circuit breaker wrapping an arbitrary async callable.

    Parameters
    ----------
    name:
        Human-readable name (used in errors and logs).
    failure_threshold:
        Number of consecutive failures in CLOSED state that trips the breaker
        to OPEN.
    recovery_timeout:
        Seconds to stay OPEN before transitioning to HALF_OPEN.
    half_open_max_calls:
        Number of probe calls allowed in HALF_OPEN.  If all succeed, the
        breaker transitions to CLOSED.  If any fail, it goes back to OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.metrics = CircuitBreakerMetrics()

        # -- internal state (to be managed by your implementation) --
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count_half_open: int = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0

    # -- public read-only properties ----------------------------------------

    @property
    def state(self) -> CircuitState:
        # TODO(human): Implement the time-based auto-transition from OPEN -> HALF_OPEN.
        #
        # WHY THIS MATTERS:
        # The circuit breaker shouldn't require an external timer to transition
        # from OPEN to HALF_OPEN.  Instead, the transition happens lazily: every
        # time someone reads the state, check whether enough time has passed since
        # the breaker opened.  If recovery_timeout has elapsed, transition to
        # HALF_OPEN.  This is the standard implementation approach used by
        # resilience4j, Polly, and Hystrix.
        #
        # WHAT TO DO:
        # 1. If the current state is OPEN, check if (now - _opened_at) >=
        #    recovery_timeout.
        # 2. If yes, transition to HALF_OPEN:
        #    - Set _state to HALF_OPEN
        #    - Reset _success_count_half_open to 0
        #    - Record the state transition in metrics.state_transitions
        #      as a tuple of (timestamp, old_state, new_state)
        #    - Print a log line so the user can see the transition
        # 3. Return self._state (whether or not a transition occurred).
        #
        # HINT:
        #   remaining = self.recovery_timeout - (time.time() - self._opened_at)
        #   if remaining <= 0: ... transition ...
        raise NotImplementedError("TODO(human): implement state property")

    # -- state transition helpers -------------------------------------------

    def _transition_to(self, new_state: CircuitState) -> None:
        """Record a state transition (logging + metrics)."""
        old = self._state
        self._state = new_state
        now = time.time()
        self.metrics.state_transitions.append((now, old, new_state))
        print(f"  [CB:{self.name}] {old.value} -> {new_state.value}")

    # -- core call method ---------------------------------------------------

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute *func* through the circuit breaker.

        If the breaker is OPEN, raise CircuitOpenError immediately.
        If CLOSED or HALF_OPEN, execute the function and track the outcome.

        Returns the function's return value on success.
        Raises CircuitBreakerCallError wrapping the original exception on failure.
        Raises CircuitOpenError when the breaker rejects the call.
        """
        # TODO(human): Implement the full circuit breaker call logic.
        #
        # WHY THIS MATTERS:
        # This is the heart of the circuit breaker pattern.  Every downstream
        # call goes through this method, which decides: should I let this
        # request through, or should I reject it immediately?  The decision
        # depends on the current state, which in turn depends on the history
        # of successes and failures.  Getting this right is the difference
        # between a system that degrades gracefully and one that cascades.
        #
        # THE ALGORITHM (step by step):
        #
        # 1. READ the current state (use self.state property -- this triggers
        #    the OPEN -> HALF_OPEN auto-transition if the timeout has elapsed).
        #
        # 2. IF state is OPEN:
        #    - Increment metrics.rejected_calls
        #    - Calculate remaining seconds until recovery:
        #        remaining = recovery_timeout - (now - _opened_at)
        #    - Raise CircuitOpenError(self.name, remaining)
        #    - This is the "fail fast" behavior: no network call, no waiting,
        #      just an immediate exception in <1ms.
        #
        # 3. IF state is CLOSED or HALF_OPEN:
        #    - Increment metrics.total_calls
        #    - Try to execute: result = await func(*args, **kwargs)
        #
        #    3a. ON SUCCESS:
        #        - Increment metrics.successful_calls
        #        - If state is CLOSED:
        #            - Reset _failure_count to 0 (success resets the counter)
        #        - If state is HALF_OPEN:
        #            - Increment _success_count_half_open
        #            - If _success_count_half_open >= half_open_max_calls:
        #                - The downstream has proven itself healthy!
        #                - Transition to CLOSED using _transition_to()
        #                - Reset _failure_count to 0
        #        - Return result
        #
        #    3b. ON FAILURE (any exception):
        #        - Increment metrics.failed_calls
        #        - Record _last_failure_time = time.time()
        #        - If state is CLOSED:
        #            - Increment _failure_count
        #            - If _failure_count >= failure_threshold:
        #                - Trip the breaker!  Transition to OPEN
        #                - Record _opened_at = time.time()
        #        - If state is HALF_OPEN:
        #            - The probe failed: downstream is still broken
        #            - Transition back to OPEN immediately
        #            - Record _opened_at = time.time()
        #        - Raise CircuitBreakerCallError(original_exception)
        #
        # IMPORTANT SUBTLETY:
        # Read self.state ONCE at the top and use that snapshot for all
        # decisions within this call.  If you re-read self.state after the
        # await, another coroutine may have changed it, leading to inconsistent
        # behavior.  In a real system you'd use a lock; for this exercise,
        # single-threaded asyncio means this is safe as long as you don't
        # yield between reading state and making the decision.
        #
        # HINT for the try/except:
        #   try:
        #       result = await func(*args, **kwargs)
        #   except Exception as exc:
        #       ... handle failure ...
        #       raise CircuitBreakerCallError(exc) from exc
        raise NotImplementedError("TODO(human): implement circuit breaker call()")


# ---------------------------------------------------------------------------
# Helper: call the unreliable service
# ---------------------------------------------------------------------------

SERVICE_URL = "http://localhost:8089"


async def call_unreliable_service(session: aiohttp.ClientSession) -> dict[str, Any]:
    """Make an HTTP GET to the unreliable service's /api/data endpoint.

    Raises on non-2xx status codes so the circuit breaker sees them as failures.
    """
    async with session.get(
        f"{SERVICE_URL}/api/data", timeout=aiohttp.ClientTimeout(total=5)
    ) as resp:
        body = await resp.json()
        if resp.status >= 500:
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=body.get("error", "Server error"),
            )
        return body


async def switch_mode(session: aiohttp.ClientSession, mode: str) -> None:
    """Switch the unreliable service to a different failure mode."""
    async with session.post(f"{SERVICE_URL}/admin/mode/{mode}") as resp:
        result = await resp.json()
        print(f"\n  [Service] Mode changed: {result['previous_mode']} -> {result['current_mode']}")


# ---------------------------------------------------------------------------
# Demo: drive the circuit breaker through its lifecycle
# ---------------------------------------------------------------------------


async def demo_circuit_breaker() -> None:
    """Drive the circuit breaker through CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    This demo:
    1. Starts with the service healthy -- calls succeed, breaker stays CLOSED
    2. Switches the service to DOWN -- calls fail, breaker trips to OPEN
    3. Waits for recovery_timeout -- breaker moves to HALF_OPEN
    4. Switches the service back to healthy -- probe calls succeed, breaker
       recovers to CLOSED
    5. Prints a summary of metrics and state transitions
    """
    # TODO(human): Implement the demo that exercises the circuit breaker lifecycle.
    #
    # WHY THIS MATTERS:
    # Seeing the circuit breaker transition through all three states builds deep
    # intuition for how the parameters (failure_threshold, recovery_timeout,
    # half_open_max_calls) affect behavior.  The timing output shows you *why*
    # the recovery_timeout exists -- without it, the breaker would either never
    # recover or would immediately re-trip.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. Create a CircuitBreaker with reasonable test parameters:
    #        name="demo", failure_threshold=3, recovery_timeout=5.0,
    #        half_open_max_calls=2
    #    (Short timeout so the demo doesn't take forever.)
    #
    # 2. async with aiohttp.ClientSession() as session:
    #
    # 3. PHASE 1 - HEALTHY (breaker stays CLOSED):
    #    - Print a header like "--- Phase 1: Healthy service (CLOSED state) ---"
    #    - Send 5 requests through the breaker:
    #        result = await breaker.call(call_unreliable_service, session)
    #    - Print each result and the breaker state
    #    - All should succeed, state should remain CLOSED
    #
    # 4. PHASE 2 - SERVICE DOWN (breaker trips to OPEN):
    #    - Switch the service to "down" mode: await switch_mode(session, "down")
    #    - Send requests in a loop (e.g., 8 iterations):
    #      - try: await breaker.call(call_unreliable_service, session)
    #      - except CircuitOpenError as e: print rejection message
    #      - except CircuitBreakerCallError as e: print failure message
    #    - The first `failure_threshold` calls will fail (CircuitBreakerCallError)
    #    - After that, calls are rejected instantly (CircuitOpenError)
    #    - Print the breaker state after each call to observe the transition
    #
    # 5. PHASE 3 - WAIT FOR RECOVERY TIMEOUT:
    #    - Print "Waiting for recovery timeout..."
    #    - await asyncio.sleep(breaker.recovery_timeout + 0.5)
    #      (the +0.5 ensures the timeout has definitely elapsed)
    #    - Print the breaker state -- it should now be HALF_OPEN
    #
    # 6. PHASE 4 - SERVICE RECOVERS (breaker returns to CLOSED):
    #    - Switch service back to healthy: await switch_mode(session, "healthy")
    #    - Send half_open_max_calls requests through the breaker
    #    - These are "probe" requests -- if they succeed, the breaker closes
    #    - Print each result and the breaker state
    #    - After enough successes, the breaker should transition to CLOSED
    #
    # 7. SUMMARY:
    #    - Print breaker.metrics: total_calls, successful, failed, rejected
    #    - Print the failure_rate
    #    - Print all state_transitions with timestamps
    #
    # HINT for nice output:
    #   for i, (ts, old, new) in enumerate(breaker.metrics.state_transitions):
    #       print(f"  {i+1}. {old.value} -> {new.value} at t={ts - start:.1f}s")
    raise NotImplementedError("TODO(human): implement demo_circuit_breaker()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 01: Circuit Breaker Pattern")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  - 00_unreliable_service.py running on port 8089")
    print()
    asyncio.run(demo_circuit_breaker())


if __name__ == "__main__":
    main()
