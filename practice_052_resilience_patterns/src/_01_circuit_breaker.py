"""Exercise 01: Circuit Breaker Pattern.

Implement the three state-machine rules of a circuit breaker:
  - OPEN -> HALF_OPEN after recovery_timeout (lazy, on state read)
  - success handling (reset / probe closing)
  - failure handling (trip from CLOSED / re-trip from HALF_OPEN)

The `call()` method, the unreliable-service HTTP helpers, and the
lifecycle demo are all scaffolded for you.

Run (with _00_unreliable_service running in another terminal):
    uv run python -m src._01_circuit_breaker
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

import aiohttp


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    def __init__(self, name: str, remaining_secs: float) -> None:
        self.name = name
        self.remaining_secs = remaining_secs
        super().__init__(f"Circuit '{name}' is OPEN (recovery in {remaining_secs:.1f}s)")


class CircuitBreakerCallError(Exception):
    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))


@dataclass
class CircuitBreakerMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: list[tuple[float, CircuitState, CircuitState]] = field(default_factory=list)


class CircuitBreaker:
    """Async circuit breaker wrapping any async callable."""

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

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count_half_open: int = 0
        self._opened_at: float = 0.0

    def _transition_to(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        self.metrics.state_transitions.append((time.time(), old, new_state))
        print(f"  [CB:{self.name}] {old.value} -> {new_state.value}")

    # -- TODO 1 ----------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return current state, auto-transitioning OPEN -> HALF_OPEN if the
        recovery timeout has elapsed.

        The transition is lazy (no background timer): every read checks the
        clock. When transitioning, reset `_success_count_half_open` and use
        `_transition_to()` to log and record the transition.
        """
        # TODO(human): if self._state is OPEN and enough time has passed since
        # self._opened_at, transition to HALF_OPEN (reset the half-open success
        # counter, then _transition_to). Return self._state either way.
        raise NotImplementedError("TODO(human): implement state property")

    # -- TODO 2 ----------------------------------------------------------------

    def _handle_success(self) -> None:
        """Update state/counters after a successful call.

        - CLOSED: reset the consecutive failure counter to 0.
        - HALF_OPEN: increment the probe success counter. If it reaches
          `half_open_max_calls`, the downstream has proven healthy — close the
          breaker and reset counters.
        """
        # TODO(human): implement the CLOSED and HALF_OPEN branches described above.
        raise NotImplementedError("TODO(human): implement _handle_success")

    # -- TODO 3 ----------------------------------------------------------------

    def _handle_failure(self) -> None:
        """Update state/counters after a failed call.

        - CLOSED: increment the failure counter. If it reaches
          `failure_threshold`, trip to OPEN and record `_opened_at`.
        - HALF_OPEN: a probe failed — go back to OPEN immediately and record
          `_opened_at` (the probe window is over; start a new recovery wait).
        """
        # TODO(human): implement the CLOSED and HALF_OPEN branches described above.
        raise NotImplementedError("TODO(human): implement _handle_failure")

    # -- Scaffolded orchestrator -----------------------------------------------

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute *func* through the breaker. Fail fast when OPEN."""
        current_state = self.state  # triggers the OPEN -> HALF_OPEN check

        if current_state == CircuitState.OPEN:
            self.metrics.rejected_calls += 1
            remaining = self.recovery_timeout - (time.time() - self._opened_at)
            raise CircuitOpenError(self.name, max(remaining, 0.0))

        self.metrics.total_calls += 1
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            self.metrics.failed_calls += 1
            self._handle_failure()
            raise CircuitBreakerCallError(exc) from exc

        self.metrics.successful_calls += 1
        self._handle_success()
        return result


# -- HTTP helpers (scaffolded) ---------------------------------------------

SERVICE_URL = "http://localhost:8089"


async def call_unreliable_service(session: aiohttp.ClientSession) -> dict[str, Any]:
    async with session.get(
        f"{SERVICE_URL}/api/data", timeout=aiohttp.ClientTimeout(total=5)
    ) as resp:
        body = await resp.json()
        if resp.status >= 500:
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=body.get("error", "server error"),
            )
        return body


async def switch_mode(session: aiohttp.ClientSession, mode: str) -> None:
    async with session.post(f"{SERVICE_URL}/admin/mode/{mode}") as resp:
        result = await resp.json()
        print(f"\n  [service] {result['previous_mode']} -> {result['current_mode']}")


# -- Demo (scaffolded) ----------------------------------------------------


async def _drive_healthy(breaker: CircuitBreaker, session: aiohttp.ClientSession) -> None:
    print("\n--- Phase 1: healthy service (breaker stays CLOSED) ---")
    for i in range(5):
        try:
            await breaker.call(call_unreliable_service, session)
            print(f"  req {i + 1}: OK  state={breaker._state.value}")
        except CircuitBreakerCallError as e:
            print(f"  req {i + 1}: FAIL ({e})  state={breaker._state.value}")


async def _drive_failing(breaker: CircuitBreaker, session: aiohttp.ClientSession) -> None:
    print("\n--- Phase 2: service down (breaker trips to OPEN) ---")
    await switch_mode(session, "down")
    for i in range(8):
        try:
            await breaker.call(call_unreliable_service, session)
            print(f"  req {i + 1}: OK  state={breaker._state.value}")
        except CircuitOpenError as e:
            print(f"  req {i + 1}: REJECTED ({e.remaining_secs:.1f}s)  state={breaker._state.value}")
        except CircuitBreakerCallError:
            print(f"  req {i + 1}: FAIL  state={breaker._state.value}")


async def _wait_for_half_open(breaker: CircuitBreaker) -> None:
    print(f"\n--- Phase 3: waiting {breaker.recovery_timeout}s for recovery ---")
    await asyncio.sleep(breaker.recovery_timeout + 0.3)
    # Reading state triggers OPEN -> HALF_OPEN transition
    print(f"  state after wait: {breaker.state.value}")


async def _drive_recovery(breaker: CircuitBreaker, session: aiohttp.ClientSession) -> None:
    print("\n--- Phase 4: service recovers (breaker closes) ---")
    await switch_mode(session, "healthy")
    for i in range(breaker.half_open_max_calls):
        try:
            await breaker.call(call_unreliable_service, session)
            print(f"  probe {i + 1}: OK  state={breaker._state.value}")
        except Exception as e:
            print(f"  probe {i + 1}: FAIL ({e})  state={breaker._state.value}")


def _print_summary(breaker: CircuitBreaker, t0: float) -> None:
    m = breaker.metrics
    print("\n--- Summary ---")
    print(f"  total={m.total_calls}  success={m.successful_calls}  "
          f"failed={m.failed_calls}  rejected={m.rejected_calls}")
    for i, (ts, old, new) in enumerate(m.state_transitions, 1):
        print(f"  {i}. t={ts - t0:5.1f}s  {old.value} -> {new.value}")


async def demo_circuit_breaker() -> None:
    breaker = CircuitBreaker(
        name="demo", failure_threshold=3, recovery_timeout=5.0, half_open_max_calls=2,
    )
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        await switch_mode(session, "healthy")
        await _drive_healthy(breaker, session)
        await _drive_failing(breaker, session)
        await _wait_for_half_open(breaker)
        await _drive_recovery(breaker, session)
    _print_summary(breaker, t0)


def main() -> None:
    print("Exercise 01: Circuit Breaker")
    print("Prereq: _00_unreliable_service running on :8089\n")
    asyncio.run(demo_circuit_breaker())


if __name__ == "__main__":
    main()
