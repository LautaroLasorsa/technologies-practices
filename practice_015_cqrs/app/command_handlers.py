"""Command handlers --- the write-side application layer.

Command handlers orchestrate the flow:
    1. Load the aggregate (rehydrate from event store)
    2. Execute the command on the aggregate (validates + emits events)
    3. Persist new events to the event store (with optimistic concurrency)
    4. Publish events to Redpanda (for the query side)

Command handlers are the "glue" between HTTP endpoints and the domain model.
They do NOT contain business logic --- that lives in the aggregate.
"""

from __future__ import annotations

import logging

from aggregate import BankAccount
from event_publisher import EventPublisher
from event_store import EventStore

logger = logging.getLogger(__name__)


class CommandHandlers:
    """Coordinates commands between aggregate, event store, and publisher."""

    def __init__(self, event_store: EventStore, publisher: EventPublisher) -> None:
        self._store = event_store
        self._publisher = publisher

    async def handle_open_account(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
    ) -> str:
        """Handle the OpenAccount command.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches the command handler pattern: orchestrating aggregate,
        # event store, and event publisher. The handler contains NO business logic—
        # it just coordinates. Persisting to the event store BEFORE publishing ensures
        # the event store remains the source of truth. If publishing fails, a separate
        # outbox processor can retry, but events are never lost.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        Steps:
            1. Create a new BankAccount instance: account = BankAccount()
            2. Call account.open_account(owner_name, initial_balance)
            3. Get pending events: events = account.collect_pending_events()
            4. Persist to event store:
               self._store.append(events, expected_version=0)
               (expected_version=0 because this is a new aggregate)
            5. Publish events to Redpanda:
               await self._publisher.publish(events)
            6. Return account.account_id

        Why persist BEFORE publishing? If publishing fails, the events are
        still in the event store (source of truth). A separate process can
        re-publish them later. If you publish first and persist fails, you'd
        have phantom events in Redpanda with no backing store.

        Hint: This is ~6 lines of straightforward orchestration code.
        """
        raise NotImplementedError("TODO(human): implement handle_open_account()")

    async def handle_deposit(
        self,
        account_id: str,
        amount: float,
        description: str = "",
    ) -> None:
        """Handle the Deposit command.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches optimistic concurrency control in Event Sourcing.
        # The expected_version check prevents lost updates: if two commands try to
        # modify the same aggregate concurrently, one will fail with a version mismatch.
        # This is how event-sourced systems enforce consistency without pessimistic
        # locking (no row locks, no distributed transactions).
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        Steps:
            1. Load the aggregate from the event store:
               account = BankAccount.load(account_id, self._store)
            2. Save the current version for concurrency check:
               expected_version = account.version
            3. Execute the command: account.deposit(amount, description)
            4. Collect pending events
            5. Persist to event store with expected_version
            6. Publish events to Redpanda

        Hint: Steps 4-6 are the same persist-then-publish pattern as open_account.
              Notice how the handler doesn't know about balances or validation ---
              all business rules live in the aggregate.
        """
        raise NotImplementedError("TODO(human): implement handle_deposit()")

    async def handle_withdraw(
        self,
        account_id: str,
        amount: float,
        description: str = "",
    ) -> None:
        """Handle the Withdraw command.

        TODO(human): Implement this method.

        Steps:
            1. Load the aggregate from the event store
            2. Save expected_version = account.version
            3. Execute: account.withdraw(amount, description)
            4. Collect, persist, publish (same pattern)

        Hint: This is structurally identical to handle_deposit().
              The aggregate's withdraw() enforces the "no overdraft" rule.
        """
        raise NotImplementedError("TODO(human): implement handle_withdraw()")
