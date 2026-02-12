"""BankAccount aggregate --- the core domain model.

An aggregate is the consistency boundary in Event Sourcing. It:
1. Receives commands (open, deposit, withdraw)
2. Validates business rules against current state
3. Emits domain events (never mutates state directly from commands)
4. Rebuilds its state by replaying events (via `apply`)

The key insight: commands produce events, events produce state.
    command --> validate --> [Event] --> apply --> new state

This is where the most important learning happens.
"""

from __future__ import annotations

import uuid

from events import AccountOpened, DomainEvent, MoneyDeposited, MoneyWithdrawn
from event_store import EventStore


class BankAccount:
    """Bank account aggregate that enforces business invariants.

    State is NEVER set directly --- only through apply() after replaying events.
    Commands validate rules, then emit events. Events are the source of truth.
    """

    def __init__(self) -> None:
        self.account_id: str = ""
        self.owner_name: str = ""
        self.balance: float = 0.0
        self.is_open: bool = False
        self.version: int = 0
        self._pending_events: list[DomainEvent] = []

    # ------------------------------------------------------------------
    # Event application (state transitions)
    # ------------------------------------------------------------------

    def apply(self, event: DomainEvent) -> None:
        """Apply a single event to update the aggregate's state.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches the core of Event Sourcing: state is derived from
        # events, not stored directly. The apply() method is a pure function (no
        # side effects, no I/O) that transitions state. Keeping it separate from
        # command validation enables temporal queries and event replay after rule
        # changes. This is the foundation of event-sourced domain modeling.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        This is the ONLY place where aggregate state changes. Each event type
        maps to a specific state transition:

        - AccountOpened:  set account_id, owner_name, balance (from initial_balance),
                          mark is_open = True
        - MoneyDeposited: add amount to balance
        - MoneyWithdrawn: subtract amount from balance

        Update self.version to event.version after applying.

        Hint: Use isinstance() or match the event.event_type string to dispatch.
              Keep it simple --- no validation here, just state mutation.

        Why no validation in apply()? Because apply() replays historical events
        that already passed validation. If you add validation here, replaying
        old events could fail when rules change.
        """
        raise NotImplementedError("TODO(human): implement apply()")

    # ------------------------------------------------------------------
    # Commands (business operations that emit events)
    # ------------------------------------------------------------------

    def open_account(self, owner_name: str, initial_balance: float = 0.0) -> None:
        """Command: open a new bank account.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches command validation and event emission. Commands
        # enforce business rules (initial balance >= 0, no double-open) and emit
        # events describing what happened. The command doesn't mutate state directly;
        # it emits an event, which apply() then processes. This separation is key
        # to maintaining an immutable event log that can be replayed later.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        Steps:
            1. Validate: initial_balance must be >= 0
               (raise ValueError if not)
            2. Validate: account must not already be open
               (raise ValueError if self.is_open)
            3. Generate a new account_id: str(uuid.uuid4())
            4. Create an AccountOpened event with:
               - aggregate_id = the new account_id
               - owner_name = owner_name
               - initial_balance = initial_balance
            5. Call self._emit(event) to apply + queue the event

        Hint: self._emit() both applies the event AND adds it to pending events.
        """
        raise NotImplementedError("TODO(human): implement open_account()")

    def deposit(self, amount: float, description: str = "") -> None:
        """Command: deposit money into the account.

        TODO(human): Implement this method.

        Steps:
            1. Validate: account must be open (raise ValueError if not)
            2. Validate: amount must be > 0 (raise ValueError if not)
            3. Create a MoneyDeposited event with:
               - aggregate_id = self.account_id
               - amount = amount
               - description = description
            4. Call self._emit(event)
        """
        raise NotImplementedError("TODO(human): implement deposit()")

    def withdraw(self, amount: float, description: str = "") -> None:
        """Command: withdraw money from the account.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches invariant enforcement in Event Sourcing. The withdraw
        # command validates against current state (self.balance, derived from replayed
        # events). If validation passes, it emits a MoneyWithdrawn event. This pattern—
        # validate against replayed state, emit event if valid—is how event-sourced
        # aggregates enforce business rules without locking the entire event stream.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        Steps:
            1. Validate: account must be open (raise ValueError if not)
            2. Validate: amount must be > 0 (raise ValueError if not)
            3. Validate: sufficient funds (self.balance >= amount)
               (raise ValueError("Insufficient funds") if not)
            4. Create a MoneyWithdrawn event with:
               - aggregate_id = self.account_id
               - amount = amount
               - description = description
            5. Call self._emit(event)

        Key insight: the validation checks CURRENT state (self.balance),
        which was built by replaying past events. This is how Event Sourcing
        enforces invariants.
        """
        raise NotImplementedError("TODO(human): implement withdraw()")

    # ------------------------------------------------------------------
    # Rehydration (rebuild state from event history)
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, account_id: str, event_store: EventStore) -> BankAccount:
        """Rehydrate a BankAccount aggregate from its event history.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches aggregate rehydration (event replay). Loading an
        # aggregate means replaying all its events through apply() to rebuild state.
        # This is "sourcing" in Event Sourcing: the event log is the source of truth,
        # not a database row. Production systems optimize with snapshots (replay from
        # snapshot + recent events), but the principle is the same.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this class method.

        Steps:
            1. Create a new BankAccount instance: account = cls()
            2. Load events from the event store:
               events = event_store.load_events(account_id)
            3. Replay each event: call account.apply(event) for each
            4. Return the account

        If no events exist, the account will be in its default state
        (is_open=False), which commands can check to prevent invalid operations.

        This is the "sourcing" in Event Sourcing --- state is derived from events,
        not stored directly.
        """
        raise NotImplementedError("TODO(human): implement load()")

    # ------------------------------------------------------------------
    # Internal helpers (fully implemented)
    # ------------------------------------------------------------------

    def _emit(self, event: DomainEvent) -> None:
        """Apply an event and add it to the pending events list.

        Pending events will be persisted to the event store and published
        to Redpanda by the command handler.
        """
        self.apply(event)
        self._pending_events.append(event)

    def collect_pending_events(self) -> list[DomainEvent]:
        """Return and clear pending events (called by command handler after persist)."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events
