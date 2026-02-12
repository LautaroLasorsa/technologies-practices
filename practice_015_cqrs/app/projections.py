"""Read-model projections --- the query-side data layer.

Projections consume domain events and build queryable views (read models).
Each projection is a "materialised view" optimized for a specific query pattern.

Two projections in this practice:
1. AccountBalance --- current balance per account (for GET /accounts/{id})
2. TransactionLog --- chronological list of deposits/withdrawals
                      (for GET /accounts/{id}/transactions)

The projection store uses a SEPARATE SQLite database from the event store,
reinforcing the CQRS separation: write-side and read-side have independent
data stores.
"""

from __future__ import annotations

import logging
import sqlite3

from events import AccountOpened, DomainEvent, MoneyDeposited, MoneyWithdrawn

logger = logging.getLogger(__name__)


class ProjectionStore:
    """SQLite-backed read model for account queries."""

    def __init__(self, db_path: str = "query_store.db") -> None:
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create projection tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_balances (
                    account_id TEXT PRIMARY KEY,
                    owner_name TEXT NOT NULL,
                    balance    REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transaction_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id  TEXT    NOT NULL,
                    event_type  TEXT    NOT NULL,
                    amount      REAL    NOT NULL,
                    balance     REAL    NOT NULL,
                    description TEXT    NOT NULL DEFAULT '',
                    occurred_at TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_txlog_account
                ON transaction_log(account_id, occurred_at)
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Event projection handlers
    # ------------------------------------------------------------------

    def project_event(self, event: DomainEvent) -> None:
        """Route an event to the correct projection handler.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches projection event dispatching. Projections consume
        # events to build read models optimized for queries. Unlike aggregates
        # (which enforce invariants), projections are denormalized views. Multiple
        # projections can consume the same events to build different read models
        # (balance view, transaction log, analytics dashboard).
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this dispatcher.

        Steps:
            1. Check the event type using isinstance():
               - AccountOpened  -> call self._project_account_opened(event)
               - MoneyDeposited -> call self._project_money_deposited(event)
               - MoneyWithdrawn -> call self._project_money_withdrawn(event)
            2. Log a warning for unknown event types (don't crash --- new event
               types may be added later, and old projections should be resilient).

        Hint: A simple if/elif/else chain works. In production you might use
              a registry pattern, but for learning, explicit dispatch is clearer.
        """
        raise NotImplementedError("TODO(human): implement project_event()")

    def _project_account_opened(self, event: AccountOpened) -> None:
        """Project an AccountOpened event into the read model.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches building a read model from events. The projection
        # transforms domain events into a queryable schema (account_balances table).
        # This is the "Query" side of CQRS: a denormalized view optimized for fast
        # reads. The write side (event store) remains normalized and immutable.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        Steps:
            1. INSERT a new row into account_balances:
               account_id = event.aggregate_id
               owner_name = event.owner_name
               balance = event.initial_balance
               updated_at = event.occurred_at
            2. If initial_balance > 0, also INSERT a row into transaction_log:
               event_type = "AccountOpened"
               amount = event.initial_balance
               balance = event.initial_balance  (balance after this event)
               description = "Initial deposit"
               occurred_at = event.occurred_at

        Hint: Use self._connect() as conn, then conn.execute(sql, params).
              The context manager auto-commits on exit.
        """
        raise NotImplementedError("TODO(human): implement _project_account_opened()")

    def _project_money_deposited(self, event: MoneyDeposited) -> None:
        """Project a MoneyDeposited event into the read model.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches incremental projection updates. Each event updates
        # the read model incrementally (UPDATE balance, INSERT transaction log entry).
        # This is more efficient than rebuilding the entire read model on every event.
        # Production systems handle projection failures by storing consumer offsets
        # and restarting from the last successfully processed event.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement this method.

        Steps:
            1. UPDATE account_balances:
               SET balance = balance + event.amount,
                   updated_at = event.occurred_at
               WHERE account_id = event.aggregate_id
            2. Read the NEW balance from account_balances
               (SELECT balance WHERE account_id = ...)
            3. INSERT into transaction_log:
               event_type = "MoneyDeposited"
               amount = event.amount
               balance = <the new balance you just read>
               description = event.description
               occurred_at = event.occurred_at

        Hint: Do all three queries inside one `with self._connect() as conn:`
              block so they share the same transaction.
        """
        raise NotImplementedError(
            "TODO(human): implement _project_money_deposited()"
        )

    def _project_money_withdrawn(self, event: MoneyWithdrawn) -> None:
        """Project a MoneyWithdrawn event into the read model.

        TODO(human): Implement this method.

        Steps:
            1. UPDATE account_balances:
               SET balance = balance - event.amount,
                   updated_at = event.occurred_at
               WHERE account_id = event.aggregate_id
            2. Read the NEW balance
            3. INSERT into transaction_log with the new balance

        Hint: This is structurally identical to _project_money_deposited,
              but subtracts instead of adds.
        """
        raise NotImplementedError(
            "TODO(human): implement _project_money_withdrawn()"
        )

    # ------------------------------------------------------------------
    # Query methods (read model access)
    # ------------------------------------------------------------------

    def get_account(self, account_id: str) -> dict | None:
        """Get current account balance and info."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM account_balances WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_all_accounts(self) -> list[dict]:
        """Get all account balances."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM account_balances ORDER BY owner_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_transactions(
        self, account_id: str, limit: int = 50
    ) -> list[dict]:
        """Get transaction history for an account (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, amount, balance, description, occurred_at
                FROM transaction_log
                WHERE account_id = ?
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                (account_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
