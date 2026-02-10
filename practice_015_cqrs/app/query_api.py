"""Query-side FastAPI application (read model).

This service exposes HTTP endpoints for reads:
    GET /accounts           --- list all accounts with balances
    GET /accounts/{id}      --- single account balance
    GET /accounts/{id}/transactions --- transaction history

It also runs an event consumer in the background that listens to Redpanda
and updates the read-model projections.

This is the entry point for the read side. Run with:
    uvicorn query_api:app --port 8002
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from event_consumer import EventConsumer
from projections import ProjectionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QRY] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

projection_store = ProjectionStore(db_path="query_store.db")
consumer = EventConsumer(projection_store)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start/stop the event consumer with the application."""
    await consumer.start()
    logger.info("Query API ready (projections consumer running)")
    yield
    await consumer.stop()
    logger.info("Query API shutting down")


app = FastAPI(
    title="Bank CQRS - Query API",
    description="Read side: queries against projected read models",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/accounts")
async def list_accounts() -> list[dict]:
    """List all accounts with current balances.

    TODO(human): Implement this endpoint.

    Steps:
        1. Call projection_store.get_all_accounts()
        2. Return the result directly (it's already a list of dicts)

    Hint: One line of code.
    """
    raise NotImplementedError("TODO(human): implement list_accounts()")


@app.get("/accounts/{account_id}")
async def get_account(account_id: str) -> dict:
    """Get a single account's current balance and info.

    TODO(human): Implement this endpoint.

    Steps:
        1. Call projection_store.get_account(account_id)
        2. If None, raise HTTPException(status_code=404, detail="Account not found")
        3. Otherwise, return the dict

    Hint: Three lines of code.
    """
    raise NotImplementedError("TODO(human): implement get_account()")


@app.get("/accounts/{account_id}/transactions")
async def get_transactions(account_id: str, limit: int = 50) -> list[dict]:
    """Get transaction history for an account (newest first).

    TODO(human): Implement this endpoint.

    Steps:
        1. First verify the account exists:
           call projection_store.get_account(account_id)
           and raise 404 if None
        2. Call projection_store.get_transactions(account_id, limit)
        3. Return the result

    Hint: Four lines of code.
    """
    raise NotImplementedError("TODO(human): implement get_transactions()")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "query-api"}
