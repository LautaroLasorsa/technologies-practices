"""Command-side FastAPI application (write model).

This service exposes HTTP endpoints that accept commands:
    POST /commands/open-account
    POST /commands/deposit
    POST /commands/withdraw

Each endpoint validates the request shape (via Pydantic), delegates to
the appropriate command handler, and returns a result.

This is the entry point for the write side. Run with:
    uvicorn command_api:app --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from command_handlers import CommandHandlers
from event_publisher import EventPublisher
from event_store import EventStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CMD] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/response models (HTTP contract, not domain)
# ---------------------------------------------------------------------------

class OpenAccountRequest(BaseModel):
    owner_name: str = Field(..., min_length=1, examples=["Alice"])
    initial_balance: float = Field(default=0.0, ge=0)


class OpenAccountResponse(BaseModel):
    account_id: str
    message: str = "Account opened"


class DepositRequest(BaseModel):
    account_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0, examples=[100.0])
    description: str = ""


class WithdrawRequest(BaseModel):
    account_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0, examples=[50.0])
    description: str = ""


class CommandResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

event_store = EventStore(db_path="command_store.db")
publisher = EventPublisher()
handlers = CommandHandlers(event_store, publisher)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start/stop the Kafka producer with the application."""
    await publisher.start()
    logger.info("Command API ready")
    yield
    await publisher.stop()
    logger.info("Command API shutting down")


app = FastAPI(
    title="Bank CQRS - Command API",
    description="Write side: accepts commands that produce domain events",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/commands/open-account", response_model=OpenAccountResponse)
async def open_account(req: OpenAccountRequest) -> OpenAccountResponse:
    """Open a new bank account."""
    try:
        account_id = await handlers.handle_open_account(
            owner_name=req.owner_name,
            initial_balance=req.initial_balance,
        )
        return OpenAccountResponse(account_id=account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/commands/deposit", response_model=CommandResponse)
async def deposit(req: DepositRequest) -> CommandResponse:
    """Deposit money into an account."""
    try:
        await handlers.handle_deposit(
            account_id=req.account_id,
            amount=req.amount,
            description=req.description,
        )
        return CommandResponse(success=True, message="Deposit successful")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/commands/withdraw", response_model=CommandResponse)
async def withdraw(req: WithdrawRequest) -> CommandResponse:
    """Withdraw money from an account."""
    try:
        await handlers.handle_withdraw(
            account_id=req.account_id,
            amount=req.amount,
            description=req.description,
        )
        return CommandResponse(success=True, message="Withdrawal successful")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "command-api"}
