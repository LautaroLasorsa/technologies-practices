"""Domain models for the wallet service."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAW = "withdraw"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"


@dataclass
class Transaction:
    type: TransactionType
    amount: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    id: str = field(default_factory=lambda: uuid4().hex[:8])


@dataclass
class Wallet:
    owner: str
    balance: float = 0.0
    transactions: list[Transaction] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:8])
