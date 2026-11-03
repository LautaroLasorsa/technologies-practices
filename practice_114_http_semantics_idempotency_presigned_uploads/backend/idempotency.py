"""In-memory Idempotency Store and Protocol Evaluator.

Implements the IETF Idempotency-Key specification and Stripe-style
deduplication protocol.

Key Protocol Concepts:
1. Fingerprinting: Canonical SHA-256 hash of the request payload.
2. In-Flight Locking: Prevents concurrent requests with the same key.
3. Response Replay: Cached HTTP response replay on retry.
4. Tampering Detection: Rejection with 422 if key is reused with altered payload.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

IdempotencyStatus = Literal["started", "completed", "failed"]
EvaluationResult = Literal["NEW", "IN_FLIGHT", "REPLAY", "PAYLOAD_MISMATCH"]


@dataclass
class IdempotencyRecord:
    key: str
    status: IdempotencyStatus
    payload_hash: str
    status_code: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# Global in-memory storage guarded by a thread lock
_store: dict[str, IdempotencyRecord] = {}
_lock = threading.Lock()


def compute_payload_hash(payload: Any) -> str:
    """Compute a deterministic, canonical SHA-256 fingerprint of the request payload.

    Dicts are sorted by keys and formatted without whitespace separators.
    """
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        # Canonical JSON serialization
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        raw = canonical.encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def evaluate_idempotency(
    key: str,
    payload_hash: str,
) -> tuple[EvaluationResult, Optional[IdempotencyRecord]]:
    """Evaluate incoming Idempotency-Key against the active store.

    ========================================================================
    TODO(human): Exercise 1 (Backend) - Implement Idempotency-Key Evaluation
    ========================================================================
    Context:
    When a client issues a POST mutation with an `Idempotency-Key` header,
    the server must atomically inspect its cache to determine if this is:
    1. A brand new mutation ("NEW").
    2. A duplicate request arriving while the original is still processing ("IN_FLIGHT").
    3. A legitimate retry of a previously completed mutation whose response should
       be replayed without re-executing business logic ("REPLAY").
    4. An illegal key reuse where the client sent a different payload under
       a previously used key ("PAYLOAD_MISMATCH").

    Instructions:
    Under `with _lock:`:
    - If `key not in _store`:
        Create a new `IdempotencyRecord(key=key, status="started", payload_hash=payload_hash)`.
        Save it into `_store[key]` and return `("NEW", None)`.
    - Retrieve `record = _store[key]`.
    - If `record.status == "started"`:
        Return `("IN_FLIGHT", record)`.
    - If `record.status == "completed"`:
        Check if `record.payload_hash == payload_hash`:
          - If they match: return `("REPLAY", record)`
          - If they differ: return `("PAYLOAD_MISMATCH", record)`
    - If `record.status == "failed"`:
        Reset `record.status = "started"`, update `record.payload_hash = payload_hash`,
        and return `("NEW", None)`.
    ========================================================================
    """
    with _lock:
        # --- IMPLEMENTATION GOES HERE ---
        raise NotImplementedError(
            "Exercise 1: Implement evaluate_idempotency in backend/idempotency.py"
        )


def store_idempotency_response(
    key: str,
    status_code: int,
    body: Any,
    headers: Optional[dict[str, str]] = None,
) -> None:
    """Store the final completed HTTP response for an idempotency key."""
    with _lock:
        record = _store.get(key)
        if record:
            record.status = "completed"
            record.status_code = status_code
            record.body = body
            record.headers = headers or {}


def mark_idempotency_failed(key: str) -> None:
    """Mark an idempotency key as failed so future retries may execute."""
    with _lock:
        record = _store.get(key)
        if record:
            record.status = "failed"


def get_all_records() -> list[dict[str, Any]]:
    """Return all idempotency records for telemetry / debug inspection."""
    with _lock:
        return [
            {
                "key": r.key,
                "status": r.status,
                "payload_hash": r.payload_hash[:12] + "...",
                "status_code": r.status_code,
                "created_at": r.created_at,
            }
            for r in _store.values()
        ]


def clear_idempotency_store() -> None:
    """Clear all stored idempotency keys (used for chaos resets)."""
    with _lock:
        _store.clear()
