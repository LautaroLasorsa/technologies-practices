"""End-to-end test script for the Bank CQRS system.

Sends commands to the command API and queries the query API to verify
the full pipeline: command -> event store -> Redpanda -> projection -> query.

Usage:
    1. Start Redpanda:       docker compose up -d
    2. Start command API:    uv run uvicorn command_api:app --port 8001
    3. Start query API:      uv run uvicorn query_api:app --port 8002
    4. Run this script:      uv run python test_e2e.py
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import json
import sys


COMMAND_URL = "http://localhost:8001"
QUERY_URL = "http://localhost:8002"


def post(url: str, data: dict) -> dict:
    """Send a POST request with JSON body."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        print(f"  ERROR {exc.code}: {detail}")
        sys.exit(1)


def get(url: str) -> dict | list:
    """Send a GET request."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        print(f"  ERROR {exc.code}: {detail}")
        sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("CQRS & Event Sourcing - End-to-End Test")
    print("=" * 60)

    # --- 1. Open accounts ---
    print("\n--- Step 1: Open accounts ---")
    resp = post(f"{COMMAND_URL}/commands/open-account", {
        "owner_name": "Alice",
        "initial_balance": 1000.0,
    })
    alice_id = resp["account_id"]
    print(f"  Alice's account: {alice_id}")

    resp = post(f"{COMMAND_URL}/commands/open-account", {
        "owner_name": "Bob",
        "initial_balance": 500.0,
    })
    bob_id = resp["account_id"]
    print(f"  Bob's account:   {bob_id}")

    # --- 2. Make deposits ---
    print("\n--- Step 2: Deposits ---")
    post(f"{COMMAND_URL}/commands/deposit", {
        "account_id": alice_id,
        "amount": 250.0,
        "description": "Salary",
    })
    print("  Alice deposited $250 (Salary)")

    post(f"{COMMAND_URL}/commands/deposit", {
        "account_id": bob_id,
        "amount": 100.0,
        "description": "Freelance payment",
    })
    print("  Bob deposited $100 (Freelance payment)")

    # --- 3. Make withdrawals ---
    print("\n--- Step 3: Withdrawals ---")
    post(f"{COMMAND_URL}/commands/withdraw", {
        "account_id": alice_id,
        "amount": 75.0,
        "description": "Groceries",
    })
    print("  Alice withdrew $75 (Groceries)")

    # --- 4. Wait for projections to catch up (eventual consistency!) ---
    print("\n--- Step 4: Waiting for projections (eventual consistency) ---")
    time.sleep(2)
    print("  Waited 2 seconds for query side to process events")

    # --- 5. Query balances ---
    print("\n--- Step 5: Query account balances ---")
    alice = get(f"{QUERY_URL}/accounts/{alice_id}")
    print(f"  Alice: balance=${alice['balance']:.2f} (expected: $1175.00)")

    bob = get(f"{QUERY_URL}/accounts/{bob_id}")
    print(f"  Bob:   balance=${bob['balance']:.2f} (expected: $600.00)")

    # --- 6. Query transaction history ---
    print("\n--- Step 6: Query transaction history ---")
    txns = get(f"{QUERY_URL}/accounts/{alice_id}/transactions")
    print(f"  Alice's transactions ({len(txns)} total):")
    for tx in txns:
        sign = "+" if tx["event_type"] != "MoneyWithdrawn" else "-"
        print(
            f"    {tx['event_type']:20s}  {sign}${tx['amount']:.2f}"
            f"  -> balance=${tx['balance']:.2f}"
            f"  ({tx['description']})"
        )

    # --- 7. List all accounts ---
    print("\n--- Step 7: List all accounts ---")
    accounts = get(f"{QUERY_URL}/accounts")
    for acct in accounts:
        print(f"  {acct['owner_name']:10s}  ${acct['balance']:.2f}")

    # --- 8. Verify expected balances ---
    print("\n--- Verification ---")
    alice_ok = abs(alice["balance"] - 1175.0) < 0.01
    bob_ok = abs(bob["balance"] - 600.0) < 0.01
    if alice_ok and bob_ok:
        print("  All balances correct!")
    else:
        print("  MISMATCH detected --- check event processing")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
