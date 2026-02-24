"""
Test client for the SAGA pattern practice.

Sends orders to the Order Service and polls for saga completion.
Uses only stdlib (no extra dependencies needed).

Usage:
    python test_client.py                  # Run all tests (happy + failure path)
    python test_client.py happy            # Only happy path (< $500)
    python test_client.py fail             # Only failure path (> $500)
    python test_client.py list             # List all orders
    python test_client.py status <order_id> # Check specific order status
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8001"
POLL_INTERVAL = 1.0
POLL_TIMEOUT = 15.0


def post_order(customer_id: str, item: str, quantity: int, price: float) -> dict:
    """Create a new order via the Order Service API."""
    payload = json.dumps({
        "customer_id": customer_id,
        "item": item,
        "quantity": quantity,
        "price": price,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/orders",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_order(order_id: str) -> dict:
    """Fetch a single order by ID."""
    with urllib.request.urlopen(f"{BASE_URL}/orders/{order_id}") as resp:
        return json.loads(resp.read())


def list_orders() -> list[dict]:
    """List all orders."""
    with urllib.request.urlopen(f"{BASE_URL}/orders") as resp:
        return json.loads(resp.read())


def poll_until_terminal(order_id: str) -> dict:
    """Poll an order until its status is no longer 'pending'."""
    elapsed = 0.0
    while elapsed < POLL_TIMEOUT:
        order = get_order(order_id)
        if order["status"] != "pending":
            return order
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    return get_order(order_id)


def print_order(order: dict, label: str = "") -> None:
    """Pretty-print an order."""
    if label:
        print(f"\n--- {label} ---")
    print(f"  order_id:    {order['order_id']}")
    print(f"  saga_id:     {order['saga_id']}")
    print(f"  customer_id: {order['customer_id']}")
    print(f"  item:        {order['item']}")
    print(f"  quantity:    {order['quantity']}")
    print(f"  price:       ${order['price']}")
    total = order["price"] * order["quantity"]
    print(f"  total:       ${total}")
    print(f"  status:      {order['status']}")


def test_happy_path() -> None:
    """Submit an order under $500 -- should complete successfully."""
    print("\n======== HAPPY PATH (total=$200, under $500 limit) ========")
    order = post_order("cust-1", "laptop", 1, 200.0)
    print_order(order, "Order created")

    print("\nPolling for saga completion...")
    final = poll_until_terminal(order["order_id"])
    print_order(final, "Final status")

    if final["status"] == "completed":
        print("\n>> PASS: Saga completed successfully")
    else:
        print(f"\n>> UNEXPECTED: status is '{final['status']}' (expected 'completed')")


def test_failure_path() -> None:
    """Submit an order over $500 -- should trigger compensation and fail."""
    print("\n======== FAILURE PATH (total=$600, over $500 limit) ========")
    order = post_order("cust-2", "laptop", 1, 600.0)
    print_order(order, "Order created")

    print("\nPolling for saga completion (expecting failure + compensation)...")
    final = poll_until_terminal(order["order_id"])
    print_order(final, "Final status")

    if final["status"] == "failed":
        print("\n>> PASS: Saga failed with compensation (as expected)")
    else:
        print(f"\n>> UNEXPECTED: status is '{final['status']}' (expected 'failed')")


def show_all_orders() -> None:
    """List all orders in the system."""
    orders = list_orders()
    if not orders:
        print("\nNo orders found.")
        return
    print(f"\n======== ALL ORDERS ({len(orders)}) ========")
    for order in orders:
        print_order(order, f"Order {order['order_id'][:8]}...")


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args else "all"

    try:
        if command == "all":
            test_happy_path()
            test_failure_path()
            print("\n======== SUMMARY ========")
            show_all_orders()
        elif command == "happy":
            test_happy_path()
        elif command == "fail":
            test_failure_path()
        elif command == "list":
            show_all_orders()
        elif command == "status" and len(args) > 1:
            order = get_order(args[1])
            print_order(order, "Order status")
        else:
            print(__doc__)
    except urllib.error.URLError as e:
        print(f"\nERROR: Could not connect to Order Service at {BASE_URL}")
        print(f"       Make sure all services are running (docker compose up)")
        print(f"       Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
