"""Manual test script for the tracing demo.

Sends requests to the Order API and prints results.
Run AFTER both services are up:
    Terminal 1: uv run uvicorn order_api:app --port 8000 --reload
    Terminal 2: uv run uvicorn payment_service:app --port 8001 --reload
    Terminal 3: uv run python test_tracing.py

Then open Jaeger UI at http://localhost:16686 to inspect traces.
"""

import httpx

ORDER_API = "http://localhost:8000"


def test_health() -> None:
    """Phase 2: Verify basic tracing with a health check."""
    print("=== Health Check ===")
    resp = httpx.get(f"{ORDER_API}/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Body:   {resp.json()}")
    print("  -> Check Jaeger for a 'health-check' span under 'order-api'\n")


def test_valid_order() -> None:
    """Phase 3-5: Create a valid order — should produce a multi-span trace."""
    print("=== Valid Order ===")
    order = {
        "customer_id": "CUST-42",
        "item": "Mechanical Keyboard",
        "quantity": 2,
        "amount": 149.99,
    }
    resp = httpx.post(f"{ORDER_API}/orders", json=order, timeout=10.0)
    print(f"  Status: {resp.status_code}")
    print(f"  Body:   {resp.json()}")
    print("  -> Check Jaeger: trace should span order-api AND payment-service\n")


def test_invalid_order() -> None:
    """Phase 4: Create an invalid order — should produce an error span."""
    print("=== Invalid Order (negative amount) ===")
    order = {
        "customer_id": "CUST-99",
        "item": "Ghost Item",
        "quantity": 1,
        "amount": -50.0,
    }
    resp = httpx.post(f"{ORDER_API}/orders", json=order, timeout=10.0)
    print(f"  Status: {resp.status_code}")
    print(f"  Body:   {resp.json()}")
    print("  -> Check Jaeger: 'validate-order' span should show ERROR status\n")


def test_fraud_order() -> None:
    """Phase 5: Create a high-value order — triggers fraud check failure."""
    print("=== Fraud Order (amount > 10,000) ===")
    order = {
        "customer_id": "CUST-SHADY",
        "item": "Luxury Watch",
        "quantity": 1,
        "amount": 25_000.0,
    }
    resp = httpx.post(f"{ORDER_API}/orders", json=order, timeout=10.0)
    print(f"  Status: {resp.status_code}")
    print(f"  Body:   {resp.json()}")
    print("  -> Check Jaeger: payment-service 'check-fraud' span should show ERROR\n")


def main() -> None:
    print("Sending test requests to Order API...\n")
    test_health()
    test_valid_order()
    test_invalid_order()
    test_fraud_order()
    print("Done! Open http://localhost:16686 to explore traces in Jaeger.")


if __name__ == "__main__":
    main()
