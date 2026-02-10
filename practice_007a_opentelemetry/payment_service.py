"""Payment Service — second service in the distributed tracing demo.

Receives payment requests from the Order API and simulates payment processing.
Demonstrates extracting trace context from incoming HTTP headers so that spans
from this service appear as children of the Order API's spans in the same trace.

Run (in a separate terminal, after docker compose up -d):
    uv run uvicorn payment_service:app --port 8001 --reload

Endpoints:
    POST /pay — process a payment request

Docs: https://opentelemetry.io/docs/languages/python/propagation/#extracting-context
"""

import time
import uuid

from fastapi import FastAPI, Request

from models import PaymentRequest, PaymentResponse
from propagation import extract_context
from tracing import get_tracer, init_tracer

# ── OpenTelemetry imports you'll need ────────────────────────────────
from opentelemetry import trace
from opentelemetry.trace import StatusCode

# ── Initialize tracing for this service ──────────────────────────────
init_tracer(service_name="payment-service")
tracer = get_tracer("payment_service")

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="Payment Service")


# ── Payment endpoint (Phase 5) ───────────────────────────────────────


@app.post("/pay")
async def process_payment(payment: PaymentRequest, request: Request) -> PaymentResponse:
    """Process a payment request.

    TODO(human): Implement context extraction and tracing in this function.

    This is where the "distributed" part of distributed tracing happens.
    The Order API injected a `traceparent` header into its HTTP request.
    You need to extract that context so your spans join the same trace.

    Steps:
      1. Extract the trace context from the incoming request headers:
             ctx = extract_context(dict(request.headers))
         `dict(request.headers)` converts Starlette's Headers object to a
         plain dict that the propagator can read.

      2. Create a SERVER span using the extracted context:
             with tracer.start_as_current_span(
                 "process-payment",
                 context=ctx,
                 kind=trace.SpanKind.SERVER,
             ) as span:
         Passing `context=ctx` makes this span a CHILD of the Order API's
         "request-payment" CLIENT span. Without it, this would start a new
         unrelated trace.

      3. Add attributes to the span:
             span.set_attribute("payment.order_id", payment.order_id)
             span.set_attribute("payment.customer_id", payment.customer_id)
             span.set_attribute("payment.amount", payment.amount)

      4. Call the processing helpers (already implemented below):
             fraud_ok = check_fraud(payment)
             if not fraud_ok:
                 span.set_status(StatusCode.ERROR, "Fraud check failed")
                 return PaymentResponse(order_id=..., status="declined", transaction_id="")
             result = charge_payment(payment)
             return result

    Hint — full skeleton:
        ctx = extract_context(dict(request.headers))
        with tracer.start_as_current_span("process-payment", context=ctx, kind=trace.SpanKind.SERVER) as span:
            span.set_attribute("payment.order_id", payment.order_id)
            span.set_attribute("payment.customer_id", payment.customer_id)
            span.set_attribute("payment.amount", payment.amount)
            fraud_ok = check_fraud(payment)
            if not fraud_ok:
                span.set_status(StatusCode.ERROR, "Fraud check failed")
                return PaymentResponse(order_id=payment.order_id, status="declined", transaction_id="")
            return charge_payment(payment)

    Docs: https://opentelemetry.io/docs/languages/python/propagation/#extracting-context
    """
    # TODO(human): Extract context and wrap in a SERVER span.
    # For now, this works without tracing:
    fraud_ok = check_fraud(payment)
    if not fraud_ok:
        return PaymentResponse(
            order_id=payment.order_id,
            status="declined",
            transaction_id="",
        )
    return charge_payment(payment)


# ── Internal helpers (already traced) ────────────────────────────────


def check_fraud(payment: PaymentRequest) -> bool:
    """Simulate a fraud check.

    Amounts over 10,000 are flagged as fraudulent (for demo purposes).
    This function already creates its own child span — no TODO here.
    """
    with tracer.start_as_current_span("check-fraud") as span:
        span.set_attribute("payment.amount", payment.amount)
        time.sleep(0.03)  # simulate processing

        is_fraudulent = payment.amount > 10_000
        span.set_attribute("fraud.detected", is_fraudulent)

        if is_fraudulent:
            span.add_event("fraud-detected", {"reason": "amount exceeds threshold"})
            span.set_status(StatusCode.ERROR, "Fraud detected")
            return False

        span.add_event("fraud-check-passed")
        return True


def charge_payment(payment: PaymentRequest) -> PaymentResponse:
    """Simulate charging the payment.

    This function already creates its own child span — no TODO here.
    """
    with tracer.start_as_current_span("charge-payment") as span:
        transaction_id = f"TXN-{uuid.uuid4().hex[:8]}"
        span.set_attribute("payment.transaction_id", transaction_id)
        span.set_attribute("payment.amount", payment.amount)

        time.sleep(0.08)  # simulate gateway latency
        span.add_event("payment-charged")

        return PaymentResponse(
            order_id=payment.order_id,
            status="approved",
            transaction_id=transaction_id,
        )
