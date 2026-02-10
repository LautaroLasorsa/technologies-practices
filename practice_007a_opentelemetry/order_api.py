"""Order API — the entry-point service for the tracing demo.

Receives POST /orders, validates the order, "saves" it to a simulated DB,
then calls the Payment Service to process payment.

Run (after docker compose up -d):
    uv run uvicorn order_api:app --port 8000 --reload

Endpoints:
    GET  /health  — simple health check (Phase 2: verify tracing works)
    POST /orders  — create an order (Phase 3-5: spans, attributes, propagation)

Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#creating-spans
"""

import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from models import OrderRequest, PaymentRequest, PaymentResponse
from propagation import inject_context
from tracing import get_tracer, init_tracer

# ── OpenTelemetry imports you'll need ────────────────────────────────
from opentelemetry import trace
from opentelemetry.trace import StatusCode

# ── Initialize tracing on module load ────────────────────────────────
init_tracer(service_name="order-api")
tracer = get_tracer("order_api")

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="Order API")

PAYMENT_SERVICE_URL = "http://localhost:8001/pay"


# ── Health check (Phase 2) ───────────────────────────────────────────


@app.get("/health")
def health_check() -> dict:
    """Simple health endpoint to verify tracing setup.

    This endpoint already has a span created around it (below).
    Once you implement init_tracer() and get_tracer(), hitting this
    endpoint should produce a span visible in Jaeger.
    """
    with tracer.start_as_current_span("health-check"):
        return {"status": "ok", "service": "order-api"}


# ── Create order (Phases 3-5) ────────────────────────────────────────


@app.post("/orders")
async def create_order(order: OrderRequest) -> dict:
    """Create an order: validate, save, and request payment.

    This function orchestrates three operations, each of which should
    be its own span. The overall request is the parent span.
    """
    order_id = f"ORD-{uuid.uuid4().hex[:8]}"

    # --- Parent span wrapping the entire request ---
    # This is already done for you. Your child spans will nest inside it.
    with tracer.start_as_current_span("create-order") as parent_span:
        parent_span.set_attribute("order.id", order_id)

        # Phase 3: Validate the order
        validate_order(order, parent_span)

        # Phase 3: Simulate saving to database
        save_order_to_db(order_id, order)

        # Phase 5: Call payment service with context propagation
        payment = await request_payment(order_id, order)

        return {
            "order_id": order_id,
            "status": "created",
            "payment": payment.model_dump(),
        }


# ── Phase 3: Validation span ────────────────────────────────────────


def validate_order(order: OrderRequest, parent_span: trace.Span) -> None:
    """Validate the incoming order.

    TODO(human): Implement the tracing logic in this function.

    The validation rules are already written (amount > 0, quantity > 0).
    Your job is to wrap this logic in a span and enrich it.

    Steps:
      1. Create a child span named "validate-order" using:
             with tracer.start_as_current_span("validate-order") as span:
         This automatically becomes a child of the current active span
         (the "create-order" span above). You don't need to pass the parent
         explicitly — OpenTelemetry tracks the active span in a context variable.

      2. Inside the span, add attributes describing the order:
             span.set_attribute("order.customer_id", order.customer_id)
             span.set_attribute("order.item", order.item)
             span.set_attribute("order.amount", order.amount)
             span.set_attribute("order.quantity", order.quantity)
         Attributes are indexed key-value pairs — you can search/filter by them
         in Jaeger.

      3. If validation fails (amount <= 0 or quantity <= 0):
         a. Record the exception on the span:
                span.record_exception(error)
            This adds an "exception" event to the span with the error message,
            type, and stack trace. It does NOT change the span's status.
         b. Set the span status to ERROR:
                span.set_status(StatusCode.ERROR, str(error))
            This marks the span red in Jaeger. record_exception() alone only
            adds an event — the span would still show as "OK" without set_status.
         c. Raise the HTTPException so FastAPI returns a 400.

      4. If validation passes, add an event:
             span.add_event("validation-passed")
         Events are timestamped log-like annotations within a span.

    Hint — skeleton:
        with tracer.start_as_current_span("validate-order") as span:
            span.set_attribute("order.customer_id", order.customer_id)
            # ... more attributes ...
            try:
                if order.amount <= 0:
                    raise ValueError(f"Invalid amount: {order.amount}")
                if order.quantity <= 0:
                    raise ValueError(f"Invalid quantity: {order.quantity}")
                span.add_event("validation-passed")
            except ValueError as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))
                raise HTTPException(status_code=400, detail=str(e))

    Docs:
      - Attributes: https://opentelemetry.io/docs/languages/python/instrumentation/#span-attributes
      - Events: https://opentelemetry.io/docs/languages/python/instrumentation/#span-events
      - Exceptions: https://opentelemetry.io/docs/languages/python/instrumentation/#record-exceptions
    """
    # TODO(human): Wrap the validation below in a span with attributes.
    # For now this just validates without tracing:
    if order.amount <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid amount: {order.amount}")
    if order.quantity <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid quantity: {order.quantity}")


# ── Phase 3: Database simulation span ────────────────────────────────


def save_order_to_db(order_id: str, order: OrderRequest) -> None:
    """Simulate saving the order to a database.

    TODO(human): Implement the tracing logic in this function.

    Steps:
      1. Create a span named "save-order-db".
      2. Add attributes: "db.system" = "postgresql", "db.operation" = "INSERT",
         "order.id" = order_id.
         These follow OpenTelemetry's semantic conventions for database spans:
         https://opentelemetry.io/docs/specs/semconv/database/
      3. The sleep(0.05) simulates a 50ms DB write — leave it as-is.
      4. After the sleep, add an event: "order-saved".

    Hint:
        with tracer.start_as_current_span("save-order-db") as span:
            span.set_attribute("db.system", "postgresql")
            span.set_attribute("db.operation", "INSERT")
            span.set_attribute("order.id", order_id)
            time.sleep(0.05)  # simulate DB latency
            span.add_event("order-saved")

    Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#creating-spans
    """
    # TODO(human): Wrap this in a span with DB semantic attributes.
    time.sleep(0.05)  # simulate DB latency


# ── Phase 5: Payment call with context propagation ───────────────────


async def request_payment(order_id: str, order: OrderRequest) -> PaymentResponse:
    """Call the Payment Service, propagating the trace context.

    TODO(human): Implement context propagation in this function.

    Steps:
      1. Create a span named "request-payment" with kind=trace.SpanKind.CLIENT.
         SpanKind.CLIENT indicates this span represents an outgoing request.
         (The payment service will create a SpanKind.SERVER span on its end.)
             with tracer.start_as_current_span(
                 "request-payment", kind=trace.SpanKind.CLIENT
             ) as span:

      2. Add attributes describing the outgoing call:
             span.set_attribute("http.method", "POST")
             span.set_attribute("http.url", PAYMENT_SERVICE_URL)
             span.set_attribute("payment.order_id", order_id)

      3. Inject the trace context into HTTP headers:
             headers = inject_context({})
         This calls the function you implemented in propagation.py. It writes
         the `traceparent` header into the dict so the payment service can
         link its spans to this trace.

      4. Make the HTTP POST call with httpx, passing the headers:
             async with httpx.AsyncClient() as client:
                 response = await client.post(
                     PAYMENT_SERVICE_URL,
                     json=payload.model_dump(),
                     headers=headers,
                     timeout=5.0,
                 )

      5. Record the response status on the span:
             span.set_attribute("http.status_code", response.status_code)

      6. If the response is an error (status >= 400), set the span status to ERROR.

      7. Parse and return the PaymentResponse.

    Hint — full skeleton:
        with tracer.start_as_current_span("request-payment", kind=trace.SpanKind.CLIENT) as span:
            span.set_attribute("http.method", "POST")
            span.set_attribute("http.url", PAYMENT_SERVICE_URL)
            headers = inject_context({})
            payload = PaymentRequest(order_id=order_id, customer_id=order.customer_id, amount=order.amount)
            async with httpx.AsyncClient() as client:
                resp = await client.post(PAYMENT_SERVICE_URL, json=payload.model_dump(), headers=headers, timeout=5.0)
            span.set_attribute("http.status_code", resp.status_code)
            if resp.status_code >= 400:
                span.set_status(StatusCode.ERROR, f"Payment failed: {resp.status_code}")
            return PaymentResponse(**resp.json())

    Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#creating-spans
    """
    # TODO(human): Add a CLIENT span and inject trace context into headers.
    payload = PaymentRequest(
        order_id=order_id,
        customer_id=order.customer_id,
        amount=order.amount,
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            PAYMENT_SERVICE_URL,
            json=payload.model_dump(),
            timeout=5.0,
        )
    response.raise_for_status()
    return PaymentResponse(**response.json())
