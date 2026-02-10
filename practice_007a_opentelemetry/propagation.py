"""Trace context propagation helpers.

Provides inject/extract functions for passing trace context across HTTP
boundaries using W3C Trace Context headers (traceparent, tracestate).

How it works:
    Service A creates a span, then INJECTS its context into HTTP headers.
    Service B receives those headers, EXTRACTS the context, and creates a
    child span linked to Service A's span. The result: both spans appear
    in the same trace in Jaeger.

    Service A                         Service B
    ┌─────────┐   HTTP + headers     ┌─────────┐
    │ span-A  │ ──────────────────>  │ span-B  │
    └─────────┘   traceparent:       └─────────┘
                  00-<trace_id>-       (child of
                  <span_id>-01         span-A)

The W3C traceparent header looks like:
    00-<32-hex-trace-id>-<16-hex-span-id>-<2-hex-flags>
    e.g., 00-a9c3b99a95cc045e573e163c3ac80a77-d99d251a8caecd06-01

Docs: https://opentelemetry.io/docs/languages/python/propagation/
"""

from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject


# ── TODO(human): Implement these two functions ───────────────────────


def inject_context(carrier: dict[str, str]) -> dict[str, str]:
    """Inject the current trace context into a carrier dict (HTTP headers).

    TODO(human): Implement this function.

    Steps:
      1. Call inject(carrier) — this writes the `traceparent` (and optionally
         `tracestate`) header into the carrier dict using the globally
         configured propagator (W3C TraceContext by default).
      2. Return the carrier.

    Usage in the calling service:
        headers = inject_context({})
        response = httpx.post("http://payment:8001/pay", headers=headers, ...)

    The inject() function reads the current active span from Python's
    context and serializes its trace_id + span_id into the traceparent header.
    If there's no active span, inject() is a no-op (headers stay empty).

    Hint:
        inject(carrier)
        return carrier

    Docs: https://opentelemetry.io/docs/languages/python/propagation/#injecting-context
    """
    raise NotImplementedError("TODO(human): implement inject_context")


def extract_context(headers: dict[str, str]) -> Context:
    """Extract trace context from incoming HTTP headers.

    TODO(human): Implement this function.

    Steps:
      1. Call extract(carrier=headers) — this reads the `traceparent` header
         and returns a Context object containing the remote span's trace_id
         and span_id.
      2. Return that Context.

    Usage in the receiving service:
        ctx = extract_context(dict(request.headers))
        with tracer.start_as_current_span("handle-payment", context=ctx):
            ...

    By passing the extracted context to start_as_current_span(), the new span
    becomes a CHILD of the remote span — linking both services in one trace.

    Hint:
        return extract(carrier=headers)

    Docs: https://opentelemetry.io/docs/languages/python/propagation/#extracting-context
    """
    raise NotImplementedError("TODO(human): implement extract_context")
