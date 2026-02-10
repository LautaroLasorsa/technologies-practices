"""Order Service -- OpenTelemetry Metrics Practice (007b).

This FastAPI app simulates an order-processing service and exposes custom
metrics for Prometheus to scrape. The metrics endpoint is served on the same
port (8000) at ``/metrics``.

Architecture:
    App (:8000/metrics) --(scrape)--> Prometheus (:9090) --(query)--> Grafana (:3000)

Boilerplate (provided): FastAPI setup, OTel provider wiring, Prometheus exporter,
    simulation endpoint, app entry-point.

TODO(human): Create metric instruments and record values in the marked sections.
"""

from __future__ import annotations

import asyncio
import random
import time

from fastapi import FastAPI, Query

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import REGISTRY

# ---------------------------------------------------------------------------
# OTel setup (boilerplate -- do NOT modify)
# ---------------------------------------------------------------------------

resource = Resource.create(attributes={SERVICE_NAME: "order-service"})
prometheus_reader = PrometheusMetricReader()
provider = MeterProvider(resource=resource, metric_readers=[prometheus_reader])
metrics.set_meter_provider(provider)

meter = metrics.get_meter("order-service", version="0.1.0")

# ---------------------------------------------------------------------------
# Metric instruments -- TODO(human)
# ---------------------------------------------------------------------------
# The ``meter`` object above is your factory for creating instruments.
# Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#creating-metrics
#
# You need to create THREE instruments and store them in module-level variables
# so the endpoint handlers below can use them.


def create_metrics(
    m: metrics.Meter,
) -> tuple[metrics.Counter, metrics.Histogram, metrics.UpDownCounter]:
    """Create and return the three metric instruments for this service.

    TODO(human): Implement this function. Use the ``meter`` methods:
        - ``m.create_counter(...)``        for counting orders processed
        - ``m.create_histogram(...)``      for recording processing duration
        - ``m.create_up_down_counter(...)`` for tracking queue depth

    Each instrument needs:
        - ``name``  : a snake_case metric name (e.g. "orders_processed")
        - ``unit``  : measurement unit (e.g. "1" for counts, "s" for seconds)
        - ``description``: human-readable explanation

    Returns:
        A tuple of (order_counter, duration_histogram, queue_depth_gauge).

    Hint -- expected metric names (must match the Grafana dashboard & alerts):
        - orders_processed    (counter, unit="1")
        - order_duration_seconds (histogram, unit="s")
        - order_queue_depth   (up-down counter, unit="1")
    """
    # TODO(human): Replace these None values with real instruments.
    order_counter: metrics.Counter = None  # type: ignore[assignment]
    duration_histogram: metrics.Histogram = None  # type: ignore[assignment]
    queue_gauge: metrics.UpDownCounter = None  # type: ignore[assignment]

    return order_counter, duration_histogram, queue_gauge


order_counter, duration_histogram, queue_gauge = create_metrics(meter)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Order Service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/orders")
async def process_order(
    fail: bool = Query(False, description="Force this order to fail"),
) -> dict[str, str]:
    """Simulate processing a single order.

    TODO(human): Record metrics inside this handler.

    Steps:
        1. Increment ``queue_gauge`` by 1 (order enters the queue).
        2. Record the start time.
        3. Simulate processing with a random sleep (0.05 -- 1.5 s).
           If ``fail`` is True or random chance < 10%, treat as failure.
        4. Compute elapsed time and record it on ``duration_histogram``.
        5. Increment ``order_counter`` with a ``status`` attribute
           ("success" or "failure").
        6. Decrement ``queue_gauge`` by 1 (order leaves the queue).
    """
    # TODO(human): Increment queue_gauge by +1 here (order enters queue).

    start = time.monotonic()

    # Simulate variable processing time
    processing_time = random.uniform(0.05, 1.5)
    await asyncio.sleep(processing_time)

    # Determine outcome
    is_failure = fail or random.random() < 0.10
    status = "failure" if is_failure else "success"
    elapsed = time.monotonic() - start

    # TODO(human): Record elapsed time on duration_histogram.
    #   Hint: duration_histogram.record(elapsed)

    # TODO(human): Increment order_counter with attribute {"status": status}.
    #   Hint: order_counter.add(1, {"status": status})

    # TODO(human): Decrement queue_gauge by -1 here (order leaves queue).
    #   Hint: queue_gauge.add(-1)

    return {"status": status, "duration_seconds": f"{elapsed:.3f}"}


@app.post("/simulate")
async def simulate(
    count: int = Query(20, ge=1, le=500, description="Number of orders"),
    error_rate: float = Query(0.1, ge=0.0, le=1.0, description="Fraction of forced failures"),
) -> dict[str, str]:
    """Generate synthetic load by sending ``count`` orders concurrently.

    This endpoint is fully implemented -- it calls ``/orders`` internally.
    Use it to generate traffic so Prometheus has data to scrape.
    """

    async def _one_order() -> str:
        fail = random.random() < error_rate
        result = await process_order(fail=fail)
        return result["status"]

    results = await asyncio.gather(*[_one_order() for _ in range(count)])
    successes = results.count("success")
    failures = results.count("failure")
    return {
        "total": str(count),
        "successes": str(successes),
        "failures": str(failures),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Start the Prometheus metrics HTTP server on the SAME port as FastAPI.
    # The PrometheusMetricReader hooks into the default prometheus_client
    # registry, and FastAPI + uvicorn will serve /metrics via the
    # prometheus_client ASGI middleware below.
    #
    # Alternatively, you could run prometheus_client.start_http_server on a
    # separate port, but co-hosting is simpler for this practice.

    from prometheus_client import make_asgi_app

    metrics_app = make_asgi_app(registry=REGISTRY)
    app.mount("/metrics", metrics_app)

    print("Starting Order Service on http://0.0.0.0:8000")
    print("Metrics at http://0.0.0.0:8000/metrics")
    print("Prometheus UI at http://localhost:9090")
    print("Grafana UI at http://localhost:3000 (admin/admin)")

    uvicorn.run(app, host="0.0.0.0", port=8000)
