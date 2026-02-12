# Practice 007a: OpenTelemetry — Instrumentation

## Technologies

- **OpenTelemetry** — Vendor-neutral observability framework for traces, metrics, and logs (CNCF graduated project)
- **opentelemetry-api / opentelemetry-sdk** — Python API and SDK for manual instrumentation
- **opentelemetry-exporter-otlp-proto-grpc** — OTLP exporter that sends telemetry data over gRPC
- **Jaeger** — Open-source distributed tracing backend with web UI
- **FastAPI / Uvicorn** — Lightweight async HTTP framework (used as the app under instrumentation)
- **httpx** — Async HTTP client (for inter-service calls with context propagation)

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

OpenTelemetry is a vendor-neutral observability framework that provides APIs, SDKs, and tools to instrument applications and collect telemetry data (traces, metrics, and logs). A distributed trace records the path taken by a single request as it propagates through multiple services, improving visibility into complex distributed systems.

**How Distributed Tracing Works:**

At its core, distributed tracing connects causally-related operations across service boundaries. A **trace** is made of one or more **spans**, with the first span representing the root span. Each span represents a single unit of work or operation within a trace, containing a unique SpanID and linking to its parent span to form a hierarchical view of the request lifecycle. Spans carry **attributes** (indexed metadata), **events** (timestamped annotations), and **status** (success/error markers).

**Context Propagation Mechanism:**

When Service A calls Service B, Service A includes a trace ID and span ID as part of the context in HTTP headers (using W3C Trace Context standard via the `traceparent` header). Service B extracts these values to create a new span that belongs to the same trace, setting the span from Service A as its parent. The OpenTelemetry SDK handles this serialization/deserialization automatically through inject/extract APIs, making it possible to track the full flow of a request across service boundaries.

The `traceparent` header format is: `00-<trace-id>-<span-id>-<flags>` where the trace-id is a 32-hex-digit identifier linking all spans in one request, and span-id identifies the current operation. When a service receives a request with this header, it extracts the context and creates a child span, forming a parent-child relationship visible as a waterfall diagram in tracing UIs like Jaeger.

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **Trace** | End-to-end journey of a request across all services, identified by a unique trace_id |
| **Span** | Single operation within a trace (function call, DB query, HTTP request) with start/end timestamps |
| **Trace Context** | Metadata (trace_id, span_id, flags) passed between services to link spans into a single trace |
| **Attributes** | Indexed key-value pairs attached to spans (e.g., `http.method=POST`) — searchable and filterable |
| **Events** | Timestamped log-like annotations within a span (e.g., "cache-hit", "retry-attempt") |
| **Status** | Span outcome indicator (OK, ERROR, UNSET) — sets visual markers in tracing UIs |
| **TracerProvider** | Global factory for creating Tracers; configured once at application startup |
| **SpanProcessor** | Component that batches and exports spans to a backend (BatchSpanProcessor vs SimpleSpanProcessor) |
| **Propagators** | Serialize/deserialize context into headers (W3C Trace Context is the default standard) |

**Ecosystem Context:**

OpenTelemetry is a CNCF graduated project and the industry standard for observability instrumentation. Unlike proprietary solutions (Datadog, New Relic), OpenTelemetry is vendor-neutral—instrumentation code remains the same regardless of backend (Jaeger, Zipkin, Tempo, AWS X-Ray, Google Cloud Trace). This practice uses Jaeger as the backend, but the instrumentation transfers to any OTLP-compatible system. The OTLP (OpenTelemetry Protocol) is a gRPC/HTTP protocol that all major observability vendors now support, ensuring future portability.

## Description

Build a **two-service order system** (Order API + Payment Service) that demonstrates distributed tracing with OpenTelemetry. Traces flow from the API through an inter-service HTTP call, showing how a single user request produces a multi-span trace visible in Jaeger.

### What you'll learn

1. **Tracing fundamentals** — TracerProvider, Resource, spans, and the span lifecycle
2. **Manual span creation** — `tracer.start_as_current_span()` for wrapping business logic
3. **Span enrichment** — Adding attributes, events, and status codes to spans
4. **Exception recording** — Capturing errors inside spans with `record_exception` and `set_status`
5. **Context propagation** — Injecting/extracting W3C `traceparent` headers across HTTP boundaries
6. **OTLP export to Jaeger** — Sending spans via gRPC to Jaeger's OTLP collector
7. **Reading traces in Jaeger UI** — Navigating the waterfall view, filtering by service/operation

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Start Jaeger with `docker compose up -d` (exposes UI at http://localhost:16686)
2. Run `uv sync` inside the project directory to install dependencies
3. Understand the OpenTelemetry data model: **Trace** (end-to-end request) > **Span** (single operation) > **Attributes** (key-value metadata)
4. Key question: How does a "trace" differ from a "log line"? What information does a trace capture that logs typically don't?

### Phase 2: Tracer Setup & First Span (~20 min)

1. Open `tracing.py` — the TracerProvider setup is scaffolded
2. **User implements:** `init_tracer()` — configure a `TracerProvider` with a `Resource`, attach a `BatchSpanProcessor` with the OTLP exporter, and set it as the global provider
3. **User implements:** `get_tracer()` — return a named tracer from the global provider
4. Run `order_api.py`, hit http://localhost:8000/health, check Jaeger UI for the "order-api" service
5. Key question: Why does OpenTelemetry use a `BatchSpanProcessor` instead of exporting each span immediately?

### Phase 3: Manual Spans & Attributes (~20 min)

1. Open `order_api.py` — the `/orders` endpoint is scaffolded with TODO markers
2. **User implements:** Create a span around the order validation logic, add attributes (`order.customer_id`, `order.item`, `order.amount`)
3. **User implements:** Create a child span for the "database save" simulation, add an event marking completion
4. Test: `POST /orders` with sample data, verify spans appear in Jaeger with attributes
5. Key question: When should you use span attributes vs span events? What's the difference?

### Phase 4: Exception Recording (~15 min)

1. The `/orders` endpoint has a validation path that rejects invalid orders
2. **User implements:** Record the exception on the span and set the span status to ERROR
3. Test: send an invalid order, verify the error span in Jaeger shows the exception details
4. Key question: Why do we call both `record_exception()` AND `set_status(ERROR)`? What does each do?

### Phase 5: Context Propagation Across Services (~25 min)

1. Open `payment_service.py` — a second FastAPI app running on port 8001
2. Open `propagation.py` — helpers for injecting/extracting trace context into HTTP headers
3. **User implements:** `inject_context()` — inject the current span's context into a carrier dict
4. **User implements:** `extract_context()` — extract context from incoming HTTP headers
5. **User implements:** In `order_api.py`, propagate context when calling the payment service
6. **User implements:** In `payment_service.py`, extract context from the incoming request and create a child span
7. Test: `POST /orders` and see a single trace spanning both services in Jaeger
8. Key question: What HTTP header carries the trace context? What happens if a service doesn't propagate it?

### Phase 6: Exploration & Review (~10 min)

1. Use Jaeger UI to explore: filter by service, find slow spans, compare error vs success traces
2. Discuss: How would auto-instrumentation (`opentelemetry-instrumentation-fastapi`) differ from what you built manually?
3. Discuss: What would you add for production use? (sampling, baggage, metrics)

## Motivation

- **Industry standard**: OpenTelemetry is the CNCF-graduated observability standard — it's the de facto way to instrument services in 2025+
- **Debugging distributed systems**: Traces are essential for understanding request flow, latency, and failures across microservices
- **Vendor-neutral**: Skills transfer to any backend (Jaeger, Zipkin, Datadog, Grafana Tempo, AWS X-Ray)
- **Complements 007b**: This practice builds instrumentation fundamentals; 007b adds dashboards, metrics, and alerting

## References

- [OpenTelemetry Python — Instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [OpenTelemetry Python — Getting Started](https://opentelemetry.io/docs/languages/python/getting-started/)
- [OpenTelemetry — Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
- [OpenTelemetry Python — Propagation](https://opentelemetry.io/docs/languages/python/propagation/)
- [Jaeger — Getting Started](https://www.jaegertracing.io/docs/1.76/getting-started/)
- [opentelemetry-exporter-otlp (PyPI)](https://pypi.org/project/opentelemetry-exporter-otlp/)

## Commands

### Phase 1: Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Jaeger container (OTLP gRPC on :4317, UI on :16686) |
| `uv sync` | Install Python dependencies from pyproject.toml |

### Phase 2: Run services

| Command | Description |
|---------|-------------|
| `uv run uvicorn order_api:app --port 8000 --reload` | Start the Order API (terminal 1) |
| `uv run uvicorn payment_service:app --port 8001 --reload` | Start the Payment Service (terminal 2) |

### Phase 3-5: Test endpoints

| Command | Description |
|---------|-------------|
| `uv run python test_tracing.py` | Run all manual test cases (health, valid order, invalid order, fraud order) |
| `curl http://localhost:8000/health` | Phase 2: verify basic tracing works |
| `curl -X POST http://localhost:8000/orders -H "Content-Type: application/json" -d "{\"customer_id\":\"CUST-42\",\"item\":\"Keyboard\",\"quantity\":2,\"amount\":149.99}"` | Phase 3: create a valid order (multi-span trace) |
| `curl -X POST http://localhost:8000/orders -H "Content-Type: application/json" -d "{\"customer_id\":\"CUST-99\",\"item\":\"Ghost\",\"quantity\":1,\"amount\":-50}"` | Phase 4: create an invalid order (error span) |
| `curl -X POST http://localhost:8000/orders -H "Content-Type: application/json" -d "{\"customer_id\":\"CUST-SHADY\",\"item\":\"Watch\",\"quantity\":1,\"amount\":25000}"` | Phase 5: create a fraud order (amount > 10,000) |

### Observe & Teardown

| Command | Description |
|---------|-------------|
| `http://localhost:16686` | Open Jaeger UI in browser to inspect traces |
| `docker compose down` | Stop and remove Jaeger container |

## State

`not-started`
