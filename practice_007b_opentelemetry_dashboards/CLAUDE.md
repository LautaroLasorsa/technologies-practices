# Practice 007b: OpenTelemetry -- Dashboards & Alerting

## Technologies

- **OpenTelemetry Python SDK** -- Vendor-neutral metrics instrumentation (counters, histograms, gauges)
- **Prometheus** -- Time-series database that scrapes `/metrics` endpoints
- **Grafana** -- Visualization platform for building dashboards and alert rules
- **prometheus_client** -- Python HTTP server exposing metrics in Prometheus text format
- **Docker Compose** -- Orchestrates the multi-container stack (app + Prometheus + Grafana)

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Description

Build a **simulated order-processing service** that emits custom OpenTelemetry metrics (request count, latency histograms, queue depth gauge), then visualize them in Grafana dashboards backed by Prometheus and configure alerting rules -- all running locally in Docker.

### What you'll learn

1. **OTel metric instruments** -- Counter, Histogram, UpDownCounter/ObservableGauge: when to use each
2. **Prometheus scrape model** -- pull-based collection, `/metrics` endpoint, scrape intervals, label cardinality
3. **PromQL basics** -- `rate()`, `histogram_quantile()`, aggregation operators for dashboard panels
4. **Grafana provisioning** -- auto-loading datasources and dashboards from JSON/YAML at container startup
5. **Alerting rules** -- Prometheus recording/alerting rules, `for` duration, severity labels
6. **End-to-end observability** -- from instrumentation in code to visual dashboard to firing alert

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Run `docker compose up -d` to start Prometheus and Grafana
2. Run `uv sync` inside `app/` to install Python dependencies
3. Understand the pipeline: **App emits metrics -> Prometheus scrapes -> Grafana queries Prometheus**
4. Open Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000` (admin/admin)
5. Key question: Why does Prometheus *pull* metrics instead of having apps *push* them? What are the trade-offs?

### Phase 2: Instrument the Python App (~30 min)

1. Open `app/main.py` -- the boilerplate (FastAPI app, OTel provider, Prometheus exporter) is ready
2. **User implements:** Create three metric instruments in `create_metrics()`:
   - A **Counter** for total orders processed (with `status` label: success/failure)
   - A **Histogram** for order processing duration in seconds
   - An **UpDownCounter** (or ObservableGauge) for current queue depth
3. **User implements:** Record metric values in the endpoint handlers using the instruments
4. Run the app: `uv run python main.py` and verify metrics appear at `http://localhost:8000/metrics`
5. Key question: Why use a Histogram instead of just averaging latencies? What does `histogram_quantile(0.95, ...)` tell you that `avg()` doesn't?

### Phase 3: Load-Test & Verify Prometheus (~10 min)

1. Use the built-in `/simulate` endpoint to generate synthetic traffic
2. Check Prometheus targets at `http://localhost:9090/targets` -- the app should show as UP
3. Run PromQL queries in the Prometheus UI:
   - `rate(orders_processed_total[1m])` -- per-second order rate
   - `histogram_quantile(0.95, rate(order_duration_seconds_bucket[5m]))` -- p95 latency
   - `order_queue_depth` -- current queue depth
4. Key question: What happens to `rate()` if the app restarts and the counter resets to zero?

### Phase 4: Grafana Dashboard (~20 min)

1. A pre-provisioned dashboard skeleton is loaded at startup (check Dashboards in Grafana)
2. The dashboard has placeholder panels -- **User implements:** Edit each panel's PromQL query:
   - Panel "Order Rate": `rate(orders_processed_total[1m])`
   - Panel "P95 Latency": `histogram_quantile(0.95, rate(order_duration_seconds_bucket[5m]))`
   - Panel "Queue Depth": `order_queue_depth`
   - Panel "Error Rate": `rate(orders_processed_total{status="failure"}[1m]) / rate(orders_processed_total[1m])`
3. Adjust time ranges, legend format, thresholds
4. Key question: Why is label cardinality (e.g., adding `customer_id` as a label) dangerous in Prometheus?

### Phase 5: Alerting Rules (~20 min)

1. Open `prometheus/alerts.yml` -- contains rule group skeleton with `TODO(human)` placeholders
2. **User implements:** Two alerting rules:
   - `HighErrorRate`: fires when error rate > 20% for 2 minutes
   - `HighLatency`: fires when p95 latency > 2 seconds for 1 minute
3. Restart Prometheus (`docker compose restart prometheus`) to load the new rules
4. Trigger alerts by using the `/simulate?error_rate=0.5` endpoint
5. Check firing alerts at `http://localhost:9090/alerts`
6. Key question: What is the `for` clause in alerting rules? Why not fire immediately?

### Phase 6: Discussion (~10 min)

1. How would you add Alertmanager to route alerts to Slack/PagerDuty?
2. Push vs pull: when would you use the OTel Collector as an intermediary?
3. What metrics would you instrument in a real production service?

## Motivation

- **Industry standard**: Prometheus + Grafana is the dominant open-source monitoring stack; OpenTelemetry is the CNCF standard for instrumentation
- **Production essential**: Every production service needs dashboards and alerting -- this is a core SRE/backend skill
- **Vendor-neutral**: OTel metrics work with Prometheus, Datadog, New Relic, etc. -- learn once, use anywhere
- **Complements 007a**: Tracing (007a) shows request flow; metrics (007b) show aggregate health -- together they form the observability foundation

## References

- [OpenTelemetry Python Metrics](https://opentelemetry.io/docs/languages/python/exporters/)
- [OpenTelemetry Metric Instruments](https://opentelemetry.io/docs/specs/otel/metrics/api/)
- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [OTel Prometheus Exporter (PyPI)](https://pypi.org/project/opentelemetry-exporter-prometheus/)

## Commands

### Phase 1: Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Prometheus (:9090) and Grafana (:3000) containers |
| `cd app && uv sync` | Install Python dependencies inside the `app/` directory |

### Phase 2: Run the app

| Command | Description |
|---------|-------------|
| `cd app && uv run python main.py` | Start the Order Service on :8000 (exposes /metrics for Prometheus) |

### Phase 3: Generate traffic & verify Prometheus

| Command | Description |
|---------|-------------|
| `curl -X POST http://localhost:8000/simulate` | Generate 20 synthetic orders (default settings) |
| `curl -X POST "http://localhost:8000/simulate?count=100&error_rate=0.1"` | Generate 100 orders with 10% error rate |
| `curl -X POST http://localhost:8000/orders` | Process a single order (success) |
| `curl -X POST "http://localhost:8000/orders?fail=true"` | Process a single order (forced failure) |
| `curl http://localhost:8000/metrics` | View raw Prometheus metrics exposed by the app |
| `curl http://localhost:8000/health` | Health check |

### Phase 5: Alerting

| Command | Description |
|---------|-------------|
| `docker compose restart prometheus` | Reload Prometheus to pick up changes to alerts.yml |
| `curl -X POST "http://localhost:8000/simulate?count=50&error_rate=0.5"` | Generate high-error traffic to trigger HighErrorRate alert |

### Observe & Teardown

| Command | Description |
|---------|-------------|
| `http://localhost:9090` | Open Prometheus UI in browser (query, targets, alerts) |
| `http://localhost:9090/targets` | Check Prometheus scrape targets (app should show as UP) |
| `http://localhost:9090/alerts` | Check firing alerts in Prometheus |
| `http://localhost:3000` | Open Grafana UI in browser (admin/admin) |
| `docker compose down` | Stop and remove Prometheus and Grafana containers |

## State

`not-started`
