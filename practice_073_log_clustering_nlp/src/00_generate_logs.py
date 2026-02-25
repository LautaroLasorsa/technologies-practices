#!/usr/bin/env python3
"""Generate synthetic distributed-system logs from 5 microservices.

Produces:
  - data/logs.txt        : one raw log line per row
  - data/logs_metadata.csv: columns [timestamp, service, level, raw_log, is_anomaly]

The logs simulate realistic patterns:
  - 5 services: api-gateway, auth-service, order-service, payment-service, notification-service
  - Each service has 10-15 templates with variable parts (IPs, UUIDs, durations, paths)
  - Distribution: ~80% INFO, ~15% WARN, ~5% ERROR
  - Anomaly burst: 200 unusual payment-service errors injected in a 5-minute window
"""

import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Template definitions per service
# ---------------------------------------------------------------------------
# Each template: (level, format_string, variable_generator_fn)
# The generator returns a dict of variable values to fill the template.


def _random_ip() -> str:
    return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _random_uuid() -> str:
    return str(uuid.uuid4())


def _random_duration_ms() -> int:
    return random.randint(1, 5000)


def _random_user_id() -> str:
    return f"user-{random.randint(1000, 9999)}"


def _random_order_id() -> str:
    return f"ord-{random.randint(10000, 99999)}"


def _random_path() -> str:
    paths = [
        "/api/v1/orders", "/api/v1/users", "/api/v1/products",
        "/api/v1/payments", "/api/v1/notifications", "/api/v2/orders",
        "/health", "/metrics", "/api/v1/auth/login", "/api/v1/auth/refresh",
    ]
    return random.choice(paths)


def _random_status_code() -> int:
    return random.choices([200, 201, 204, 400, 401, 403, 404, 500, 502, 503],
                          weights=[50, 10, 5, 8, 5, 3, 8, 4, 3, 4])[0]


def _random_port() -> int:
    return random.choice([3306, 5432, 6379, 8080, 8443, 9092, 27017])


def _random_bytes() -> int:
    return random.randint(128, 65536)


# ---- api-gateway templates ----

API_GATEWAY_TEMPLATES: list[tuple[str, str, callable]] = [
    ("INFO", "Received {method} request to {path} from {client_ip}",
     lambda: {"method": random.choice(["GET", "POST", "PUT", "DELETE"]),
              "path": _random_path(), "client_ip": _random_ip()}),
    ("INFO", "Response sent for {path} status={status_code} latency={latency}ms",
     lambda: {"path": _random_path(), "status_code": _random_status_code(),
              "latency": _random_duration_ms()}),
    ("INFO", "Rate limit check passed for client {client_ip} token={token}",
     lambda: {"client_ip": _random_ip(), "token": _random_uuid()}),
    ("WARN", "Rate limit exceeded for client {client_ip} limit=100 window=60s",
     lambda: {"client_ip": _random_ip()}),
    ("WARN", "Slow upstream response from {upstream_ip}:{port} latency={latency}ms threshold=2000ms",
     lambda: {"upstream_ip": _random_ip(), "port": _random_port(),
              "latency": random.randint(2001, 8000)}),
    ("INFO", "Health check passed for upstream {upstream_ip}:{port}",
     lambda: {"upstream_ip": _random_ip(), "port": _random_port()}),
    ("ERROR", "Connection refused to upstream {upstream_ip}:{port}",
     lambda: {"upstream_ip": _random_ip(), "port": _random_port()}),
    ("INFO", "TLS handshake completed with {client_ip} cipher=TLS_AES_256_GCM_SHA384",
     lambda: {"client_ip": _random_ip()}),
    ("WARN", "Request body too large from {client_ip} size={size} bytes max=1048576",
     lambda: {"client_ip": _random_ip(), "size": random.randint(1048577, 5000000)}),
    ("INFO", "Routing request {request_id} to {service} via round-robin",
     lambda: {"request_id": _random_uuid(),
              "service": random.choice(["auth-service", "order-service", "payment-service"])}),
    ("ERROR", "Circuit breaker OPEN for {service} failures={failures} threshold=5",
     lambda: {"service": random.choice(["auth-service", "order-service", "payment-service"]),
              "failures": random.randint(5, 20)}),
    ("INFO", "Connection pool stats: active={active} idle={idle} max={max_conn}",
     lambda: {"active": random.randint(1, 50), "idle": random.randint(0, 20),
              "max_conn": 100}),
]

# ---- auth-service templates ----

AUTH_SERVICE_TEMPLATES: list[tuple[str, str, callable]] = [
    ("INFO", "Authentication successful for user {user_id} from {client_ip}",
     lambda: {"user_id": _random_user_id(), "client_ip": _random_ip()}),
    ("WARN", "Authentication failed for user {user_id} from {client_ip} reason=invalid_password",
     lambda: {"user_id": _random_user_id(), "client_ip": _random_ip()}),
    ("INFO", "JWT token issued for user {user_id} expires_in=3600s",
     lambda: {"user_id": _random_user_id()}),
    ("INFO", "JWT token refreshed for user {user_id} session={session_id}",
     lambda: {"user_id": _random_user_id(), "session_id": _random_uuid()}),
    ("WARN", "JWT token expired for user {user_id} token_id={token_id}",
     lambda: {"user_id": _random_user_id(), "token_id": _random_uuid()}),
    ("ERROR", "Redis connection failed to {redis_ip}:{port} retrying in {delay}ms",
     lambda: {"redis_ip": _random_ip(), "port": 6379,
              "delay": random.choice([100, 500, 1000, 2000])}),
    ("INFO", "Session created for user {user_id} session={session_id}",
     lambda: {"user_id": _random_user_id(), "session_id": _random_uuid()}),
    ("INFO", "Session invalidated for user {user_id} reason=logout",
     lambda: {"user_id": _random_user_id()}),
    ("WARN", "Brute force detection triggered for {client_ip} attempts={attempts}",
     lambda: {"client_ip": _random_ip(), "attempts": random.randint(5, 50)}),
    ("INFO", "Password hash verified for user {user_id} algo=argon2id latency={latency}ms",
     lambda: {"user_id": _random_user_id(), "latency": random.randint(50, 300)}),
    ("ERROR", "LDAP connection timeout to {ldap_ip}:{port} after {timeout}ms",
     lambda: {"ldap_ip": _random_ip(), "port": 389,
              "timeout": random.randint(5000, 30000)}),
]

# ---- order-service templates ----

ORDER_SERVICE_TEMPLATES: list[tuple[str, str, callable]] = [
    ("INFO", "Order {order_id} created by user {user_id} total={total}",
     lambda: {"order_id": _random_order_id(), "user_id": _random_user_id(),
              "total": round(random.uniform(10, 500), 2)}),
    ("INFO", "Order {order_id} status changed from {from_status} to {to_status}",
     lambda: {"order_id": _random_order_id(),
              "from_status": random.choice(["PENDING", "CONFIRMED", "PROCESSING"]),
              "to_status": random.choice(["CONFIRMED", "PROCESSING", "SHIPPED"])}),
    ("INFO", "Inventory check passed for order {order_id} items={items}",
     lambda: {"order_id": _random_order_id(), "items": random.randint(1, 10)}),
    ("WARN", "Inventory low for product {product_id} remaining={remaining} threshold=10",
     lambda: {"product_id": f"prod-{random.randint(100, 999)}",
              "remaining": random.randint(1, 10)}),
    ("ERROR", "Order {order_id} failed validation: {reason}",
     lambda: {"order_id": _random_order_id(),
              "reason": random.choice(["missing_address", "invalid_quantity",
                                       "product_unavailable", "duplicate_order"])}),
    ("INFO", "Database query executed in {latency}ms rows_affected={rows}",
     lambda: {"latency": _random_duration_ms(), "rows": random.randint(0, 100)}),
    ("INFO", "Order {order_id} confirmed and sent to payment-service",
     lambda: {"order_id": _random_order_id()}),
    ("WARN", "Saga compensation triggered for order {order_id} step={step}",
     lambda: {"order_id": _random_order_id(),
              "step": random.choice(["inventory_reserve", "payment_charge"])}),
    ("INFO", "Bulk order import completed: processed={processed} failed={failed}",
     lambda: {"processed": random.randint(50, 500), "failed": random.randint(0, 5)}),
    ("ERROR", "Database connection pool exhausted max={max_conn} active={active}",
     lambda: {"max_conn": 50, "active": 50}),
    ("INFO", "Cache hit for order {order_id} ttl_remaining={ttl}s",
     lambda: {"order_id": _random_order_id(), "ttl": random.randint(10, 300)}),
    ("INFO", "Event published to kafka topic=order-events partition={partition}",
     lambda: {"partition": random.randint(0, 5)}),
]

# ---- payment-service templates ----

PAYMENT_SERVICE_TEMPLATES: list[tuple[str, str, callable]] = [
    ("INFO", "Payment initiated for order {order_id} amount={amount} currency=USD",
     lambda: {"order_id": _random_order_id(),
              "amount": round(random.uniform(10, 500), 2)}),
    ("INFO", "Payment {payment_id} processed successfully via {provider}",
     lambda: {"payment_id": _random_uuid(),
              "provider": random.choice(["stripe", "paypal", "braintree"])}),
    ("WARN", "Payment retry attempt {attempt} for order {order_id}",
     lambda: {"attempt": random.randint(1, 3), "order_id": _random_order_id()}),
    ("ERROR", "Payment declined for order {order_id} reason={reason}",
     lambda: {"order_id": _random_order_id(),
              "reason": random.choice(["insufficient_funds", "card_expired",
                                       "fraud_detected", "network_error"])}),
    ("INFO", "Refund {refund_id} issued for order {order_id} amount={amount}",
     lambda: {"refund_id": _random_uuid(), "order_id": _random_order_id(),
              "amount": round(random.uniform(10, 500), 2)}),
    ("INFO", "Payment gateway health check OK provider={provider} latency={latency}ms",
     lambda: {"provider": random.choice(["stripe", "paypal", "braintree"]),
              "latency": random.randint(20, 200)}),
    ("WARN", "Payment gateway slow response provider={provider} latency={latency}ms threshold=1000ms",
     lambda: {"provider": random.choice(["stripe", "paypal", "braintree"]),
              "latency": random.randint(1001, 5000)}),
    ("ERROR", "Payment gateway unreachable provider={provider} timeout={timeout}ms",
     lambda: {"provider": random.choice(["stripe", "paypal", "braintree"]),
              "timeout": random.randint(5000, 30000)}),
    ("INFO", "Transaction {tx_id} committed to ledger",
     lambda: {"tx_id": _random_uuid()}),
    ("INFO", "Webhook received from {provider} event={event}",
     lambda: {"provider": random.choice(["stripe", "paypal"]),
              "event": random.choice(["payment.completed", "refund.processed",
                                      "dispute.created"])}),
    ("WARN", "Idempotency key collision detected for order {order_id}",
     lambda: {"order_id": _random_order_id()}),
]

# ---- notification-service templates ----

NOTIFICATION_SERVICE_TEMPLATES: list[tuple[str, str, callable]] = [
    ("INFO", "Email sent to {email} subject=Order Confirmation order={order_id}",
     lambda: {"email": f"user{random.randint(1,999)}@example.com",
              "order_id": _random_order_id()}),
    ("INFO", "SMS sent to {phone} template=order_shipped order={order_id}",
     lambda: {"phone": f"+1{random.randint(2000000000,9999999999)}",
              "order_id": _random_order_id()}),
    ("WARN", "Email delivery delayed to {email} queue_depth={depth}",
     lambda: {"email": f"user{random.randint(1,999)}@example.com",
              "depth": random.randint(100, 1000)}),
    ("ERROR", "SMTP connection failed to {smtp_ip}:{port} error=connection_refused",
     lambda: {"smtp_ip": _random_ip(), "port": 587}),
    ("INFO", "Push notification sent to device {device_id} user={user_id}",
     lambda: {"device_id": _random_uuid(), "user_id": _random_user_id()}),
    ("INFO", "Notification preference loaded for user {user_id} channels={channels}",
     lambda: {"user_id": _random_user_id(),
              "channels": random.choice(["email,sms", "email", "sms,push", "email,sms,push"])}),
    ("WARN", "Notification rate limit hit for user {user_id} limit=10 per_hour",
     lambda: {"user_id": _random_user_id()}),
    ("INFO", "Template rendered template={template} locale={locale} latency={latency}ms",
     lambda: {"template": random.choice(["order_confirm", "password_reset",
                                          "shipping_update", "welcome"]),
              "locale": random.choice(["en-US", "es-MX", "fr-FR"]),
              "latency": random.randint(5, 100)}),
    ("ERROR", "Failed to render template={template} error=missing_variable var={var}",
     lambda: {"template": random.choice(["order_confirm", "shipping_update"]),
              "var": random.choice(["customer_name", "order_total", "tracking_number"])}),
    ("INFO", "Webhook callback sent to {callback_url} status={status}",
     lambda: {"callback_url": f"https://{_random_ip()}/webhooks/notify",
              "status": random.choice([200, 201, 202])}),
    ("INFO", "Batch digest email queued for {count} users",
     lambda: {"count": random.randint(10, 500)}),
]

# ---- Anomaly templates (unusual payment-service errors) ----

ANOMALY_TEMPLATES: list[tuple[str, str, callable]] = [
    ("ERROR", "CRITICAL: Payment database replication lag detected lag={lag}s threshold=1s",
     lambda: {"lag": round(random.uniform(5, 60), 1)}),
    ("ERROR", "CRITICAL: Payment encryption key rotation failed key_id={key_id}",
     lambda: {"key_id": _random_uuid()}),
    ("ERROR", "CRITICAL: Payment service memory pressure high used={used}MB max={max_mem}MB",
     lambda: {"used": random.randint(3500, 4000), "max_mem": 4096}),
    ("ERROR", "CRITICAL: Deadlock detected in payment transaction pool tx={tx_id}",
     lambda: {"tx_id": _random_uuid()}),
    ("ERROR", "CRITICAL: Payment audit log write failed disk_space=0 path=/var/log/payment/audit.log",
     lambda: {}),
    ("ERROR", "CRITICAL: SSL certificate expiring in {hours} hours for payment gateway endpoint",
     lambda: {"hours": random.randint(1, 24)}),
    ("ERROR", "CRITICAL: Payment reconciliation mismatch amount_expected={expected} amount_actual={actual}",
     lambda: {"expected": round(random.uniform(1000, 50000), 2),
              "actual": round(random.uniform(1000, 50000), 2)}),
]

# ---------------------------------------------------------------------------
# Service registry
# ---------------------------------------------------------------------------

SERVICES: dict[str, list[tuple[str, str, callable]]] = {
    "api-gateway": API_GATEWAY_TEMPLATES,
    "auth-service": AUTH_SERVICE_TEMPLATES,
    "order-service": ORDER_SERVICE_TEMPLATES,
    "payment-service": PAYMENT_SERVICE_TEMPLATES,
    "notification-service": NOTIFICATION_SERVICE_TEMPLATES,
}

# Weights control how often each service logs (api-gateway is noisiest)
SERVICE_WEIGHTS = {
    "api-gateway": 30,
    "auth-service": 20,
    "order-service": 20,
    "payment-service": 15,
    "notification-service": 15,
}


# ---------------------------------------------------------------------------
# Log generation
# ---------------------------------------------------------------------------


def _format_log_line(timestamp: datetime, service: str, level: str, message: str) -> str:
    """Format a single log line in a common structured format."""
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"{ts_str} [{level:<5}] [{service}] {message}"


def _pick_template(
    templates: list[tuple[str, str, callable]],
) -> tuple[str, str, dict]:
    """Pick a random template and generate its variable values.

    Within each service, weight selection by level:
      INFO templates ~80%, WARN ~15%, ERROR ~5%
    """
    level_weights = {"INFO": 80, "WARN": 15, "ERROR": 5}

    weighted_templates = []
    weights = []
    for level, fmt, gen in templates:
        weighted_templates.append((level, fmt, gen))
        weights.append(level_weights.get(level, 10))

    level, fmt, gen = random.choices(weighted_templates, weights=weights, k=1)[0]
    variables = gen()
    message = fmt.format(**variables)
    return level, message, variables


def generate_normal_logs(
    n: int,
    start_time: datetime,
) -> list[tuple[datetime, str, str, str, bool]]:
    """Generate n normal log lines spread over a time range.

    Returns list of (timestamp, service, level, raw_log, is_anomaly).
    """
    rows: list[tuple[datetime, str, str, str, bool]] = []

    # Spread logs over ~2 hours
    time_range_seconds = 2 * 60 * 60  # 2 hours

    service_names = list(SERVICES.keys())
    service_w = [SERVICE_WEIGHTS[s] for s in service_names]

    for _ in range(n):
        offset = timedelta(seconds=random.uniform(0, time_range_seconds))
        timestamp = start_time + offset

        service = random.choices(service_names, weights=service_w, k=1)[0]
        templates = SERVICES[service]
        level, message, _ = _pick_template(templates)

        raw_log = _format_log_line(timestamp, service, level, message)
        rows.append((timestamp, service, level, raw_log, False))

    return rows


def generate_anomaly_burst(
    n: int,
    burst_center: datetime,
    burst_window_minutes: int = 5,
) -> list[tuple[datetime, str, str, str, bool]]:
    """Generate n anomalous log lines concentrated in a time window.

    All anomaly logs come from payment-service with unusual ERROR patterns.
    """
    rows: list[tuple[datetime, str, str, str, bool]] = []
    half_window = timedelta(minutes=burst_window_minutes / 2)

    for _ in range(n):
        offset = timedelta(seconds=random.uniform(
            -half_window.total_seconds(), half_window.total_seconds()))
        timestamp = burst_center + offset

        level, fmt, gen = random.choice(ANOMALY_TEMPLATES)
        variables = gen()
        message = fmt.format(**variables)

        raw_log = _format_log_line(timestamp, "payment-service", level, message)
        rows.append((timestamp, "payment-service", level, raw_log, True))

    return rows


def main() -> None:
    """Generate synthetic logs and save to data/."""
    random.seed(42)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime(2025, 6, 15, 8, 0, 0)
    burst_center = start_time + timedelta(minutes=75)  # ~09:15

    print("Generating 10,000 normal log lines...")
    normal_logs = generate_normal_logs(10_000, start_time)

    print("Generating 200 anomaly burst lines...")
    anomaly_logs = generate_anomaly_burst(200, burst_center, burst_window_minutes=5)

    all_logs = normal_logs + anomaly_logs
    all_logs.sort(key=lambda r: r[0])  # sort by timestamp

    # Save raw logs (one line per row)
    logs_path = DATA_DIR / "logs.txt"
    with open(logs_path, "w", encoding="utf-8") as f:
        for _, _, _, raw_log, _ in all_logs:
            f.write(raw_log + "\n")
    print(f"  Wrote {len(all_logs)} lines to {logs_path}")

    # Save metadata CSV
    meta_path = DATA_DIR / "logs_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "service", "level", "raw_log", "is_anomaly"])
        for ts, service, level, raw_log, is_anomaly in all_logs:
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                service,
                level,
                raw_log,
                is_anomaly,
            ])
    print(f"  Wrote metadata to {meta_path}")

    # Summary statistics
    n_anomaly = sum(1 for r in all_logs if r[4])
    n_normal = len(all_logs) - n_anomaly
    services_seen = set(r[1] for r in all_logs)
    levels = {}
    for _, _, level, _, _ in all_logs:
        levels[level] = levels.get(level, 0) + 1

    print(f"\nSummary:")
    print(f"  Total lines:   {len(all_logs)}")
    print(f"  Normal lines:  {n_normal}")
    print(f"  Anomaly lines: {n_anomaly}")
    print(f"  Services:      {sorted(services_seen)}")
    print(f"  Level counts:  {levels}")
    print(f"  Anomaly burst: ~{burst_center.strftime('%H:%M')} +/- 2.5 min")


if __name__ == "__main__":
    main()
