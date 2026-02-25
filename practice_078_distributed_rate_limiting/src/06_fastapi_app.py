"""Exercise 06: FastAPI Application with Rate Limiting Middleware

A FastAPI service that applies distributed rate limiting as middleware.
When deployed as multiple replicas behind a load balancer (Nginx), all
replicas share rate limit state via Redis.

The middleware intercepts every request, checks the rate limiter, and
returns HTTP 429 Too Many Requests with proper headers when the limit
is exceeded.

Run locally (single replica):
    uv run uvicorn src.06_fastapi_app:app --host 0.0.0.0 --port 8000

Run via Docker Compose (3 replicas behind Nginx):
    docker compose up --build -d
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.common import load_lua_script


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
RATE_LIMIT_ALGORITHM = os.environ.get("RATE_LIMIT_ALGORITHM", "token_bucket")
RATE_LIMIT_CAPACITY = int(os.environ.get("RATE_LIMIT_CAPACITY", "20"))
RATE_LIMIT_REFILL_RATE = float(os.environ.get("RATE_LIMIT_REFILL_RATE", "10"))
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "10"))


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Holds shared resources: Redis client and registered Lua scripts."""

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.token_bucket_script = None
        self.replica_id: str = f"replica-{os.getpid()}"
        self.request_count: int = 0


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Redis connection and register Lua scripts on startup."""
    state.redis = aioredis.from_url(REDIS_URL, decode_responses=False)
    await state.redis.ping()
    print(f"[{state.replica_id}] Connected to Redis at {REDIS_URL}")

    # Register the token bucket Lua script
    lua_source = load_lua_script("token_bucket")
    state.token_bucket_script = state.redis.register_script(lua_source)
    print(f"[{state.replica_id}] Lua scripts registered")

    yield

    if state.redis:
        await state.redis.aclose()
        print(f"[{state.replica_id}] Redis connection closed")


# ---------------------------------------------------------------------------
# Rate Limit Middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that applies distributed rate limiting to all requests.

    Extracts the client identifier from the X-API-Key header (falling back
    to the client IP), checks the rate limiter, and either passes the
    request through or returns 429 with standard rate limit headers.

    HTTP response headers (per RFC 6585 and draft-ietf-httpapi-ratelimit-headers):
        X-RateLimit-Limit:     The maximum number of requests allowed
        X-RateLimit-Remaining: Approximate tokens remaining
        X-RateLimit-Reset:     Seconds until the bucket is full again
        Retry-After:           Seconds to wait before retrying (on 429)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # TODO(human): Implement the rate limiting middleware.
        #
        # WHY THIS MATTERS:
        # This is where theory meets production. A rate limit middleware
        # must be:
        # (a) Fast -- adds <2ms latency (single Redis round-trip).
        # (b) Resilient -- if Redis is down, decide fail-open or fail-closed.
        # (c) Informative -- return standard headers so clients can adapt.
        # (d) Correct -- use the client identifier (not just IP, because
        #     multiple clients share IPs behind NATs/proxies).
        #
        # WHAT TO DO:
        #
        # 1. SKIP HEALTH ENDPOINT:
        #    if request.url.path == "/health":
        #        return await call_next(request)
        #
        # 2. EXTRACT CLIENT IDENTIFIER:
        #    Use the X-API-Key header if present, otherwise fall back to
        #    the client's IP address.
        #    client_id = request.headers.get("x-api-key") or request.client.host
        #
        # 3. CHECK RATE LIMIT using the token bucket Lua script:
        #    redis_key = f"ratelimit:api:tokenbucket:{client_id}"
        #    now = time.time()
        #
        #    try:
        #        result = await state.token_bucket_script(
        #            keys=[redis_key],
        #            args=[
        #                RATE_LIMIT_CAPACITY,
        #                RATE_LIMIT_REFILL_RATE,
        #                now,
        #                1,  # cost = 1 token per request
        #            ],
        #        )
        #        allowed = result[0] == 1
        #        tokens_remaining = float(result[1])
        #    except Exception as exc:
        #        # Redis is down -- fail-open (allow the request)
        #        # In production, you'd log this and potentially switch to
        #        # a local in-memory rate limiter as fallback.
        #        print(f"[{state.replica_id}] Redis error: {exc}. Failing open.")
        #        return await call_next(request)
        #
        # 4. IF REJECTED -- return 429 with headers:
        #    if not allowed:
        #        retry_after = max(1.0, (1 - tokens_remaining) / RATE_LIMIT_REFILL_RATE)
        #        return JSONResponse(
        #            status_code=429,
        #            content={
        #                "error": "rate_limit_exceeded",
        #                "message": f"Rate limit exceeded for client '{client_id}'",
        #                "retry_after": round(retry_after, 2),
        #            },
        #            headers={
        #                "X-RateLimit-Limit": str(RATE_LIMIT_CAPACITY),
        #                "X-RateLimit-Remaining": "0",
        #                "Retry-After": str(int(retry_after)),
        #            },
        #        )
        #
        # 5. IF ALLOWED -- pass through with rate limit headers:
        #    response = await call_next(request)
        #    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_CAPACITY)
        #    response.headers["X-RateLimit-Remaining"] = str(int(tokens_remaining))
        #    return response
        #
        # DESIGN DECISION -- FAIL-OPEN VS FAIL-CLOSED:
        # We fail-open (allow requests when Redis is down) because
        # self-inflicted outages (rate limiter blocking all traffic) are
        # worse than briefly allowing unrated traffic. Most production
        # systems (Stripe, Cloudflare) fail-open. Some security-critical
        # systems (banking APIs) fail-closed.

        # Placeholder -- passes all requests through without rate limiting
        return await call_next(request)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Rate Limited API",
    description="Distributed rate limiting demo with Redis-backed token bucket",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)


@app.get("/health")
async def health():
    """Health check endpoint (not rate limited)."""
    return {
        "status": "ok",
        "replica": state.replica_id,
        "requests_served": state.request_count,
    }


@app.get("/api/data")
async def get_data():
    """Example data endpoint (rate limited)."""
    state.request_count += 1
    return {
        "data": {"message": "Hello from the rate-limited API!"},
        "replica": state.replica_id,
        "request_number": state.request_count,
    }


@app.get("/api/expensive")
async def get_expensive():
    """Expensive endpoint that costs 5 tokens (rate limited, higher cost)."""
    state.request_count += 1
    return {
        "data": {"message": "This was an expensive operation!"},
        "replica": state.replica_id,
        "cost": 5,
    }


@app.get("/api/stats")
async def get_stats():
    """Return rate limiting statistics."""
    return {
        "replica": state.replica_id,
        "requests_served": state.request_count,
        "algorithm": RATE_LIMIT_ALGORITHM,
        "capacity": RATE_LIMIT_CAPACITY,
        "refill_rate": RATE_LIMIT_REFILL_RATE,
    }
