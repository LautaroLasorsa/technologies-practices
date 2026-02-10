"""FastAPI application entry point.

Configures the app with:
- Custom OpenAPI metadata (title, description, version, tags)
- API versioning via URL prefix (/v1/)
- Route registration
- Startup event to seed sample data
- Middleware registration

Run with: uv run uvicorn app.main:app --reload
Docs at:  http://localhost:8000/docs (Swagger UI)
          http://localhost:8000/redoc (ReDoc)
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app import store
from app.middleware import RateLimitMiddleware
from app.models import HealthResponse
from app.routes import authors, books


# ---------------------------------------------------------------------------
# OpenAPI customization
# ---------------------------------------------------------------------------

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Service health check.",
    },
    {
        "name": "Authors",
        "description": "Manage authors. Each author can have many books.",
    },
    {
        "name": "Books",
        "description": (
            "Manage books. Supports pagination, filtering, sorting, "
            "HATEOAS links, and idempotent creation."
        ),
    },
]


# ---------------------------------------------------------------------------
# Application lifespan (seed data on startup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store.seed()
    yield


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bookshelf API",
    description=(
        "A REST API demonstrating professional design principles: "
        "resource naming, HTTP semantics, RFC 9457 errors, "
        "pagination, HATEOAS, rate limiting, and idempotency."
    ),
    version="1.0.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(RateLimitMiddleware)


# ---------------------------------------------------------------------------
# TODO(human): Wire the RFC 9457 exception handler
# ---------------------------------------------------------------------------
#
# After implementing problem_detail_handler in app/errors.py, register it here:
#
#   from fastapi.exceptions import HTTPException
#   from app.errors import problem_detail_handler
#   app.exception_handler(HTTPException)(problem_detail_handler)
#
# This replaces FastAPI's default error format with RFC 9457 Problem Details.


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(authors.router)
app.include_router(books.router)


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
    operation_id="healthCheck",
)
async def health() -> HealthResponse:
    return HealthResponse()
