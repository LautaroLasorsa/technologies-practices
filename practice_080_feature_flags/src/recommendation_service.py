# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi>=0.115",
#     "uvicorn>=0.34",
#     "httpx>=0.28",
#     "flagsmith>=3.7",
#     "openfeature-sdk>=0.7",
#     "openfeature-provider-flagsmith>=0.2",
# ]
# ///
"""
Recommendation Service -- downstream microservice demonstrating flag context propagation.

This service receives flag evaluation context from the upstream Product API via
HTTP headers, and uses it to evaluate flags locally. This ensures both services
see the same flag values for the same user in the same request chain.

Key learning: in a microservices architecture, flag context must propagate across
service boundaries just like distributed tracing context (OpenTelemetry baggage).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

# Ensure src/ is on the path for sibling module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from flag_client import create_openfeature_client, build_evaluation_context
from hooks import LoggingHook, MetricsHook, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RECO-SVC] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
flag_client = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize OpenFeature client on startup."""
    global flag_client

    flag_client = create_openfeature_client(domain="recommendation-service")

    # Register hooks (same pattern as Product API)
    # TODO(human): Register LoggingHook and MetricsHook on the flag_client.
    # Use: flag_client.add_hooks([LoggingHook(), MetricsHook()])

    logger.info("Recommendation Service ready (OpenFeature + Flagsmith)")
    yield


app = FastAPI(title="Recommendation Service", lifespan=lifespan)


# =============================================================================
# Simulated recommendation data
# =============================================================================

RECOMMENDATIONS = {
    "collaborative": [
        {"id": "r1", "name": "Users like you also bought: Keyboard", "score": 0.95},
        {"id": "r2", "name": "Trending in your segment: Monitor", "score": 0.87},
    ],
    "content_based": [
        {"id": "r3", "name": "Similar to your history: USB Hub", "score": 0.92},
        {"id": "r4", "name": "Based on your preferences: Webcam", "score": 0.85},
    ],
    "hybrid": [
        {"id": "r5", "name": "Best match: Docking Station", "score": 0.97},
        {"id": "r6", "name": "Personalized pick: Headset Stand", "score": 0.91},
        {"id": "r7", "name": "Trending + personal: Cable Organizer", "score": 0.83},
    ],
}


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "recommendation-service"}


@app.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Expose flag evaluation metrics for this service."""
    return metrics.summary()


@app.get("/recommendations")
async def get_recommendations(request: Request) -> JSONResponse:
    """
    Return recommendations using the flag-selected algorithm.

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint is the downstream counterpart to the Product API's
    # /recommendations proxy. It demonstrates:
    #
    # 1. RECEIVING PROPAGATED FLAG CONTEXT: The user's attributes arrive as
    #    X-User-* HTTP headers, set by the upstream Product API. This service
    #    extracts them and builds an EvaluationContext, ensuring it evaluates
    #    flags for the SAME user as the upstream service.
    #
    # 2. LOCAL FLAG EVALUATION: Using the propagated context, this service
    #    evaluates the "recommendation_algorithm" flag locally (via its own
    #    OpenFeature client and Flagsmith provider). The result should be
    #    identical to what the Product API would get for the same user --
    #    because both services use the same Flagsmith environment and the
    #    same targeting_key.
    #
    # 3. CONSISTENT CROSS-SERVICE BEHAVIOR: If the flag says "hybrid" for
    #    user-1 in the Product API, it must also say "hybrid" for user-1 in
    #    the Recommendation Service. This consistency comes from:
    #    a) Same Flagsmith environment (same flag rules)
    #    b) Same targeting_key (propagated via X-User-Id header)
    #    c) Deterministic evaluation (no randomness, only consistent hashing)
    #
    # The propagation pattern:
    #   Client -> Product API (extracts user from auth, sets headers) ->
    #   Recommendation Service (reads headers, builds context, evaluates flags)
    # --------------------------------------------------------------------------

    TODO(human): Implement flag-controlled recommendations with propagated context.

    Steps:
    1. Extract user context from propagated headers:
       - user_id = request.headers.get("X-User-Id", "anonymous")
       - plan = request.headers.get("X-User-Plan", "free")
       - country = request.headers.get("X-User-Country", "US")
       - beta_tester = request.headers.get("X-User-Beta", "false").lower() == "true"
       - email = request.headers.get("X-User-Email", "")

    2. Build EvaluationContext using build_evaluation_context() with those values

    3. Evaluate the "recommendation_algorithm" flag:
       - Use: flag_client.get_string_value("recommendation_algorithm", "collaborative", context)

    4. Look up recommendations from the RECOMMENDATIONS dict using the algorithm value
       - Default to "collaborative" if the algorithm isn't found

    5. Evaluate a boolean flag "show_sponsored" with default False
       - If True, add a sponsored recommendation to the list:
         {"id": "sponsored-1", "name": "SPONSORED: Premium Headphones", "score": 1.0}

    6. Return JSONResponse with:
       {"algorithm": algorithm, "recommendations": recommendations,
        "user_id": user_id, "show_sponsored": show_sponsored_value,
        "service": "recommendation-service"}
    """
    raise NotImplementedError("TODO(human): Implement flag-controlled recommendations with propagated context")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "recommendation_service:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        reload_dirs=["src"],
    )
