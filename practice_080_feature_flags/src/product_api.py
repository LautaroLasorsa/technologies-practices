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
Product API -- FastAPI service with feature-flag-controlled behavior.

This service demonstrates the core feature flag patterns:
- Kill switch (maintenance mode)
- Boolean release toggle (new search)
- Multivariate remote config (recommendation algorithm, UI config)
- Percentage-based A/B experiment (checkout flow)
- User-targeted flags (pricing by plan)
- Cross-service flag propagation (calling Recommendation Service)

All flag evaluations go through the OpenFeature SDK, making them
vendor-agnostic. The actual flag management backend is Flagsmith.
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

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from flag_client import create_openfeature_client, build_evaluation_context
from hooks import LoggingHook, MetricsHook, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCT-API] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RECOMMENDATION_SERVICE_URL = os.environ.get(
    "RECOMMENDATION_SERVICE_URL", "http://localhost:8002"
)

# ---------------------------------------------------------------------------
# Application state (initialized during lifespan)
# ---------------------------------------------------------------------------
flag_client = None  # OpenFeature client, set during startup
http_client = None  # httpx async client for calling recommendation service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize OpenFeature client and HTTP client on startup."""
    global flag_client, http_client

    flag_client = create_openfeature_client(domain="product-api")

    # Register hooks on the client for logging and metrics
    # TODO(human): Register the LoggingHook and MetricsHook on the flag_client.
    #
    # -- Exercise Context ----------------------------------------------------------
    # OpenFeature hooks can be registered at three levels:
    # 1. Global (api.add_hooks) -- affects ALL clients
    # 2. Client (client.add_hooks) -- affects this client only
    # 3. Invocation (per flag evaluation call) -- affects single evaluation
    #
    # For this practice, register at the client level so all flag evaluations
    # from the Product API are logged and metricated, but hooks don't leak into
    # other services that might share the same OpenFeature API.
    #
    # Use: flag_client.add_hooks([LoggingHook(), MetricsHook()])
    # --------------------------------------------------------------------------

    http_client = httpx.AsyncClient(base_url=RECOMMENDATION_SERVICE_URL, timeout=5.0)
    logger.info("Product API ready (OpenFeature + Flagsmith)")

    yield

    await http_client.aclose()


app = FastAPI(title="Product API", lifespan=lifespan)


# =============================================================================
# Helper: extract user context from HTTP headers
# =============================================================================

def user_context_from_headers(request: Request) -> dict[str, Any]:
    """Extract user attributes from X-User-* HTTP headers.

    This is the standard pattern for propagating user context in microservices.
    The API gateway (or auth middleware) sets these headers after authentication,
    and downstream services use them for flag evaluation and authorization.
    """
    return {
        "user_id": request.headers.get("X-User-Id", "anonymous"),
        "plan": request.headers.get("X-User-Plan", "free"),
        "country": request.headers.get("X-User-Country", "US"),
        "beta_tester": request.headers.get("X-User-Beta", "false").lower() == "true",
        "email": request.headers.get("X-User-Email", ""),
    }


# =============================================================================
# Simulated product data
# =============================================================================

PRODUCTS = [
    {"id": "p1", "name": "Wireless Headphones", "price": 79.99, "category": "electronics"},
    {"id": "p2", "name": "Python Cookbook", "price": 45.00, "category": "books"},
    {"id": "p3", "name": "Standing Desk", "price": 399.99, "category": "furniture"},
    {"id": "p4", "name": "Mechanical Keyboard", "price": 149.99, "category": "electronics"},
    {"id": "p5", "name": "Noise Cancelling Earbuds", "price": 199.99, "category": "electronics"},
]


def legacy_search(query: str) -> list[dict]:
    """Legacy search: simple substring match on product name."""
    if not query:
        return PRODUCTS
    return [p for p in PRODUCTS if query.lower() in p["name"].lower()]


def new_search(query: str) -> list[dict]:
    """New search: matches on both name AND category (the 'improved' version)."""
    if not query:
        return PRODUCTS
    q = query.lower()
    return [p for p in PRODUCTS if q in p["name"].lower() or q in p["category"].lower()]


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "product-api"}


@app.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Expose flag evaluation metrics (from MetricsHook)."""
    return metrics.summary()


@app.get("/products")
async def get_products(request: Request, q: str = "") -> JSONResponse:
    """
    Get products, with flag-controlled search algorithm and maintenance mode.

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint demonstrates TWO flag patterns:
    #
    # 1. KILL SWITCH (maintenance_mode):
    #    A boolean flag that, when enabled, returns HTTP 503 for ALL requests.
    #    This is the most critical flag pattern in production: it lets you instantly
    #    disable a feature or entire service without redeployment. Every high-risk
    #    feature should ship behind a kill switch.
    #
    #    In practice: during an incident, an on-call engineer opens the Flagsmith
    #    dashboard, flips maintenance_mode to ON, and the API immediately starts
    #    returning 503. No deploy, no CI/CD pipeline, no waiting.
    #
    # 2. RELEASE TOGGLE (new_search_enabled):
    #    A boolean flag that controls which search algorithm to use. The new code
    #    is deployed but hidden behind the flag. When the team is confident the new
    #    search works correctly, they enable the flag. If bugs are found, they
    #    disable it instantly.
    #
    # The flag evaluation uses an EvaluationContext built from the user's HTTP
    # headers. This means different users can see different flag values based on
    # targeting rules configured in Flagsmith.
    # --------------------------------------------------------------------------

    TODO(human): Implement flag-controlled product search.

    Steps:
    1. Extract user context from request headers using user_context_from_headers()
    2. Build an EvaluationContext using build_evaluation_context(**user_context)
    3. KILL SWITCH: Evaluate boolean flag "maintenance_mode" with default False.
       - Use: flag_client.get_boolean_value("maintenance_mode", False, context)
       - If True, return JSONResponse with status 503 and body:
         {"error": "Service is in maintenance mode. Please try again later."}
    4. RELEASE TOGGLE: Evaluate boolean flag "new_search_enabled" with default False.
       - If True, use new_search(q) for the results
       - If False, use legacy_search(q) for the results
    5. Return JSONResponse with:
       {"products": results, "search_engine": "new" or "legacy", "user_id": user_id}
    """
    raise NotImplementedError("TODO(human): Implement kill switch + release toggle for product search")


@app.get("/algorithm")
async def get_recommendation_algorithm(request: Request) -> JSONResponse:
    """
    Get the active recommendation algorithm (multivariate flag).

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint demonstrates MULTIVARIATE FLAGS (remote configuration).
    #
    # Instead of a boolean on/off, the flag returns one of several string values:
    # "collaborative", "content_based", or "hybrid". This is the remote config
    # pattern: you change application behavior by changing the flag value in the
    # dashboard, without any code deployment.
    #
    # Real-world example: Netflix uses this pattern to switch recommendation
    # algorithms per-region. A data scientist changes the flag value for "eu-west"
    # to test a new algorithm, without involving engineers or deployments.
    #
    # The evaluation uses get_string_value() instead of get_boolean_value().
    # The default value ("collaborative") is returned if the flag doesn't exist
    # or evaluation fails -- this is the safe fallback.
    # --------------------------------------------------------------------------

    TODO(human): Implement multivariate flag for algorithm selection.

    Steps:
    1. Extract user context from headers and build EvaluationContext
    2. Evaluate string flag "recommendation_algorithm" with default "collaborative"
       - Use: flag_client.get_string_value("recommendation_algorithm", "collaborative", context)
    3. Return JSONResponse with:
       {"algorithm": algorithm_value, "user_id": user_context["user_id"]}
    """
    raise NotImplementedError("TODO(human): Implement multivariate flag for recommendation algorithm")


@app.get("/ui-config")
async def get_ui_config(request: Request) -> JSONResponse:
    """
    Get UI configuration (JSON multivariate flag / remote config).

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint demonstrates JSON remote configuration via feature flags.
    #
    # The flag "ui_config" stores a JSON string that controls UI behavior:
    # {"theme": "dark", "max_results": 20, "show_banner": true}
    #
    # This pattern is powerful for dynamic configuration: change the theme,
    # adjust pagination limits, toggle UI elements -- all without deployment.
    # The flag value is a string containing JSON, which we parse into a dict.
    #
    # Flagsmith supports storing arbitrary string values in flags. By convention,
    # JSON strings are used for structured remote config. The OpenFeature spec
    # also has get_object_value() for typed object evaluation, but the Flagsmith
    # provider currently maps it through string values.
    # --------------------------------------------------------------------------

    TODO(human): Implement JSON remote config flag.

    Steps:
    1. Extract user context and build EvaluationContext
    2. Define a default config as a JSON string:
       '{"theme": "light", "max_results": 10, "show_banner": false}'
    3. Evaluate string flag "ui_config" with the default JSON string
       - Use: flag_client.get_string_value("ui_config", default_json, context)
    4. Parse the result with json.loads()
    5. Return JSONResponse with:
       {"ui_config": parsed_config, "user_id": user_context["user_id"]}

    Note: Wrap the json.loads() in a try/except. If parsing fails (bad JSON in
    the flag value), fall back to the default config and log a warning.
    """
    raise NotImplementedError("TODO(human): Implement JSON remote config flag")


@app.get("/checkout")
async def get_checkout_flow(request: Request) -> JSONResponse:
    """
    Get checkout flow variant (percentage-based A/B experiment).

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint demonstrates PERCENTAGE-BASED A/B EXPERIMENTS.
    #
    # The flag "checkout_experiment" assigns users to one of three variants:
    # - "control" (current checkout flow) -- 50% of users
    # - "streamlined" (fewer steps) -- 25% of users
    # - "one_click" (single-page checkout) -- 25% of users
    #
    # The flag system uses consistent hashing on the targeting_key (user ID) to
    # assign each user to a variant. This means:
    # - The SAME user always gets the SAME variant (sticky bucketing)
    # - DIFFERENT users are randomly distributed according to percentages
    # - No server-side session state is needed
    #
    # Sticky bucketing is essential for valid experiments: if a user sees the
    # "streamlined" checkout on Monday and "control" on Tuesday, the experiment
    # data is corrupted. The targeting_key ensures consistency.
    #
    # In production, you would log the variant assignment to an analytics pipeline
    # (Segment, Amplitude, BigQuery) and later analyze conversion rates per variant.
    # The MetricsHook tracks variant distribution automatically.
    # --------------------------------------------------------------------------

    TODO(human): Implement A/B experiment with percentage-based rollout.

    Steps:
    1. Extract user context and build EvaluationContext
    2. Evaluate string flag "checkout_experiment" with default "control"
       - Use: flag_client.get_string_value("checkout_experiment", "control", context)
    3. Simulate different checkout flows based on the variant:
       - "control": {"steps": 4, "flow": "multi-page"}
       - "streamlined": {"steps": 2, "flow": "two-step"}
       - "one_click": {"steps": 1, "flow": "single-page"}
    4. Return JSONResponse with:
       {"variant": variant, "checkout_config": config, "user_id": user_context["user_id"]}
    """
    raise NotImplementedError("TODO(human): Implement percentage-based A/B experiment")


@app.get("/pricing")
async def get_pricing_tier(request: Request) -> JSONResponse:
    """
    Get pricing tier (user-targeted flag based on plan attribute).

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint demonstrates USER-TARGETED FLAGS with evaluation context.
    #
    # The flag "premium_features" is evaluated against the user's plan attribute:
    # - Users with plan "enterprise" get premium features enabled
    # - Users with plan "starter" get basic features
    # - Users with plan "free" get minimal features
    #
    # This works because the EvaluationContext carries the user's plan as an
    # attribute. In Flagsmith, you configure "segments" that match on traits:
    # - Segment "enterprise_users": trait "plan" equals "enterprise"
    # - Segment "starter_users": trait "plan" equals "starter"
    #
    # Each segment can have different flag values (segment overrides). The flag
    # system evaluates segments in priority order and returns the first match.
    #
    # This is the PERMISSION TOGGLE pattern: different users see different
    # feature sets based on their subscription level. Unlike percentage rollouts,
    # this is deterministic based on user attributes, not random bucketing.
    # --------------------------------------------------------------------------

    TODO(human): Implement user-targeted flag based on plan.

    Steps:
    1. Extract user context and build EvaluationContext
    2. Evaluate string flag "premium_features" with default "basic"
       - Use: flag_client.get_string_value("premium_features", "basic", context)
    3. Define feature sets for each tier:
       - "premium": {"analytics": True, "api_rate_limit": 10000, "support": "priority"}
       - "standard": {"analytics": True, "api_rate_limit": 1000, "support": "email"}
       - "basic": {"analytics": False, "api_rate_limit": 100, "support": "community"}
    4. Look up the feature set by the flag value (default to "basic" tier)
    5. Return JSONResponse with:
       {"tier": tier_value, "features": feature_set, "user_id": user_context["user_id"],
        "plan": user_context["plan"]}
    """
    raise NotImplementedError("TODO(human): Implement user-targeted flag for pricing tiers")


@app.get("/recommendations")
async def get_recommendations(request: Request) -> JSONResponse:
    """
    Get recommendations by calling the Recommendation Service with flag context propagation.

    # -- Exercise Context ----------------------------------------------------------
    # This endpoint demonstrates MULTI-SERVICE FLAG PROPAGATION.
    #
    # When the Product API calls the Recommendation Service, it needs to forward
    # the user's flag context (user ID, plan, country, etc.) in HTTP headers.
    # This ensures the downstream service evaluates flags for the SAME user,
    # giving consistent behavior across the request chain.
    #
    # Without propagation, the Recommendation Service would evaluate flags with
    # no user context (anonymous), potentially returning different flag values
    # than the Product API -- breaking the user experience.
    #
    # The pattern: upstream service extracts user context from its request headers,
    # then includes that context as X-User-* headers when calling downstream services.
    # This is analogous to distributed tracing context propagation (W3C Trace Context,
    # OpenTelemetry baggage).
    # --------------------------------------------------------------------------

    TODO(human): Implement cross-service flag context propagation.

    Steps:
    1. Extract user context from request headers using user_context_from_headers()
    2. Build propagation headers to forward to the Recommendation Service:
       headers = {
           "X-User-Id": user_context["user_id"],
           "X-User-Plan": user_context["plan"],
           "X-User-Country": user_context["country"],
           "X-User-Beta": str(user_context["beta_tester"]).lower(),
           "X-User-Email": user_context["email"],
       }
    3. Make an HTTP GET request to the Recommendation Service:
       response = await http_client.get("/recommendations", headers=headers)
    4. Return the Recommendation Service's response as JSON, wrapped with:
       {"source": "product-api-proxy", "downstream_response": response.json(),
        "user_id": user_context["user_id"]}
    5. Handle errors: if the downstream call fails, return a fallback response
       with status 200 and body:
       {"source": "product-api-fallback", "error": str(e),
        "recommendations": [], "user_id": user_context["user_id"]}
    """
    raise NotImplementedError("TODO(human): Implement cross-service flag context propagation")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "product_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=["src"],
    )
