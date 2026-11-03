"""Exercise 1: Explicit CORS Rules, Preflight Caching & Fetch Metadata Inspection.

This module implements HTTP security boundaries before request processing reaches endpoints.
"""
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse

# Trusted origin for the victim SPA
TRUSTED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforces CORS policies and inspects Sec-Fetch-Site metadata."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin")
        sec_fetch_site = request.headers.get("sec-fetch-site")

        # -------------------------------------------------------------------------
        # Exercise 1 - Task A: Handle CORS Preflight (OPTIONS)
        # -------------------------------------------------------------------------
        # When the browser encounters a complex request (e.g. Content-Type: application/json
        # or a custom header like X-CSRF-Token), it issues an HTTP OPTIONS preflight.
        #
        # TODO(human): Handle preflight requests properly:
        # 1. If request.method == "OPTIONS":
        #    a. Check if the 'origin' is in TRUSTED_ORIGINS.
        #    b. If valid, return a 204 No Content response with:
        #       - "Access-Control-Allow-Origin": origin (never "*" when credentials=true!)
        #       - "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        #       - "Access-Control-Allow-Headers": "Content-Type, X-CSRF-Token, Authorization"
        #       - "Access-Control-Allow-Credentials": "true"
        #       - "Access-Control-Max-Age": "86400" (caches preflight for 24h)
        #    c. If not trusted origin, return 403 Forbidden.
        #
        # NAIVE / UNHARDENED STARTER (Vulnerable to cross-origin abuse or lacking headers):
        if request.method == "OPTIONS":
            # [TODO(human): Replace naive preflight with strict trusted-origin verification]
            headers = {
                "Access-Control-Allow-Origin": origin or "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
            return Response(status_code=200, headers=headers)

        # -------------------------------------------------------------------------
        # Exercise 1 - Task B: Validate Sec-Fetch-Site & Origin on Mutating Requests
        # -------------------------------------------------------------------------
        # Sec-Fetch-Site is an automated browser metadata header:
        # - "same-origin": Initiated from http://localhost:8115 itself
        # - "same-site": Initiated from a sibling domain
        # - "cross-site": Initiated from another domain (e.g., http://localhost:6660)
        #
        # TODO(human): Inspect mutating requests (POST, PUT, DELETE, PATCH):
        # 1. If request.method not in SAFE_METHODS:
        #    a. If sec_fetch_site == "cross-site":
        #       return PlainTextResponse("Blocked: Cross-site request forbidden by Sec-Fetch-Site", status_code=403)
        #    b. If origin is present and origin not in TRUSTED_ORIGINS:
        #       return PlainTextResponse("Blocked: Untrusted Origin", status_code=403)
        #
        # [TODO(human): Insert Sec-Fetch-Site and Origin verification here]

        # Process the downstream request
        response = await call_next(request)

        # -------------------------------------------------------------------------
        # Exercise 1 - Task C: Add Explicit CORS Response Headers
        # -------------------------------------------------------------------------
        # TODO(human): Ensure that responses returned to trusted origins include:
        # - "Access-Control-Allow-Origin": origin
        # - "Access-Control-Allow-Credentials": "true"
        # - "Vary": "Origin" (instructs HTTP caches that responses differ by Origin)
        #
        # NAIVE / UNHARDENED STARTER:
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response
