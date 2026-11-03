"""Exercise 4: Content Security Policy (CSP) & Clickjacking Defense Middleware.

This module injects strict Content-Security-Policy headers into responses
and provides diagnostic hooks for browser violation telemetry.
"""
from typing import Callable, Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Default CSP directives dictionary
DEFAULT_CSP_DIRECTIVES: Dict[str, List[str]] = {
    "default-src": ["'self'"],
    "script-src": ["'self'"],
    "style-src": ["'self'", "'unsafe-inline'"],
    "img-src": ["'self'", "data:"],
    "connect-src": ["'self'", "http://localhost:8115", "http://localhost:5173"],
    "frame-ancestors": ["'none'"],  # Blocks framing in <iframe> (Clickjacking protection)
    "report-uri": ["/api/csp-report"],
}


def build_csp_header(directives: Dict[str, List[str]]) -> str:
    """Serializes a dictionary of CSP directives into an HTTP header string.

    Format: "directive1 value1 value2; directive2 value1; ..."
    """
    # -------------------------------------------------------------------------
    # Exercise 4 - Task A: Build CSP Header String
    # -------------------------------------------------------------------------
    # TODO(human):
    # 1. Iterate over directives.items():
    #    join each directive and its values with spaces: f"{directive} {' '.join(values)}"
    # 2. Join the directive statements with semicolons: "; "
    # 3. Return the resulting policy string.
    #
    # NAIVE / UNHARDENED STARTER (Empty policy - offers no browser protection):
    return ""


class CSPMiddleware(BaseHTTPMiddleware):
    """Injects Content-Security-Policy and framing protection headers into responses."""

    def __init__(self, app, directives: Dict[str, List[str]] = None):
        super().__init__(app)
        self.directives = directives or DEFAULT_CSP_DIRECTIVES

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # -------------------------------------------------------------------------
        # Exercise 4 - Task B: Inject CSP and Defensive Headers
        # -------------------------------------------------------------------------
        # TODO(human):
        # 1. Build CSP string using build_csp_header(self.directives).
        # 2. If the CSP string is non-empty, inject header:
        #    response.headers["Content-Security-Policy"] = csp_string
        # 3. Also inject legacy defense-in-depth headers:
        #    response.headers["X-Frame-Options"] = "DENY"
        #    response.headers["X-Content-Type-Options"] = "nosniff"
        #
        # NAIVE / UNHARDENED STARTER:
        csp_string = build_csp_header(self.directives)
        if csp_string:
            response.headers["Content-Security-Policy"] = csp_string
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"

        return response
