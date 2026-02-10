"""Rate limiting middleware.

Implements per-client rate limiting using a fixed-window counter strategy.
Returns standard rate-limit headers on every response and 429 when exceeded.

References:
    - https://restfulapi.net/rest-api-rate-limit-guidelines/
    - https://www.ietf.org/archive/id/draft-polli-ratelimit-headers-02.html
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.models import ProblemDetail


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RATE_LIMIT = 30          # max requests per window
WINDOW_SECONDS = 60      # window duration in seconds


# ---------------------------------------------------------------------------
# TODO(human): Implement RateLimitMiddleware
# ---------------------------------------------------------------------------
#
# Create a class that extends BaseHTTPMiddleware and overrides:
#
#   async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
#
# Strategy: Fixed-window counter per client IP.
#
# You'll need:
#   - A dict to track per-IP state: {ip: {"count": int, "window_start": float}}
#   - On each request:
#       1. Get client IP from request.client.host
#       2. Determine the current window (using time.time() and WINDOW_SECONDS)
#       3. If the IP's window has expired, reset count to 0 and update window_start
#       4. Increment count
#       5. If count > RATE_LIMIT:
#          - Return a 429 JSONResponse with RFC 9457 ProblemDetail body
#          - Set "Retry-After" header (seconds until window resets)
#          - Content-Type: "application/problem+json"
#       6. Otherwise, call `response = await call_next(request)` to proceed
#       7. Add these headers to EVERY response (even successful ones):
#          - X-RateLimit-Limit: {RATE_LIMIT}
#          - X-RateLimit-Remaining: {remaining}
#          - X-RateLimit-Reset: {unix_timestamp_when_window_resets}
#       8. Return the response
#
# Hints:
#   - time.time() gives current unix timestamp as float
#   - Window resets at: window_start + WINDOW_SECONDS
#   - remaining = max(0, RATE_LIMIT - count)
#   - For the 429 body, use ProblemDetail(
#         type="https://api.bookshelf.local/problems/rate-limit-exceeded",
#         title="Rate Limit Exceeded",
#         status=429,
#         detail=f"Rate limit of {RATE_LIMIT} requests per {WINDOW_SECONDS}s exceeded.",
#         instance=str(request.url.path),
#     ).model_dump()
#
# Note: This in-memory approach works for learning. In production you'd use
# Redis or a similar shared store for distributed rate limiting.


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter per client IP."""

    def __init__(self, app):
        super().__init__(app)
        # TODO(human): initialize the per-IP tracking dict here
        pass

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # TODO(human): implement the rate limiting logic described above
        response = await call_next(request)
        return response
