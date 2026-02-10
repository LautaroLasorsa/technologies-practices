"""RFC 9457 Problem Details — error handling for the API.

This module defines the structured error format and the FastAPI
exception handler that converts errors into RFC 9457 JSON responses.

References:
    - https://www.rfc-editor.org/rfc/rfc9457.html
    - https://datatracker.ietf.org/doc/html/rfc7807
"""

from typing import Never

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.models import ProblemDetail


# ---------------------------------------------------------------------------
# Problem type URIs (would be real documentation URLs in production)
# ---------------------------------------------------------------------------

PROBLEM_TYPES = {
    400: "https://api.bookshelf.local/problems/bad-request",
    404: "https://api.bookshelf.local/problems/not-found",
    409: "https://api.bookshelf.local/problems/conflict",
    422: "https://api.bookshelf.local/problems/validation-error",
    429: "https://api.bookshelf.local/problems/rate-limit-exceeded",
}

PROBLEM_TITLES = {
    400: "Bad Request",
    404: "Resource Not Found",
    409: "Conflict",
    422: "Validation Error",
    429: "Rate Limit Exceeded",
}


# ---------------------------------------------------------------------------
# raise_problem — starter stub (works, but not RFC 9457 yet)
# ---------------------------------------------------------------------------
#
# TODO(human): Enhance this function to return RFC 9457 ProblemDetail bodies.
#
# Currently it raises a plain HTTPException with a string detail.
# Your task: make it raise an HTTPException whose detail is a ProblemDetail
# dict (using .model_dump()), so the response body follows RFC 9457.
#
# Steps:
#   1. Build a ProblemDetail using PROBLEM_TYPES and PROBLEM_TITLES dicts above.
#      - type = PROBLEM_TYPES.get(status, "about:blank")
#      - title = PROBLEM_TITLES.get(status, "Error")
#      - status = status
#      - detail = detail
#      - instance = instance
#   2. Raise HTTPException with:
#      - status_code = status
#      - detail = the ProblemDetail dict (use .model_dump())
#
# Hint: HTTPException accepts `detail` as Any — you can pass a dict.

def raise_problem(status: int, detail: str, instance: str | None = None) -> Never:
    """Raise an HTTP error. Enhance this to use RFC 9457 ProblemDetail format."""
    raise HTTPException(status_code=status, detail=detail)


# ---------------------------------------------------------------------------
# TODO(human): Implement the exception handler
# ---------------------------------------------------------------------------
#
# Write an async function to register with app.exception_handler(HTTPException).
#
# Signature hint:
#
#   async def problem_detail_handler(request: Request, exc: HTTPException) -> JSONResponse:
#
# Steps:
#   1. Check if exc.detail is a dict (structured ProblemDetail) or a plain string.
#   2. If it's a dict, return JSONResponse with:
#      - status_code = exc.status_code
#      - content = exc.detail
#      - media_type = "application/problem+json"  (the RFC 9457 content type!)
#   3. If it's a plain string (e.g., from FastAPI's default 422 handler),
#      wrap it in a ProblemDetail structure, then return the same JSONResponse.
#
# The media_type "application/problem+json" is what tells clients this is
# a structured RFC 9457 error, not a generic JSON error.
