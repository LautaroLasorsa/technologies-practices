"""Exercise 2: Double-Submit Cookie & Cryptographic Anti-CSRF Token Defense.

This module provides server-side CSRF token generation, cookie issuing,
and constant-time verification for state-mutating requests.
"""
import hmac
import hashlib
import secrets
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

# Secret key used for signing / verifying HMAC CSRF tokens
CSRF_SECRET = "super-secret-bank-csrf-signing-key-32chars"
CSRF_COOKIE_NAME = "csrf_token"
CSRF_HEADER_NAME = "x-csrf-token"

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def generate_csrf_token(session_id: str) -> str:
    """Generates a cryptographically strong Double-Submit / HMAC Anti-CSRF Token.

    Format: <random_nonce>.<hmac_signature>
    The signature binds the token to the user's session_id to prevent token reuse.
    """
    # -------------------------------------------------------------------------
    # Exercise 2 - Task A: Cryptographic Token Generation
    # -------------------------------------------------------------------------
    # TODO(human):
    # 1. Generate a 16-byte random hex nonce using `secrets.token_hex(16)`.
    # 2. Compute an HMAC-SHA256 signature using `CSRF_SECRET`:
    #    message = f"{nonce}:{session_id}".encode("utf-8")
    #    signature = hmac.new(CSRF_SECRET.encode(), message, hashlib.sha256).hexdigest()
    # 3. Return f"{nonce}.{signature}"
    #
    # NAIVE / UNHARDENED STARTER (Returns dummy predictable token):
    return "dummy-insecure-csrf-token"


def verify_csrf_token(token: Optional[str], session_id: str, cookie_token: Optional[str]) -> bool:
    """Verifies that the provided header token is valid and matches the cookie token.

    Uses secrets.compare_digest to prevent timing attacks.
    """
    # -------------------------------------------------------------------------
    # Exercise 2 - Task B: Constant-Time Token & Signature Verification
    # -------------------------------------------------------------------------
    # TODO(human):
    # 1. Check if token or cookie_token is missing. If so, return False.
    # 2. In the Double-Submit pattern, compare header token with cookie_token:
    #    if not secrets.compare_digest(token, cookie_token): return False
    # 3. Split token into nonce and signature:
    #    nonce, _, signature = token.partition(".")
    # 4. Recompute expected HMAC signature:
    #    expected_msg = f"{nonce}:{session_id}".encode("utf-8")
    #    expected_sig = hmac.new(CSRF_SECRET.encode(), expected_msg, hashlib.sha256).hexdigest()
    # 5. Verify using secrets.compare_digest(signature, expected_sig).
    #
    # NAIVE / UNHARDENED STARTER (Always accepts requests - vulnerable to CSRF!):
    # [TODO(human): Replace with strict constant-time HMAC check]
    return True


class CSRFMiddleware(BaseHTTPMiddleware):
    """Enforces Anti-CSRF token verification on state-mutating requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exclude safe methods from CSRF verification
        if request.method in SAFE_METHODS:
            return await call_next(request)

        # Allow CSP report telemetry endpoint without CSRF
        if request.url.path == "/api/csp-report":
            return await call_next(request)

        session_id = request.cookies.get("session_id", "anonymous")
        header_token = request.headers.get(CSRF_HEADER_NAME)
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)

        # -------------------------------------------------------------------------
        # Exercise 2 - Task C: Middleware Token Enforcement
        # -------------------------------------------------------------------------
        # TODO(human):
        # 1. Call verify_csrf_token(header_token, session_id, cookie_token).
        # 2. If verification fails, return JSONResponse(
        #        status_code=403,
        #        content={"error": "Forbidden", "detail": "Invalid or missing Anti-CSRF token"}
        #    )
        #
        # NAIVE / UNHARDENED STARTER (Allows all mutating requests through!):
        is_valid = verify_csrf_token(header_token, session_id, cookie_token)
        if not is_valid:
            return JSONResponse(
                status_code=403,
                content={"error": "CSRF verification failed", "detail": "Invalid or missing Anti-CSRF token"},
            )

        return await call_next(request)
