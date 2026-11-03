"""FastAPI Treasury Backend (Port 8115) for Practice 115a.

Demonstrates SOP, CORS, CSRF, and CSP behaviors in an authentic banking environment.
"""
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from security_middleware import SecurityMiddleware
from csrf import CSRFMiddleware, generate_csrf_token, CSRF_COOKIE_NAME
from csp import CSPMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("treasury-backend")

app = FastAPI(title="Treasury Banking Backend", version="1.0.0")

# Register security middlewares
app.add_middleware(CSPMiddleware)
app.add_middleware(CSRFMiddleware)
app.add_middleware(SecurityMiddleware)

# In-memory banking state
DEFAULT_BALANCE = 10000.0
bank_state = {
    "user": "Alice Cooper (Executive Treasury)",
    "session_id": "alice_vault_session_8844",
    "balance": DEFAULT_BALANCE,
    "transactions": [
        {"id": 1, "recipient": "Payroll Department", "amount": 2500.0, "status": "COMPLETED"},
        {"id": 2, "recipient": "Cloud Hosting Provider", "amount": 420.0, "status": "COMPLETED"},
    ],
    "comments": [
        {
            "id": 1,
            "author": "Security Officer Bob",
            "author_url": "https://company.internal/bob",
            "body": "Quarterly audit completed. <b>All accounts in balance.</b>",
        },
        {
            "id": 2,
            "author": "Audit Analyst Carol",
            "author_url": "https://audit.internal/carol",
            "body": "Verified reserve ratios against <i>compliance standards</i>.",
        },
    ],
    "csp_reports": [],
}


class TransferRequest(BaseModel):
    recipient: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0)


class CommentRequest(BaseModel):
    author: str = Field(..., min_length=1, max_length=100)
    author_url: str = Field(..., max_length=500)
    body: str = Field(..., max_length=2000)


@app.get("/")
async def root():
    return {
        "service": "Treasury Banking Backend",
        "status": "online",
        "docs": "/docs",
        "endpoints": ["/api/me", "/api/transfer", "/api/comments", "/api/csrf-token", "/api/csp-report"],
    }


@app.get("/api/me")
async def get_current_user(request: Request, response: Response):
    """Returns profile and sets ambient authentication cookie and CSRF token."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = bank_state["session_id"]
        # Set ambient session cookie (simulating legitimate user login)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,   # Protected from JS reads (XSS cookie theft)
            samesite="lax",  # Default modern browser behavior
            path="/",
        )

    # Issue or refresh Anti-CSRF token cookie (readable by SPA JavaScript)
    csrf_token = generate_csrf_token(session_id)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,  # Legitimate SPA needs to read this to attach X-CSRF-Token
        samesite="lax",
        path="/",
    )

    return {
        "user": bank_state["user"],
        "balance": bank_state["balance"],
        "transactions": bank_state["transactions"],
        "csrf_token": csrf_token,
    }


@app.get("/api/csrf-token")
async def get_csrf_token(request: Request, response: Response):
    """Dedicated endpoint to retrieve and set Anti-CSRF token."""
    session_id = request.cookies.get("session_id", bank_state["session_id"])
    token = generate_csrf_token(session_id)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,
        samesite="lax",
        path="/",
    )
    return {"csrf_token": token}


@app.post("/api/transfer")
async def transfer_funds(req: TransferRequest, request: Request):
    """Executes a financial transfer. Highly sensitive state-mutating endpoint!"""
    logger.info(
        "Transfer request received: recipient=%s, amount=%.2f, Origin=%s, Cookie=%s",
        req.recipient,
        req.amount,
        request.headers.get("origin"),
        request.cookies.get("session_id"),
    )

    if bank_state["balance"] < req.amount:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Insufficient funds", "balance": bank_state["balance"]},
        )

    bank_state["balance"] -= req.amount
    tx = {
        "id": len(bank_state["transactions"]) + 1,
        "recipient": req.recipient,
        "amount": req.amount,
        "status": "COMPLETED",
    }
    bank_state["transactions"].append(tx)

    return {
        "message": f"Successfully transferred ${req.amount:.2f} to {req.recipient}",
        "new_balance": bank_state["balance"],
        "transaction": tx,
    }


@app.get("/api/comments")
async def list_comments():
    """Lists audit comments."""
    return bank_state["comments"]


@app.post("/api/comments")
async def add_comment(req: CommentRequest):
    """Adds a new audit comment. Demonstrates client-side XSS sinks."""
    new_comment = {
        "id": len(bank_state["comments"]) + 1,
        "author": req.author,
        "author_url": req.author_url,
        "body": req.body,
    }
    bank_state["comments"].append(new_comment)
    return {"message": "Comment recorded", "comment": new_comment}


@app.post("/api/csp-report")
async def receive_csp_report(request: Request):
    """Receives CSP violation telemetry from browsers."""
    try:
        report = await request.json()
        logger.warning("🚨 CSP VIOLATION DETECTED: %s", report)
        bank_state["csp_reports"].append(report)
        return {"status": "recorded"}
    except Exception as exc:
        logger.error("Failed to parse CSP report: %s", exc)
        return {"status": "error"}


@app.get("/api/csp-reports")
async def list_csp_reports():
    """Lists received CSP reports."""
    return bank_state["csp_reports"]


@app.post("/api/reset")
async def reset_state(response: Response):
    """Resets the bank state for reproducible testing."""
    bank_state["balance"] = DEFAULT_BALANCE
    bank_state["transactions"] = bank_state["transactions"][:2]
    bank_state["comments"] = bank_state["comments"][:2]
    bank_state["csp_reports"].clear()
    return {"message": "State reset successfully", "balance": bank_state["balance"]}
