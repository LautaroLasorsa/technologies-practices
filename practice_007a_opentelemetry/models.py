"""Shared data models for the order system.

Used by both order_api.py and payment_service.py.
"""

from pydantic import BaseModel


class OrderRequest(BaseModel):
    """Incoming order from the client."""

    customer_id: str
    item: str
    quantity: int
    amount: float


class PaymentRequest(BaseModel):
    """Payment request sent from Order API to Payment Service."""

    order_id: str
    customer_id: str
    amount: float


class PaymentResponse(BaseModel):
    """Payment result returned by the Payment Service."""

    order_id: str
    status: str  # "approved" or "declined"
    transaction_id: str
