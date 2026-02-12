"""
Order Service -- Accepts HTTP orders and triggers sagas.

This service has two responsibilities:
1. HTTP API: Accepts POST /orders requests and publishes SAGA_START events
2. Status consumer: Listens on order.status for saga completion/failure
   and updates the in-memory order store.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import (
    GROUP_ORDER_SERVICE,
    TOPIC_ORDER_STATUS,
    TOPIC_SAGA_EVENTS,
)
from shared.events import MessageType, OrderData, SagaMessage
from shared.kafka_utils import (
    consume_loop,
    create_producer,
    publish,
    wait_for_redpanda,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORDER-SVC] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# In-memory order store (simulates a database)
# =============================================================================

orders: dict[str, dict] = {}


# =============================================================================
# Request / Response models
# =============================================================================

class CreateOrderRequest(BaseModel):
    customer_id: str
    item: str
    quantity: int
    price: float


class OrderResponse(BaseModel):
    order_id: str
    saga_id: str
    status: str
    customer_id: str
    item: str
    quantity: int
    price: float


# =============================================================================
# Status consumer (background task)
# =============================================================================

async def status_consumer() -> None:
    """
    Background consumer that listens for saga completion/failure.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches event-driven status updates. The order service
    # doesn't poll the orchestrator; it reacts to saga lifecycle events. This
    # asynchronous, event-driven pattern is foundational for scalable microservices:
    # services are decoupled and can scale independently.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement the handler that updates order status.

    When a SAGA_COMPLETED message arrives:
    - Look up the order by saga_id in the `orders` dict
    - Update its status to "completed"
    - Log the completion

    When a SAGA_FAILED message arrives:
    - Look up the order by saga_id
    - Update its status to "failed"
    - Log the failure

    Hint: message.saga_id is the key in the `orders` dict.
          message.message_type tells you which event it is.
    """

    async def handle_status(message: SagaMessage) -> list[SagaMessage] | None:
        # TODO(human): Implement status update logic
        raise NotImplementedError("Implement handle_status()")

    await consume_loop(
        topic=TOPIC_ORDER_STATUS,
        group_id=GROUP_ORDER_SERVICE,
        handler=handle_status,
    )


# =============================================================================
# FastAPI app with lifespan
# =============================================================================

producer = None  # set during startup


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start Kafka producer and background status consumer on app startup."""
    global producer

    await wait_for_redpanda()
    producer = await create_producer()
    logger.info("Order service ready")

    # Run status consumer in background
    consumer_task = asyncio.create_task(status_consumer())

    yield

    consumer_task.cancel()
    await producer.stop()


app = FastAPI(title="Order Service", lifespan=lifespan)


# =============================================================================
# HTTP endpoints
# =============================================================================

@app.post("/orders", response_model=OrderResponse, status_code=201)
async def create_order(request: CreateOrderRequest) -> OrderResponse:
    """
    Create a new order and start a saga.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches saga initiation from a synchronous HTTP request.
    # The HTTP call returns immediately (201 Created) without waiting for saga
    # completion. This fire-and-forget pattern is critical for avoiding long-lived
    # HTTP connections and ensuring the API remains responsive under load.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement the order creation and saga initiation.

    Steps:
    1. Generate a unique order_id and saga_id (use str(uuid.uuid4()))
    2. Create an OrderData instance from the request
    3. Store the order in the `orders` dict with status "pending"
       (store: order_id, saga_id, status, and all order fields)
    4. Create a SAGA_START message with the order data as payload
    5. Publish it to TOPIC_SAGA_EVENTS using the `publish()` helper
    6. Return an OrderResponse

    Hint: OrderData has a .to_dict() method for the message payload.
          The saga_id is used as the correlation key across all services.
    """
    # TODO(human): Implement order creation and saga start
    raise NotImplementedError("Implement create_order()")


@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str) -> OrderResponse:
    """Look up an order by ID."""
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse(**orders[order_id])


@app.get("/orders")
async def list_orders() -> list[OrderResponse]:
    """List all orders."""
    return [OrderResponse(**o) for o in orders.values()]
