"""
Saga Orchestrator -- The central coordinator of the distributed transaction.

This is the CORE of the practice. The orchestrator:
1. Receives a SAGA_START event from the order service
2. Drives the saga forward by sending commands (reserve inventory, process payment)
3. Listens for success/failure events from each service
4. On failure, triggers compensating transactions in reverse order
5. Tracks saga state via a state machine

Think of it like a workflow engine: it knows the sequence of steps, manages
transitions, and handles rollback.
"""

from __future__ import annotations

import logging
from dataclasses import field

from shared.events import (
    MessageType,
    OrderSaga,
    SagaMessage,
    SagaState,
    OrderData
)
import datetime
logger = logging.getLogger(__name__)


class SagaOrchestrator:
    """
    Manages all active saga instances and their state transitions.

    Each saga is identified by saga_id and tracked in the `sagas` dict.
    """

    def __init__(self) -> None:
        self.sagas: dict[str, OrderSaga] = {}

    async def handle_event(self, message: SagaMessage) -> list[SagaMessage]:
        """
        Process an incoming event and decide the next action.

        This is the STATE MACHINE -- the most important method in the practice.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches orchestration-based saga coordination. The state
        # machine is the heart of the SAGA pattern: it sequences forward steps,
        # detects failures, and triggers compensations. Understanding this explicit
        # coordination logic is key to debugging and extending sagas in production.
        # Production systems often persist this state machine to a database for
        # crash recovery and replay.
        # ──────────────────────────────────────────────────────────────────────

        TODO(human): Implement the full state machine logic.

        The orchestrator should handle these incoming message types:

        1. SAGA_START:
           - Create a new OrderSaga instance and store it in self.sagas
           - Transition state to RESERVING_INVENTORY
           - Return a RESERVE_INVENTORY command

        2. INVENTORY_RESERVED:
           - Transition state to PROCESSING_PAYMENT
           - Return a PROCESS_PAYMENT command

        3. INVENTORY_RESERVE_FAILED:
           - No compensation needed (nothing to undo yet)
           - Transition directly to FAILED
           - Return a SAGA_FAILED event to order.status topic

        4. PAYMENT_PROCESSED:
           - Transition state to COMPLETED
           - Return a SAGA_COMPLETED event

        5. PAYMENT_FAILED:
           - Start compensation: transition to COMPENSATING_INVENTORY
           - Return a RELEASE_INVENTORY command (undo the reservation)

        6. INVENTORY_RELEASED (compensation complete):
           - Transition to FAILED
           - Return a SAGA_FAILED event

        For each transition:
        - Look up the saga by saga_id (from message.saga_id)
        - Update saga.state
        - Log the transition
        - Return the appropriate command/event as a SagaMessage

        Hint: Use SagaMessage.create(MessageType.XXX, saga_id, payload)
              The payload should carry the order data (message.payload).

        Returns:
            List of SagaMessages to publish (commands to saga.commands,
            or status events to order.status).
        """
        # TODO(human): Implement the state machine
        new_state: SagaState
        next_message : MessageType

        match message.message_type:
            case MessageType.SAGA_START.value:
                self._create_saga(message.saga_id,message.payload)
                new_state = SagaState.INVENTORY_STARTED
                next_message = MessageType.RESERVE_INVENTORY
            case MessageType.INVENTORY_RESERVED.value:
                new_state = SagaState.PAYMENT_STARTED
                next_message = MessageType.PROCESS_PAYMENT

            case MessageType.INVENTORY_RESERVE_FAILED.value:
                new_state = SagaState.FAILED
                next_message = MessageType.SAGA_FAILED

            case MessageType.PAYMENT_PROCESSED.value:
                new_state = SagaState.COMPLETED
                next_message = MessageType.SAGA_COMPLETED

            case MessageType.PAYMENT_FAILED.value:
                new_state = SagaState.INVENTORY_COMPENSATION_STARTED
                next_message = MessageType.RELEASE_INVENTORY

            case MessageType.INVENTORY_RELEASED.value:
                new_state = SagaState.FAILED
                next_message = MessageType.SAGA_FAILED

            case _: raise ValueError(f"Unexpected message: {message.message_type}")

        self._transition(self.sagas[message.saga_id], new_state)
        return [SagaMessage.create(next_message, message.saga_id, message.payload)]

    def _create_saga(self, saga_id: str, payload: dict) -> OrderSaga:
        """
        Create and register a new saga instance.

        TODO(human): Instantiate an OrderSaga with the provided saga_id
        and payload, store it in self.sagas, and return it.
        """
        order_saga = OrderSaga(
            order_details = OrderData(**payload),
            saga_id=saga_id,
            timestamp = datetime.datetime.now()
        )
        self.sagas[saga_id] = order_saga
        return order_saga

    def _transition(self, saga: OrderSaga, new_state: SagaState) -> None:
        """
        Transition a saga to a new state.

        TODO(human): Update saga.state to new_state and append the
        transition to the saga's history for audit trail.
        Log the old state -> new state transition.
        """
        logger.info(f"Saga {saga.saga_id} from {saga.saga_state} to {new_state}")
        saga.saga_state = new_state
        saga.states_history.append(new_state)
