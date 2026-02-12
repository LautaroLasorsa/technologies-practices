# Practice 014: SAGA Pattern

## Technologies

- **SAGA Pattern** -- Distributed transaction management without 2PC (two-phase commit)
- **Redpanda** -- Kafka-compatible event streaming platform (lightweight, no JVM/ZooKeeper)
- **aiokafka** -- Async Python Kafka client (works with Redpanda's Kafka API)
- **FastAPI** -- HTTP endpoints for triggering orders and inspecting state
- **Docker Compose** -- Multi-service orchestration

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### SAGA Pattern: Distributed Transactions Without Locks

The SAGA pattern solves the distributed transaction problem without distributed locks or two-phase commit (2PC). In a microservices architecture where each service owns its database, traditional ACID transactions spanning multiple services are infeasible. A SAGA breaks a long-running business transaction into a sequence of local transactions, each published as an event. If any step fails, compensating transactions undo the effects of previous steps in reverse order. This achieves eventual consistency instead of strong consistency.

SAGAs emerged from Hector Garcia-Molina's 1987 paper "Sagas" and became mainstream with microservices adoption. The pattern trades immediate consistency for availability and partition tolerance (CAP theorem). Unlike 2PC—which requires all participants to lock resources during a voting phase, causing high latency and blocking under failures—SAGAs use **compensating transactions** (semantic undo) that execute asynchronously. For example, if an order saga reserves inventory, processes payment, and ships the product, but payment fails, the saga runs compensations: cancel shipment (if started) and release inventory reservation. These compensations are business-logic rollbacks, not database rollbacks.

Two coordination strategies exist: **orchestration** and **choreography**. In **orchestration** (used in this practice), a central coordinator (the orchestrator) sends commands to services and listens for replies, explicitly managing the state machine. In **choreography**, services listen for events and autonomously decide their actions, with no central coordinator. Orchestration provides centralized observability and easier debugging (saga state is tracked in one place), but introduces a single point of failure. Choreography is more decentralized and resilient but harder to reason about (saga logic is implicit across services) and prone to cyclic dependencies if not carefully designed.

**Key concepts**:

| Concept | Description |
|---------|-------------|
| **Compensating Transaction** | A semantic undo operation that reverses a committed local transaction (e.g., release reserved inventory) |
| **Forward Recovery** | Retry failed steps until they succeed (assumes transient failures) |
| **Backward Recovery** | Execute compensating transactions to undo completed steps (used when forward recovery is not possible) |
| **Idempotency** | Handlers must produce the same result when called multiple times (critical for at-least-once event delivery) |
| **Saga State** | Tracks progress through the transaction sequence (STARTED → RESERVING_INVENTORY → PROCESSING_PAYMENT → COMPLETED) |
| **Pivot Transaction** | The point of no return; after this step, the saga must complete forward (no compensations can run) |

**Ecosystem context**: SAGA implementations exist in frameworks like Temporal (workflow-as-code, supports both orchestration and choreography), Camunda (BPMN-based orchestration), Eventuate Tram (Java event-driven sagas), and NServiceBus (.NET message-based sagas). Cloud-native solutions include AWS Step Functions (state machine orchestration), Azure Durable Functions (orchestrator functions), and Google Cloud Workflows. Trade-offs: dedicated workflow engines (Temporal, Camunda) provide rich features (timers, versioning, retries) but add operational complexity. Lightweight patterns (like this practice) using Kafka/Redpanda are simpler but require manual state management. SAGAs shine when failures are rare and compensations are well-defined; for high contention or complex rollback logic, consider re-modeling bounded contexts to avoid distributed transactions altogether.

## Description

Build an **Order Processing System** with three microservices (Order, Payment, Inventory) coordinated by a central **Saga Orchestrator** via Redpanda events. The system demonstrates:

1. **Orchestration-based SAGA** -- A central orchestrator drives the transaction sequence, sending commands and listening for replies.
2. **Compensating transactions** -- When a step fails (e.g., payment declined), previous steps are undone in reverse order (e.g., release reserved inventory).
3. **Idempotent handlers** -- Each service processes commands idempotently, so retries are safe.
4. **State machine** -- The orchestrator tracks saga state transitions (PENDING -> RESERVING_INVENTORY -> PROCESSING_PAYMENT -> COMPLETED / COMPENSATING -> FAILED).

### Architecture

```
                         ┌──────────────┐
   POST /orders ───────> │ Order Service│ ──> saga.commands ──> ┌──────────────┐
                         └──────┬───────┘                      │ Orchestrator │
                                │                              └──────┬───────┘
                         saga.events                                  │
                                │                           saga.commands
                    ┌───────────┴──────────┐                         │
                    │                      │              ┌──────────┴──────────┐
              ┌─────┴─────┐         ┌──────┴──────┐      │                     │
              │ Inventory │         │   Payment   │ <────┘              saga.events
              │  Service  │         │   Service   │ ─────────────────────────>│
              └───────────┘         └─────────────┘
```

**Happy path:** Order Created -> Reserve Inventory -> Process Payment -> Order Completed
**Failure path (payment fails):** Order Created -> Reserve Inventory -> Process Payment (FAIL) -> Release Inventory (compensate) -> Order Failed

### What you'll learn

1. **SAGA pattern fundamentals** -- Why distributed transactions need sagas instead of 2PC
2. **Orchestration vs choreography** -- Trade-offs between centralized and event-driven coordination
3. **Compensating transactions** -- How to "undo" completed steps when a later step fails
4. **State machine design** -- Modeling saga lifecycle as explicit state transitions
5. **Idempotent event handlers** -- Ensuring at-least-once delivery doesn't cause duplicates
6. **Event-driven architecture** -- Commands vs events, topic design, message schemas

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Start Redpanda and Redpanda Console with `docker compose up -d`
2. Verify Redpanda is running: visit `http://localhost:8080` (Console UI)
3. Initialize each service with `uv sync`
4. Key question: Why can't we use a single database transaction across microservices? What problem does SAGA solve that 2PC doesn't?

### Phase 2: Event Schemas & Shared Models (~15 min)

1. Review the shared event models in `shared/events.py`
2. **User implements:** Complete the `SagaState` enum with all saga lifecycle states
3. **User implements:** Complete the `OrderSaga` dataclass that tracks saga progress
4. Key question: Why do we separate "commands" (do this) from "events" (this happened)?

### Phase 3: Order Service (~15 min)

1. Review the FastAPI order service skeleton
2. **User implements:** The `POST /orders` endpoint that creates an order and publishes a `SagaStarted` command to Redpanda
3. **User implements:** The event consumer that listens for saga completion/failure and updates order status
4. Key question: Should the order service wait synchronously for the saga to complete? Why or why not?

### Phase 4: Saga Orchestrator (~25 min) -- Core of the practice

1. Review the orchestrator skeleton with its state machine
2. **User implements:** The `handle_event()` method -- the state machine that decides the next step based on current state and incoming event
3. **User implements:** The `compensate()` method -- reverse-order compensation when a step fails
4. **User implements:** The Redpanda consumer/producer loop that reads events and emits commands
5. Key question: What happens if the orchestrator crashes mid-saga? How would you make it recoverable?

### Phase 5: Inventory & Payment Services (~20 min)

1. Review the service skeletons
2. **User implements:** Inventory service -- `reserve_inventory` and `release_inventory` (compensating) handlers
3. **User implements:** Payment service -- `process_payment` and `refund_payment` (compensating) handlers
4. Simulate failures: Payment service rejects orders above $500
5. Key question: Why must compensating transactions be idempotent?

### Phase 6: End-to-End Test (~15 min)

1. Start all services: `docker compose up`
2. Submit a valid order (< $500) -- verify happy path completes
3. Submit a failing order (> $500) -- verify compensation runs
4. Check Redpanda Console to trace the full event flow
5. Discussion: How would choreography differ from this orchestration approach? What are the trade-offs?

## Motivation

- **Microservices essential**: SAGA is the standard pattern for distributed transactions in microservice architectures -- required knowledge for any backend engineer working with distributed systems
- **Event-driven design**: Understanding commands, events, and compensating transactions is foundational for event sourcing, CQRS, and reactive architectures
- **Production relevance**: Systems like order processing, payment flows, and booking systems all use saga patterns in production (Uber, Netflix, Airbnb)
- **Complements practices 003 (Kafka) and 015 (CQRS)**: Builds on event streaming knowledge and leads into event sourcing

## References

- [Microservices.io: Saga Pattern](https://microservices.io/patterns/data/saga.html)
- [Microsoft Azure: Saga Design Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/saga)
- [Baeldung: Saga Pattern in Microservices](https://www.baeldung.com/cs/saga-pattern-microservices)
- [ByteByteGo: Saga Pattern Demystified](https://blog.bytebytego.com/p/saga-pattern-demystified-orchestration)
- [Redpanda Quickstart](https://docs.redpanda.com/current/get-started/quick-start/)
- [Redpanda Python Tutorial](https://www.redpanda.com/blog/python-redpanda-kafka-api-tutorial)
- [aiokafka Documentation](https://aiokafka.readthedocs.io/en/stable/)

## Commands

### Phase 1: Infrastructure Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Redpanda broker, Console UI, and create Kafka topics |
| `docker compose up -d redpanda console init-topics` | Start only Redpanda infrastructure (without app services) |
| `docker compose ps` | Check status of all containers |
| `docker compose logs redpanda` | View Redpanda broker logs |
| `docker compose logs init-topics` | Verify topic creation completed successfully |

### Phase 1: Dependency Installation (per service, for local dev)

| Command | Description |
|---------|-------------|
| `cd order_service && uv sync` | Install order service dependencies |
| `cd orchestrator && uv sync` | Install orchestrator dependencies |
| `cd inventory_service && uv sync` | Install inventory service dependencies |
| `cd payment_service && uv sync` | Install payment service dependencies |

### Phase 3-5: Run Services Locally (each in a separate terminal)

| Command | Description |
|---------|-------------|
| `cd order_service && uv run uvicorn order_service.main:app --host 0.0.0.0 --port 8001` | Start order service (HTTP API on port 8001) |
| `cd orchestrator && uv run python -m orchestrator.main` | Start saga orchestrator (event consumer/producer) |
| `cd inventory_service && uv run python -m inventory_service.main` | Start inventory service (event consumer/producer) |
| `cd payment_service && uv run python -m payment_service.main` | Start payment service (event consumer/producer) |

### Phase 6: Run All via Docker Compose

| Command | Description |
|---------|-------------|
| `docker compose up` | Start all services with logs in foreground (Redpanda + all 4 app services) |
| `docker compose up -d` | Start all services in background |
| `docker compose up --build` | Rebuild Docker images and start all services |
| `docker compose logs -f order-service` | Follow order service logs |
| `docker compose logs -f orchestrator` | Follow orchestrator logs |
| `docker compose logs -f inventory-service` | Follow inventory service logs |
| `docker compose logs -f payment-service` | Follow payment service logs |

### Phase 6: End-to-End Testing (curl)

| Command | Description |
|---------|-------------|
| `curl -X POST http://localhost:8001/orders -H "Content-Type: application/json" -d "{\"customer_id\": \"cust-1\", \"item\": \"laptop\", \"quantity\": 1, \"price\": 200}"` | Submit a valid order (< $500, happy path) |
| `curl -X POST http://localhost:8001/orders -H "Content-Type: application/json" -d "{\"customer_id\": \"cust-2\", \"item\": \"laptop\", \"quantity\": 1, \"price\": 600}"` | Submit a failing order (> $500, triggers compensation) |
| `curl http://localhost:8001/orders` | List all orders and their statuses |
| `curl http://localhost:8001/orders/{order_id}` | Get a specific order by ID |

### Inspection & Cleanup

| Command | Description |
|---------|-------------|
| Open `http://localhost:8080` in browser | Redpanda Console -- inspect topics, messages, consumer groups |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop, remove containers, and delete Redpanda data volume |

## State

`not-started`
