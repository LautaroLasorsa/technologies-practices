# Practice 015: CQRS & Event Sourcing

## Technologies

- **CQRS** (Command Query Responsibility Segregation) --- Separate write and read models optimized for their concerns
- **Event Sourcing** --- Persist state changes as immutable domain events instead of mutable current state
- **Redpanda** --- Kafka-compatible streaming platform (event bus between command and query sides)
- **FastAPI** --- Async HTTP API for commands (writes) and queries (reads)
- **aiokafka** --- Async Python Kafka client (producer/consumer against Redpanda)
- **SQLite** --- Lightweight event store (append-only) and read model (projections)
- **Redpanda Console** --- Web UI for inspecting topics and messages

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### CQRS & Event Sourcing: Separated Concerns for Complex Domains

CQRS (Command Query Responsibility Segregation) and Event Sourcing are complementary patterns often used together. **CQRS** separates write operations (commands) from read operations (queries), using independent models optimized for each concern. The write model enforces business invariants and produces events; the read model consumes events to build denormalized views optimized for queries. This separation enables independent scaling (read-heavy workloads can scale reads without burdening writes), polyglot persistence (different databases for write and read sides), and specialized optimizations (write model uses normalized relational schema, read model uses NoSQL document store).

**Event Sourcing** stores state changes as immutable events instead of mutable current state. Every state mutation is recorded as a domain event (e.g., `AccountOpened`, `MoneyDeposited`). The current state is derived by replaying all events from the beginning (or from a snapshot). This design provides a complete audit trail (every historical state is queryable), temporal queries (what was the balance on January 15th?), event-driven integration (other systems consume events), and natural support for undo/replay (rebuild state after a bug fix by replaying events through corrected logic).

The aggregate is the consistency boundary in Event Sourcing. An aggregate receives commands, validates them against current state (derived from replayed events), and emits new events. The `apply()` method is a pure state transition function: `apply(event) → new state`. This method must be side-effect-free and idempotent (replaying the same events twice produces the same state). Commands validate business rules and emit events; `apply()` only mutates state. Separating command validation from event application is critical: validation rules may change over time, but old events must remain replayable without modification.

CQRS+ES introduces eventual consistency: after a command executes (persisting events to the write-side event store), there's a brief delay before the read-side projection updates. This is acceptable for many business domains (e.g., banking: your balance might lag by milliseconds, but consistency is guaranteed eventually). For strong consistency requirements (e.g., preventing double-spending), the write-side enforces invariants synchronously; reads can tolerate staleness.

**Key concepts**:

| Concept | Description |
|---------|-------------|
| **Command** | An intent to change state (imperative, validated before execution, can be rejected) |
| **Event** | A fact that state changed (past tense, immutable, always accepted once persisted) |
| **Aggregate** | Consistency boundary that validates commands and emits events |
| **Event Store** | Append-only log of domain events (the source of truth) |
| **Projection** | A read model built by consuming and transforming events (optimized for queries) |
| **Rehydration** | Rebuilding aggregate state by replaying its event history |
| **Optimistic Concurrency** | Detect concurrent modifications using event version numbers (prevent lost updates) |

**Ecosystem context**: Event-sourcing frameworks include Axon (Java, with Axon Server event store), EventStoreDB (dedicated event store with projections), Marten (.NET, PostgreSQL as event store), and Akka Persistence (JVM, actor-based ES). Cloud-native solutions: AWS DynamoDB Streams (change data capture), Azure Cosmos DB change feed, Google Cloud Firestore listeners. Trade-offs: Event Sourcing adds complexity (event versioning, schema evolution, replay performance) and storage overhead (full event history). It shines in domains with complex business logic, auditing requirements, or frequent state queries across time (financial systems, healthcare, order management). For simple CRUD apps with no audit or temporal queries, traditional CRUD with database snapshots suffices. CQRS without Event Sourcing is also viable (separate read/write models, but store current state instead of events).

## Description

Build a **Bank Account System** that demonstrates CQRS and Event Sourcing end-to-end:

- **Command side**: receives commands (OpenAccount, Deposit, Withdraw), validates them against the aggregate, appends domain events to an event store (SQLite), and publishes events to Redpanda.
- **Query side**: consumes events from Redpanda, updates read-model projections (account balances, transaction history) in a separate SQLite database.
- **Two separate FastAPI services**: one for commands, one for queries --- illustrating the physical separation of read and write models.

### What you'll learn

1. **Event Sourcing fundamentals** --- events as the source of truth, aggregate rehydration by replaying events
2. **CQRS separation** --- independent write model (event store) and read model (projections), each optimized for its purpose
3. **Domain events** --- designing events that capture business intent (`AccountOpened`, `MoneyDeposited`, `MoneyWithdrawn`)
4. **Aggregates** --- `BankAccount` aggregate that enforces invariants (no overdraft, positive deposit) before emitting events
5. **Projections** --- consuming events to build queryable read models (balance view, transaction log)
6. **Eventual consistency** --- the read model lags behind writes; understanding and embracing this tradeoff
7. **Event bus** --- Redpanda as the bridge between command and query services

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Start infrastructure: `docker compose up -d` (Redpanda + Console)
2. Initialize Python project: `cd app && uv sync`
3. Verify Redpanda Console at `http://localhost:8080`
4. Review the domain model: `BankAccount` aggregate, three event types
5. Key question: Why store events instead of just the current balance? What can you do with an event log that you cannot do with a snapshot?

### Phase 2: Event Store & Aggregate (~30 min)

1. Review the `EventStore` class (SQLite append-only table)
2. **User implements:** `BankAccount.apply()` --- mutate aggregate state based on each event type
3. **User implements:** `BankAccount.open_account()`, `deposit()`, `withdraw()` --- validate business rules, emit events
4. **User implements:** `BankAccount.load()` class method --- rehydrate aggregate by replaying events from the store
5. Test: open an account, deposit, withdraw, verify balance by replaying events
6. Key question: What happens if you replay the same events twice? Why must `apply()` be a pure state transition?

### Phase 3: Command Handlers & Event Publishing (~20 min)

1. Review the command handler scaffold (receives HTTP commands, coordinates aggregate + event store)
2. **User implements:** `handle_open_account()` --- create aggregate, persist events, publish to Redpanda
3. **User implements:** `handle_deposit()` and `handle_withdraw()` --- load aggregate, execute command, persist + publish
4. Test via `curl` or the provided test script
5. Key question: Why persist to the event store *before* publishing to Redpanda? What could go wrong if you reverse the order?

### Phase 4: Projections & Query Service (~25 min)

1. Review the event consumer scaffold (reads from Redpanda, updates projections)
2. **User implements:** `project_account_opened()` --- insert a new row in the balance projection
3. **User implements:** `project_money_deposited()` and `project_money_withdrawn()` --- update balance, append to transaction log
4. **User implements:** Query endpoints --- `GET /accounts/{id}` (balance), `GET /accounts/{id}/transactions` (history)
5. Observe eventual consistency: command returns immediately, query reflects the change after a brief delay
6. Key question: How would you handle a projection that crashes mid-update? What guarantees does consumer offset tracking give you?

### Phase 5: End-to-End & Discussion (~15 min)

1. Run the full test script: open accounts, deposit, withdraw, query balances and history
2. Inspect events in Redpanda Console (topic `bank-events`)
3. Discuss: How would you add a new projection (e.g., "daily summary") without changing the command side?
4. Discuss: How would you handle event schema evolution (adding fields to events)?
5. Discuss: When is CQRS+ES overkill? When does it shine?

## Motivation

- **Architectural pattern literacy**: CQRS and Event Sourcing are foundational patterns in event-driven microservices, used at scale in banking, e-commerce, and logistics
- **Audit trail by design**: Event sourcing gives you a complete, immutable history --- critical for financial systems and compliance
- **Scalability understanding**: Separating reads and writes allows independent scaling and optimization of each side
- **Complements Practice 014 (SAGA)**: Together, SAGA + CQRS/ES form the backbone of event-driven distributed systems
- **Industry demand**: These patterns appear frequently in senior backend and systems design interviews

## References

- [Event Sourcing --- Martin Fowler](https://martinfowler.com/eaaDev/EventSourcing.html)
- [CQRS --- Martin Fowler](https://martinfowler.com/bliki/CQRS.html)
- [Event Sourcing Pattern --- Microsoft](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing)
- [CQRS Pattern --- Microsoft](https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs)
- [Event Sourcing in Python (eventsourcing library)](https://eventsourcing.readthedocs.io/)
- [CQRS and Event Sourcing --- microservices.io](https://microservices.io/patterns/data/event-sourcing.html)
- [Redpanda Docker Compose](https://docs.redpanda.com/redpanda-labs/docker-compose/single-broker/)
- [aiokafka Documentation](https://aiokafka.readthedocs.io/)
- [Why a bank account is not the best example of Event Sourcing?](https://event-driven.io/en/bank_account_event_sourcing/) (good nuance on when ES fits)

## Commands

### Phase 1: Infrastructure Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Redpanda broker and Console UI |
| `docker compose ps` | Check status of Redpanda containers |
| `docker compose logs redpanda` | View Redpanda broker logs |
| `cd app && uv sync` | Install Python dependencies |

### Phase 2-3: Run Command API (write side)

| Command | Description |
|---------|-------------|
| `cd app && uv run uvicorn command_api:app --port 8001` | Start the command API (write model) on port 8001 |

### Phase 4: Run Query API (read side, in separate terminal)

| Command | Description |
|---------|-------------|
| `cd app && uv run uvicorn query_api:app --port 8002` | Start the query API (read model + event consumer) on port 8002 |

### Phase 3-4: Manual Testing (curl)

| Command | Description |
|---------|-------------|
| `curl -X POST http://localhost:8001/commands/open-account -H "Content-Type: application/json" -d "{\"owner_name\": \"Alice\", \"initial_balance\": 1000}"` | Open a new bank account |
| `curl -X POST http://localhost:8001/commands/deposit -H "Content-Type: application/json" -d "{\"account_id\": \"<id>\", \"amount\": 250, \"description\": \"Salary\"}"` | Deposit money into an account |
| `curl -X POST http://localhost:8001/commands/withdraw -H "Content-Type: application/json" -d "{\"account_id\": \"<id>\", \"amount\": 75, \"description\": \"Groceries\"}"` | Withdraw money from an account |
| `curl http://localhost:8002/accounts` | List all accounts with balances (query side) |
| `curl http://localhost:8002/accounts/{account_id}` | Get a single account balance (query side) |
| `curl http://localhost:8002/accounts/{account_id}/transactions` | Get transaction history for an account (query side) |
| `curl http://localhost:8001/health` | Health check for command API |
| `curl http://localhost:8002/health` | Health check for query API |

### Phase 5: End-to-End Test Script

| Command | Description |
|---------|-------------|
| `cd app && uv run python test_e2e.py` | Run full end-to-end test (requires both APIs running) |

### Inspection & Cleanup

| Command | Description |
|---------|-------------|
| Open `http://localhost:8080` in browser | Redpanda Console -- inspect `bank-events` topic and messages |
| `docker compose down` | Stop and remove Redpanda containers |
| `docker compose down -v` | Stop, remove containers, and delete Redpanda data |

## State

`not-started`
