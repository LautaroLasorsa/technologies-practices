# Tier 4 — Specialized System Design Problems

Modern, niche, and domain-specific problems. For targeted interview prep: HFT, ML infrastructure, event-driven systems.

---

## Problem 16: Design an Order Matching Engine (HFT/Exchange)

**Tests:** Ultra-low latency, lock-free data structures, deterministic execution, FIFO fairness
**Scale:** 1M orders/sec, < 10μs matching latency, zero message loss, deterministic replay

### Key Questions
- Data structure for the order book? (sorted price levels, FIFO queue per level)
- Lock-free vs single-threaded event loop — which for lowest latency?
- Matching algorithms: price-time priority (FIFO) vs pro-rata — when each?
- How do you handle market orders vs limit orders vs stop orders?
- Deterministic replay: how do you replay the exact sequence for auditing?
- How do you handle network jitter / message ordering from multiple gateways?
- Market data dissemination: how do you fan-out price updates to subscribers?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Order book data structure: -->


<!-- Matching algorithm: -->


<!-- Architecture (single-threaded vs multi-threaded): -->


<!-- Message sequencing + deterministic replay: -->


<!-- Market data fan-out: -->


<!-- Failure recovery (engine crash, how to restore state): -->


---

## Problem 17: Design an LLM Serving Infrastructure

**Tests:** GPU resource management, batching, KV cache, multi-tenancy, cost optimization
**Scale:** 10K concurrent requests, 100+ models, < 500ms time-to-first-token, multi-region

### Key Questions
- How do you batch requests to maximize GPU utilization? (continuous batching)
- KV cache management: memory pressure vs latency — when do you evict?
- Model routing: how do you decide which GPU/instance serves a request?
- Multi-tenancy: how do you isolate customers while sharing GPU resources?
- Auto-scaling: GPU instances are expensive and slow to start — how do you right-size?
- Long-context requests: how do you handle requests that don't fit in one GPU's memory?
- Streaming responses: how do you deliver tokens as they're generated?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Request routing + load balancing: -->


<!-- Batching strategy: -->


<!-- High-level design: -->


<!-- KV cache management: -->


<!-- Multi-tenancy isolation: -->


<!-- Auto-scaling strategy: -->


<!-- Cost optimization (spot instances, model sharing): -->


---

## Problem 18: Design an Event Sourcing + CQRS System

**Tests:** Event store design, projections, eventual consistency, replay, schema evolution
**Scale:** 100K events/sec, years of retention, < 50ms read latency, multiple read models

### Key Questions
- Event store: append-only log — which storage? (Kafka, EventStoreDB, PostgreSQL)
- How do you build and maintain read projections (materialized views)?
- Projection rebuild: a new read model needs all historical events — how fast?
- Schema evolution: how do you handle event schema changes over years of data?
- Snapshotting: when and how to snapshot aggregate state to avoid replaying all events?
- Exactly-once projection processing: consumer crash mid-batch — how do you recover?
- How do you query across aggregates? (projections, cross-aggregate views)

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Event store design: -->


<!-- Write path (command → event): -->


<!-- High-level design: -->


<!-- Projection/read model strategy: -->


<!-- Schema evolution approach: -->


<!-- Snapshotting strategy: -->


<!-- Consistency between write + read models: -->


---

## Problem 19: Design a Real-Time Fraud Detection System

**Tests:** Stream processing, ML inference at scale, feature engineering, low latency, false positive management
**Scale:** 10K transactions/sec, < 100ms decision latency, < 0.1% false positive rate

### Key Questions
- What features do you compute in real-time vs batch? (velocity, geo, device fingerprint)
- How do you combine rule-based and ML-based detection?
- Feature store: how do you serve pre-computed features with < 10ms latency?
- Model serving: how do you run inference inline without blocking the transaction?
- How do you handle model updates (A/B testing, shadow scoring, gradual rollout)?
- Feedback loop: how do disputed charges improve the model?
- Explainability: why was this transaction flagged? (regulatory requirement)

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Feature engineering pipeline (real-time + batch): -->


<!-- Detection architecture (rules + ML): -->


<!-- High-level design: -->


<!-- Model serving + low-latency inference: -->


<!-- Feedback loop + model retraining: -->


<!-- Explainability strategy: -->


---

## Problem 20: Design a Distributed Transaction Coordinator (Saga)

**Tests:** Saga pattern, compensating transactions, idempotency, orchestration vs choreography
**Scale:** 100K distributed transactions/day across 5-10 services, < 5s end-to-end

### Key Questions
- Orchestration (central coordinator) vs choreography (event-driven) — when each?
- Compensating transactions: how do you "undo" a payment if shipping fails?
- Idempotency: a compensating action is retried — how do you ensure it's safe?
- Timeout handling: step 3 of 5 hasn't responded in 30s — what happens?
- Observability: how do you track the state of a distributed transaction across services?
- Partial failures: 3 of 5 steps succeeded, 4th failed — state is inconsistent. Recovery?
- How does this compare to 2PC? When would you use 2PC instead of Saga?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Saga pattern choice (orchestration vs choreography): -->


<!-- Step definition + compensating actions: -->


<!-- High-level design: -->


<!-- Idempotency implementation per step: -->


<!-- Timeout + retry policy: -->


<!-- Observability (tracking saga state): -->


<!-- 2PC vs Saga trade-off analysis: -->
