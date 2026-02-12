# Practice 002: Cloud Pub/Sub

## Technologies

- **Google Cloud Pub/Sub** — Managed asynchronous messaging service (at-least-once delivery, MVCC)
- **google-cloud-pubsub** — Official Python client library
- **Pub/Sub Emulator** — Local Docker emulator (no GCP credentials needed)

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

Google Cloud Pub/Sub is a fully-managed asynchronous messaging service designed for building event-driven systems and decoupling microservices. It provides a serverless, horizontally scalable message queue with guaranteed **at-least-once delivery** semantics and global replication across Google's infrastructure. Unlike self-hosted systems (Kafka, RabbitMQ), Pub/Sub handles all infrastructure concerns — you pay per message without managing brokers, partitions, or rebalancing.

### Architecture

Pub/Sub follows a **topic-subscription model**:
1. **Publishers** send messages to a **topic** (a named resource)
2. Each **subscription** to that topic receives a copy of every message (fan-out)
3. **Subscribers** pull messages from their subscription and explicitly acknowledge them

Messages are stored redundantly across multiple zones. When a subscriber acknowledges a message, it's marked as processed for that subscription only — other subscriptions still see it. If a message isn't acknowledged within the **ack deadline** (default 10s, configurable up to 600s), Pub/Sub automatically redelivers it.

### Delivery Semantics

Pub/Sub guarantees **at-least-once delivery**: a message will be delivered at least once, but may be delivered multiple times (e.g., if ack is lost in transit). This requires subscribers to be **idempotent** — processing the same message twice must produce the same result. Pub/Sub does not guarantee exactly-once delivery or ordering by default, but provides **ordering keys** to enable per-key ordering within a subscription.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Topic** | Named resource where publishers send messages. Decouples producers from consumers. |
| **Subscription** | Represents a stream of messages from a topic. Multiple subscriptions = fan-out. |
| **Message** | Data payload + optional attributes (metadata key-value pairs). |
| **Acknowledgment (ACK)** | Subscriber confirms successful processing. Removes message from subscription queue. |
| **NACK** | Negative acknowledgment. Redelivers message immediately (or after backoff). |
| **Ack Deadline** | Time window for ack/nack. After expiration, message is redelivered. |
| **Ordering Key** | Optional string to ensure messages with the same key are delivered in order. |
| **Dead-Letter Topic** | Destination for messages that fail repeatedly (after max retries). |

### Push vs Pull Subscriptions

Pub/Sub supports two subscription modes:
- **Pull**: Subscriber calls `pull()` to fetch messages (synchronous or streaming). Gives subscriber control over flow.
- **Push**: Pub/Sub posts messages to a subscriber's HTTP endpoint. Simpler but requires a public webhook.

This practice uses **pull** (both synchronous and streaming) since it works locally without exposing endpoints.

### Ecosystem Context

Pub/Sub competes with AWS SNS/SQS (similar managed model, AWS-only), Apache Kafka (self-hosted, higher throughput, replay capability, more operational overhead), and RabbitMQ (self-hosted, AMQP protocol, lower latency for small messages). Choose Pub/Sub when you need **zero operational overhead**, **global scale**, and **serverless billing**. Choose Kafka when you need **replay semantics**, **total ordering**, or **higher throughput** (Kafka can handle millions of messages/sec per broker). Pub/Sub's managed nature makes it ideal for cloud-native applications on GCP, while Kafka dominates data-intensive pipelines and on-prem deployments.

## Description

Build an **Order Processing System** that demonstrates core Pub/Sub patterns: publish/subscribe, fan-out, message ordering, acknowledgment semantics, and dead-letter queues — all running locally against the Pub/Sub emulator in Docker.

### What you'll learn

1. **Pub/Sub core model** — topics, subscriptions, publishers, subscribers
2. **Delivery semantics** — at-least-once delivery, ACK/NACK, ack deadlines, idempotency
3. **Pull vs streaming pull** — synchronous pull vs callback-based streaming pull
4. **Fan-out pattern** — one topic, multiple subscriptions (each gets all messages)
5. **Ordering keys** — guaranteed order per key within a subscription
6. **Dead-letter topics** — routing undeliverable messages after max retries
7. **Pub/Sub vs Kafka** — when to choose each (managed vs self-hosted, throughput, retention)

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Set up Docker Compose with the Pub/Sub emulator (`gcr.io/google.com/cloudsdktool/google-cloud-cli:emulators`)
2. Initialize Python project with `uv`, install `google-cloud-pubsub`
3. Understand the Pub/Sub model: topics → subscriptions → subscribers
4. Key question: How does Pub/Sub's "at-least-once" delivery differ from "exactly-once"? Why does this matter for your subscriber code?

### Phase 2: Publisher & Basic Subscriber (~30 min)

1. Create a topic (`orders`) and a pull subscription (`inventory-sub`)
2. **User implements:** Publisher function that sends order messages with attributes (order_id, item, quantity)
3. **User implements:** Synchronous pull subscriber that receives and ACKs messages
4. Test: publish 5 orders, verify all received
5. Key question: What happens if you don't call `message.ack()`?

### Phase 3: Streaming Pull & Fan-Out (~30 min)

1. Create a second subscription (`notification-sub`) on the same `orders` topic
2. **User implements:** Streaming pull subscriber with callback for notifications
3. **User implements:** Fan-out test — publish 5 orders, verify both subscribers receive all messages
4. Simulate slow processing — observe redelivery after ack_deadline
5. Key question: Why does fan-out use separate subscriptions, not separate consumers on the same subscription?

### Phase 4: Ordering Keys & Dead-Letter (~30 min)

1. Create a subscription with `enable_message_ordering=True`
2. **User implements:** Publisher with ordering keys (group orders by customer_id)
3. Verify: messages with same key arrive in order
4. Create a dead-letter topic and subscription with `max_delivery_attempts=3`
5. **User implements:** Subscriber that NACKs specific "poison" messages to trigger dead-letter routing
6. Verify: poison messages appear in dead-letter subscription after 3 retries
7. Key question: What happens to ordering when a message with an ordering key is NACKed?

### Phase 5: Discussion & Comparison (~10 min)

1. Compare Pub/Sub vs Kafka: when would you choose each?
2. Discuss: How would you handle deduplication with at-least-once delivery?
3. Discuss: Push subscriptions vs pull — when is each appropriate?

## Motivation

- **Cloud-native messaging**: Pub/Sub is a core GCP service used in event-driven architectures, data pipelines, and microservice communication
- **Pattern literacy**: Fan-out, dead-letter queues, and ordering guarantees are universal messaging concepts (transferable to Kafka, RabbitMQ, SQS)
- **Production readiness**: Understanding ACK semantics, idempotency, and poison message handling prevents common production bugs
- **Complements Kafka practice (003)**: Understanding both managed and self-hosted messaging enables informed architecture decisions

## References

- [Pub/Sub Overview — Google Cloud](https://cloud.google.com/pubsub/docs/overview)
- [Pub/Sub Emulator Setup](https://cloud.google.com/pubsub/docs/emulator)
- [Python Client Library](https://docs.cloud.google.com/python/docs/reference/pubsub/latest)
- [Ordering Keys](https://cloud.google.com/pubsub/docs/ordering)
- [Dead-Letter Topics](https://docs.cloud.google.com/pubsub/docs/dead-letter-topics)
- [Pub/Sub vs Kafka Comparison](https://systemdesignschool.io/blog/pub-sub-vs-kafka)

## Commands

All commands run from `practice_002_cloud_pubsub/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start the Pub/Sub emulator container (gRPC server on port 8085) |
| `docker compose down` | Stop and remove the emulator container |
| `docker compose logs -f` | Stream emulator logs (useful for debugging connection issues) |

### Project Setup

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from `pyproject.toml` into `.venv` |
| `cd app && uv run python main.py` | Create all topics and subscriptions in the emulator, then print next steps |
| `cd app && uv run python setup_resources.py` | Create topics/subscriptions only (same as `main.py` but without the banner) |

### Phase 2: Publisher & Synchronous Pull

| Command | Description |
|---------|-------------|
| `cd app && uv run python publisher.py` | Publish 5 sample orders (basic, no ordering) + 5 orders with ordering keys |
| `cd app && uv run python subscriber_pull.py` | Pull and ACK messages synchronously from `inventory-sub` (up to 3 rounds) |

### Phase 3: Streaming Pull & Fan-Out

| Command | Description |
|---------|-------------|
| `cd app && uv run python subscriber_streaming.py` | Start a streaming pull listener on `notification-sub` (30s timeout, Ctrl+C to stop) |
| `cd app && uv run python test_fanout.py` | Publish 3 test orders, then verify both `inventory-sub` and `notification-sub` received all of them |

### Phase 4: Ordering Keys & Dead-Letter

| Command | Description |
|---------|-------------|
| `cd app && uv run python subscriber_deadletter.py` | Listen on `ordered-sub`, NACK "Monitor" orders (poison), then check `dead-letter-sub` for routed messages |

## Notes

- **User observation:** Pub/Sub's ordered publish returns independent futures per ordering key — handling failures as values (`str | Exception`) instead of raising preserves the batch length invariant, mirroring Rust's `Result<T, E>` / monadic error handling pattern. This is preferable in batch/pipeline scenarios where you want to resolve all futures before propagating errors.
- **Streaming pull timeout as control flow:** `StreamingPullFuture.result(timeout=N)` raises `TimeoutError` by design — the future never resolves because the gRPC stream is infinite. The exception is the intended shutdown signal, not a failure. Same pattern as `KeyboardInterrupt`.
- **Batch ack vs per-message ack:** Sync pull batches ack_ids and calls `acknowledge()` once (fewer RPCs, but crash = full batch redelivery). Streaming pull acks per-message inside callbacks because callbacks run on a thread pool concurrently — batching wouldn't make sense.
- **Dead-letter bypass:** Pub/Sub has no API to send a message directly to the DLQ. The server counts delivery attempts. Workaround: ack the message and publish it to the dead-letter topic yourself — faster and allows adding `failure_reason` metadata. The built-in dead-letter policy is a safety net for unexpected failures.
- **From discussion (deduplication):** User identified three strategies: (1) message ID + Redis idempotency store (standard but has race conditions), (2) partition/shard-based dedup with single consumer per key (industry standard in Kafka, Pub/Sub equivalent is ordering keys), (3) idempotent operations (`INSERT ON CONFLICT DO NOTHING`). Strategy 2 eliminates coordination overhead entirely.
- **From discussion (push vs pull):** User's initial model (push=OLTP, pull=OLAP) captures latency-vs-throughput, but the real axis is **who controls flow** — push means Pub/Sub drives pace (serverless, webhooks), pull means consumer drives pace (batch, backpressure).

## State

`completed`
