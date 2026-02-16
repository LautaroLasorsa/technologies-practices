"""Exercise 5: Protocol Comparison -- MQTT vs Kafka vs RabbitMQ vs Cloud Pub/Sub.

Demonstrates:
  - Structured analysis of messaging protocols across key dimensions
  - Code-driven comparison framework producing a formatted report
  - Trade-off analysis for architecture decisions

This exercise produces a comparison report by implementing analysis
functions for each dimension. The analysis is code + knowledge, not
just documentation -- each function computes or structures data that
feeds into the final report.

Run:
    uv run python protocol_comparison.py
"""

from dataclasses import dataclass, field


# ── Data model ──────────────────────────────────────────────────────


@dataclass
class ProtocolProfile:
    """Complete profile of a messaging protocol across all dimensions."""

    name: str
    category: str  # e.g., "IoT Pub/Sub", "Event Streaming"
    overhead: "OverheadAnalysis | None" = None
    delivery: "DeliveryAnalysis | None" = None
    scalability: "ScalabilityAnalysis | None" = None
    use_cases: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class OverheadAnalysis:
    """Protocol overhead breakdown."""

    min_header_bytes: int
    typical_message_bytes: int  # For a ~50-byte payload
    connection_setup: str  # Description of connection setup cost
    keepalive_mechanism: str
    bandwidth_efficiency: str  # "high", "medium", "low"


@dataclass
class DeliveryAnalysis:
    """Delivery guarantee analysis."""

    guarantees: list[str]  # List of supported guarantee levels
    default_guarantee: str
    exactly_once: bool
    exactly_once_mechanism: str  # How exactly-once is achieved (or "N/A")
    message_ordering: str  # Description of ordering guarantees
    persistence_model: str  # How messages are persisted


@dataclass
class ScalabilityAnalysis:
    """Scalability and load balancing analysis."""

    consumer_group_mechanism: str
    max_throughput: str  # Approximate throughput description
    horizontal_scaling: str  # How to scale consumers
    partitioning: str  # Partitioning model
    replay_capability: bool
    replay_mechanism: str


@dataclass
class UseCaseScenario:
    """A scenario for protocol recommendation."""

    name: str
    description: str
    requirements: list[str]
    recommended_protocol: str
    reasoning: str
    runner_up: str
    runner_up_reasoning: str


# ── TODO(human): Implement these functions ───────────────────────────


def analyze_overhead() -> dict[str, OverheadAnalysis]:
    """Analyze and compare protocol overhead across MQTT, Kafka, RabbitMQ, and Pub/Sub.

    TODO(human): Implement this function.

    Return a dict mapping protocol name -> OverheadAnalysis.

    For each protocol, create an OverheadAnalysis with accurate data:

    MQTT:
      - min_header_bytes: 2 (1 byte control packet type + flags, 1 byte remaining length)
        The fixed header is the smallest of any messaging protocol.
        Variable header adds topic length (2 bytes) + topic string + packet ID (2 bytes for QoS 1/2).
        A complete PUBLISH with "t" topic and "x" payload is just 5 bytes total.
      - typical_message_bytes: ~60 (2 fixed + ~30 topic + 2 packet ID + ~20-50 payload)
        With a topic like "sensors/temperature/room1" (26 chars) and a 50-byte JSON payload.
      - connection_setup: "Single TCP connection with CONNECT/CONNACK handshake (~4 packets).
        Connection is persistent -- no per-message connection overhead."
      - keepalive_mechanism: "PINGREQ/PINGRESP every N seconds (configurable). Broker detects
        dead clients after 1.5x keepalive interval."
      - bandwidth_efficiency: "high"

    Kafka:
      - min_header_bytes: 24+ (record header in a batch: length, attributes, timestamp delta,
        offset delta, key length, value length, headers count). Records are batched.
      - typical_message_bytes: ~100+ (record header + key + value + batch overhead amortized)
        Kafka optimizes for throughput via batching -- per-message overhead is lower in
        large batches but higher for individual messages.
      - connection_setup: "TCP connection + API version negotiation + metadata fetch.
        Persistent connections with consumer poll loop."
      - keepalive_mechanism: "Consumer heartbeats to group coordinator (heartbeat.interval.ms).
        Session timeout triggers rebalance."
      - bandwidth_efficiency: "medium" (high throughput but heavier per-message framing)

    RabbitMQ:
      - min_header_bytes: 8 (AMQP 0-9-1 frame header: type + channel + size + frame-end)
        A message requires 3 frames: method + content-header + content-body = ~24+ bytes overhead.
      - typical_message_bytes: ~120+ (AMQP framing + exchange/routing key + properties + payload)
      - connection_setup: "TCP + AMQP handshake (protocol header + Connection.Start/Tune/Open).
        Channel multiplexing over a single TCP connection."
      - keepalive_mechanism: "AMQP heartbeat frames at negotiated interval."
      - bandwidth_efficiency: "medium"

    Cloud Pub/Sub:
      - min_header_bytes: 200+ (HTTP/2 or gRPC framing + headers + authentication)
        Every publish/pull is an HTTP request with OAuth tokens and metadata.
      - typical_message_bytes: ~300+ (HTTP headers + protobuf wrapper + payload)
        Batching reduces amortized overhead for publish, but pull still requires HTTP round-trips.
      - connection_setup: "HTTP/2 connection + TLS handshake + OAuth token exchange.
        Streaming pull maintains a persistent gRPC stream."
      - keepalive_mechanism: "gRPC keepalive pings. Streaming pull has built-in flow control."
      - bandwidth_efficiency: "low" (HTTP/gRPC overhead, designed for cloud-native, not constrained devices)

    Return: {"MQTT": OverheadAnalysis(...), "Kafka": ..., "RabbitMQ": ..., "Cloud Pub/Sub": ...}

    Sources:
      - MQTT packet format: https://www.steves-internet-guide.com/mqtt-protocol-messages-overview/
      - Kafka record format: https://kafka.apache.org/documentation/#recordbatch
      - AMQP frame format: https://www.rabbitmq.com/amqp-0-9-1-reference.html
      - MQTT vs HTTP overhead: https://www.emqx.com/en/blog/mqtt-vs-http
    """
    raise NotImplementedError(
        "TODO(human): Create OverheadAnalysis for each protocol"
    )


def analyze_delivery_guarantees() -> dict[str, DeliveryAnalysis]:
    """Analyze delivery guarantees across all four protocols.

    TODO(human): Implement this function.

    Return a dict mapping protocol name -> DeliveryAnalysis.

    For each protocol, create a DeliveryAnalysis:

    MQTT:
      - guarantees: ["QoS 0 (at-most-once)", "QoS 1 (at-least-once)", "QoS 2 (exactly-once)"]
      - default_guarantee: "QoS 0" (depends on publisher choice)
      - exactly_once: True
      - exactly_once_mechanism: "QoS 2: 4-step handshake (PUBLISH -> PUBREC -> PUBREL -> PUBCOMP).
        The PUBREL packet acts as a deduplication boundary. Per-message choice."
      - message_ordering: "Per-topic ordering (single broker). No partitioning, so all messages
        on a topic are ordered. With shared subscriptions, order is NOT guaranteed across workers."
      - persistence_model: "Retained: only LAST message per topic stored. QoS 1/2 messages
        queued for offline persistent-session clients. No message log or replay."

    Kafka:
      - guarantees: ["At-least-once (default)", "At-most-once", "Exactly-once (transactions)"]
      - default_guarantee: "At-least-once"
      - exactly_once: True
      - exactly_once_mechanism: "Idempotent producer (enable.idempotence=true) + transactions
        (transactional.id). Requires both producer and consumer to use transactions.
        Topic-level, not per-message."
      - message_ordering: "Per-partition ordering. Messages with the same key go to the same
        partition (deterministic). No cross-partition ordering."
      - persistence_model: "Append-only commit log with configurable retention (time or size).
        Consumers track offsets and can replay by seeking. Messages survive broker restarts."

    RabbitMQ:
      - guarantees: ["At-most-once (no acks)", "At-least-once (acks + persistent queues)"]
      - default_guarantee: "At-least-once (with publisher confirms + consumer acks)"
      - exactly_once: False
      - exactly_once_mechanism: "Not natively supported. Must implement application-level
        deduplication (e.g., message IDs + idempotent consumers)."
      - message_ordering: "Per-queue FIFO. Ordering can break with competing consumers,
        redeliveries, or priority queues."
      - persistence_model: "Persistent queues store messages to disk. Messages deleted after
        consumer ack. Dead-letter exchanges for rejected messages. No replay after ack."

    Cloud Pub/Sub:
      - guarantees: ["At-least-once (default)"]
      - default_guarantee: "At-least-once"
      - exactly_once: False (effectively)
      - exactly_once_mechanism: "Exactly-once delivery available per subscription but with
        significant throughput reduction. Requires application-level idempotency for most cases."
      - message_ordering: "Unordered by default. Ordering keys provide per-key ordering within
        a region, but add latency constraints."
      - persistence_model: "Messages retained until acknowledged (max 7 days). No offset-based
        replay. Seek-to-timestamp available. Fully managed storage."

    Return: {"MQTT": DeliveryAnalysis(...), "Kafka": ..., "RabbitMQ": ..., "Cloud Pub/Sub": ...}

    Sources:
      - MQTT QoS: https://www.emqx.com/en/blog/introduction-to-mqtt-qos
      - Kafka guarantees: https://kafka.apache.org/documentation/#semantics
      - RabbitMQ reliability: https://www.rabbitmq.com/docs/reliability
      - Pub/Sub delivery: https://cloud.google.com/pubsub/docs/exactly-once-delivery
    """
    raise NotImplementedError(
        "TODO(human): Create DeliveryAnalysis for each protocol"
    )


def analyze_scalability_patterns() -> dict[str, ScalabilityAnalysis]:
    """Analyze scalability and consumer patterns across all four protocols.

    TODO(human): Implement this function.

    Return a dict mapping protocol name -> ScalabilityAnalysis.

    For each protocol:

    MQTT:
      - consumer_group_mechanism: "Shared subscriptions (MQTT 5.0): $share/<group>/<topic>.
        Broker distributes messages round-robin across group members. No partition assignment."
      - max_throughput: "10K-100K+ msg/sec per broker (QoS dependent). Designed for many
        small messages from constrained devices, not bulk data transfer."
      - horizontal_scaling: "Add more shared subscription members. Broker handles distribution.
        For broker scaling: use clustered brokers (EMQX, HiveMQ) or bridge multiple Mosquittos."
      - partitioning: "None built-in. All messages on a topic go through the same broker.
        Sharding requires application-level topic design (e.g., sensors/<region>/<type>)."
      - replay_capability: False
      - replay_mechanism: "No replay. Messages are ephemeral after delivery. Only retained
        messages (last per topic) are stored. No consumer offsets."

    Kafka:
      - consumer_group_mechanism: "Consumer groups with automatic partition assignment.
        Each partition assigned to exactly one consumer in the group. Rebalancing on join/leave."
      - max_throughput: "Millions of msg/sec per broker. Designed for high-throughput event
        streaming. Throughput scales linearly with partitions and brokers."
      - horizontal_scaling: "Add partitions + consumers. Max parallelism = number of partitions.
        Add brokers for storage and throughput."
      - partitioning: "Topics divided into partitions. Key-based or round-robin assignment.
        Partitions are the unit of parallelism, ordering, and replication."
      - replay_capability: True
      - replay_mechanism: "Consumer offset seeking. Can replay from beginning, specific offset,
        or timestamp. Retention-based (time or size). Core feature of Kafka."

    RabbitMQ:
      - consumer_group_mechanism: "Competing consumers: multiple consumers on the same queue.
        Broker round-robins messages. No built-in group coordination or rebalancing."
      - max_throughput: "10K-50K msg/sec per node. Optimized for flexible routing and reliable
        delivery, not raw throughput."
      - horizontal_scaling: "Add more consumers to a queue. Use consistent hash exchange for
        sharding. Cluster nodes for HA. Quorum queues for replicated queues."
      - partitioning: "No native partitioning. Use consistent hash exchange or application-level
        routing to distribute across queues."
      - replay_capability: False
      - replay_mechanism: "No replay. Messages deleted after ack. Dead-letter queues capture
        rejected messages. Streams (RabbitMQ 3.9+) offer limited log-based replay."

    Cloud Pub/Sub:
      - consumer_group_mechanism: "Multiple subscribers to a subscription. Pub/Sub auto-distributes
        messages. Flow control via max outstanding messages."
      - max_throughput: "Scales elastically (managed). 100K+ msg/sec per subscription.
        Google infrastructure handles scaling transparently."
      - horizontal_scaling: "Add more subscriber instances. Pub/Sub automatically distributes.
        No manual partition management."
      - partitioning: "No user-visible partitions. Ordering keys provide logical partitioning
        for ordered delivery."
      - replay_capability: True (limited)
      - replay_mechanism: "Seek to timestamp (not offset). Replay unacknowledged messages.
        Cannot replay already-acknowledged messages. Max retention 7 days."

    Return: {"MQTT": ScalabilityAnalysis(...), "Kafka": ..., "RabbitMQ": ..., "Cloud Pub/Sub": ...}

    Sources:
      - MQTT shared subs: https://www.hivemq.com/blog/mqtt5-essentials-part7-shared-subscriptions/
      - Kafka architecture: https://kafka.apache.org/documentation/#design
      - RabbitMQ scaling: https://www.rabbitmq.com/docs/clustering
      - Pub/Sub scaling: https://cloud.google.com/pubsub/docs/overview
    """
    raise NotImplementedError(
        "TODO(human): Create ScalabilityAnalysis for each protocol"
    )


def analyze_use_case_fit() -> list[UseCaseScenario]:
    """Create use-case scenarios and recommend the best protocol for each.

    TODO(human): Implement this function.

    Return a list of UseCaseScenario objects. Create at least 5 scenarios
    covering different domains. For each, recommend the best protocol and
    a runner-up with reasoning.

    Suggested scenarios (implement these or create your own):

    1. IoT Fleet Telemetry
       - description: "100K sensors publishing temperature/humidity every 5 seconds"
       - requirements: ["Low per-message overhead", "Constrained device support",
                        "Device presence tracking", "Tolerates occasional data loss"]
       - recommended_protocol: "MQTT"
       - reasoning: "2-byte header minimizes bandwidth. LWT provides device presence.
         QoS 0 handles high-frequency sensor data efficiently. Shared subscriptions
         enable backend processing scaling."
       - runner_up: "Kafka" (if data needs replay for analytics)

    2. Real-Time Event Streaming Pipeline
       - description: "E-commerce platform processing orders, clicks, and inventory updates"
       - requirements: ["High throughput (1M+ events/sec)", "Message replay for reprocessing",
                        "Exactly-once processing", "Stream processing integration"]
       - recommended_protocol: "Kafka"
       - reasoning: ...
       - runner_up: "Cloud Pub/Sub" (if managed infrastructure preferred)

    3. Microservice Task Queue
       - description: "Backend services distributing async tasks (email, PDF generation, payments)"
       - requirements: ["Flexible routing (direct, fanout, topic)", "Message acknowledgment",
                        "Dead-letter handling", "Priority queues"]
       - recommended_protocol: "RabbitMQ"
       - reasoning: ...
       - runner_up: "Cloud Pub/Sub" (simpler but less routing flexibility)

    4. Cloud-Native Event Bus
       - description: "Serverless functions triggered by cross-service events on GCP"
       - requirements: ["Zero infrastructure management", "Auto-scaling",
                        "Integration with cloud services", "At-least-once delivery"]
       - recommended_protocol: "Cloud Pub/Sub"
       - reasoning: ...
       - runner_up: "Kafka" (if self-hosted with more control needed)

    5. Connected Vehicle Platform
       - description: "50K vehicles reporting GPS, diagnostics, and receiving OTA commands"
       - requirements: ["Bidirectional communication", "Unreliable network tolerance",
                        "Device presence (online/offline)", "Low bandwidth on cellular"]
       - recommended_protocol: "MQTT"
       - reasoning: ...
       - runner_up: "Custom (MQTT for edge + Kafka for backend processing)"

    6. Financial Transaction Audit Log
       - description: "Recording every trade execution for compliance and replay"
       - requirements: ["Exactly-once delivery", "Immutable log", "Long retention",
                        "Replay from any point in time"]
       - recommended_protocol: "Kafka"
       - reasoning: ...
       - runner_up: "RabbitMQ" (with streams for limited replay)

    Feel free to add more scenarios or modify these based on your experience.

    Sources:
      - Kafka use cases: https://kafka.apache.org/powered-by
      - MQTT use cases: https://www.hivemq.com/mqtt/use-cases/
      - RabbitMQ use cases: https://www.rabbitmq.com/docs/getstarted
    """
    raise NotImplementedError(
        "TODO(human): Create use-case scenarios with protocol recommendations"
    )


def generate_comparison_report(
    overhead: dict[str, OverheadAnalysis],
    delivery: dict[str, DeliveryAnalysis],
    scalability: dict[str, ScalabilityAnalysis],
    scenarios: list[UseCaseScenario],
) -> str:
    """Generate a formatted comparison report from all analyses.

    TODO(human): Implement this function.

    Steps:
      1. Build a report string with clear sections. Use a list of strings
         and join at the end for efficiency:
             lines: list[str] = []

      2. Add a header:
             lines.append("=" * 70)
             lines.append("MESSAGING PROTOCOL COMPARISON REPORT")
             lines.append("MQTT vs Kafka vs RabbitMQ vs Cloud Pub/Sub")
             lines.append("=" * 70)

      3. Section 1 -- Protocol Overhead:
         For each protocol in overhead:
             lines.append(f"\\n  {name}:")
             lines.append(f"    Min header: {analysis.min_header_bytes} bytes")
             lines.append(f"    Typical message: {analysis.typical_message_bytes} bytes")
             lines.append(f"    Connection: {analysis.connection_setup}")
             lines.append(f"    Efficiency: {analysis.bandwidth_efficiency}")

      4. Section 2 -- Delivery Guarantees:
         For each protocol in delivery:
             lines.append(f"\\n  {name}:")
             lines.append(f"    Guarantees: {', '.join(analysis.guarantees)}")
             lines.append(f"    Default: {analysis.default_guarantee}")
             lines.append(f"    Exactly-once: {'Yes' if analysis.exactly_once else 'No'}")
             lines.append(f"    Ordering: {analysis.message_ordering}")

      5. Section 3 -- Scalability & Replay:
         For each protocol in scalability:
             lines.append(f"\\n  {name}:")
             lines.append(f"    Consumer groups: {analysis.consumer_group_mechanism}")
             lines.append(f"    Max throughput: {analysis.max_throughput}")
             lines.append(f"    Replay: {'Yes' if analysis.replay_capability else 'No'} "
                          f"({analysis.replay_mechanism})")

      6. Section 4 -- Use Case Recommendations:
         For each scenario in scenarios:
             lines.append(f"\\n  Scenario: {scenario.name}")
             lines.append(f"    {scenario.description}")
             lines.append(f"    Requirements: {', '.join(scenario.requirements)}")
             lines.append(f"    Recommended: {scenario.recommended_protocol}")
             lines.append(f"      Why: {scenario.reasoning}")
             lines.append(f"    Runner-up: {scenario.runner_up}")
             lines.append(f"      Why: {scenario.runner_up_reasoning}")

      7. Section 5 -- Key Takeaways:
             lines.append("\\n" + "=" * 70)
             lines.append("KEY TAKEAWAYS")
             lines.append("=" * 70)
             lines.append("  1. MQTT: Unbeatable for IoT/edge (tiny overhead, LWT, retained msgs)")
             lines.append("  2. Kafka: Unbeatable for event streaming (replay, throughput, exactly-once)")
             lines.append("  3. RabbitMQ: Best for flexible routing & task queues (AMQP, exchanges)")
             lines.append("  4. Pub/Sub: Best for cloud-native with zero ops (managed, auto-scaling)")
             lines.append("  5. These are COMPLEMENTARY, not competing -- many architectures use 2+")

      8. Join and return:
             return "\\n".join(lines)

    The goal is a report that could be used as a reference document for
    architecture decisions. It should be factual, concise, and structured
    for easy scanning.
    """
    raise NotImplementedError(
        "TODO(human): Generate formatted comparison report from all analyses"
    )


# ── Orchestration (boilerplate) ──────────────────────────────────────


def build_protocol_profiles(
    overhead: dict[str, OverheadAnalysis],
    delivery: dict[str, DeliveryAnalysis],
    scalability: dict[str, ScalabilityAnalysis],
    scenarios: list[UseCaseScenario],
) -> dict[str, ProtocolProfile]:
    """Assemble complete protocol profiles from individual analyses."""
    protocols = {
        "MQTT": ProtocolProfile(
            name="MQTT",
            category="IoT / Lightweight Pub/Sub",
            strengths=[
                "Smallest protocol overhead (2-byte header)",
                "Built-in device presence (LWT + retained)",
                "Per-message QoS choice (0/1/2)",
                "Persistent connections, push-based delivery",
            ],
            weaknesses=[
                "No message replay or log",
                "Single broker is SPOF (clustering is broker-specific)",
                "No native partitioning",
                "Shared subscriptions only in MQTT 5.0",
            ],
        ),
        "Kafka": ProtocolProfile(
            name="Kafka",
            category="Event Streaming Platform",
            strengths=[
                "Highest throughput (millions msg/sec)",
                "Full message replay via offset seeking",
                "Exactly-once with transactions",
                "Rich ecosystem (Streams, Connect, ksqlDB)",
            ],
            weaknesses=[
                "High operational complexity (cluster management)",
                "Higher per-message overhead than MQTT",
                "Pull-based model adds latency",
                "No built-in device presence or LWT",
            ],
        ),
        "RabbitMQ": ProtocolProfile(
            name="RabbitMQ",
            category="Enterprise Message Broker",
            strengths=[
                "Flexible routing (exchanges: direct, fanout, topic, headers)",
                "Priority queues and dead-letter exchanges",
                "Multiple protocol support (AMQP, STOMP, MQTT)",
                "Mature plugin ecosystem",
            ],
            weaknesses=[
                "No message replay after acknowledgment",
                "No native exactly-once delivery",
                "Lower throughput than Kafka",
                "Complex exchange/queue topology",
            ],
        ),
        "Cloud Pub/Sub": ProtocolProfile(
            name="Cloud Pub/Sub",
            category="Managed Cloud Messaging",
            strengths=[
                "Zero operational overhead (fully managed)",
                "Auto-scaling with no configuration",
                "Deep GCP integration (Cloud Functions, Dataflow)",
                "Global availability and durability",
            ],
            weaknesses=[
                "Higher per-message overhead (HTTP/gRPC)",
                "Vendor lock-in (Google Cloud)",
                "No offset-based replay (only seek-to-timestamp)",
                "Higher latency than MQTT or RabbitMQ",
            ],
        ),
    }

    # Attach analysis results
    for name in protocols:
        if name in overhead:
            protocols[name].overhead = overhead[name]
        if name in delivery:
            protocols[name].delivery = delivery[name]
        if name in scalability:
            protocols[name].scalability = scalability[name]

    # Attach use cases from scenarios
    for scenario in scenarios:
        proto = scenario.recommended_protocol
        if proto in protocols:
            protocols[proto].use_cases.append(scenario.name)

    return protocols


def main() -> None:
    print("=" * 70)
    print("EXERCISE 5: Protocol Comparison Report")
    print("MQTT vs Kafka vs RabbitMQ vs Cloud Pub/Sub")
    print("=" * 70)

    # Run all analyses
    print("\n[1/4] Analyzing protocol overhead...")
    overhead = analyze_overhead()

    print("[2/4] Analyzing delivery guarantees...")
    delivery = analyze_delivery_guarantees()

    print("[3/4] Analyzing scalability patterns...")
    scalability = analyze_scalability_patterns()

    print("[4/4] Analyzing use-case fit...")
    scenarios = analyze_use_case_fit()

    # Build profiles
    profiles = build_protocol_profiles(overhead, delivery, scalability, scenarios)

    # Generate report
    print("\nGenerating comparison report...\n")
    report = generate_comparison_report(overhead, delivery, scalability, scenarios)
    print(report)

    # Print protocol profiles summary
    print("\n" + "=" * 70)
    print("PROTOCOL PROFILES")
    print("=" * 70)
    for name, profile in profiles.items():
        print(f"\n  {name} ({profile.category})")
        print(f"    Strengths:  {', '.join(profile.strengths[:2])}...")
        print(f"    Weaknesses: {', '.join(profile.weaknesses[:2])}...")
        print(f"    Best for:   {', '.join(profile.use_cases) if profile.use_cases else 'See scenarios'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
