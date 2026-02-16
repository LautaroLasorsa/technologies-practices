"""Exercise 2: MQTT QoS Levels -- Delivery Guarantees.

Demonstrates:
  - QoS 0 (at most once): fire and forget, no acknowledgment
  - QoS 1 (at least once): PUBACK handshake, possible duplicates
  - QoS 2 (exactly once): 4-step handshake (PUBREC -> PUBREL -> PUBCOMP)
  - Timing comparison across QoS levels

Run after starting Mosquitto:
    uv run python qos_levels.py
"""

import json
import time
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion, MQTTProtocolVersion

import config

# ── Constants ────────────────────────────────────────────────────────

QOS_TOPIC_PREFIX = "qos_test"
NUM_MESSAGES_PER_QOS = 10


# ── Data tracking ───────────────────────────────────────────────────


@dataclass
class QoSMetrics:
    """Track delivery metrics for a QoS level."""

    qos_level: int
    messages_sent: int = 0
    messages_received: int = 0
    publish_acks: int = 0
    publish_times: list[float] = field(default_factory=list)
    receive_times: list[float] = field(default_factory=list)


# ── Callbacks (boilerplate) ──────────────────────────────────────────


def on_connect(
    client: mqtt.Client,
    userdata: dict[int, QoSMetrics],
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    """Log connection result."""
    if reason_code == 0:
        print(f"[CONNECTED] {client._client_id.decode()}")
    else:
        print(f"[ERROR] Connection failed: {reason_code}")


def on_message(
    client: mqtt.Client,
    userdata: dict[int, QoSMetrics],
    msg: mqtt.MQTTMessage,
) -> None:
    """Track received messages per QoS level."""
    receive_time = time.perf_counter()
    payload = json.loads(msg.payload.decode("utf-8"))
    qos_level = msg.qos

    if qos_level in userdata:
        userdata[qos_level].messages_received += 1
        userdata[qos_level].receive_times.append(receive_time)

    print(
        f"  [RECV] topic={msg.topic}  qos={qos_level}  "
        f"seq={payload.get('seq')}  received_at={receive_time:.6f}"
    )


def on_publish(
    client: mqtt.Client,
    userdata: dict[int, QoSMetrics],
    mid: int,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    """Called when a PUBLISH is acknowledged by the broker.

    For QoS 0: called when the message leaves the client (no broker ack).
    For QoS 1: called when PUBACK is received from the broker.
    For QoS 2: called when the PUBCOMP (final step) is received.
    """
    # We track acks, but we can't easily map mid -> qos here without
    # extra bookkeeping. The per-QoS timing is tracked in the exercise.
    pass


# ── TODO(human): Implement these functions ───────────────────────────


def demonstrate_qos0(
    pub_client: mqtt.Client,
    sub_client: mqtt.Client,
    metrics: QoSMetrics,
) -> None:
    """Demonstrate QoS 0 -- At Most Once (Fire and Forget).

    TODO(human): Implement this function.

    Steps:
      1. Subscribe the sub_client to the QoS 0 test topic:
             topic = f"{QOS_TOPIC_PREFIX}/qos0"
             sub_client.subscribe(topic, qos=0)

         When subscribing with QoS 0, the subscriber tells the broker:
         "I accept QoS 0 delivery for messages on this topic." Even if
         a publisher sends with QoS 1, the broker downgrades to
         min(publisher_qos, subscriber_qos) = 0 for this subscriber.

      2. Wait briefly for the subscription to be established:
             time.sleep(0.5)

      3. Publish NUM_MESSAGES_PER_QOS messages with QoS 0:
         For each i in range(NUM_MESSAGES_PER_QOS):
           a. Build the payload:
                  payload = json.dumps({"seq": i, "qos": 0, "data": "qos0_test"})
           b. Record the send time:
                  send_time = time.perf_counter()
                  metrics.publish_times.append(send_time)
           c. Publish:
                  pub_client.publish(topic, payload, qos=0)
           d. Increment metrics.messages_sent

         QoS 0 protocol flow:
           Publisher ---PUBLISH---> Broker ---PUBLISH---> Subscriber
           (one packet each direction, no acknowledgment)

         The publish() call returns immediately. There is NO wait_for_publish()
         guarantee with QoS 0 -- the message might be lost if the TCP
         connection drops between client and broker, or between broker
         and subscriber. The broker does NOT store QoS 0 messages.

      4. Wait for messages to arrive:
             time.sleep(1.0)

      5. Unsubscribe:
             sub_client.unsubscribe(topic)

      6. Print results:
             print(f"  Sent: {metrics.messages_sent}, Received: {metrics.messages_received}")
             print(f"  QoS 0 guarantee: at-most-once (some messages MAY be lost)")

    Key insight:
      On a local broker (localhost), QoS 0 messages are almost never lost
      because the TCP connection is reliable. The "fire and forget" behavior
      becomes apparent over unreliable networks (WiFi, cellular, satellite).
      QoS 0 is ideal for high-frequency, low-value data (e.g., sensor
      readings every second where losing one reading doesn't matter).

    Docs: https://www.emqx.com/en/blog/introduction-to-mqtt-qos
    """
    raise NotImplementedError(
        "TODO(human): Demonstrate QoS 0 -- fire and forget publishing"
    )


def demonstrate_qos1(
    pub_client: mqtt.Client,
    sub_client: mqtt.Client,
    metrics: QoSMetrics,
) -> None:
    """Demonstrate QoS 1 -- At Least Once (Acknowledged Delivery).

    TODO(human): Implement this function.

    Steps:
      1. Subscribe the sub_client to the QoS 1 test topic:
             topic = f"{QOS_TOPIC_PREFIX}/qos1"
             sub_client.subscribe(topic, qos=1)
             time.sleep(0.5)

      2. Publish NUM_MESSAGES_PER_QOS messages with QoS 1:
         For each i in range(NUM_MESSAGES_PER_QOS):
           a. Build payload: {"seq": i, "qos": 1, "data": "qos1_test"}
           b. Record send time in metrics.publish_times
           c. Publish with qos=1:
                  result = pub_client.publish(topic, json.dumps(payload), qos=1)
           d. Call result.wait_for_publish(timeout=5)
              This blocks until PUBACK is received from the broker.
           e. Increment metrics.messages_sent and metrics.publish_acks

         QoS 1 protocol flow:
           Publisher ---PUBLISH(QoS1)---> Broker
           Publisher <---PUBACK---------- Broker
                                          Broker ---PUBLISH(QoS1)---> Subscriber
                                          Broker <---PUBACK---------- Subscriber

         The PUBLISH packet has a Packet Identifier (a 16-bit integer).
         The PUBACK references this same identifier. If the sender doesn't
         receive PUBACK within a timeout, it retransmits the PUBLISH with
         the DUP flag set. This can cause duplicate delivery if PUBACK
         was lost but the broker already processed the message.

      3. Wait for subscriber to receive messages:
             time.sleep(1.0)

      4. Unsubscribe: sub_client.unsubscribe(topic)

      5. Print results including ack count:
             print(f"  Sent: {metrics.messages_sent}, "
                   f"Acked: {metrics.publish_acks}, "
                   f"Received: {metrics.messages_received}")
             print(f"  QoS 1 guarantee: at-least-once (duplicates possible)")

    Key insight:
      QoS 1 is the most commonly used level in production IoT systems.
      It provides good reliability (messages won't silently disappear)
      with acceptable overhead (2 packets instead of 1). The risk of
      duplicates is manageable with idempotent processing (e.g., using
      message IDs to deduplicate).

    Docs: https://www.hivemq.com/blog/mqtt-essentials-part-6-mqtt-quality-of-service-levels/
    """
    raise NotImplementedError(
        "TODO(human): Demonstrate QoS 1 -- at-least-once with PUBACK"
    )


def demonstrate_qos2(
    pub_client: mqtt.Client,
    sub_client: mqtt.Client,
    metrics: QoSMetrics,
) -> None:
    """Demonstrate QoS 2 -- Exactly Once (Four-Step Handshake).

    TODO(human): Implement this function.

    Steps:
      1. Subscribe the sub_client to the QoS 2 test topic:
             topic = f"{QOS_TOPIC_PREFIX}/qos2"
             sub_client.subscribe(topic, qos=2)
             time.sleep(0.5)

      2. Publish NUM_MESSAGES_PER_QOS messages with QoS 2:
         For each i in range(NUM_MESSAGES_PER_QOS):
           a. Build payload: {"seq": i, "qos": 2, "data": "qos2_test"}
           b. Record send time in metrics.publish_times
           c. Publish with qos=2:
                  result = pub_client.publish(topic, json.dumps(payload), qos=2)
           d. Call result.wait_for_publish(timeout=10)
              This blocks until the FULL 4-step handshake completes
              (PUBCOMP is received). Takes longer than QoS 1.
           e. Increment metrics.messages_sent and metrics.publish_acks

         QoS 2 protocol flow (4-step handshake):
           Publisher ---PUBLISH(QoS2)---> Broker    (step 1: send message)
           Publisher <---PUBREC---------- Broker    (step 2: received confirmation)
           Publisher ---PUBREL----------> Broker    (step 3: release/commit)
           Publisher <---PUBCOMP--------- Broker    (step 4: complete)

         Why 4 steps instead of 2?
           The extra PUBREL/PUBCOMP exchange solves the duplicate problem:
           - After PUBREC, both sides know the message was received.
           - PUBREL signals "I got your PUBREC, you can now process/forward."
           - The PUBREL acts as a BOUNDARY: any PUBLISH received before it
             is a retransmission (duplicate); any after is a new message.
           - PUBCOMP confirms the handshake is fully done.
           This 2-phase approach ensures exactly-once even if individual
           packets are lost and retransmitted.

      3. Wait for subscriber to receive messages:
             time.sleep(1.5)

      4. Unsubscribe: sub_client.unsubscribe(topic)

      5. Print results:
             print(f"  Sent: {metrics.messages_sent}, "
                   f"Acked: {metrics.publish_acks}, "
                   f"Received: {metrics.messages_received}")
             print(f"  QoS 2 guarantee: exactly-once (no loss, no duplicates)")

    Key insight:
      QoS 2 is rarely used in high-throughput IoT scenarios because the
      4-packet overhead per message significantly reduces throughput. It's
      reserved for critical messages where duplicates cause real harm:
      billing events, actuator commands (e.g., "open valve"), or financial
      transactions. Most IoT systems use QoS 1 + idempotent processing.

    Docs: https://www.emqx.com/en/blog/introduction-to-mqtt-qos
    """
    raise NotImplementedError(
        "TODO(human): Demonstrate QoS 2 -- exactly-once with 4-step handshake"
    )


def compare_qos_levels(
    all_metrics: dict[int, QoSMetrics],
) -> None:
    """Compare timing and delivery results across all three QoS levels.

    TODO(human): Implement this function.

    Steps:
      1. Print a comparison header:
             print("\\n" + "=" * 60)
             print("QoS LEVEL COMPARISON")
             print("=" * 60)

      2. For each QoS level (0, 1, 2), using all_metrics[qos]:
         a. Calculate delivery rate:
                delivery_rate = metrics.messages_received / max(metrics.messages_sent, 1) * 100

         b. Calculate average round-trip time (if both publish_times and
            receive_times are available):
            If len(metrics.publish_times) > 0 and len(metrics.receive_times) > 0:
                avg_time = (metrics.receive_times[-1] - metrics.publish_times[0]) / max(metrics.messages_received, 1)
            This is an approximation -- true per-message latency would
            require correlating individual message timestamps.

         c. Print a formatted row for this QoS level:
                print(f"  QoS {qos}: sent={metrics.messages_sent}, "
                      f"received={metrics.messages_received}, "
                      f"delivery={delivery_rate:.1f}%, "
                      f"acks={metrics.publish_acks}")

      3. Print the trade-off summary:
             print("\\n  Trade-offs:")
             print("    QoS 0: Fastest, lowest overhead, messages may be lost")
             print("    QoS 1: Reliable, 2-packet overhead, duplicates possible")
             print("    QoS 2: Strongest guarantee, 4-packet overhead, slowest")

    Key insight:
      On a reliable local network, all three QoS levels likely show 100%
      delivery. The differences become apparent under:
      - Network packet loss (QoS 0 loses messages, QoS 1/2 retry)
      - High throughput (QoS 2's 4x packet overhead becomes a bottleneck)
      - Constrained devices (QoS 0 uses less CPU/memory for state tracking)

    Docs: https://cedalo.com/blog/understanding-mqtt-qos/
    """
    raise NotImplementedError(
        "TODO(human): Compare delivery metrics across QoS levels"
    )


# ── Orchestration (boilerplate) ──────────────────────────────────────


def create_client(
    client_id: str,
    metrics: dict[int, QoSMetrics],
) -> mqtt.Client:
    """Create an MQTT 5.0 client with QoS metrics tracking."""
    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=MQTTProtocolVersion.MQTTv5,
        userdata=metrics,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    return client


def main() -> None:
    print("=" * 60)
    print("EXERCISE 2: MQTT QoS Levels -- Delivery Guarantees")
    print("=" * 60)

    # Shared metrics dict -- both publisher and subscriber update it
    metrics: dict[int, QoSMetrics] = {
        0: QoSMetrics(qos_level=0),
        1: QoSMetrics(qos_level=1),
        2: QoSMetrics(qos_level=2),
    }

    # Create publisher and subscriber clients
    pub_client = create_client("qos-publisher", metrics)
    sub_client = create_client("qos-subscriber", metrics)

    pub_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
    sub_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)

    pub_client.loop_start()
    sub_client.loop_start()

    time.sleep(0.5)  # Wait for both clients to connect

    # Run each QoS demonstration
    print("\n--- QoS 0: At Most Once (Fire and Forget) ---")
    demonstrate_qos0(pub_client, sub_client, metrics[0])

    print("\n--- QoS 1: At Least Once (PUBACK) ---")
    demonstrate_qos1(pub_client, sub_client, metrics[1])

    print("\n--- QoS 2: Exactly Once (4-Step Handshake) ---")
    demonstrate_qos2(pub_client, sub_client, metrics[2])

    # Compare results
    compare_qos_levels(metrics)

    # Cleanup
    pub_client.loop_stop()
    pub_client.disconnect()
    sub_client.loop_stop()
    sub_client.disconnect()

    print("\nDone. Both clients disconnected cleanly.")


if __name__ == "__main__":
    main()
