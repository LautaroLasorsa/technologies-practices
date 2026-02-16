"""Exercise 1: Basic MQTT Publish/Subscribe with Topic Hierarchies and Wildcards.

Demonstrates:
  - Connecting to an MQTT 5.0 broker using paho-mqtt v2 API
  - Publishing messages to hierarchical topics (sensors/temperature/room1)
  - Subscribing with single-level (+) and multi-level (#) wildcards
  - Observing which messages match which subscription patterns

Run after starting Mosquitto:
    uv run python basic_pubsub.py
"""

import json
import time

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion, MQTTProtocolVersion

import config

# ── Constants ────────────────────────────────────────────────────────

# Sample sensor readings to publish across the topic hierarchy
SENSOR_READINGS: list[dict[str, str | float]] = [
    {"topic": "sensors/temperature/room1", "value": 22.5, "unit": "C"},
    {"topic": "sensors/temperature/room2", "value": 23.1, "unit": "C"},
    {"topic": "sensors/temperature/room3", "value": 21.8, "unit": "C"},
    {"topic": "sensors/humidity/room1", "value": 45.2, "unit": "%"},
    {"topic": "sensors/humidity/room2", "value": 52.7, "unit": "%"},
    {"topic": "sensors/pressure/room1", "value": 1013.25, "unit": "hPa"},
    {"topic": "sensors/pressure/room2", "value": 1012.80, "unit": "hPa"},
    {"topic": "sensors/battery/room1", "value": 87.0, "unit": "%"},
    {"topic": "sensors/battery/room3", "value": 42.0, "unit": "%"},
]

# Wildcard subscription patterns to demonstrate
WILDCARD_PATTERNS: list[dict[str, str]] = [
    {
        "pattern": "sensors/temperature/+",
        "description": "All temperature sensors (single-level wildcard)",
    },
    {
        "pattern": "sensors/+/room1",
        "description": "All sensor types for room1 (single-level wildcard)",
    },
    {
        "pattern": "sensors/#",
        "description": "Everything under sensors/ (multi-level wildcard)",
    },
    {
        "pattern": "sensors/humidity/#",
        "description": "All humidity data (multi-level wildcard)",
    },
]


# ── Callbacks (boilerplate) ──────────────────────────────────────────


def on_connect(
    client: mqtt.Client,
    userdata: dict[str, list[str]],
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    """Called when the client connects to the broker.

    For paho-mqtt v2 (CallbackAPIVersion.VERSION2), the signature includes
    reason_code (a ReasonCode object, not an int) and properties.
    """
    if reason_code == 0:
        print(f"[CONNECTED] Client '{client._client_id.decode()}' connected successfully")
    else:
        print(f"[ERROR] Connection failed: {reason_code}")


def on_message(
    client: mqtt.Client,
    userdata: dict[str, list[str]],
    msg: mqtt.MQTTMessage,
) -> None:
    """Called when a message matching a subscription is received.

    The userdata dict is used to track which messages each subscription
    pattern received, so we can compare wildcard matching results.
    """
    payload = json.loads(msg.payload.decode("utf-8"))
    print(
        f"  [RECV] topic={msg.topic}  "
        f"value={payload.get('value')}  "
        f"unit={payload.get('unit')}  "
        f"qos={msg.qos}  "
        f"retained={msg.retain}"
    )
    # Track message for comparison
    for pattern, messages in userdata.items():
        if mqtt.topic_matches_sub(pattern, msg.topic):
            messages.append(msg.topic)


# ── TODO(human): Implement these functions ───────────────────────────


def create_client(client_id: str, userdata: dict[str, list[str]] | None = None) -> mqtt.Client:
    """Create and configure an MQTT 5.0 client using paho-mqtt v2 API.

    TODO(human): Implement this function.

    Steps:
      1. Create an mqtt.Client instance with these parameters:
             mqtt.Client(
                 callback_api_version=CallbackAPIVersion.VERSION2,
                 client_id=client_id,
                 protocol=MQTTProtocolVersion.MQTTv5,
                 userdata=userdata,
             )

         - CallbackAPIVersion.VERSION2: Uses the modern callback signatures
           where on_connect receives (client, userdata, flags, reason_code, properties)
           instead of the deprecated v1 signature with just an int rc.
         - MQTTProtocolVersion.MQTTv5: Enables MQTT 5.0 features (shared subs,
           message expiry, user properties, response topics).
         - userdata: Arbitrary object passed to all callbacks. We use a dict
           to track which wildcard patterns matched which messages.

      2. Assign the callback functions:
             client.on_connect = on_connect
             client.on_message = on_message

      3. Return the client (do NOT connect yet -- the caller decides when).

    Why MQTTv5?
      MQTT 5.0 adds shared subscriptions, message expiry, user properties,
      response topics, and reason codes. Even if you don't use these features
      immediately, connecting with v5 ensures you get proper reason codes
      in all ACK packets instead of the limited v3.1.1 return codes.

    Docs:
      - Client constructor: https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html
      - CallbackAPIVersion: https://eclipse.dev/paho/files/paho.mqtt.python/html/migrations.html
    """
    raise NotImplementedError(
        "TODO(human): Create an MQTT 5.0 client with CallbackAPIVersion.VERSION2"
    )


def publish_sensor_data(
    client: mqtt.Client,
    readings: list[dict[str, str | float]],
) -> None:
    """Publish sensor readings to hierarchical MQTT topics.

    TODO(human): Implement this function.

    Steps:
      1. Iterate over each reading in `readings`. Each reading dict has:
             {"topic": "sensors/temperature/room1", "value": 22.5, "unit": "C"}

      2. Build the JSON payload from the reading (exclude the "topic" key):
             payload = {k: v for k, v in reading.items() if k != "topic"}

      3. Publish the message:
             result = client.publish(
                 topic=reading["topic"],
                 payload=json.dumps(payload),
                 qos=1,
             )

         - topic: The hierarchical MQTT topic string. Topics are created
           implicitly on first publish -- no need to pre-create them.
           The hierarchy uses "/" as the separator.
         - payload: Must be bytes, str, or None. json.dumps() returns str,
           which paho converts to UTF-8 bytes automatically.
         - qos=1: At-least-once delivery. The broker sends PUBACK to confirm.
           For this exercise QoS 1 is a good default -- we want confirmation
           that messages were received by the broker.

      4. Call result.wait_for_publish(timeout=5) to block until the broker
         acknowledges the message. This ensures all publishes complete before
         we check subscription results.

         The publish() method is asynchronous -- it queues the message in the
         client's internal buffer. wait_for_publish() blocks until the
         message has actually been sent and (for QoS 1/2) acknowledged.

      5. Print the publish result:
             print(f"  [PUB] {reading['topic']} -> {payload}")

    Key concept -- MQTT topic hierarchy:
      Topics like "sensors/temperature/room1" create an implicit tree:
          sensors/
            temperature/
              room1
              room2
            humidity/
              room1
            pressure/
              room1
      This hierarchy enables powerful wildcard subscriptions (see below).
      Unlike Kafka (flat topics with partitions) or RabbitMQ (exchange/queue
      bindings), MQTT's routing is built directly into the topic structure.

    Docs: https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html#paho.mqtt.client.Client.publish
    """
    raise NotImplementedError(
        "TODO(human): Publish sensor readings to hierarchical MQTT topics"
    )


def subscribe_with_wildcards(
    client: mqtt.Client,
    patterns: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Subscribe to multiple wildcard patterns and report which topics each matches.

    TODO(human): Implement this function.

    Steps:
      1. Create a results dict to track matches:
             results = {p["pattern"]: [] for p in patterns}

      2. Set client.user_data_set(results) so the on_message callback
         can record which patterns matched which topics. The userdata
         object is passed to every callback as the second argument.

      3. For each pattern in `patterns`:
         a. Print the pattern and its description:
                print(f"\n--- Subscribing: {p['pattern']} ({p['description']}) ---")

         b. Subscribe to the pattern:
                client.subscribe(p["pattern"], qos=1)

            - Wildcards are ONLY valid in subscriptions, never in publishes.
            - "+" matches exactly ONE topic level:
                  "sensors/+/room1" matches "sensors/temperature/room1"
                  but NOT "sensors/a/b/room1"
            - "#" matches ZERO OR MORE levels (must be the last character):
                  "sensors/#" matches "sensors", "sensors/temperature",
                  "sensors/temperature/room1", etc.

         c. Wait briefly for messages to arrive (time.sleep(1.5)).
            MQTT is push-based -- the broker sends matching messages as
            they arrive. The paho client's background network loop (started
            by client.loop_start()) processes them and fires on_message.

         d. Unsubscribe so the next pattern starts fresh:
                client.unsubscribe(p["pattern"])

         e. Print how many messages matched:
                matched = results[p["pattern"]]
                print(f"  Matched {len(matched)} topics: {matched}")

      4. Return the results dict.

    Key concept -- Wildcard matching:
      Given these published topics:
        sensors/temperature/room1, sensors/temperature/room2,
        sensors/humidity/room1, sensors/pressure/room1

      - "sensors/temperature/+"  -> matches room1, room2 (2 matches)
      - "sensors/+/room1"        -> matches temperature, humidity, pressure (3 matches)
      - "sensors/#"              -> matches ALL (4+ matches)
      - "sensors/humidity/#"     -> matches only humidity topics

      This is more expressive than Kafka (no wildcards) and different from
      RabbitMQ's routing keys (which use "." and "*"/"#" similarly but
      require explicit exchange/queue bindings).

    Docs: https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html#paho.mqtt.client.Client.subscribe
    """
    raise NotImplementedError(
        "TODO(human): Subscribe with wildcards and track which topics match"
    )


# ── Orchestration (boilerplate) ──────────────────────────────────────


def main() -> None:
    # Phase 1: Create a publisher client and publish sensor data
    print("=" * 60)
    print("EXERCISE 1: Basic MQTT Pub/Sub with Topic Hierarchies")
    print("=" * 60)

    pub_client = create_client("publisher-001")
    pub_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
    pub_client.loop_start()

    print(f"\n--- Publishing {len(SENSOR_READINGS)} sensor readings ---")
    publish_sensor_data(pub_client, SENSOR_READINGS)

    # Phase 2: Create a subscriber client and test wildcard patterns
    print("\n--- Testing wildcard subscriptions ---")
    sub_client = create_client("subscriber-001")
    sub_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
    sub_client.loop_start()

    # Small delay to ensure subscriber is connected before re-publishing
    time.sleep(0.5)

    # Re-publish so the subscriber can receive messages
    # (MQTT is push-based -- messages published before subscription are lost
    # unless retained. We re-publish here for the wildcard demonstration.)
    print("\n--- Re-publishing for wildcard demonstration ---")

    results = subscribe_with_wildcards(sub_client, WILDCARD_PATTERNS)

    # Phase 3: Summary
    print("\n" + "=" * 60)
    print("WILDCARD MATCH SUMMARY")
    print("=" * 60)
    for pattern, matched in results.items():
        print(f"  {pattern:30s} -> {len(matched)} matches")

    # Cleanup
    pub_client.loop_stop()
    pub_client.disconnect()
    sub_client.loop_stop()
    sub_client.disconnect()

    print("\nDone. Both clients disconnected cleanly.")


if __name__ == "__main__":
    main()
