"""Exercise 3: Retained Messages & Last Will and Testament (LWT).

Demonstrates:
  - Retained messages: broker stores last message per topic, delivers to new subscribers
  - Last Will and Testament: broker publishes a pre-set message on ungraceful disconnect
  - Device presence pattern: combining retained + LWT for online/offline tracking

Run after starting Mosquitto:
    uv run python retained_lastwill.py
"""

import json
import time
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion, MQTTProtocolVersion

import config

# ── Constants ────────────────────────────────────────────────────────

DEVICE_IDS = ["sensor-001", "sensor-002", "sensor-003"]
STATUS_TOPIC_TEMPLATE = "devices/{device_id}/status"
TELEMETRY_TOPIC_TEMPLATE = "devices/{device_id}/telemetry"


# ── Data tracking ───────────────────────────────────────────────────


@dataclass
class DeviceState:
    """Track observed state for a device."""

    device_id: str
    last_status: str = "unknown"
    last_telemetry: dict | None = None
    status_updates: list[str] = field(default_factory=list)


# ── Callbacks (boilerplate) ──────────────────────────────────────────


def on_connect(
    client: mqtt.Client,
    userdata: dict[str, DeviceState] | None,
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
    userdata: dict[str, DeviceState] | None,
    msg: mqtt.MQTTMessage,
) -> None:
    """Process received messages, noting retained flag."""
    payload_str = msg.payload.decode("utf-8")
    retained_tag = " [RETAINED]" if msg.retain else ""

    # Parse the device_id from the topic: devices/<device_id>/status or /telemetry
    parts = msg.topic.split("/")
    if len(parts) >= 3:
        device_id = parts[1]
        msg_type = parts[2]  # "status" or "telemetry"
    else:
        device_id = "unknown"
        msg_type = "unknown"

    print(
        f"  [RECV{retained_tag}] topic={msg.topic}  "
        f"payload={payload_str}  qos={msg.qos}"
    )

    # Track state if userdata is provided
    if userdata and device_id in userdata:
        if msg_type == "status":
            userdata[device_id].last_status = payload_str
            userdata[device_id].status_updates.append(payload_str)
        elif msg_type == "telemetry":
            try:
                userdata[device_id].last_telemetry = json.loads(payload_str)
            except json.JSONDecodeError:
                pass


def on_disconnect(
    client: mqtt.Client,
    userdata: dict[str, DeviceState] | None,
    flags: mqtt.DisconnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    """Log disconnection reason."""
    client_id = client._client_id.decode()
    if reason_code == 0:
        print(f"[DISCONNECTED] {client_id} (clean)")
    else:
        print(f"[DISCONNECTED] {client_id} (unexpected: {reason_code})")


# ── TODO(human): Implement these functions ───────────────────────────


def publish_retained_status(
    client: mqtt.Client,
    device_id: str,
    status: str,
) -> None:
    """Publish a device status as a retained message.

    TODO(human): Implement this function.

    Steps:
      1. Build the topic string:
             topic = STATUS_TOPIC_TEMPLATE.format(device_id=device_id)
         Example: "devices/sensor-001/status"

      2. Publish with the retain flag set to True:
             result = client.publish(
                 topic=topic,
                 payload=status,
                 qos=1,
                 retain=True,
             )

         - retain=True: This is the critical flag. When set, the broker
           STORES this message as the "retained message" for this topic.
           Only ONE retained message is stored per topic -- each new
           retained publish replaces the previous one.

         - When ANY client subscribes to this topic (or a wildcard that
           matches it), the broker IMMEDIATELY delivers the retained
           message. The subscriber doesn't have to wait for the next
           publish. This is the key difference from normal publishing.

         - The subscriber receives the message with msg.retain = True,
           so it can distinguish retained messages from live ones.

      3. Wait for publish acknowledgment:
             result.wait_for_publish(timeout=5)

      4. Print the result:
             print(f"  [PUB RETAINED] {topic} = {status}")

    Key concept -- retained messages:
      Retained messages solve the "late joiner" problem:
      - Without retained: A new subscriber must wait for the next publish
        to learn the current state. If publishes are infrequent (e.g.,
        device status changes), this could be minutes or hours.
      - With retained: The subscriber gets the current state immediately
        upon subscribing, then receives live updates as they happen.

      To CLEAR a retained message, publish an empty payload (b"") with
      retain=True to the same topic. The broker removes the retained
      message and stops delivering it to new subscribers.

      Retained messages are independent of sessions -- they persist
      even if no clients are connected, surviving broker restarts if
      persistence is enabled (which we configured in mosquitto.conf).

    Docs: https://www.hivemq.com/blog/mqtt-essentials-part-8-retained-messages/
    """
    raise NotImplementedError(
        "TODO(human): Publish device status as a retained message"
    )


def setup_last_will(
    client: mqtt.Client,
    device_id: str,
) -> None:
    """Configure Last Will and Testament (LWT) for a device client.

    TODO(human): Implement this function.

    IMPORTANT: will_set() MUST be called BEFORE connect(). The will
    message is registered with the broker during the CONNECT handshake.
    Calling will_set() after connect() has no effect.

    Steps:
      1. Build the will topic:
             will_topic = STATUS_TOPIC_TEMPLATE.format(device_id=device_id)

      2. Build the will payload. This is what the broker publishes when
         this client disconnects ungracefully:
             will_payload = "offline"

      3. Set the will message on the client:
             client.will_set(
                 topic=will_topic,
                 payload=will_payload,
                 qos=1,
                 retain=True,
             )

         - topic: The topic where the will message will be published.
           We use the same status topic so the "offline" message replaces
           the "online" retained message.
         - payload: The message content. "offline" indicates the device
           is no longer reachable.
         - qos=1: At-least-once delivery for the will message. We want
           to make sure subscribers are notified.
         - retain=True: CRITICAL. By setting the will as retained, the
           "offline" status becomes the new retained message for this
           topic. Any NEW subscriber will see "offline" immediately.
           Without retain=True, only currently-connected subscribers
           would see the "offline" message.

         How LWT works internally:
         - During CONNECT, the client sends the will topic, payload, QoS,
           and retain flag to the broker as part of the CONNECT packet.
         - The broker stores this will message in memory.
         - If the client disconnects CLEANLY (sends DISCONNECT packet),
           the broker DISCARDS the will message.
         - If the client disconnects UNGRACEFULLY (TCP drops, keepalive
           timeout, protocol error), the broker PUBLISHES the will.
         - In MQTT 5.0, a Will Delay Interval can be set to wait N
           seconds before publishing the will, giving the client a
           chance to reconnect.

      4. Print confirmation:
             print(f"  [LWT SET] {will_topic} = {will_payload} (retain=True)")

    Key concept -- device presence pattern:
      The combination of retained messages + LWT creates a robust
      device presence system:
        1. Client calls will_set(topic="devices/X/status", payload="offline", retain=True)
        2. Client connects and publishes "online" (retained) to "devices/X/status"
        3. While connected, publishes telemetry normally.
        4. If client crashes: broker publishes "offline" (retained).
        5. Any monitoring dashboard subscribing to "devices/+/status"
           immediately gets the current state of ALL devices.

      This pattern is one of MQTT's killer features -- no equivalent
      exists in Kafka, RabbitMQ, or Cloud Pub/Sub without custom logic.

    Docs: https://www.hivemq.com/blog/mqtt-essentials-part-9-last-will-and-testament/
    """
    raise NotImplementedError(
        "TODO(human): Configure Last Will and Testament before connecting"
    )


def demonstrate_device_presence(
    monitor_client: mqtt.Client,
    device_states: dict[str, DeviceState],
) -> None:
    """Demonstrate the full device presence pattern with retained + LWT.

    TODO(human): Implement this function.

    This function orchestrates a complete device presence scenario:
    a device comes online, publishes telemetry, then crashes (simulated
    by closing the socket without DISCONNECT), triggering the LWT.

    Steps:
      1. Subscribe the monitor_client to all device status topics:
             monitor_client.subscribe("devices/+/status", qos=1)
             time.sleep(0.5)
         The "+" wildcard matches any device_id, so we see status
         updates from all devices.

      2. For each device_id in DEVICE_IDS[:2] (use first 2 devices):
         a. Create a new device client:
                device_client = mqtt.Client(
                    callback_api_version=CallbackAPIVersion.VERSION2,
                    client_id=device_id,
                    protocol=MQTTProtocolVersion.MQTTv5,
                )

         b. Configure the Last Will BEFORE connecting:
                setup_last_will(device_client, device_id)

         c. Connect the device to the broker:
                device_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
                device_client.loop_start()
                time.sleep(0.3)

         d. Publish "online" as retained to the status topic:
                publish_retained_status(device_client, device_id, "online")
                time.sleep(0.5)

         e. Publish some telemetry data (not retained, normal messages):
                telemetry_topic = TELEMETRY_TOPIC_TEMPLATE.format(device_id=device_id)
                device_client.publish(
                    telemetry_topic,
                    json.dumps({"temp": 22.5, "humidity": 45.0}),
                    qos=1,
                )

         f. Print the device status:
                print(f"  Device {device_id}: online, sending telemetry")

      3. Simulate an ungraceful disconnect for the first device:
         The key is to close the underlying socket WITHOUT sending
         a DISCONNECT packet. This triggers the LWT.

             print(f"\\n--- Simulating crash of {DEVICE_IDS[0]} ---")

         To simulate a crash, close the socket directly:
             first_device_client._sock_close()

         OR use the internal disconnect with a non-zero reason code.
         The simplest approach: call loop_stop() then close the socket:
             first_device_client.loop_stop()
             # Force-close the underlying socket (bypass clean DISCONNECT)
             if first_device_client._sock is not None:
                 first_device_client._sock.close()
                 first_device_client._sock = None

         Note: This is intentionally "wrong" -- in production you'd never
         do this. We're simulating a device crash to trigger the LWT.

         Wait for the broker to detect the disconnect and publish the will.
         The broker uses the keepalive interval (default 60s) to detect
         dead connections. For faster detection in testing, the keepalive
         was set to a lower value. However, ungraceful socket close is
         typically detected quickly (within a few seconds).
             time.sleep(3)

      4. Cleanly disconnect the second device:
             second_device_client.disconnect()
             second_device_client.loop_stop()
             time.sleep(1)

         A clean DISCONNECT does NOT trigger the LWT -- the broker
         discards the will message.

      5. Now demonstrate the "late joiner" behavior:
         Create a NEW monitoring client and subscribe to device status:
             late_client = mqtt.Client(
                 callback_api_version=CallbackAPIVersion.VERSION2,
                 client_id="late-monitor",
                 protocol=MQTTProtocolVersion.MQTTv5,
             )
             late_client.on_message = on_message
             late_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
             late_client.loop_start()
             late_client.subscribe("devices/+/status", qos=1)
             time.sleep(1.5)

         The late joiner should IMMEDIATELY receive the retained status
         messages: "offline" for the crashed device, "online" for the
         cleanly disconnected one (or "offline" if you also published
         retained "offline" on clean disconnect).

         Print what the late joiner sees:
             print("  Late joiner should see retained status for all devices")

      6. Cleanup:
             late_client.loop_stop()
             late_client.disconnect()

      7. Clean up retained messages (good practice for testing):
         For each device_id, publish empty payload with retain=True
         to clear the retained message:
             for device_id in DEVICE_IDS[:2]:
                 topic = STATUS_TOPIC_TEMPLATE.format(device_id=device_id)
                 monitor_client.publish(topic, payload=b"", qos=1, retain=True)

    Key insight:
      The device presence pattern shows MQTT's unique strength: the
      BROKER manages presence state, not the application. In Kafka or
      RabbitMQ, you'd need custom logic (heartbeat topics, timeout
      detection, state stores) to achieve the same result. In MQTT,
      it's built into the protocol with retained messages + LWT.

    Docs:
      - Retained: https://www.emqx.com/en/blog/mqtt5-features-retain-message
      - LWT: https://www.hivemq.com/blog/mqtt-essentials-part-9-last-will-and-testament/
    """
    raise NotImplementedError(
        "TODO(human): Orchestrate the device presence demonstration"
    )


# ── Orchestration (boilerplate) ──────────────────────────────────────


def create_monitor_client(
    device_states: dict[str, DeviceState],
) -> mqtt.Client:
    """Create a monitoring client that tracks device states."""
    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="device-monitor",
        protocol=MQTTProtocolVersion.MQTTv5,
        userdata=device_states,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    return client


def main() -> None:
    print("=" * 60)
    print("EXERCISE 3: Retained Messages & Last Will (Device Presence)")
    print("=" * 60)

    # Initialize device state tracking
    device_states = {did: DeviceState(device_id=did) for did in DEVICE_IDS}

    # Create the monitoring client
    monitor_client = create_monitor_client(device_states)
    monitor_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
    monitor_client.loop_start()

    time.sleep(0.5)

    # Part 1: Demonstrate retained messages
    print("\n--- Part 1: Retained Messages ---")
    print("Publishing retained status for devices...\n")
    publish_retained_status(monitor_client, "sensor-001", "online")
    publish_retained_status(monitor_client, "sensor-002", "online")
    publish_retained_status(monitor_client, "sensor-003", "standby")
    time.sleep(1)

    # Create a NEW subscriber to show it gets retained messages immediately
    print("\n--- New subscriber joining (should get retained messages) ---")
    late_sub = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="late-subscriber",
        protocol=MQTTProtocolVersion.MQTTv5,
    )
    late_sub.on_message = on_message
    late_sub.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
    late_sub.loop_start()
    late_sub.subscribe("devices/+/status", qos=1)
    time.sleep(1.5)
    late_sub.loop_stop()
    late_sub.disconnect()

    # Part 2: Demonstrate LWT with device presence
    print("\n--- Part 2: Last Will & Device Presence ---")
    demonstrate_device_presence(monitor_client, device_states)

    # Summary
    print("\n" + "=" * 60)
    print("DEVICE STATE SUMMARY")
    print("=" * 60)
    for did, state in device_states.items():
        print(f"  {did}: status={state.last_status}, updates={state.status_updates}")

    # Cleanup
    # Clear all retained messages
    for did in DEVICE_IDS:
        topic = STATUS_TOPIC_TEMPLATE.format(device_id=did)
        monitor_client.publish(topic, payload=b"", qos=1, retain=True)

    time.sleep(0.5)
    monitor_client.loop_stop()
    monitor_client.disconnect()

    print("\nDone. All retained messages cleared. Monitor disconnected.")


if __name__ == "__main__":
    main()
