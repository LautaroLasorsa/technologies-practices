"""Multiple schemas in a single Kafka topic.

Demonstrates the RecordNameStrategy for subject naming, which allows
different event types (with different schemas) to coexist in the same
topic. This is a common pattern for event-driven architectures where
a single topic carries multiple event types (e.g., "sensor-events"
contains both SensorReading and SensorAlert).

Subject naming strategies:
  - TopicNameStrategy (default): subject = "<topic>-value"
    One schema per topic. Simple, but limiting.
  - RecordNameStrategy: subject = "<record_fullname>"
    Subject is the fully-qualified Avro record name (namespace.name).
    Multiple schemas per topic.
  - TopicRecordNameStrategy: subject = "<topic>-<record_fullname>"
    Combines both. Different schemas per topic, but the same record
    type in different topics gets different subjects.

Run after admin.py has created topics:
    uv run python multi_schema_topic.py
"""

import json
import random
import signal
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.schema_registry.common import schema_registry_client
from confluent_kafka.serialization import (
    MessageField,
    SerializationContext,
    StringDeserializer,
    StringSerializer,
)
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer

import config
import schemas


# ── Graceful shutdown flag ───────────────────────────────────────────

_running = True


def _signal_handler(signum, frame) -> None:
    global _running
    print("\nShutdown requested...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


# ── Helpers (boilerplate) ────────────────────────────────────────────


def identity_to_dict(obj: dict, ctx: SerializationContext) -> dict:
    """Identity function for Avro serializer to_dict callback."""
    return obj


def identity_from_dict(data: dict, ctx: SerializationContext) -> dict:
    """Identity function for Avro deserializer from_dict callback."""
    return data


def delivery_report(err, msg) -> None:
    """Delivery callback for produced messages."""
    if err is not None:
        print(f"  FAILED: {err}")
    else:
        print(
            f"  Delivered: partition={msg.partition()} "
            f"offset={msg.offset()} "
            f"key={msg.key()} "
            f"size={len(msg.value()) if msg.value() else 0}B"
        )


def generate_sensor_readings(count: int) -> list[dict]:
    """Generate sample sensor reading events."""
    sensors = ["sensor-001", "sensor-002", "sensor-003", "sensor-004"]
    readings = []
    for _ in range(count):
        readings.append({
            "sensor_id": random.choice(sensors),
            "temperature": round(random.uniform(15.0, 45.0), 2),
            "timestamp": int(time.time() * 1000),
        })
    return readings


def generate_sensor_alerts(count: int) -> list[dict]:
    """Generate sample sensor alert events."""
    sensors = ["sensor-001", "sensor-002", "sensor-003"]
    alert_types = ["HIGH_TEMP", "LOW_TEMP", "RATE_OF_CHANGE", "SENSOR_OFFLINE"]
    alerts = []
    for _ in range(count):
        alerts.append({
            "sensor_id": random.choice(sensors),
            "alert_type": random.choice(alert_types),
            "temperature": round(random.uniform(40.0, 60.0), 2),
            "timestamp": int(time.time() * 1000),
        })
    return alerts


# ── TODO(human): Implement these functions ───────────────────────────


def configure_record_name_strategy() -> (
    tuple[Producer, AvroSerializer, AvroSerializer]
):
    """Set up producer with RecordNameStrategy for multi-schema topics.

    TODO(human): Implement this function.

    By default, the AvroSerializer uses TopicNameStrategy, which derives
    the subject name from the topic: "sensor-readings-value". This means
    every message on that topic must use the same schema.

    RecordNameStrategy instead derives the subject from the Avro record's
    fully-qualified name: "com.practice.sensors.SensorReading" and
    "com.practice.sensors.SensorAlert". This allows different event types
    to live in the same topic, each with their own schema and evolution path.

    Steps:
      1. Create a SchemaRegistryClient:
             schema_registry = SchemaRegistryClient({"url": config.SCHEMA_REGISTRY_URL})

      2. Create an AvroSerializer for SensorReading:
             reading_serializer = AvroSerializer(
                 schema_registry_client=schema_registry,
                 schema_str=json.dumps(schemas.SENSOR_READING),
                 to_dict=identity_to_dict,
                 conf={"subject.name.strategy": "record_name_strategy"},
             )
         The key configuration is "subject.name.strategy". By setting it
         to "record_name_strategy", the serializer will use the Avro
         record's namespace + name as the subject instead of the topic name.
         For SensorReading, this means the subject will be:
             "com.practice.sensors.SensorReading"

      3. Create an AvroSerializer for SensorAlert (same pattern):
             alert_serializer = AvroSerializer(
                 schema_registry_client=schema_registry,
                 schema_str=json.dumps(schemas.SENSOR_ALERT),
                 to_dict=identity_to_dict,
                 conf={"subject.name.strategy": "record_name_strategy"},
             )

      4. Create a Producer:
             producer = Producer({"bootstrap.servers": config.BOOTSTRAP_SERVERS})

      5. Return the tuple (producer, reading_serializer, alert_serializer).

    Why this matters:
      In event-driven architectures, a single topic often carries multiple
      event types (e.g., "order-events" with OrderCreated, OrderShipped,
      OrderCancelled). RecordNameStrategy lets each event type have its
      own schema versioning and compatibility rules, while still benefiting
      from Kafka's ordering guarantees within the topic.

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#subject-name-strategy
    """
    from confluent_kafka.schema_registry import record_subject_name_strategy

    schema_registry = SchemaRegistryClient({"url":config.SCHEMA_REGISTRY_URL})
    reading_seralizer = AvroSerializer(
        schema_registry_client = schema_registry,
        schema_str = json.dumps(schemas.SENSOR_READING),
        to_dict = identity_to_dict,
        conf={"subject.name.strategy": record_subject_name_strategy}
    )
    alert_serializer = AvroSerializer(
        schema_registry_client = schema_registry,
        schema_str = json.dumps(schemas.SENSOR_ALERT),
        to_dict = identity_to_dict,
        conf={"subject.name.strategy": record_subject_name_strategy}
    )
    producer = Producer({"bootstrap.servers": config.BOOTSTRAP_SERVERS})
    return (producer, reading_seralizer, alert_serializer)

def produce_mixed_events(
    producer: Producer,
    reading_serializer: AvroSerializer,
    alert_serializer: AvroSerializer,
    num_readings: int = 8,
    num_alerts: int = 4,
) -> int:
    """Produce both SensorReading and SensorAlert events to the same topic.

    TODO(human): Implement this function.

    This demonstrates the key benefit of RecordNameStrategy: different
    event types with different schemas flowing through the same topic.
    Each event type gets its own subject in the registry, with independent
    versioning and compatibility rules.

    Steps:
      1. Create a StringSerializer for keys:
             key_serializer = StringSerializer("utf_8")

      2. Generate sample data:
             readings = generate_sensor_readings(num_readings)
             alerts = generate_sensor_alerts(num_alerts)

      3. Interleave readings and alerts for a realistic event stream.
         A simple approach: combine into a list of (event, serializer) tuples:
             events = [(r, reading_serializer) for r in readings]
             events += [(a, alert_serializer) for a in alerts]
             random.shuffle(events)

      4. For each (event, serializer) pair:
         a. Create a SerializationContext:
                ctx = SerializationContext(
                    config.SENSOR_READINGS_TOPIC, MessageField.VALUE
                )

         b. Serialize the value:
                value_bytes = serializer(event, ctx)

         c. Serialize the key (use sensor_id):
                key_bytes = key_serializer(event["sensor_id"])

         d. Produce to the sensor-readings topic:
                producer.produce(
                    topic=config.SENSOR_READINGS_TOPIC,
                    key=key_bytes,
                    value=value_bytes,
                    callback=delivery_report,
                )

         e. Call producer.poll(0).

      5. Flush: producer.flush(timeout=10)
      6. Return total count of events produced.

    Why this matters:
      In practice, you'd have different microservices producing different
      event types to the same topic. The Schema Registry ensures each
      service's schema evolves independently. A sensor monitoring service
      produces SensorReadings while an alerting service produces
      SensorAlerts -- both on the same Kafka topic for unified processing.

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#how-the-naming-strategies-work
    """

    key_serializer = StringSerializer("utf_8")
    readings = generate_sensor_readings(num_readings)
    alerts = generate_sensor_alerts(num_alerts)
    events = [(r, reading_serializer) for r in readings] + [(a, alert_serializer) for a in alerts]
    random.shuffle(events)
    for (event, serializer) in events:
        ctx = SerializationContext(config.SENSOR_READINGS_TOPIC, MessageField.VALUE)
        value_bytes = serializer(event, ctx)
        key = key_serializer(event["sensor_id"])
        producer.produce(
            topic = config.SENSOR_READINGS_TOPIC,
            key = key,
            value = value_bytes,
            callback=delivery_report
        )
        producer.poll(0)

    producer.flush(timeout=10)
    return len(events)

def consume_mixed_events(max_messages: int = 30) -> list[dict]:
    """Consume mixed event types from a single topic.

    TODO(human): Implement this function.

    The consumer side of multi-schema topics is more complex because
    each message might have a different schema. You can't use a single
    AvroDeserializer with a fixed reader schema -- instead, you either:
      a) Use a generic deserializer that reads whatever schema the
         message was written with (no reader schema specified), OR
      b) Inspect the schema ID from the wire format and select the
         appropriate deserializer.

    For this exercise, use approach (a): create an AvroDeserializer
    WITHOUT specifying a reader schema_str. It will use the writer's
    schema (fetched from registry via the wire format schema ID) to
    deserialize each message.

    Steps:
      1. Create a SchemaRegistryClient.

      2. Create an AvroDeserializer WITHOUT schema_str:
             deserializer = AvroDeserializer(
                 schema_registry_client=schema_registry,
                 from_dict=identity_from_dict,
             )
         Without a reader schema, the deserializer uses the writer's
         schema as-is. This works for consuming mixed types because
         each message carries its own schema ID.

      3. Create a StringDeserializer for keys.

      4. Create a Consumer with a unique group ID:
             consumer = Consumer({
                 "bootstrap.servers": config.BOOTSTRAP_SERVERS,
                 "group.id": config.SENSOR_GROUP,
                 "auto.offset.reset": "earliest",
                 "enable.auto.commit": False,
             })

      5. Subscribe to the sensor-readings topic.

      6. Poll loop (same pattern as avro_consumer.py):
         For each valid message:
         a. Deserialize the value.
         b. Determine the event type from the deserialized dict:
            - If "alert_type" in event -> SensorAlert
            - Else -> SensorReading
         c. Print the event with its type.
         d. Commit offset.
         e. Append to results.

      7. Close consumer in a finally block.
      8. Return results.

    Why this matters:
      Consuming multi-schema topics requires the consumer to handle
      polymorphic events. This is the consumer-side pattern for event
      sourcing and CQRS architectures where a single stream carries
      diverse event types. The consumer uses field inspection or
      external type indicators to dispatch events to the right handler.

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#subject-name-strategy
    """

    schema_registry = SchemaRegistryClient({"url":config.SCHEMA_REGISTRY_URL})
    deserializer = AvroDeserializer(
        schema_registry_client = schema_registry,
        from_dict = identity_from_dict
    )
    key_deserializer = StringDeserializer("utf_8")
    consumer = Consumer({
        "bootstrap.servers": config.BOOTSTRAP_SERVERS,
        "group.id": config.SENSOR_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False
    })

    consumer.subscribe([config.SENSOR_READINGS_TOPIC])
    responses = []
    consecutive_none = 0

    while _running and len(responses) < max_messages and consecutive_none < 30:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            consecutive_none += 1
            continue

        consecutive_none = 0

        error = msg.error()
        if error is not None:
            if error == KafkaError._PARTITION_EOF:
                continue
            raise KafkaException(error)

        event: dict = deserializer(msg.value(), SerializationContext(
            config.SENSOR_READINGS_TOPIC,
            MessageField.VALUE
        ))
        key = key_deserializer(msg.key(), SerializationContext(
          config.SENSOR_READINGS_TOPIC,
          MessageField.KEY
        ))

        type = "alert" if "alert_type" in event else "reading"
        print(type, event)
        consumer.commit(message = msg, asynchronous=False)
        responses.append(event)

    return responses
# ── Orchestration (boilerplate) ──────────────────────────────────────


def main() -> None:
    print("=== Multi-Schema Topic: RecordNameStrategy Demo ===\n")
    print(f"Topic: {config.SENSOR_READINGS_TOPIC}")
    print(f"Schemas: SensorReading + SensorAlert in same topic")
    print(f"Strategy: RecordNameStrategy\n")

    # ── Produce mixed events ─────────────────────────────────────
    print("--- Producing mixed events ---\n")
    producer, reading_ser, alert_ser = configure_record_name_strategy()
    count = produce_mixed_events(producer, reading_ser, alert_ser)
    print(f"\nProduced {count} mixed events.\n")

    # ── Consume mixed events ─────────────────────────────────────
    print("--- Consuming mixed events ---\n")
    events = consume_mixed_events()
    print(f"\nConsumed {len(events)} events total.")

    # ── Summary ──────────────────────────────────────────────────
    readings = [e for e in events if "alert_type" not in e]
    alerts = [e for e in events if "alert_type" in e]
    print(f"\n  SensorReadings: {len(readings)}")
    print(f"  SensorAlerts:   {len(alerts)}")


if __name__ == "__main__":
    main()
