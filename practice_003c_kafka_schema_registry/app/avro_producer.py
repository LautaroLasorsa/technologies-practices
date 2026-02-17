"""Avro-serialized Kafka producer.

Demonstrates producing Kafka messages with Avro serialization via the
Confluent Schema Registry. Instead of manually serializing JSON, the
AvroSerializer handles:
  1. Schema registration (auto-registers on first produce)
  2. Avro binary encoding (compact, typed, schema-aware)
  3. Wire format prefix (magic byte + 4-byte schema ID)

The consumer then uses the schema ID from each message to fetch the
correct schema and deserialize -- even if the producer's schema has
evolved since the message was written.

Run after admin.py has created topics:
    uv run python avro_producer.py
"""

import json
import time

from confluent_kafka import Producer
from confluent_kafka.serialization import (
    MessageField,
    SerializationContext,
    StringSerializer,
)
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

import config
import schemas


# ── Sample data ──────────────────────────────────────────────────────

SAMPLE_USERS = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": None},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
    {"id": 4, "name": "Diana", "email": None},
    {"id": 5, "name": "Eve", "email": "eve@example.com"},
    {"id": 6, "name": "Frank", "email": "frank@example.com"},
    {"id": 7, "name": "Grace", "email": None},
    {"id": 8, "name": "Heidi", "email": "heidi@example.com"},
    {"id": 9, "name": "Ivan", "email": None},
    {"id": 10, "name": "Judy", "email": "judy@example.com"},
]


# ── Helpers (boilerplate) ────────────────────────────────────────────


def user_to_dict(user: dict, ctx: SerializationContext) -> dict:
    """Convert a user dict to a dict for Avro serialization.

    The AvroSerializer requires a `to_dict` callable that transforms
    your domain object into a plain dict matching the Avro schema fields.
    Since our data is already a dict, this is an identity function.

    In production, this would convert a dataclass/Pydantic model to dict:
        def user_to_dict(user: User, ctx) -> dict:
            return {"id": user.id, "name": user.name, "email": user.email}

    Args:
        user: The user data dict.
        ctx: SerializationContext (contains topic, field info). Unused here
             but required by the AvroSerializer callback signature.
    """
    return user


def delivery_report(err, msg) -> None:
    """Delivery callback for produced messages."""
    if err is not None:
        print(f"  FAILED: {err}")
    else:
        # msg.value() contains the raw bytes (wire format + Avro)
        raw_size = len(msg.value()) if msg.value() else 0
        print(
            f"  Delivered: partition={msg.partition()} "
            f"offset={msg.offset()} "
            f"key={msg.key()} "
            f"size={raw_size}B"
        )


# ── TODO(human): Implement these functions ───────────────────────────


def create_avro_producer(
    schema_str: str,
) -> tuple[Producer, AvroSerializer]:
    """Create a Kafka producer with Avro serialization support.

    TODO(human): Implement this function.

    This function wires together three components:
      1. SchemaRegistryClient -- connects to the registry REST API
      2. AvroSerializer -- handles schema registration + Avro encoding
      3. Producer -- the confluent-kafka producer for sending messages

    Steps:
      1. Create a SchemaRegistryClient:
             schema_registry = SchemaRegistryClient({"url": config.SCHEMA_REGISTRY_URL})
         This client talks to http://localhost:8081 to register/fetch schemas.

      2. Create an AvroSerializer:
             serializer = AvroSerializer(
                 schema_registry_client=schema_registry,
                 schema_str=schema_str,
                 to_dict=user_to_dict,
             )
         Parameters:
           - schema_registry_client: the client from step 1
           - schema_str: the Avro schema as a JSON string (use json.dumps())
           - to_dict: callable that converts your object to a dict for
             serialization. The AvroSerializer calls this before encoding.

      3. Create a Producer:
             producer = Producer({"bootstrap.servers": config.BOOTSTRAP_SERVERS})

      4. Return the tuple (producer, serializer).

    Why this matters:
      The AvroSerializer auto-registers the schema with the registry on
      first use. It also prepends the 5-byte Confluent wire format
      (0x00 + 4-byte schema ID) before the Avro binary data. This is
      what enables consumers to look up the correct schema for each message.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#avroserializer
    """
    raise NotImplementedError("TODO(human)")


def produce_users(
    producer: Producer,
    serializer: AvroSerializer,
    users: list[dict],
) -> int:
    """Produce user events with Avro serialization.

    TODO(human): Implement this function.

    Unlike plain JSON production (practice 003a), here you pass the
    serialized bytes to producer.produce() as the value. The
    AvroSerializer handles the conversion from dict -> Avro binary
    (with wire format prefix).

    Steps:
      1. Create a StringSerializer for keys:
             key_serializer = StringSerializer("utf_8")
         Keys are still plain strings (user ID), not Avro-encoded.

      2. For each user dict in the users list:
         a. Create a SerializationContext for the value:
                ctx = SerializationContext(
                    config.USERS_TOPIC, MessageField.VALUE
                )
            This tells the serializer which topic and field (key vs value)
            it's serializing for. The serializer uses this to determine
            the subject name (e.g., "users-value") for schema registration.

         b. Serialize the value using the AvroSerializer:
                value_bytes = serializer(user, ctx)
            This calls user_to_dict internally, then Avro-encodes the
            result and prepends the wire format bytes.

         c. Serialize the key:
                key_bytes = key_serializer(str(user["id"]))

         d. Produce the message:
                producer.produce(
                    topic=config.USERS_TOPIC,
                    key=key_bytes,
                    value=value_bytes,
                    callback=delivery_report,
                )

         e. Call producer.poll(0) to trigger delivery callbacks.

      3. After the loop, call producer.flush(timeout=10) to ensure all
         messages are delivered before returning.
      4. Return the count of users produced.

    Why this matters:
      This is the standard pattern for producing Avro messages in the
      Confluent ecosystem. The serializer abstracts away schema
      registration and binary encoding, but understanding the steps
      helps debug issues (wrong subject name, schema mismatch, etc.).

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.serialization.SerializationContext
    """
    raise NotImplementedError("TODO(human)")


# ── Orchestration (boilerplate) ──────────────────────────────────────


def main() -> None:
    schema_str = json.dumps(schemas.USER_V2)

    print("=== Avro Producer: Producing users with Schema Registry ===\n")
    print(f"Topic: {config.USERS_TOPIC}")
    print(f"Schema Registry: {config.SCHEMA_REGISTRY_URL}")
    print(f"Schema: User v2 (id, name, email)\n")

    producer, serializer = create_avro_producer(schema_str)
    count = produce_users(producer, serializer, SAMPLE_USERS)
    print(f"\nDone. Produced {count} users with Avro serialization.")


if __name__ == "__main__":
    main()
