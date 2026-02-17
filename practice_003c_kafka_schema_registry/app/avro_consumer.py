"""Avro-deserialized Kafka consumer.

Demonstrates consuming Kafka messages that were produced with Avro
serialization. The AvroDeserializer:
  1. Reads the 5-byte wire format prefix to extract the schema ID
  2. Fetches the writer's schema from the registry (cached locally)
  3. Decodes the Avro binary payload using the writer's schema
  4. Optionally applies schema evolution (reader vs writer schema)

Run after avro_producer.py has produced messages:
    uv run python avro_consumer.py
"""

import json
import signal
import time

from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.serialization import (
    MessageField,
    SerializationContext,
    StringDeserializer,
)
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer

import config
import schemas


# ── Graceful shutdown flag ───────────────────────────────────────────

_running = True


def _signal_handler(signum, frame) -> None:
    """Set the shutdown flag on Ctrl+C."""
    global _running
    print("\nShutdown requested...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


# ── Helpers (boilerplate) ────────────────────────────────────────────


def dict_to_user(data: dict, ctx: SerializationContext) -> dict:
    """Convert deserialized Avro dict back to a user dict.

    The AvroDeserializer requires a `from_dict` callable that transforms
    the deserialized Avro data into your domain object. Since we're
    working with plain dicts, this is an identity function.

    In production, this would construct a dataclass/Pydantic model:
        def dict_to_user(data: dict, ctx) -> User:
            return User(**data)
    """
    return data


# ── TODO(human): Implement these functions ───────────────────────────


def create_avro_consumer(
    schema_str: str, group_id: str
) -> tuple[Consumer, AvroDeserializer]:
    """Create a Kafka consumer with Avro deserialization support.

    TODO(human): Implement this function.

    This mirrors the producer setup but for deserialization. The key
    insight is that the consumer can use a DIFFERENT schema (the
    "reader schema") than what the producer used (the "writer schema").
    Avro handles the translation automatically -- this is schema
    evolution in action.

    Steps:
      1. Create a SchemaRegistryClient:
             schema_registry = SchemaRegistryClient({"url": config.SCHEMA_REGISTRY_URL})

      2. Create an AvroDeserializer:
             deserializer = AvroDeserializer(
                 schema_registry_client=schema_registry,
                 schema_str=schema_str,
                 from_dict=dict_to_user,
             )
         Parameters:
           - schema_str: the reader's schema. If this differs from the
             writer's schema, Avro applies resolution rules (e.g., fills
             in default values for new fields, ignores removed fields).
           - from_dict: callable that converts the deserialized dict to
             your domain object.

      3. Create a Consumer:
             consumer = Consumer({
                 "bootstrap.servers": config.BOOTSTRAP_SERVERS,
                 "group.id": group_id,
                 "auto.offset.reset": "earliest",
                 "enable.auto.commit": False,
             })

      4. Return the tuple (consumer, deserializer).

    Why this matters:
      The consumer doesn't need to know which schema version the
      producer used. The wire format prefix tells the deserializer
      which schema was used to write, and the reader schema tells it
      what the consumer expects. Avro bridges the gap automatically.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#avrodeserializer
    """
    raise NotImplementedError("TODO(human)")


def consume_users(
    consumer: Consumer,
    deserializer: AvroDeserializer,
    topic: str,
    max_messages: int = 50,
) -> list[dict]:
    """Consume and deserialize Avro messages from a topic.

    TODO(human): Implement this function.

    This follows the same poll-loop pattern from practice 003a, but
    instead of json.loads() you use the AvroDeserializer to decode
    message values. The deserializer handles fetching the writer's
    schema from the registry and applying schema evolution rules.

    Steps:
      1. Create a StringDeserializer for keys:
             key_deserializer = StringDeserializer("utf_8")

      2. Subscribe to the topic:
             consumer.subscribe([topic])

      3. Create an empty list to collect results.

      4. Enter a poll loop (while _running and count < max_messages):
         a. Call consumer.poll(timeout=1.0).

         b. If None: increment a consecutive-None counter. After 30
            consecutive Nones, break. (Remember from 003a: the first
            few polls return None while the consumer joins the group.)

         c. If the message has an error:
            - KafkaError._PARTITION_EOF: print info and continue.
            - Otherwise: raise KafkaException.

         d. Reset the consecutive-None counter.

         e. Deserialize the key:
                key = key_deserializer(msg.key(), SerializationContext(
                    topic, MessageField.KEY
                ))

         f. Deserialize the value using the AvroDeserializer:
                user = deserializer(msg.value(), SerializationContext(
                    topic, MessageField.VALUE
                ))
            The deserializer:
              1. Reads bytes 0: magic byte (0x00)
              2. Reads bytes 1-4: schema ID (big-endian int)
              3. Fetches the writer schema from registry by ID (cached)
              4. Decodes the Avro payload using writer + reader schemas
              5. Calls from_dict() on the result

         g. Print the deserialized user: partition, offset, key, user dict.

         h. Append the user dict to results.

         i. Commit the offset:
                consumer.commit(message=msg, asynchronous=False)

      5. Return the results list.

    Why this matters:
      This demonstrates the full round-trip: producer writes with
      schema X, consumer reads with schema Y, and Avro resolves
      differences. Understanding this flow is essential for evolving
      schemas in production without downtime.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Consumer.poll
    """
    raise NotImplementedError("TODO(human)")


# ── Orchestration (boilerplate) ──────────────────────────────────────


def main() -> None:
    schema_str = json.dumps(schemas.USER_V2)

    print("=== Avro Consumer: Consuming users from Schema Registry ===\n")
    print(f"Topic: {config.USERS_TOPIC}")
    print(f"Group: {config.USERS_GROUP}")
    print(f"Schema Registry: {config.SCHEMA_REGISTRY_URL}\n")

    consumer, deserializer = create_avro_consumer(schema_str, config.USERS_GROUP)

    try:
        users = consume_users(consumer, deserializer, config.USERS_TOPIC)
        print(f"\nConsumed {len(users)} users total.")
    finally:
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
