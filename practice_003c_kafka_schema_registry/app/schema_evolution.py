"""Schema evolution demonstrations.

Demonstrates how the Schema Registry enforces compatibility rules
when schemas change over time. This is the core "data contract"
concept: schemas are contracts between producers and consumers,
and the registry ensures changes don't break existing applications.

Compatibility modes control what changes are allowed:
  - BACKWARD: new schema can READ data written with old schema
  - FORWARD: old schema can READ data written with new schema
  - FULL: both BACKWARD and FORWARD
  - NONE: no compatibility checks (dangerous in production)
  - *_TRANSITIVE: checks against ALL previous versions, not just latest

Also demonstrates parsing the Confluent wire format -- the 5-byte
prefix that every Avro-serialized Kafka message carries.

Run after registry_explorer.py:
    uv run python schema_evolution.py
"""

import json
import struct

import requests

import config
import schemas


# ── Constants ────────────────────────────────────────────────────────

REGISTRY_CONTENT_TYPE = "application/vnd.schemaregistry.v1+json"


# ── Helpers (boilerplate) ────────────────────────────────────────────


def register_schema(registry_url: str, subject: str, schema: dict) -> int:
    """Register a schema and return the schema ID. (Helper -- fully implemented.)"""
    resp = requests.post(
        f"{registry_url}/subjects/{subject}/versions",
        headers={"Content-Type": REGISTRY_CONTENT_TYPE},
        json={"schema": json.dumps(schema)},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def check_compat(
    registry_url: str, subject: str, schema: dict
) -> tuple[bool, list[str]]:
    """Check compatibility. (Helper -- fully implemented.)"""
    resp = requests.post(
        f"{registry_url}/compatibility/subjects/{subject}/versions/latest",
        headers={"Content-Type": REGISTRY_CONTENT_TYPE},
        json={"schema": json.dumps(schema)},
        params={"verbose": "true"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("is_compatible", False), data.get("messages", [])


def get_subject_compatibility(registry_url: str, subject: str) -> str:
    """Get the compatibility level for a subject. (Helper -- fully implemented.)"""
    try:
        resp = requests.get(f"{registry_url}/config/{subject}")
        resp.raise_for_status()
        return resp.json().get("compatibilityLevel", "UNKNOWN")
    except requests.exceptions.HTTPError:
        # Subject has no override; using global default
        resp = requests.get(f"{registry_url}/config")
        resp.raise_for_status()
        return resp.json().get("compatibilityLevel", "UNKNOWN")


def set_subject_compatibility(
    registry_url: str, subject: str, level: str
) -> str:
    """Set compatibility level for a subject. (Helper -- fully implemented.)"""
    resp = requests.put(
        f"{registry_url}/config/{subject}",
        headers={"Content-Type": REGISTRY_CONTENT_TYPE},
        json={"compatibility": level},
    )
    resp.raise_for_status()
    return resp.json().get("compatibility", level)


def delete_subject(registry_url: str, subject: str) -> None:
    """Soft-delete a subject (all versions). (Helper -- fully implemented.)"""
    resp = requests.delete(f"{registry_url}/subjects/{subject}")
    if resp.status_code == 404:
        return  # Already deleted
    resp.raise_for_status()
    # Hard delete (permanent)
    resp = requests.delete(
        f"{registry_url}/subjects/{subject}",
        params={"permanent": "true"},
    )
    # Ignore errors on hard delete (may fail if not soft-deleted yet)


def print_separator(title: str) -> None:
    """Print a section separator."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ── TODO(human): Implement these functions ───────────────────────────


def demonstrate_backward_compatibility(registry_url: str) -> None:
    """Demonstrate BACKWARD compatibility mode.

    TODO(human): Implement this function.

    BACKWARD compatibility means: a consumer using the NEW schema can
    read data written with the OLD schema. This is the default mode
    and the most common in practice.

    What's allowed under BACKWARD:
      - Adding a field WITH a default value (new consumers use default
        when reading old data that lacks the field)
      - Removing a field (new consumers simply ignore it)

    What's NOT allowed under BACKWARD:
      - Adding a REQUIRED field (no default) -- old data can't provide it
      - Changing a field's type incompatibly

    Steps:
      1. Use a fresh subject name: "compat-backward-test-value"
         First, clean up any leftover state:
             delete_subject(registry_url, subject)

      2. Register User v1 schema:
             schema_id = register_schema(registry_url, subject, schemas.USER_V1)
         Print the schema ID and fields.

      3. Check if User v2 is backward-compatible with v1:
             is_compat, messages = check_compat(registry_url, subject, schemas.USER_V2)
         Print the result. This should PASS because v2 adds "email" with
         a default of null -- old data without "email" gets null.

      4. Register User v2 (since it's compatible):
             schema_id = register_schema(registry_url, subject, schemas.USER_V2)
         Print the schema ID.

      5. Check if USER_BREAKING is compatible with v2:
             is_compat, messages = check_compat(registry_url, subject, schemas.USER_BREAKING)
         Print the result. This should FAIL because USER_BREAKING adds
         "phone" as a REQUIRED field (no default). A consumer using the
         new schema can't read old v2 data that has no "phone" field.

      6. Print a summary explaining why the break happened.

    Why this matters:
      BACKWARD is the safest default because you typically upgrade
      consumers BEFORE producers. New consumers must handle both old
      and new data during the rolling deployment window.

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#backward-compatibility
    """
    raise NotImplementedError("TODO(human)")


def demonstrate_forward_compatibility(registry_url: str) -> None:
    """Demonstrate FORWARD compatibility mode.

    TODO(human): Implement this function.

    FORWARD compatibility means: a consumer using the OLD schema can
    read data written with the NEW schema. This is the opposite of
    BACKWARD -- useful when you upgrade producers BEFORE consumers.

    What's allowed under FORWARD:
      - Removing a field (old consumers that expect it will use its
        default value from their schema, or ignore the absence)
      - Adding a field (old consumers simply ignore unknown fields)

    What's NOT allowed under FORWARD:
      - Removing a REQUIRED field that has no default in the old schema
      - Changing a field type incompatibly

    Steps:
      1. Use a fresh subject name: "compat-forward-test-value"
         Clean up: delete_subject(registry_url, subject)

      2. Set compatibility to FORWARD:
             set_subject_compatibility(registry_url, subject, "FORWARD")
         Print the current level to confirm.

      3. Register User v1:
             register_schema(registry_url, subject, schemas.USER_V1)

      4. Check if User v2 is forward-compatible:
             is_compat, messages = check_compat(registry_url, subject, schemas.USER_V2)
         Print result. Under FORWARD, adding an optional field should
         be compatible because old consumers (v1) simply ignore "email".

      5. Register User v2.

      6. Check if User v3 is forward-compatible:
             is_compat, messages = check_compat(registry_url, subject, schemas.USER_V3)
         Print result. v3 adds another optional field ("age") -- should pass.

      7. Print a summary explaining FORWARD vs BACKWARD.

    Why this matters:
      In blue/green deployments where producers are upgraded first,
      FORWARD ensures old consumers can still read the new data format.
      The choice between BACKWARD and FORWARD depends on your deployment
      strategy (consumers-first vs producers-first).

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#forward-compatibility
    """
    raise NotImplementedError("TODO(human)")


def demonstrate_full_compatibility(registry_url: str) -> None:
    """Demonstrate FULL compatibility mode.

    TODO(human): Implement this function.

    FULL compatibility = BACKWARD + FORWARD combined. Both old and new
    schemas must be able to read each other's data. This is the most
    restrictive mode and provides the strongest contract guarantees.

    What's allowed under FULL:
      - Adding an optional field with a default (works in both directions)
      - Removing an optional field with a default (both directions)

    What's NOT allowed under FULL:
      - Adding or removing REQUIRED fields (breaks one direction)
      - Any type change that's not symmetric

    Steps:
      1. Use a fresh subject name: "compat-full-test-value"
         Clean up: delete_subject(registry_url, subject)

      2. Set compatibility to FULL:
             set_subject_compatibility(registry_url, subject, "FULL")
         Print the current level.

      3. Register User v1.

      4. Check v2 compatibility. Adding optional "email" with default
         should pass FULL because:
           - BACKWARD: new consumer reads old data -> email defaults to null
           - FORWARD: old consumer reads new data -> ignores "email"

      5. Register User v2.

      6. Check if USER_BREAKING is compatible. Should FAIL:
           - BACKWARD fails: new schema has required "phone", old data lacks it
           - (FORWARD would also fail in practice for the same schema)

      7. Check v3 compatibility. Adding optional "age" should pass.

      8. Print a summary comparing BACKWARD vs FORWARD vs FULL.

    Why this matters:
      FULL compatibility is recommended when producers and consumers
      are deployed independently and you can't control upgrade order.
      It's the gold standard for data contracts in microservices.

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#full-compatibility
    """
    raise NotImplementedError("TODO(human)")


def analyze_wire_format(raw_bytes: bytes) -> dict:
    """Parse the Confluent wire format prefix from raw Avro message bytes.

    TODO(human): Implement this function.

    Every Avro-serialized message produced by the Confluent serializer
    starts with a 5-byte prefix BEFORE the actual Avro data:

        Byte 0:      Magic byte (always 0x00)
        Bytes 1-4:   Schema ID (4-byte big-endian unsigned int)
        Bytes 5+:    Avro binary payload

    This wire format is how consumers know which schema was used to
    encode the message. The consumer reads the schema ID, fetches that
    schema from the registry, and uses it as the "writer schema" for
    Avro decoding.

    Steps:
      1. Validate that raw_bytes is at least 5 bytes long.
         If not, return {"error": "Message too short for wire format"}.

      2. Extract the magic byte:
             magic_byte = raw_bytes[0]
         Verify it equals 0 (0x00). The magic byte identifies this as
         a Confluent-encoded Avro message (vs plain Avro or other formats).

      3. Extract the schema ID using struct.unpack:
             schema_id = struct.unpack(">I", raw_bytes[1:5])[0]
         ">I" means big-endian unsigned 4-byte integer.

      4. The remaining bytes are the Avro payload:
             avro_payload = raw_bytes[5:]

      5. Return a dict:
             {
                 "magic_byte": magic_byte,
                 "schema_id": schema_id,
                 "avro_payload_size": len(avro_payload),
                 "total_size": len(raw_bytes),
                 "overhead_bytes": 5,
             }

    Why this matters:
      Understanding the wire format is crucial for debugging serialization
      issues. If a consumer can't deserialize a message, inspecting the
      wire format tells you: is the magic byte correct (is this Avro at
      all)? What schema ID was used? Does that ID exist in the registry?
      This is also the first thing you check when diagnosing "schema not
      found" errors in production.

    Docs: https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#wire-format
    """
    raise NotImplementedError("TODO(human)")


# ── Orchestration (boilerplate) ──────────────────────────────────────


def main() -> None:
    url = config.SCHEMA_REGISTRY_URL

    print("=== Schema Evolution Demonstrations ===\n")
    print(f"Registry: {url}\n")

    # ── Part 1: BACKWARD compatibility (default) ─────────────────
    print_separator("Part 1: BACKWARD Compatibility (default)")
    demonstrate_backward_compatibility(url)

    # ── Part 2: FORWARD compatibility ────────────────────────────
    print_separator("Part 2: FORWARD Compatibility")
    demonstrate_forward_compatibility(url)

    # ── Part 3: FULL compatibility ───────────────────────────────
    print_separator("Part 3: FULL Compatibility")
    demonstrate_full_compatibility(url)

    # ── Part 4: Wire format analysis ─────────────────────────────
    print_separator("Part 4: Wire Format Analysis")

    # Construct a sample wire format message for analysis
    # Magic byte (0x00) + schema ID (1 as big-endian) + some Avro bytes
    sample_wire = b"\x00" + struct.pack(">I", 1) + b"\x02\x0aAlice"
    result = analyze_wire_format(sample_wire)
    print(f"Sample wire format analysis:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Test with a real message if possible (requires avro_producer to have run)
    print("\nTo analyze real messages, run avro_producer.py first,")
    print("then inspect raw message bytes with avro_consumer.py.")

    print_separator("Done!")


if __name__ == "__main__":
    main()
