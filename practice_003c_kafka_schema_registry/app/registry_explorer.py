"""Schema Registry REST API exploration.

Demonstrates direct interaction with the Confluent Schema Registry
REST API to understand how schemas are registered, versioned, and
validated for compatibility.

The Schema Registry exposes a REST API at http://localhost:8081.
All requests that include a schema body must use:
    Content-Type: application/vnd.schemaregistry.v1+json

Key concepts:
  - A "subject" is a scope under which schemas are versioned.
    Default naming: <topic>-key or <topic>-value (TopicNameStrategy).
  - Each subject can have multiple versions (1, 2, 3, ...).
  - The registry enforces compatibility rules when registering new versions.

Run after docker compose up:
    uv run python registry_explorer.py
"""

import json
import sys

import requests

import config
import schemas


# ── Constants ────────────────────────────────────────────────────────

# The Schema Registry requires this specific Content-Type header for
# all requests that include schema data in the body. Using plain
# "application/json" will result in a 422 Unprocessable Entity error.
REGISTRY_CONTENT_TYPE = "application/vnd.schemaregistry.v1+json"


# ── TODO(human): Implement these functions ───────────────────────────


def list_subjects(registry_url: str) -> list[str]:
    """List all subjects registered in the Schema Registry.

    TODO(human): Implement this function.

    The Schema Registry organizes schemas by "subject". A subject is
    essentially a namespace for schema versions. By default, when you
    produce Avro data to a topic called "users", two subjects are
    created: "users-key" and "users-value" (TopicNameStrategy).

    Steps:
      1. Make a GET request to: {registry_url}/subjects
         No special headers needed for GET requests.
      2. Check that the response status code is 200 (raise_for_status()).
      3. Parse the JSON response -- it returns a plain list of strings:
             ["users-value", "sensor-readings-value", ...]
      4. Return the list.

    Why this matters:
      Listing subjects is the first step to auditing what schemas exist
      in your registry. In production, you'd use this to verify that
      all expected services have registered their schemas, and to detect
      orphaned subjects from decommissioned services.

    Docs: https://docs.confluent.io/platform/current/schema-registry/develop/api.html#get--subjects
    """
    raise NotImplementedError("TODO(human)")


def register_schema(registry_url: str, subject: str, schema: dict) -> int:
    """Register a new schema version under a subject.

    TODO(human): Implement this function.

    This is how schemas enter the registry. When you register a schema,
    the registry:
      1. Checks if the exact same schema already exists (deduplication).
      2. If new, validates it against the latest version for compatibility.
      3. Assigns a globally unique schema ID and a per-subject version number.

    Steps:
      1. Build the request body as a JSON dict:
             {"schema": json.dumps(schema)}
         NOTE: The "schema" field value must be a JSON *string*, not a
         dict. The registry expects the Avro schema as an escaped JSON
         string inside the request body. This double-encoding is a
         common gotcha.
      2. Make a POST request to: {registry_url}/subjects/{subject}/versions
         Headers: {"Content-Type": REGISTRY_CONTENT_TYPE}
         Body: the JSON dict from step 1
      3. Check response status (raise_for_status()).
      4. Parse the JSON response -- it returns: {"id": <schema_id>}
         The schema ID is a global integer that uniquely identifies this
         schema across ALL subjects. It's the ID embedded in the Confluent
         wire format prefix of every Avro message.
      5. Return the schema ID (int).

    Why this matters:
      The schema ID returned here is what gets embedded in every Kafka
      message's 5-byte prefix. Consumers use this ID to fetch the
      writer's schema from the registry, enabling deserialization even
      when the consumer has a different (compatible) reader schema.

    Docs: https://docs.confluent.io/platform/current/schema-registry/develop/api.html#post--subjects-(string-%20subject)-versions
    """
    raise NotImplementedError("TODO(human)")


def check_compatibility(
    registry_url: str, subject: str, schema: dict
) -> tuple[bool, list[str]]:
    """Check if a new schema is compatible with the latest version.

    TODO(human): Implement this function.

    Before registering a new schema version, you can (and should) test
    whether it's compatible with the existing versions. This is like a
    "dry run" for schema evolution. The registry will tell you exactly
    what's wrong if compatibility fails.

    Steps:
      1. Build the request body (same format as register_schema):
             {"schema": json.dumps(schema)}
      2. Make a POST request to:
             {registry_url}/compatibility/subjects/{subject}/versions/latest?verbose=true
         Headers: {"Content-Type": REGISTRY_CONTENT_TYPE}
         The ?verbose=true query parameter is important -- without it,
         you only get {"is_compatible": true/false}. With verbose=true,
         you also get human-readable error messages explaining WHY a
         schema is incompatible.
      3. Check response status. NOTE: The compatibility endpoint returns
         200 even when the schema is NOT compatible -- the result is in
         the response body, not the status code.
      4. Parse the JSON response:
             {"is_compatible": true/false, "messages": ["..."]}
         The "messages" field contains detailed compatibility errors
         (e.g., "new field 'phone' has no default value").
      5. Return a tuple: (is_compatible: bool, messages: list[str]).

    Why this matters:
      In CI/CD pipelines, you'd call this endpoint before deploying a
      new service version. If the schema change would break consumers,
      you catch it before any messages are produced with the incompatible
      schema -- preventing runtime deserialization failures.

    Docs: https://docs.confluent.io/platform/current/schema-registry/develop/api.html#post--compatibility-subjects-(string-%20subject)-versions-(versionId-%20version)
    """
    raise NotImplementedError("TODO(human)")


def get_schema_versions(registry_url: str, subject: str) -> list[dict]:
    """Get all schema versions registered under a subject.

    TODO(human): Implement this function.

    This retrieves the full history of schema versions for a subject,
    letting you see how the schema has evolved over time.

    Steps:
      1. First, get the list of version numbers:
             GET {registry_url}/subjects/{subject}/versions
         Response: [1, 2, 3, ...]

      2. For each version number, fetch the full schema details:
             GET {registry_url}/subjects/{subject}/versions/{version}
         Response for each:
             {
               "subject": "users-value",
               "version": 1,
               "id": 1,
               "schema": "{\"type\":\"record\",...}"
             }
         Note: the "schema" field is a JSON string, not a parsed dict.
         You'll need json.loads() to parse it into a dict if you want
         to inspect the fields.

      3. Collect results into a list of dicts, each with keys:
             {"version": int, "id": int, "schema": dict}
         Where "schema" is the parsed Avro schema dict.
      4. Return the list sorted by version number.

    Why this matters:
      Schema version history is critical for debugging. When a consumer
      fails to deserialize a message, you check: what schema ID is in
      the message? What version is that? What changed between versions?
      This function gives you the full audit trail.

    Docs: https://docs.confluent.io/platform/current/schema-registry/develop/api.html#get--subjects-(string-%20subject)-versions
    """
    raise NotImplementedError("TODO(human)")


# ── Orchestration (boilerplate) ──────────────────────────────────────


def print_separator(title: str) -> None:
    """Print a section separator."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main() -> None:
    url = config.SCHEMA_REGISTRY_URL

    # ── Step 1: List subjects (should be empty initially) ────────
    print_separator("Step 1: List subjects (should be empty)")
    subjects = list_subjects(url)
    print(f"Subjects: {subjects}")

    # ── Step 2: Register User v1 schema ──────────────────────────
    print_separator("Step 2: Register User v1 schema")
    subject_name = "users-value"
    schema_id = register_schema(url, subject_name, schemas.USER_V1)
    print(f"Registered User v1 with schema ID: {schema_id}")

    # ── Step 3: List subjects again (should show users-value) ────
    print_separator("Step 3: List subjects after registration")
    subjects = list_subjects(url)
    print(f"Subjects: {subjects}")

    # ── Step 4: Check compatibility of v2 (should pass) ──────────
    print_separator("Step 4: Check v2 compatibility (should pass)")
    is_compat, messages = check_compatibility(url, subject_name, schemas.USER_V2)
    print(f"Compatible: {is_compat}")
    if messages:
        print(f"Messages: {messages}")

    # ── Step 5: Register User v2 schema ──────────────────────────
    print_separator("Step 5: Register User v2 schema")
    schema_id_v2 = register_schema(url, subject_name, schemas.USER_V2)
    print(f"Registered User v2 with schema ID: {schema_id_v2}")

    # ── Step 6: Check compatibility of breaking change (should fail)
    print_separator("Step 6: Check breaking change compatibility (should FAIL)")
    is_compat, messages = check_compatibility(
        url, subject_name, schemas.USER_BREAKING
    )
    print(f"Compatible: {is_compat}")
    if messages:
        print("Incompatibility reasons:")
        for msg in messages:
            print(f"  - {msg}")

    # ── Step 7: Get all versions for the subject ─────────────────
    print_separator("Step 7: Get all schema versions")
    versions = get_schema_versions(url, subject_name)
    for v in versions:
        print(f"Version {v['version']} (ID: {v['id']}):")
        print(f"  Fields: {[f['name'] for f in v['schema']['fields']]}")

    print_separator("Done!")
    print("You've explored the Schema Registry REST API.")
    print("Next: try avro_producer.py to produce Avro-serialized messages.")


if __name__ == "__main__":
    main()
