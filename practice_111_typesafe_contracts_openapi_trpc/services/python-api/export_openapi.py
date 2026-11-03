#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema to openapi.json."""
import json
from pathlib import Path
from main import app


def export_schema() -> None:
    output_path = Path(__file__).parent / "openapi.json"
    schema = app.openapi()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Exported OpenAPI 3.1 schema to {output_path}")


if __name__ == "__main__":
    export_schema()
