"""Generate Python gRPC stubs from .proto files.

Usage (from the practice root, not app/):
    cd practice_004a_grpc_python
    uv run --project app python -m scripts.generate_protos

Or more simply (from app/):
    cd app
    uv run python -m scripts.generate_protos

The script can also be run directly:
    cd practice_004a_grpc_python
    uv run --project app python scripts/generate_protos.py
"""

from pathlib import Path

from grpc_tools import protoc


def generate() -> None:
    practice_root = Path(__file__).resolve().parent.parent
    proto_dir = practice_root / "protos"
    output_dir = practice_root / "app" / "generated"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py so `app.generated` is importable as a package
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        return

    for proto_file in proto_files:
        print(f"Generating stubs for: {proto_file.name}")
        result = protoc.main([
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file),
        ])
        if result != 0:
            raise RuntimeError(
                f"protoc failed for {proto_file.name} with exit code {result}"
            )

    print(f"Stubs generated in: {output_dir}")


if __name__ == "__main__":
    generate()
