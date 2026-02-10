"""Generate Python gRPC stubs from the shared .proto file.

Usage (from python_client/):
    uv run python generate_stubs.py

This produces two files in src/:
    - inventory_pb2.py        (message classes)
    - inventory_pb2_grpc.py   (client stub and server servicer)
"""

from grpc_tools import protoc
import sys

PROTO_DIR = "../proto"
PROTO_FILE = "../proto/inventory.proto"
OUTPUT_DIR = "./src"


def main() -> None:
    exit_code = protoc.main([
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        PROTO_FILE,
    ])
    if exit_code != 0:
        print(f"protoc failed with exit code {exit_code}", file=sys.stderr)
        sys.exit(exit_code)
    print(f"Stubs generated in {OUTPUT_DIR}/:")
    print("  - inventory_pb2.py")
    print("  - inventory_pb2_grpc.py")


if __name__ == "__main__":
    main()
