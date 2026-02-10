# Practice 004b: gRPC Rust Interop

## Technologies

- **tonic** -- Async Rust gRPC framework built on Hyper and Tower
- **prost** -- Protobuf code generation for Rust (message types)
- **tonic-prost-build** -- Build-time `.proto` compiler that generates Rust server/client stubs
- **tokio** -- Async runtime for the Rust server
- **grpcio / grpcio-tools** -- Python gRPC runtime and protobuf code generator
- **protobuf (protoc)** -- Protocol Buffers compiler (required by both Rust and Python toolchains)

## Stack

- Rust (cargo, edition 2024)
- Python 3.12+ (uv)

## Description

Build a **cross-language gRPC service**: a Rust server exposes an `Inventory` service (add items, query stock, list items) via gRPC, and a Python client consumes it. Both sides share a single `.proto` contract in `proto/inventory.proto`.

This practice focuses on the **Rust side** of gRPC (tonic ecosystem) and cross-language interop -- calling a Rust server from Python using the same protobuf definitions.

### What you'll learn

1. **Protobuf service definition** -- defining messages and RPC methods in `.proto` files
2. **tonic code generation** -- how `build.rs` + `tonic-prost-build` turns `.proto` into Rust traits and types
3. **Implementing a gRPC server in Rust** -- the `#[tonic::async_trait]` pattern, `Request<T>` / `Response<T>`, `Status` error handling
4. **Server-side streaming** -- returning a stream of items from Rust to Python
5. **Cross-language interop** -- generating Python stubs from the same `.proto` and calling the Rust server
6. **Error handling** -- mapping domain errors to gRPC status codes

## Prerequisites

### protoc (Protocol Buffers compiler)

`tonic-prost-build` (Rust) requires the `protoc` binary on PATH. Python's `grpcio-tools` bundles its own `protoc`, so this is only strictly needed for the Rust build.

**Windows (winget):**
```
winget install Google.Protobuf
```

**Windows (manual):** Download from https://github.com/protocolbuffers/protobuf/releases, extract, add `bin/` to PATH.

**Verify installation:**
```
protoc --version
```

If you already have `protoc` from a previous practice (004a), you're set.

## Instructions

### Phase 1: Explore the Proto Contract (~5 min)

1. Read `proto/inventory.proto` -- understand the three RPCs: `AddItem` (unary), `GetItem` (unary), `ListItems` (server streaming)
2. Key question: Why is `ListItems` a server-streaming RPC instead of returning a repeated field?

### Phase 2: Rust Server Setup (~10 min)

1. Review `rust_server/Cargo.toml` and `rust_server/build.rs` -- already configured
2. Run `cargo build` inside `rust_server/` to trigger protobuf code generation
3. Inspect the generated code: `tonic::include_proto!("inventory")` brings in traits and types
4. Key question: Where does `tonic-prost-build` put the generated `.rs` files?

### Phase 3: Implement the Rust gRPC Server (~30 min)

1. Open `rust_server/src/main.rs` -- boilerplate is provided, TODOs mark the learning spots
2. **User implements:** `add_item` -- insert into the in-memory store, return confirmation
3. **User implements:** `get_item` -- look up by name, return `Status::NOT_FOUND` if missing
4. **User implements:** `list_items` -- stream all items back using `tokio::sync::mpsc` + `ReceiverStream`
5. Run the server: `cargo run` from `rust_server/`
6. Key question: Why does `list_items` return `Response<Self::ListItemsStream>` instead of `Response<ListItemsResponse>`?

### Phase 4: Python Client Setup (~10 min)

1. Review `python_client/pyproject.toml` -- already configured
2. Run `uv sync` inside `python_client/`
3. Generate Python stubs: `uv run python generate_stubs.py` (or manually: `uv run python -m grpc_tools.protoc -I../proto --python_out=./src --grpc_python_out=./src ../proto/inventory.proto`)
4. Verify `inventory_pb2.py` and `inventory_pb2_grpc.py` were generated in `python_client/src/`

### Phase 5: Implement the Python Client (~20 min)

1. Open `python_client/src/client.py` -- boilerplate is provided, TODOs mark the learning spots
2. **User implements:** Connect to the Rust server and call `AddItem` for several items
3. **User implements:** Call `GetItem` for an existing and a non-existing item (handle the error)
4. **User implements:** Call `ListItems` and iterate the server stream
5. Run: `uv run python src/client.py` from `python_client/`
6. Key question: How does a Python `for response in stub.ListItems(request)` correspond to the Rust `mpsc::channel` stream?

### Phase 6: Error Handling & Edge Cases (~15 min)

1. **User implements:** Add a duplicate item -- the Rust server should return `Status::ALREADY_EXISTS`
2. Test from Python: catch the `grpc.RpcError` and inspect `.code()` and `.details()`
3. Discussion: How do gRPC status codes compare to HTTP status codes? When would you use metadata vs status details?

## Motivation

- **Rust + gRPC is production-critical**: Tonic is used at Cloudflare, Discord, AWS, and many infrastructure companies for high-performance services
- **Cross-language interop is a real-world pattern**: Backend teams commonly expose Rust/Go services consumed by Python ML pipelines or web frontends
- **Protobuf schema-first design**: Understanding contract-driven development is essential for microservice architectures
- **Complements current profile**: Adds Rust service development to existing Python/FastAPI experience, showing polyglot capability

## References

- [tonic GitHub](https://github.com/hyperium/tonic)
- [tonic HelloWorld Tutorial](https://github.com/hyperium/tonic/blob/master/examples/helloworld-tutorial.md)
- [prost (Protobuf for Rust)](https://github.com/tokio-rs/prost)
- [tonic-prost-build docs](https://docs.rs/tonic-prost-build)
- [gRPC Python Basics](https://grpc.io/docs/languages/python/basics/)
- [Protocol Buffers Language Guide](https://protobuf.dev/programming-guides/proto3/)
- [gRPC Status Codes](https://grpc.github.io/grpc/core/md_doc_statuscodes.html)

## Commands

### Prerequisites

| Command | Description |
|---------|-------------|
| `protoc --version` | Verify Protocol Buffers compiler is installed and on PATH |

### Phase 2: Rust Server Build

All commands run from `rust_server/`.

| Command | Description |
|---------|-------------|
| `cargo build` | Compile the Rust server and trigger `build.rs` protobuf code generation via tonic-prost-build |
| `cargo run` | Build and start the gRPC server on `127.0.0.1:50051` |

### Phase 4: Python Client Setup

All commands run from `python_client/`.

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (grpcio, grpcio-tools) into the virtual environment |
| `uv run python generate_stubs.py` | Generate Python protobuf stubs (`inventory_pb2.py`, `inventory_pb2_grpc.py`) into `src/` |
| `uv run python -m grpc_tools.protoc -I../proto --python_out=./src --grpc_python_out=./src ../proto/inventory.proto` | Manual alternative: generate Python stubs directly via protoc |

### Phase 5: Run Python Client

All commands run from `python_client/`.

| Command | Description |
|---------|-------------|
| `uv run python src/client.py` | Run the Python gRPC client against the running Rust server |

## State

`not-started`
