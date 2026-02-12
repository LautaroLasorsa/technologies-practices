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

## Theoretical Context

### What is Tonic and What Problem Does It Solve?

[Tonic](https://github.com/hyperium/tonic) is a native Rust gRPC client and server implementation built on async/await, designed for production systems that demand high performance, type safety, and interoperability. Unlike Python's grpcio (which wraps a C++ core), tonic is **pure Rust** — no FFI overhead, full control over memory, and seamless integration with Rust's ownership model. This makes it ideal for low-latency services, high-throughput data pipelines, and infrastructure components where memory safety and performance are non-negotiable.

The problem tonic solves is **efficient cross-language microservice communication**. In polyglot architectures (e.g., Rust backend serving Python ML pipelines or JavaScript frontends), you need a shared RPC protocol that's fast, type-safe, and language-agnostic. Protocol Buffers provide the contract (schema-first development), and tonic/gRPC provide the runtime. Rust's performance (zero-cost abstractions, no GC) combined with gRPC's HTTP/2 multiplexing makes tonic a compelling choice for performance-critical services that other languages consume.

### How Tonic Works: Prost, Build.rs, and Async Traits

Tonic is composed of three layers: **prost** (protobuf code generation), **tonic-build** (service code generation via `build.rs`), and **tonic** (async gRPC runtime).

[**Prost**](https://docs.rs/prost) is a pure-Rust protobuf compiler and runtime. It generates Rust structs from `.proto` message definitions, with derived `Serialize`/`Deserialize` traits for binary encoding. Unlike Python's dynamic message classes, prost's structs are **statically typed** — field access is checked at compile time, eliminating an entire class of runtime errors. Prost's binary encoding is optimized (zero-copy where possible), making serialization/deserialization faster than Python's protobuf implementation.

[**tonic-build**](https://crates.io/crates/tonic-prost-build) hooks into [Cargo's build script system](https://www.thorsten-hans.com/grpc-services-in-rust-with-tonic/) (`build.rs`). When you run `cargo build`, Cargo executes `build.rs` **before** compiling your crate. Inside `build.rs`, you call `tonic_build::compile_protos("inventory.proto")`, which: (1) invokes prost to generate message structs, (2) generates a `InventoryServer` trait (server interface) and `InventoryClient` (client stub), and (3) places the generated code in `OUT_DIR` (accessed via `include_proto!("inventory")`). This is declarative code generation — no manual protoc invocations, no stale generated files. Every build regenerates code from the proto, ensuring sync between schema and implementation.

**Async traits** (`#[tonic::async_trait]`) are how tonic exposes service methods. In Rust, traits can't natively have async methods (they involve opaque `impl Future` return types that confuse the trait object system). The `async_trait` macro desugars `async fn add_item(...)` into `fn add_item(...) -> Box<dyn Future<Output=...> + Send>`, making it trait-compatible. You implement the generated `Inventory` trait for your service struct, and tonic's runtime calls your methods when RPCs arrive. All I/O (network reads, serialization) is async, leveraging [tokio](https://docs.rs/tokio) for efficient concurrency.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Prost** | Pure-Rust protobuf compiler and serialization runtime. Generates strongly-typed structs from `.proto` message definitions. |
| **tonic-build** | Build-time codegen that turns `.proto` files into Rust traits (server) and clients (stubs) via `build.rs`. |
| **build.rs** | Cargo's pre-compile hook. Runs arbitrary code before main compilation — tonic uses it to generate gRPC code from protos. |
| **#[tonic::async_trait]** | Macro that enables async methods in traits. Required because Rust's trait system doesn't natively support async fn in traits (pre-Rust 1.75). |
| **Request\<T\>** | Wrapper around incoming protobuf messages. Contains the message plus metadata (headers, deadlines). |
| **Response\<T\>** | Wrapper around outgoing protobuf messages. You return `Result<Response<T>, Status>` from service methods. |
| **Status** | gRPC error type. Returned when RPCs fail — includes a status code (`NOT_FOUND`, `INVALID_ARGUMENT`) and optional details string. |
| **ReceiverStream** | Adapts `tokio::sync::mpsc::Receiver` into a `Stream`. Used for server-streaming: you push messages into a channel, tonic reads from the stream. |
| **tokio::spawn** | Spawns an async task on the tokio runtime. Used to produce streaming responses concurrently (e.g., iterating items and sending them). |
| **Arc\<Mutex\<T\>\>** | Thread-safe shared state. `Arc` enables multiple tasks to own the same data; `Mutex` ensures synchronized access. Required because tonic spawns a task per RPC. |

### Ecosystem Context: Tonic vs Python grpcio

**Performance**: [Tonic outperforms Python grpcio](https://markaicode.com/building-microservices-with-rust-tonic-grpc-best-practices/) by 2-5x in throughput and tail latency due to Rust's zero-cost abstractions, lack of GC pauses, and efficient async I/O (tokio's epoll-based reactor). For services handling 100K+ RPS or strict latency SLAs (<10ms p99), Rust is the better choice.

**Type Safety**: Rust's ownership model prevents null pointer bugs, use-after-free, and data races at compile time. Python's dynamic typing defers these errors to runtime. For production services, Rust's "if it compiles, it probably works" philosophy reduces operational risk.

**Developer Experience**: Python's ecosystem (debugging, profiling, rapid iteration) is more mature. Rust's compile times are slower, error messages are dense for beginners, and the borrow checker adds friction. For rapid prototyping or teams without Rust expertise, Python is easier. For performance-critical services or systems programming, Rust's upfront cost pays dividends.

**Cross-Language Interop**: This practice demonstrates the core value proposition of gRPC — a Rust server and Python client share a `.proto` contract and communicate seamlessly. This is how polyglot teams scale: performance-critical services in Rust, business logic in Python, frontend in TypeScript, all speaking gRPC.

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
