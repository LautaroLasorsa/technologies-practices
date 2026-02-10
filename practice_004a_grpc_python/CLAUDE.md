# Practice 004a: gRPC & Protobuf — Python Service

## Technologies

- **gRPC** — High-performance RPC framework using HTTP/2, binary serialization, and code generation
- **Protocol Buffers (protobuf)** — Language-neutral IDL for defining services and message types
- **grpcio** — Python gRPC runtime library
- **grpcio-tools** — Protobuf compiler + gRPC plugin for Python code generation

## Stack

- Python 3.12+ (uv)

## Description

Build a **TaskManager gRPC service** that supports CRUD operations on tasks plus a server-streaming RPC for watching task updates in real time. This teaches gRPC's core mechanics in Python: proto file design, code generation, unary and streaming RPCs, interceptors, and error handling with status codes.

### What you'll learn

1. **Protobuf IDL** — message types, field numbers, enums, `oneof`, repeated fields
2. **Service definition** — defining RPC methods (unary, server streaming) in `.proto` files
3. **Code generation** — using `grpc_tools.protoc` to produce `_pb2.py` and `_pb2_grpc.py` stubs
4. **Server implementation** — subclassing the generated servicer, registering with `grpc.server()`
5. **Client stubs** — creating channels and calling RPCs (unary and streaming)
6. **Error handling** — `context.abort()`, `grpc.StatusCode`, catching `grpc.RpcError` on the client
7. **Interceptors** — server-side logging interceptor via `grpc.ServerInterceptor`
8. **gRPC vs REST** — when binary RPC beats JSON/HTTP, and when it doesn't

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Review the proto file `protos/task_manager.proto` — understand message types, enums, service definition
2. Run `uv run python -m grpc_tools.protoc` to generate stubs (use the provided `scripts/generate_protos.py`)
3. Inspect the generated `_pb2.py` (message classes) and `_pb2_grpc.py` (servicer + stub classes)
4. Key question: Why does gRPC use HTTP/2 instead of HTTP/1.1? What does multiplexing buy you?

### Phase 2: Server — Unary RPCs (~25 min)

1. Open `app/server.py` — boilerplate is done, service registration is wired
2. **User implements:** `CreateTask` — validate input, store task in the in-memory dict, return the created task
3. **User implements:** `GetTask` — look up by ID, return `NOT_FOUND` if missing (use `context.abort()`)
4. **User implements:** `UpdateTask` — find existing task, update fields, return updated task
5. **User implements:** `DeleteTask` — remove from store, return `NOT_FOUND` if missing
6. Test with `uv run python -m app.client` — verify CRUD operations
7. Key question: What's the difference between `context.abort()` and `context.set_code()` + `context.set_details()`?

### Phase 3: Server Streaming — WatchTasks (~20 min)

1. **User implements:** `WatchTasks` — yield task events as they happen (use a queue or polling pattern)
2. **User implements:** Client-side streaming consumer that iterates server responses
3. Test: start watcher in one terminal, create/update tasks in another, observe real-time events
4. Key question: How does the server know when the client disconnects from a stream?

### Phase 4: Interceptors & Error Handling (~20 min)

1. Open `app/interceptors.py` — the class skeleton is there
2. **User implements:** `LoggingInterceptor.intercept_service()` — log method name, timestamp, and duration
3. Wire the interceptor into the server (see the `# TODO(human)` in `server.py`)
4. **User implements:** Client-side error handling — catch `grpc.RpcError`, inspect `.code()` and `.details()`
5. Test: request a non-existent task, verify `NOT_FOUND` is caught and logged
6. Key question: In what order do multiple interceptors execute? (hint: think middleware/onion model)

### Phase 5: Discussion (~10 min)

1. Compare gRPC vs REST: when would you choose each?
2. Discuss: How would you add authentication to this service? (metadata/interceptors vs TLS)
3. Discuss: What changes for bidirectional streaming? (practice 004b explores Rust interop)

## Motivation

- **Industry standard for microservices**: gRPC is the dominant RPC framework at Google, Netflix, Slack, and many microservice architectures — especially for internal service-to-service communication
- **Performance**: Binary serialization (protobuf) + HTTP/2 multiplexing makes gRPC 2-10x faster than JSON/REST for structured data
- **Code generation**: Proto files serve as a single source of truth for API contracts, generating type-safe clients in any language — critical for polyglot systems
- **Complements REST knowledge**: Understanding when to use RPC vs REST is a key architectural decision; this practice builds that intuition
- **Foundation for 004b**: This service becomes the Python side of a cross-language gRPC system with Rust

## References

- [gRPC Python Basics Tutorial](https://grpc.io/docs/languages/python/basics/)
- [gRPC Python Quick Start](https://grpc.io/docs/languages/python/quickstart/)
- [Protocol Buffers Language Guide (proto3)](https://protobuf.dev/programming-guides/proto3/)
- [gRPC Status Codes](https://grpc.io/docs/guides/status-codes/)
- [gRPC Interceptors Guide](https://grpc.io/docs/guides/interceptors/)
- [gRPC Python API Reference](https://grpc.github.io/grpc/python/grpc.html)
- [gRPC Python Interceptor Examples (GitHub)](https://github.com/grpc/grpc/tree/master/examples/python/interceptors)

## Commands

### Phase 1: Setup & Code Generation

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies (`grpcio`, `grpcio-tools`) into the project venv |
| `uv run --project app python scripts/generate_protos.py` | Generate `_pb2.py` and `_pb2_grpc.py` stubs from `protos/task_manager.proto` into `app/generated/` |

### Phase 2-3: Server & Client

| Command | Description |
|---------|-------------|
| `uv run --project app python -m app.server` | Start the gRPC server on `localhost:50051` (blocks until Ctrl+C) |
| `uv run --project app python -m app` | Start the gRPC server (alternative — `__main__.py` calls `serve()`) |
| `uv run --project app python -m app.client` | Run the client demo: exercises CRUD, error handling, and streaming RPCs |

### Phase 4: Interceptors & Error Handling

No additional commands -- interceptor wiring is done by uncommenting code in `app/server.py`, then restarting the server and re-running the client with the same commands above.

## State

`not-started`
