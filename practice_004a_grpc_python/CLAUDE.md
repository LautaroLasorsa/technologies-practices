# Practice 004a: gRPC & Protobuf — Python Service

## Technologies

- **gRPC** — High-performance RPC framework using HTTP/2, binary serialization, and code generation
- **Protocol Buffers (protobuf)** — Language-neutral IDL for defining services and message types
- **grpcio** — Python gRPC runtime library
- **grpcio-tools** — Protobuf compiler + gRPC plugin for Python code generation

## Stack

- Python 3.12+ (uv)

## Theoretical Context

### What is gRPC and What Problem Does It Solve?

[gRPC](https://grpc.io/docs/what-is-grpc/introduction/) is a high-performance RPC (Remote Procedure Call) framework developed by Google that makes distributed systems communication efficient, type-safe, and language-agnostic. Unlike REST APIs where you send HTTP requests to URLs and parse JSON responses, gRPC lets you call remote methods on a server **as if they were local functions**. You define services and message types in a `.proto` file, generate client/server code in any language, and the framework handles serialization, networking, and error propagation.

The problem gRPC solves is **inefficient microservice communication**. Traditional REST/JSON APIs are designed for human-readable data (browser-friendly, cacheable) but suffer from performance penalties in service-to-service calls: JSON parsing is CPU-intensive, text serialization is verbose, and HTTP/1.1 lacks multiplexing (one request per TCP connection). In high-throughput microservice architectures (Google, Netflix, Slack), these overheads compound. [gRPC with HTTP/2](https://arpitbhayani.me/blogs/grpc-http2/) and Protocol Buffers delivers 2-10x performance improvements: smaller payloads, faster parsing, and connection reuse via multiplexing.

### How gRPC Works: HTTP/2, Protobuf, and Code Generation

gRPC uses three core technologies: **Protocol Buffers** (protobuf) for serialization, **HTTP/2** for transport, and **code generation** for type-safe clients/servers.

**Protocol Buffers** is a binary serialization format. You define message schemas in `.proto` files using a simple IDL (Interface Definition Language): field names, types, and numbers. The `protoc` compiler generates language-specific classes with getter/setter methods and efficient binary encoding. [Protobuf serialization](https://medium.com/@lchang1994/deep-dive-grpc-protobuf-http-2-0-74e6295f1d38) is dramatically faster than JSON: messages are 3-10x smaller (no field name duplication, compact varint encoding for integers) and parsing is 5-10x faster (binary decoding vs text parsing). Field numbers (not names) are encoded, so renaming fields doesn't break compatibility — this is protobuf's versioning superpower.

**HTTP/2** is the transport layer that [enables gRPC's streaming and multiplexing](https://www.ibm.com/think/topics/grpc). Unlike HTTP/1.1 (one request → one response per connection), HTTP/2 uses **streams**: multiple requests/responses can be interleaved over a single TCP connection. This eliminates the overhead of opening dozens of connections (TLS handshakes, slow-start) for concurrent requests. HTTP/2 also provides **flow control** (backpressure) and **header compression** (HPACK), further reducing latency. For gRPC, this means a single connection can handle thousands of concurrent RPCs — critical for microservice meshes.

**Code generation** ensures type safety and eliminates boilerplate. From `service TaskManager { rpc CreateTask(...) ... }` in the proto, `protoc` generates: (1) message classes (`Task`, `CreateTaskRequest`), (2) server base classes (`TaskManagerServicer`), and (3) client stubs (`TaskManagerStub`). You implement the servicer methods (server) or call stub methods (client) — the framework handles everything else (serialization, HTTP/2 framing, error codes). This is contract-first development: the proto is the single source of truth, and clients/servers in different languages are guaranteed to agree on the API shape.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Proto File (.proto)** | IDL file defining messages (data structures) and services (RPC methods). The contract shared by client and server. |
| **Message** | A typed data structure (like a class or struct). Defines fields with types, numbers, and optionality (required, optional, repeated). |
| **Service** | A collection of RPC methods. Each method specifies request/response message types and streaming mode (unary, client streaming, server streaming, bidirectional). |
| **Unary RPC** | Single request → single response (like a REST POST). Most common RPC pattern. |
| **Server Streaming** | Single request → stream of responses. Used for pagination, live feeds, or chunked results. |
| **Client Streaming** | Stream of requests → single response. Used for uploads or batching. |
| **Bidirectional Streaming** | Stream of requests ↔ stream of responses. Used for chat, real-time collaboration, or duplex protocols. |
| **Status Codes** | gRPC's error model. [Standard codes](https://grpc.io/docs/guides/status-codes/) like `NOT_FOUND`, `INVALID_ARGUMENT`, `UNAUTHENTICATED` replace HTTP status codes. Richer than HTTP (e.g., `DEADLINE_EXCEEDED`). |
| **Interceptor** | Middleware that wraps RPC calls. Server interceptors run before/after handlers; client interceptors wrap outgoing calls. Used for logging, auth, metrics, retries. |
| **Metadata** | Key-value headers sent with requests/responses. gRPC's equivalent of HTTP headers. Used for auth tokens, tracing IDs, etc. |
| **Servicer** | Server-side class that implements the service methods (generated base class). You subclass and override methods. |
| **Stub** | Client-side proxy that exposes service methods as Python functions. Generated by `protoc`. |

### Ecosystem Context: gRPC vs REST

**When to use gRPC**: [Internal microservices](https://aws.amazon.com/compare/the-difference-between-grpc-and-rest/), high QPS (queries per second), low latency requirements, streaming (real-time updates, telemetry), and polyglot systems (Go server + Python client). gRPC [outperforms REST by up to 7x](https://boldsign.com/blogs/grpc-vs-rest-api-performance-guide/) in microservice architectures due to binary serialization and HTTP/2 multiplexing. Companies like Google, Netflix, and Slack use gRPC for inter-service calls where performance and streaming matter more than browser accessibility.

**When to use REST**: Public APIs, third-party developers, browser-first requirements (browsers can't speak HTTP/2 binary streams natively), caching/CDN friendliness (HTTP GET is cacheable; gRPC POST is not), and debugging simplicity (curl vs gRPC clients). REST's human-readable JSON and stateless design make it ideal for external APIs where developer experience and ecosystem tooling (Postman, Swagger) matter.

**Trade-offs**: gRPC requires code generation (language-specific tooling), lacks browser support without grpc-web (a proxy layer), and is harder to debug (binary payloads vs JSON). REST is universally supported, human-readable, and has mature caching infrastructure (Cloudflare, Fastly), but sacrifices performance and streaming. Hybrid architectures are common: gRPC for backend-to-backend, REST for frontend-to-backend or external APIs.

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
