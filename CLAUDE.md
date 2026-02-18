# Technologies & Practices

Hands-on practices exploring technologies and patterns, each assisted by Claude Code in Learning Mode.

## Default Output Style: Learning Mode

**This overrides the default Claude Code behavior for this project.**

When working in this folder, Claude operates as a **guided learning assistant**, not a code-generation tool:

1. **Explain before implementing.** Briefly introduce each concept/tool before touching code. Use analogies to things the user already knows (competitive programming, Python/FastAPI, Rust).
2. **Scaffold boilerplate, leave the core.** Write config files, project setup, imports, and glue code. But for the critical logic — the part that teaches the technology — describe what needs to happen and let the user write it. Provide the function signature and a short hint if needed.
3. **Ask, don't tell.** When there's a design decision (e.g., "should this be sync or async?"), pose it as a question before proceeding. Socratic style where it adds value, but don't overdo it — respect the 60–120 min time box.
4. **Verify understanding.** After the user implements a key part, briefly confirm correctness and explain *why* it works (or what would break if done differently).
5. **Progressive disclosure.** Start simple, layer complexity. Don't dump all concepts at once.
6. **Concise explanations.** 2-4 sentences per concept, not paragraphs. Link to official docs for deep dives.

**Dual objective:**

1. Improve professional proficiency by incorporating new technologies and practices.
2. Demonstrate learning interests and topics explored (content may be AI-assisted — the value is in the topics chosen and understood).

---

## Practices

Each practice lives in `practice_<NNN>_<name>/` with its own `CLAUDE.md` specifying: Title, Technologies, Stack, Description, Instructions, and Motivation.

Multi-session topics are split into separate folders (e.g., `003a`, `003b`), each a standalone ~60–120 min guided session.

| # | Name | Stack | Category | State |
|---|------|-------|----------|-------|
| 001 | LMDB | Python, Rust | Data Stores & Messaging | `completed` |
| 002 | Cloud Pub/Sub | Python, Docker (emulator) | Data Stores & Messaging | `completed` |
| 003a | Kafka: Producers & Consumers | Python, Docker (KRaft) | Data Stores & Messaging | `completed` |
| 003b | Kafka: Streams & Processing | Python, Docker (KRaft) | Data Stores & Messaging | `not-started` |
| 003c | Kafka: Schema Registry & Data Contracts | Python, Docker (KRaft, Schema Registry) | Data Stores & Messaging | `not-started` |
| 003d | Kafka Connect & Change Data Capture | Python, Docker (KRaft, Debezium, PostgreSQL) | Data Stores & Messaging | `not-started` |
| 003e | Kafka: Internals & Performance Tuning | Python, Docker (KRaft, 3-broker) | Data Stores & Messaging | `not-started` |
| 004a | gRPC & Protobuf: Python Service | Python | API Design & Protocols | `not-started` |
| 004b | gRPC: Rust Interop | Rust, Python | API Design & Protocols | `not-started` |
| 005 | Docker Compose: Multi-Service Orchestration & Limits | Python, Docker Compose | Infra, DevOps & Observability | `not-started` |
| 006a | Kubernetes: Core Concepts & Deploy | Docker, minikube/kind | Infra, DevOps & Observability | `not-started` |
| 006b | Kubernetes: Services, Scaling & Ingress | Docker, minikube/kind | Infra, DevOps & Observability | `not-started` |
| 006c | Service Mesh: Sidecar & Envoy | Docker, minikube/kind, Envoy | Infra, DevOps & Observability | `not-started` |
| 007a | OpenTelemetry: Instrumentation | Python, Docker (Jaeger) | Infra, DevOps & Observability | `not-started` |
| 007b | OpenTelemetry: Dashboards & Alerting | Python, Docker (Grafana/Prometheus) | Infra, DevOps & Observability | `not-started` |
| 008 | Vector Databases | Python, Docker (Qdrant) | Data Stores & Messaging | `not-started` |
| 009 | GraphQL | Python (Strawberry), Docker | API Design & Protocols | `not-started` |
| 010a | Terraform: Basics & Local Resources | HCL, Docker provider | Infra, DevOps & Observability | `not-started` |
| 010b | Terraform: Multi-Resource Orchestration | HCL, Docker provider | Infra, DevOps & Observability | `not-started` |
| 011a | Spark: Core Transformations | Python, Docker | Data Engineering | `not-started` |
| 011b | Spark: Real Data Pipeline | Python, Docker | Data Engineering | `not-started` |
| 012a | C++17 Features & abseil-cpp | C++17, abseil-cpp | Systems Programming | `completed` |
| 012b | Boost Deep-Dive | C++17, Boost | Systems Programming | `completed` |
| 013 | Financial C++: QuickFIX & QuantLib | C++17, QuickFIX, QuantLib | Systems Programming | `not-started` |
| 014 | SAGA Pattern | Python, Docker Compose, Redpanda | Distributed Systems Patterns | `not-started` |
| 015 | CQRS & Event Sourcing | Python, Docker Compose, Redpanda | Distributed Systems Patterns | `not-started` |
| 016 | REST API Design Principles | Python (FastAPI), OpenAPI | API Design & Protocols | `not-started` |
| 017a | World Model RL: Dyna-Q Basics | Python (PyTorch), Gymnasium | ML Systems | `not-started` |
| 017b | World Model RL: Neural World Model | Python (PyTorch), Gymnasium | ML Systems | `not-started` |
| 017c | World Model RL: Latent Imagination (Dreamer) | Python (PyTorch), Gymnasium | ML Systems | `not-started` |
| 018a | CMake: FetchContent, Targets & Modern Build | C++17, CMake 3.16+ | Systems Programming | `not-started` |
| 018b | CMake: Toolchains, Presets & Cross-Compilation | C++17, CMake 3.21+ | Systems Programming | `not-started` |
| 019a | CUDA: Kernels, Memory Model & Parallel Patterns | C++17, CUDA 12.x, CMake | GPU Computing & HFT | `not-started` |
| 019b | CUDA for HPC/HFT: Streams, Pinned Memory & Low-Latency | C++17, CUDA 12.x, CMake | GPU Computing & HFT | `not-started` |
| 020a | HFT Low-Latency C++: Lock-Free, Cache & Memory | C++17, CMake | GPU Computing & HFT | `not-started` |
| 020b | HFT Systems: Order Book, Matching Engine & Feed Handler | C++17, abseil-cpp, CMake | GPU Computing & HFT | `not-started` |
| 021a | Solana Smart Contracts: Anchor & Accounts Model | Rust, Anchor, LiteSVM (WSL) | Blockchain | `not-started` |
| 021b | Solana Tokens: SPL, Escrow & PDA Vaults | Rust, Anchor, LiteSVM (WSL) | Blockchain | `not-started` |
| 022 | Concurrent Data Structures: moodycamel & Lock-Free Patterns | C++17, moodycamel, CMake | Systems Programming | `not-started` |
| 023 | Advanced SQL: Window Functions, CTEs & Query Optimization | Python, PostgreSQL, Docker | Data Engineering | `not-started` |
| 024a | Testing Patterns: Python | Python (pytest, Hypothesis) | Software Engineering | `not-started` |
| 024b | Testing Patterns: Rust | Rust (mockall, proptest) | Software Engineering | `not-started` |
| 025a | ML Compilers: Computation Graphs & IR from Scratch | Python (PyTorch) | ML Systems | `not-started` |
| 025b | ML Compilers: Operator Fusion & Graph Rewrites | Python (PyTorch, torch.fx) | ML Systems | `not-started` |
| 025c | ML Compilers: XLA & HLO Inspection | Python (JAX) | ML Systems | `not-started` |
| 025d | ML Compilers: Triton Custom GPU Kernels | Python (Triton, PyTorch) | ML Systems | `not-started` |
| 025e | ML Compilers: TVM Scheduling & Auto-Tuning | Python (TVM), Docker | ML Systems | `not-started` |
| 025f | ML Compilers: torch.compile Deep Dive | Python (PyTorch 2.x) | ML Systems | `not-started` |
| 026a | ETL/ELT Orchestration: Airflow Fundamentals | Python, Docker (Airflow) | Data Engineering | `not-started` |
| 026b | ETL/ELT Orchestration: Dagster Assets & dbt | Python, Docker (Dagster, dbt) | Data Engineering | `not-started` |
| 027 | Advanced System Design Interview | Markdown (design docs, no code) | Software Engineering | `not-started` |
| 028 | LangExtract: Structured Extraction from Text | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 029a | LangChain Advanced: LCEL, Tools & Structured Output | Python, Docker (Ollama) | AI & LLM Engineering | `completed` |
| 029b | LangGraph: Stateful Agent Graphs | Python, Docker (Ollama) | AI & LLM Engineering | `completed` |
| 030a | DSPy Fundamentals: Signatures, Modules & Optimization | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 030b | DSPy Advanced: RAG, Assertions & MIPROv2 | Python, Docker (Ollama, Qdrant) | AI & LLM Engineering | `not-started` |
| 030c | DSPy + LangGraph Integration | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 031a | Agentic AI: Single-Agent Design Patterns | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 031b | Agentic AI: Multi-Agent Systems | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 031c | Agentic AI: Production Patterns | Python, Docker (Ollama, Qdrant, Langfuse) | AI & LLM Engineering | `not-started` |
| 054a | Google ADK: Agents, Tools & Sessions | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 054b | Google ADK: Multi-Agent Orchestration | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 054c | Google ADK: Callbacks, Evaluation & Streaming | Python, Docker (Ollama) | AI & LLM Engineering | `not-started` |
| 032a | LP: Simplex Method | C++17, Eigen | OR: Algorithms (C++/Rust) | `not-started` |
| 032b | LP: Duality & Interior Point | C++17, Eigen | OR: Algorithms (C++/Rust) | `not-started` |
| 033a | Convex Opt: First-Order Methods | C++17, Eigen | OR: Algorithms (C++/Rust) | `not-started` |
| 033b | Convex Opt: Proximal & Second-Order | C++17, Eigen | OR: Algorithms (C++/Rust) | `not-started` |
| 034a | MIP: Branch & Bound | Rust | OR: Algorithms (C++/Rust) | `not-started` |
| 034b | MIP: Cutting Planes & Heuristics | Rust | OR: Algorithms (C++/Rust) | `not-started` |
| 035a | Non-convex: Local Search & SA | Rust | OR: Algorithms (C++/Rust) | `not-started` |
| 035b | Non-convex: Evolutionary Algorithms | Rust | OR: Algorithms (C++/Rust) | `not-started` |
| 036a | Network Opt: Flows & Assignment | C++17 | OR: Algorithms (C++/Rust) | `not-started` |
| 036b | Network Opt: TSP & VRP Heuristics | C++17 | OR: Algorithms (C++/Rust) | `not-started` |
| 037 | Constraint Programming | Rust | OR: Algorithms (C++/Rust) | `not-started` |
| 038 | Multi-Objective: NSGA-II | Rust | OR: Algorithms (C++/Rust) | `not-started` |
| 039 | Stochastic DP: MDPs | C++17, Eigen | OR: Algorithms (C++/Rust) | `not-started` |
| 040a | LP/MIP: PuLP & HiGHS | Python | OR: Applied (Python) | `not-started` |
| 040b | Advanced Modeling: Pyomo | Python | OR: Applied (Python) | `not-started` |
| 041a | Convex Opt: CVXPY | Python | OR: Applied (Python) | `not-started` |
| 041b | Conic: SOCP & SDP | Python | OR: Applied (Python) | `not-started` |
| 042a | CP: OR-Tools CP-SAT | Python | OR: Applied (Python) | `not-started` |
| 042b | Routing: OR-Tools | Python | OR: Applied (Python) | `not-started` |
| 043 | Global Opt: scipy & Optuna | Python | OR: Applied (Python) | `not-started` |
| 044 | Multi-Objective: pymoo | Python | OR: Applied (Python) | `not-started` |
| 045 | Stochastic Programming | Python | OR: Applied (Python) | `not-started` |
| 046 | Robust Optimization | Python | OR: Applied (Python) | `not-started` |
| 047 | Bayesian Optimization | Python | OR: Applied (Python) | `not-started` |
| 048 | Integrated OR: Supply Chain | Python | OR: Applied (Python) | `not-started` |
| 049a | Raft Consensus: Leader Election & Log Replication | Python | Distributed Systems Patterns | `not-started` |
| 049b | Raft Consensus: Safety & Failure Recovery | Python | Distributed Systems Patterns | `not-started` |
| 050 | CRDTs: Conflict-Free Replicated Data Types | Python | Distributed Systems Patterns | `not-started` |
| 051 | Distributed Coordination: etcd | Python, Docker (etcd) | Distributed Systems Patterns | `not-started` |
| 052 | Resilience Patterns: Circuit Breaker, Bulkhead & Rate Limiting | Python, Docker (Redis) | Distributed Systems Patterns | `not-started` |
| 053 | Consistent Hashing & DHTs | Python | Distributed Systems Patterns | `not-started` |
| 055 | RabbitMQ: Queues, Exchanges & AMQP | Python, Docker (RabbitMQ) | Data Stores & Messaging | `not-started` |
| 057 | MQTT: Lightweight Messaging & Protocol Comparison | Python, Docker (Mosquitto) | Data Stores & Messaging | `not-started` |
| 056 | DuckDB: Embedded Analytics & Columnar Queries | Python | Data Engineering | `not-started` |
| 058a | SQLModel: FastAPI Integration & Pydantic Models | Python (SQLModel, FastAPI), Docker (PostgreSQL) | Data Stores & Messaging | `not-started` |
| 058b | SQLAlchemy 2.0: Async ORM & Advanced Queries | Python (SQLAlchemy 2.0, asyncpg), Docker (PostgreSQL) | Data Stores & Messaging | `not-started` |
| 059a | SQLx: Compile-Time Checked SQL | Rust (SQLx, tokio), Docker (PostgreSQL) | Data Stores & Messaging | `not-started` |
| 059b | Diesel: Type-Safe ORM & Migrations | Rust (Diesel), Docker (PostgreSQL) | Data Stores & Messaging | `not-started` |
| 060a | Unsafe Rust & FFI: Raw Pointers, bindgen & repr(C) | Rust (bindgen, libc, cbindgen) | Systems Programming | `not-started` |
| 060b | Async Runtime Internals: Future, Waker & Tokio Architecture | Rust (tokio) | Systems Programming | `not-started` |
| 061a | Lock-Free Rust: crossbeam & Epoch-Based Reclamation | Rust (crossbeam, crossbeam-epoch) | Systems Programming | `not-started` |
| 061b | Custom Allocators: GlobalAlloc, jemalloc & Arena Patterns | Rust (tikv-jemallocator, bumpalo) | Systems Programming | `not-started` |
| 062 | SIMD in Rust: portable_simd & std::arch | Rust (nightly, std::arch) | Systems Programming | `not-started` |
| 063a | Profiling & Flamegraphs | Rust, C++ (cargo-flamegraph, samply) | Systems Programming | `not-started` |
| 063b | Memory Safety Verification: Miri & Sanitizers | Rust, C++ (Miri, ASAN) | Systems Programming | `not-started` |
| 064 | WebAssembly from Systems Languages: WASM & WASI | Rust (wasm-pack, wasmtime) | Systems Programming | `not-started` |
| 065 | eBPF Observability: Kernel Tracing with aya | Rust, Docker (Linux) | Systems Programming | `not-started` |
| 066 | io_uring: Modern Linux Async I/O | Rust (tokio-uring, io-uring), Docker (Linux) | Systems Programming | `not-started` |

**State tags:** `not-started` → `in-progress` → `completed`

---

## Constraints

- **No real external APIs or services.** Emulate everything locally using Docker containers, emulators, or local-only tools.
- **Per-practice stack.** Each practice chooses its own language/framework based on what best fits the technology.
- **Guided learning format.** Claude Code scaffolds boilerplate (config files, Docker, build systems, glue code) and explains concepts, but leaves critical/interesting/most-educational parts for the user to implement. Each session targets 60–120 minutes. **Important:** Technology setup code (creating databases, topics, subscriptions, indexes, schemas, etc.) is NOT boilerplate — it is part of the exercises. Learning to provision and configure the technology's resources is essential to understanding how it works.
- **Standalone sessions.** Each practice folder is a self-contained session — no dependency on completing previous practices (unless explicitly noted).

---

## Best practices

- **Research before creating**: When creating a new practice or its content, first do in-depth research about all relevant information related to the practice's topic.

- **Skills creation**: When you find that some information is hard or expensive to reproduce and is useful in a wide variety of situations related to the purpose of this repository, create a project-level skill containing that information in detail so it can be easily recovered in the future.

- **Use subagents**: When you have to make a task specific for a practice in isolation of the whole context, and speciall if you want to make a similar action in many diferent practices, send a subagent to do the task in each practice (one subagent asigned to one task)

- **Keep README.md in sync**: Whenever a practice is added, removed, renamed, or its state changes (`not-started` → `in-progress` → `completed`), update **both** the Practices table in this file **and** the corresponding section/table in `README.md`. The README groups practices by topic — place new practices in the matching section. For completed practices, apply `bgcolor="#d4edda"` + bold to all `<td>` cells in that row. Update the shields.io progress badge count as well.

---

## Practice CLAUDE.md Template

Each practice folder must contain a `CLAUDE.md` with:

- **Title** — Practice name
- **Technologies** — Libraries, frameworks, and tools used
- **Stack** — Language(s) and runtime
- **Theoretical Context** — **(MANDATORY)** Conceptual explanation of the technology being practiced. Should cover: what the technology is and the problem it solves, how it works internally (key mechanisms/architecture), key concepts with brief definitions, and where it fits in the ecosystem (alternatives, trade-offs). Written for a developer experienced in general programming but new to this specific technology. Thorough enough that the learner understands the "why" behind every exercise.
- **Description** — What this practice covers and builds
- **Instructions** — Step-by-step setup and execution guide. Each exercise (TODO(human) in source code) must include a brief context explanation of what it teaches and why it matters for understanding the technology.
- **Motivation** — Why this technology matters (market demand, gap in profile, complementary to current skills)
- **Commands** — **(MANDATORY)** A table of every command needed to run the practice, grouped by phase/stage. Each command must have a description. If a command is run with different options for different phases, list each variant as a separate entry explaining the option used. Commands should be runnable from the practice folder root.
- **Notes** — **(Optional, added during practice)** Notable observations, cross-domain connections, or insights that emerged while solving the practice. When a note originated from the user or from a user–Claude interaction, attribute it accordingly (e.g., "User observation:", "From discussion:"). This section captures the learner's own thinking — not the scaffold's instructions.
