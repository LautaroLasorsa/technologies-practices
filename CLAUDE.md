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

| # | Name | Stack | State |
|---|------|-------|-------|
| 001 | LMDB | Python, Rust | `completed` |
| 002 | Cloud Pub/Sub | Python, Docker (emulator) | `completed` |
| 003a | Kafka: Producers & Consumers | Python, Docker (KRaft) | `not-started` |
| 003b | Kafka: Streams & Processing | Python, Docker (KRaft) | `not-started` |
| 004a | gRPC & Protobuf: Python Service | Python | `not-started` |
| 004b | gRPC: Rust Interop | Rust, Python | `not-started` |
| 005 | Docker Compose: Multi-Service Orchestration & Limits | Python, Docker Compose | `not-started` |
| 006a | Kubernetes: Core Concepts & Deploy | Docker, minikube/kind | `not-started` |
| 006b | Kubernetes: Services, Scaling & Ingress | Docker, minikube/kind | `not-started` |
| 007a | OpenTelemetry: Instrumentation | Python, Docker (Jaeger) | `not-started` |
| 007b | OpenTelemetry: Dashboards & Alerting | Python, Docker (Grafana/Prometheus) | `not-started` |
| 008 | Vector Databases | Python, Docker (Qdrant) | `not-started` |
| 009 | GraphQL | Python (Strawberry), Docker | `not-started` |
| 010a | Terraform: Basics & Local Resources | HCL, Docker provider | `not-started` |
| 010b | Terraform: Multi-Resource Orchestration | HCL, Docker provider | `not-started` |
| 011a | Spark: Core Transformations | Python, Docker | `not-started` |
| 011b | Spark: Real Data Pipeline | Python, Docker | `not-started` |
| 012a | C++17 Features & abseil-cpp | C++17, abseil-cpp | `completed` |
| 012b | Boost Deep-Dive | C++17, Boost | `completed` |
| 013 | Financial C++: QuickFIX & QuantLib | C++17, QuickFIX, QuantLib | `not-started` |
| 014 | SAGA Pattern | Python, Docker Compose, Redpanda | `not-started` |
| 015 | CQRS & Event Sourcing | Python, Docker Compose, Redpanda | `not-started` |
| 016 | REST API Design Principles | Python (FastAPI), OpenAPI | `not-started` |
| 017a | World Model RL: Dyna-Q Basics | Python (PyTorch), Gymnasium | `not-started` |
| 017b | World Model RL: Neural World Model | Python (PyTorch), Gymnasium | `not-started` |
| 017c | World Model RL: Latent Imagination (Dreamer) | Python (PyTorch), Gymnasium | `not-started` |
| 018a | CMake: FetchContent, Targets & Modern Build | C++17, CMake 3.16+ | `not-started` |
| 018b | CMake: Toolchains, Presets & Cross-Compilation | C++17, CMake 3.21+ | `not-started` |
| 019a | CUDA: Kernels, Memory Model & Parallel Patterns | C++17, CUDA 12.x, CMake | `not-started` |
| 019b | CUDA for HPC/HFT: Streams, Pinned Memory & Low-Latency | C++17, CUDA 12.x, CMake | `not-started` |
| 020a | HFT Low-Latency C++: Lock-Free, Cache & Memory | C++17, CMake | `not-started` |
| 020b | HFT Systems: Order Book, Matching Engine & Feed Handler | C++17, abseil-cpp, CMake | `not-started` |
| 021a | Solana Smart Contracts: Anchor & Accounts Model | Rust, Anchor, LiteSVM (WSL) | `not-started` |
| 021b | Solana Tokens: SPL, Escrow & PDA Vaults | Rust, Anchor, LiteSVM (WSL) | `not-started` |
| 022 | Concurrent Data Structures: moodycamel & Lock-Free Patterns | C++17, moodycamel, CMake | `not-started` |
| 023 | Advanced SQL: Window Functions, CTEs & Query Optimization | Python, PostgreSQL, Docker | `not-started` |
| 024a | Testing Patterns: Python | Python (pytest, Hypothesis) | `not-started` |
| 024b | Testing Patterns: Rust | Rust (mockall, proptest) | `not-started` |
| 025a | ML Compilers: Computation Graphs & IR from Scratch | Python (PyTorch) | `not-started` |
| 025b | ML Compilers: Operator Fusion & Graph Rewrites | Python (PyTorch, torch.fx) | `not-started` |
| 025c | ML Compilers: XLA & HLO Inspection | Python (JAX) | `not-started` |
| 025d | ML Compilers: Triton Custom GPU Kernels | Python (Triton, PyTorch) | `not-started` |
| 025e | ML Compilers: TVM Scheduling & Auto-Tuning | Python (TVM), Docker | `not-started` |
| 025f | ML Compilers: torch.compile Deep Dive | Python (PyTorch 2.x) | `not-started` |
| 026a | ETL/ELT Orchestration: Airflow Fundamentals | Python, Docker (Airflow) | `not-started` |
| 026b | ETL/ELT Orchestration: Dagster Assets & dbt | Python, Docker (Dagster, dbt) | `not-started` |
| 027 | Advanced System Design Interview | Markdown (design docs, no code) | `not-started` |
| 028 | LangExtract: Structured Extraction from Text | Python, Docker (Ollama) | `not-started` |

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
