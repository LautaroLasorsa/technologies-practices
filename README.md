# Technologies & Practices


<p align="center">
  <img src="https://skillicons.dev/icons?i=py,rust,cpp,docker,kubernetes,terraform,kafka,graphql,pytorch,cmake,fastapi,grafana,prometheus,postgres,gcp,wasm,linux&perline=17" />
</p>

<b> _**This repository is not to show the things that I made, but the things that I have been interested into learning, and my progress learning them**_ </b>

<p align="center">
  Hands-on technology practices spanning systems programming, distributed systems, AI/ML, and operations research.<br>
  Each practice is a self-contained ~60–120 min session where I implement the core logic from scratch —<br>
  setup and boilerplate are AI-assisted, but the critical learning code is written by hand.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Progress-7%20%2F%20115%20practices-blue?style=flat-square" alt="Progress: 7/115">
</p>

> **Want to try these practices yourself?** Check out the [`template`](../../tree/template) branch — it has the same scaffolding with all exercises in `not-started` state, ready for you to implement.

---

## Data Stores & Messaging

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td><b>001</b></td>
  <td><b>LMDB</b></td>
  <td><img src="https://skillicons.dev/icons?i=py,rust" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td><b>002</b></td>
  <td><b>Cloud Pub/Sub</b></td>
  <td><img src="https://skillicons.dev/icons?i=py,gcp,docker" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td><b>003a</b></td>
  <td><b>Kafka: Producers & Consumers</b></td>
  <td><img src="https://skillicons.dev/icons?i=py,kafka,docker" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td>003b</td>
  <td>Kafka: Streams & Processing</td>
  <td><img src="https://skillicons.dev/icons?i=py,kafka,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>003c</td>
  <td>Kafka: Schema Registry & Data Contracts</td>
  <td><img src="https://skillicons.dev/icons?i=py,kafka,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>003d</td>
  <td>Kafka Connect & Change Data Capture</td>
  <td><img src="https://skillicons.dev/icons?i=py,kafka,postgres,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>003e</td>
  <td>Kafka: Internals & Performance Tuning</td>
  <td><img src="https://skillicons.dev/icons?i=py,kafka,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>008</td>
  <td>Vector Databases (Qdrant)</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>055</td>
  <td>RabbitMQ: Queues, Exchanges & AMQP</td>
  <td><img src="https://skillicons.dev/icons?i=py,rabbitmq,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>057</td>
  <td>MQTT: Lightweight Messaging & Protocol Comparison</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=mqtt" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>058a</td>
  <td>SQLModel: FastAPI Integration & Pydantic Models</td>
  <td><img src="https://skillicons.dev/icons?i=py,fastapi,postgres,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>058b</td>
  <td>SQLAlchemy 2.0: Async ORM & Advanced Queries</td>
  <td><img src="https://skillicons.dev/icons?i=py,postgres,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>059a</td>
  <td>SQLx: Compile-Time Checked SQL</td>
  <td><img src="https://skillicons.dev/icons?i=rust,postgres,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>059b</td>
  <td>Diesel: Type-Safe ORM & Migrations</td>
  <td><img src="https://skillicons.dev/icons?i=rust,postgres,docker" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## API Design & Protocols

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>004a</td>
  <td>gRPC & Protobuf: Python Service</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>004b</td>
  <td>gRPC: Rust Interop</td>
  <td><img src="https://skillicons.dev/icons?i=rust,py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>009</td>
  <td>GraphQL (Strawberry)</td>
  <td><img src="https://skillicons.dev/icons?i=py,graphql,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>016</td>
  <td>REST API Design Principles</td>
  <td><img src="https://skillicons.dev/icons?i=py,fastapi" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Infrastructure, DevOps & Observability

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>005</td>
  <td>Docker Compose: Multi-Service Orchestration & Limits</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>006a</td>
  <td>Kubernetes: Core Concepts & Deploy</td>
  <td><img src="https://skillicons.dev/icons?i=docker,kubernetes" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>006b</td>
  <td>Kubernetes: Services, Scaling & Ingress</td>
  <td><img src="https://skillicons.dev/icons?i=docker,kubernetes" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>006c</td>
  <td>Service Mesh: Sidecar & Envoy</td>
  <td><img src="https://skillicons.dev/icons?i=docker,kubernetes" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>007a</td>
  <td>OpenTelemetry: Instrumentation</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>007b</td>
  <td>OpenTelemetry: Dashboards & Alerting</td>
  <td><img src="https://skillicons.dev/icons?i=py,grafana,prometheus" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>010a</td>
  <td>Terraform: Basics & Local Resources</td>
  <td><img src="https://skillicons.dev/icons?i=terraform,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>010b</td>
  <td>Terraform: Multi-Resource Orchestration</td>
  <td><img src="https://skillicons.dev/icons?i=terraform,docker" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Data Engineering

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>011a</td>
  <td>Spark: Core Transformations</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>011b</td>
  <td>Spark: Real Data Pipeline</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>023</td>
  <td>Advanced SQL: Window Functions, CTEs & Query Optimization</td>
  <td><img src="https://skillicons.dev/icons?i=py,postgres,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>026a</td>
  <td>ETL/ELT Orchestration: Airflow Fundamentals</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>026b</td>
  <td>ETL/ELT Orchestration: Dagster Assets & dbt</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>056</td>
  <td>DuckDB: Embedded Analytics & Columnar Queries</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Distributed Systems Patterns

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>014</td>
  <td>SAGA Pattern</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>015</td>
  <td>CQRS & Event Sourcing</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>049a</td>
  <td>Raft Consensus: Leader Election & Log Replication</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>049b</td>
  <td>Raft Consensus: Safety & Failure Recovery</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>050</td>
  <td>CRDTs: Conflict-Free Replicated Data Types</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>051</td>
  <td>Distributed Coordination: etcd</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>052</td>
  <td>Resilience Patterns: Circuit Breaker, Bulkhead & Rate Limiting</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker,redis" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>053</td>
  <td>Consistent Hashing & DHTs</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Systems Programming

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td><b>012a</b></td>
  <td><b>C++17 Features & abseil-cpp</b></td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td><b>012b</b></td>
  <td><b>Boost Deep-Dive</b></td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td>013</td>
  <td>Financial C++: QuickFIX & QuantLib</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>018a</td>
  <td>CMake: FetchContent, Targets & Modern Build</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>018b</td>
  <td>CMake: Toolchains, Presets & Cross-Compilation</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>022</td>
  <td>Concurrent Data Structures: moodycamel & Lock-Free Patterns</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>060a</td>
  <td>Unsafe Rust & FFI: Raw Pointers, bindgen & repr(C)</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>060b</td>
  <td>Async Runtime Internals: Future, Waker & Tokio Architecture</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>061a</td>
  <td>Lock-Free Rust: crossbeam & Epoch-Based Reclamation</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>061b</td>
  <td>Custom Allocators: GlobalAlloc, jemalloc & Arena Patterns</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>062</td>
  <td>SIMD in Rust: portable_simd & std::arch</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>063a</td>
  <td>Profiling & Flamegraphs</td>
  <td><img src="https://skillicons.dev/icons?i=rust,cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>063b</td>
  <td>Memory Safety Verification: Miri & Sanitizers</td>
  <td><img src="https://skillicons.dev/icons?i=rust,cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>064</td>
  <td>WebAssembly from Systems Languages: WASM & WASI</td>
  <td><img src="https://skillicons.dev/icons?i=rust,wasm" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>065</td>
  <td>eBPF Observability: Kernel Tracing with aya</td>
  <td><img src="https://skillicons.dev/icons?i=rust,docker,linux" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>066</td>
  <td>io_uring: Modern Linux Async I/O</td>
  <td><img src="https://skillicons.dev/icons?i=rust,docker,linux" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## GPU Computing & High-Frequency Trading

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>019a</td>
  <td>CUDA: Kernels, Memory Model & Parallel Patterns</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=cuda" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>019b</td>
  <td>CUDA for HPC/HFT: Streams, Pinned Memory & Low-Latency</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=cuda" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>020a</td>
  <td>HFT Low-Latency C++: Lock-Free, Cache & Memory</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>020b</td>
  <td>HFT Systems: Order Book, Matching Engine & Feed Handler</td>
  <td><img src="https://skillicons.dev/icons?i=cpp,cmake" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Blockchain

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>021a</td>
  <td>Solana Smart Contracts: Anchor & Accounts Model</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=solana" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>021b</td>
  <td>Solana Tokens: SPL, Escrow & PDA Vaults</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=solana" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## AI & LLM Engineering

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>028</td>
  <td>LangExtract: Structured Extraction from Text</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td><b>029a</b></td>
  <td><b>LangChain Advanced: LCEL, Tools & Structured Output</b></td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=langchain" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td><b>029b</b></td>
  <td><b>LangGraph: Stateful Agent Graphs</b></td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=langchain" height="20"></td>
  <td>✅ Completed</td>
</tr>
<tr>
  <td>030a</td>
  <td>DSPy Fundamentals: Signatures, Modules & Optimization</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>030b</td>
  <td>DSPy Advanced: RAG, Assertions & MIPROv2</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>030c</td>
  <td>DSPy + LangGraph Integration</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>031a</td>
  <td>Agentic AI: Single-Agent Design Patterns</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>031b</td>
  <td>Agentic AI: Multi-Agent Systems</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>031c</td>
  <td>Agentic AI: Production Patterns</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>054a</td>
  <td>Google ADK: Agents, Tools & Sessions</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=google" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>054b</td>
  <td>Google ADK: Multi-Agent Orchestration</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=google" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>054c</td>
  <td>Google ADK: Callbacks, Evaluation & Streaming</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"> <img src="https://go-skill-icons.vercel.app/api/icons?i=google" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Machine Learning Systems

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>017a</td>
  <td>World Model RL: Dyna-Q Basics</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>017b</td>
  <td>World Model RL: Neural World Model</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>017c</td>
  <td>World Model RL: Latent Imagination (Dreamer)</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>025a</td>
  <td>ML Compilers: Computation Graphs & IR from Scratch</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>025b</td>
  <td>ML Compilers: Operator Fusion & Graph Rewrites</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>025c</td>
  <td>ML Compilers: XLA & HLO Inspection</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>025d</td>
  <td>ML Compilers: Triton Custom GPU Kernels</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>025e</td>
  <td>ML Compilers: TVM Scheduling & Auto-Tuning</td>
  <td><img src="https://skillicons.dev/icons?i=py,docker" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>025f</td>
  <td>ML Compilers: torch.compile Deep Dive</td>
  <td><img src="https://skillicons.dev/icons?i=py,pytorch" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Software Engineering Practices

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>024a</td>
  <td>Testing Patterns: Python (pytest, Hypothesis)</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>024b</td>
  <td>Testing Patterns: Rust (mockall, proptest)</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>027</td>
  <td>Advanced System Design Interview</td>
  <td>Markdown (design docs)</td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Operations Research: Algorithms (C++ & Rust)

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>032a</td>
  <td>LP: Simplex Method</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>032b</td>
  <td>LP: Duality & Interior Point</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>033a</td>
  <td>Convex Opt: First-Order Methods</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>033b</td>
  <td>Convex Opt: Proximal & Second-Order</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>034a</td>
  <td>MIP: Branch & Bound</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>034b</td>
  <td>MIP: Cutting Planes & Heuristics</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>035a</td>
  <td>Non-convex: Local Search & Simulated Annealing</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>035b</td>
  <td>Non-convex: Evolutionary Algorithms</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>036a</td>
  <td>Network Opt: Flows & Assignment</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>036b</td>
  <td>Network Opt: TSP & VRP Heuristics</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>037</td>
  <td>Constraint Programming</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>038</td>
  <td>Multi-Objective: NSGA-II</td>
  <td><img src="https://skillicons.dev/icons?i=rust" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>039</td>
  <td>Stochastic DP: MDPs</td>
  <td><img src="https://skillicons.dev/icons?i=cpp" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## Operations Research: Applied (Python)

<table>
<thead>
<tr><th>#<img width="50" height="1"></th><th>Practice<img width="500" height="1"></th><th>Stack<img width="200" height="1"></th><th>Status<img width="150" height="1"></th></tr>
</thead>
<tbody>
<tr>
  <td>040a</td>
  <td>LP/MIP: PuLP & HiGHS</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>040b</td>
  <td>Advanced Modeling: Pyomo</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>041a</td>
  <td>Convex Opt: CVXPY</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>041b</td>
  <td>Conic: SOCP & SDP</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>042a</td>
  <td>CP: OR-Tools CP-SAT</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>042b</td>
  <td>Routing: OR-Tools</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>043</td>
  <td>Global Opt: scipy & Optuna</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>044</td>
  <td>Multi-Objective: pymoo</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>045</td>
  <td>Stochastic Programming</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>046</td>
  <td>Robust Optimization</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>047</td>
  <td>Bayesian Optimization</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
<tr>
  <td>048</td>
  <td>Integrated OR: Supply Chain</td>
  <td><img src="https://skillicons.dev/icons?i=py" height="20"></td>
  <td>—</td>
</tr>
</tbody>
</table>

---

## About This Repository

Each practice lives in its own folder (`practice_<NNN>_<name>/`) with a `CLAUDE.md` containing theoretical context, step-by-step instructions, and exercises. The core implementation code is marked with `TODO(human)` — I write these functions myself while the surrounding infrastructure (Docker configs, project setup, test harnesses) is AI-scaffolded.

**Methodology:** All practices run locally using Docker containers and emulators — no cloud accounts or external API keys required.

**Languages:** Python (primary), C++17, Rust — chosen per practice based on what best fits the technology.
