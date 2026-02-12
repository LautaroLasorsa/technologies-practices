# Practice 025f: ML Compilers --- torch.compile Deep Dive

## Technologies

- **PyTorch 2.x** --- Deep learning framework with native compilation support via `torch.compile`
- **TorchDynamo** --- Python-level JIT that captures PyTorch operations into FX graphs by rewriting Python bytecode at runtime
- **TorchInductor** --- The default backend compiler for `torch.compile`; generates optimized Triton kernels (GPU) or C++/OpenMP code (CPU)
- **Triton** --- OpenAI's GPU kernel language; TorchInductor's code generation target for CUDA GPUs

## Stack

Python 3.12+ (uv), Docker (for full Inductor/Triton backend)

## Theoretical Context

### What is torch.compile?

**torch.compile** is PyTorch 2.x's JIT compilation system that transforms eager PyTorch code into optimized kernels. Unlike TorchScript (PyTorch's previous compiler, now deprecated), `torch.compile` operates via **Python bytecode rewriting** at runtime: it intercepts PyTorch operations *as your Python code runs*, captures them into FX graphs, and dispatches those graphs to backend compilers (TorchInductor by default). This "define-by-run" approach preserves PyTorch's dynamic, Pythonic API while achieving static-graph optimization performance.

torch.compile solves the **ease-of-use vs performance dilemma** in deep learning frameworks. Eager execution (PyTorch's default) is flexible (arbitrary Python, easy debugging) but slow (kernel-level overhead, no cross-operation optimization). Static graphs (TorchScript, XLA) are fast (operator fusion, whole-program optimization) but restrictive (limited Python support, hard to debug). torch.compile achieves **both**: you write normal PyTorch code, and Dynamo automatically extracts optimizable subgraphs without sacrificing Python flexibility.

### How torch.compile Works Internally

The torch.compile pipeline has three stages:

1. **TorchDynamo (Graph Capture)**: Dynamo rewrites Python bytecode at import time, inserting hooks that intercept PyTorch operations. When a `torch.compile(model)` calls `model(x)`, Dynamo **traces** execution: it records every tensor operation (`torch.matmul`, `torch.relu`, etc.) into an **FX graph** (torch.fx IR from Practice 025b). If Dynamo encounters unsupported operations (print statements, data-dependent control flow, non-torch operations), it emits a **graph break** — ending the current graph and starting a new one. Multiple graphs are compiled separately.

2. **AOTAutograd (Ahead-Of-Time Autograd)**: Before handing graphs to backends, PyTorch runs AOTAutograd to **precompute backward graphs**. For each forward FX graph, AOTAutograd generates the corresponding backward graph using PyTorch's autograd engine. This eliminates runtime autodiff overhead — the backward pass becomes a compiled graph too.

3. **TorchInductor (Code Generation)**: The default backend. Inductor takes FX graphs and generates device-specific code:
   - **GPU**: Emits **Triton kernels** (Practice 025d). Inductor's fusion pass identifies fusible patterns (matmul+bias+relu → single kernel) and generates corresponding Triton code. Triton compiles to PTX/CUDA.
   - **CPU**: Emits **C++ loops with OpenMP parallelization**. Uses vectorization (AVX) and tiling for cache locality.

**Graph breaks** are the most common issue when adopting torch.compile. When Dynamo cannot trace an operation (e.g., `if x.item() > 0:`), it stops the current graph, compiles what it has so far, returns to eager mode for the unsupported operation, then starts tracing a new graph. Multiple graphs → multiple kernel launches → less optimization opportunity.

**Dynamic shapes**: torch.compile supports dynamic batch sizes (shape polymorphism). When a model is first compiled with shape `[batch, 3, 224, 224]`, Dynamo generates code parameterized by batch size. Subsequent calls with different batch sizes reuse the same compiled graph (no recompilation) as long as other dimensions match. This is critical for production deployment with varying batch sizes.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **TorchDynamo** | Python bytecode rewriter that captures PyTorch operations into FX graphs at runtime. |
| **FX Graph** | torch.fx IR (from 025b) — dataflow graph of tensor operations. Dynamo's output. |
| **Graph Break** | When Dynamo encounters non-traceable code (print, control flow). Splits into multiple graphs. |
| **AOTAutograd** | Precomputes backward graphs from forward graphs before compilation. |
| **TorchInductor** | Default backend. Generates Triton (GPU) or C++/OpenMP (CPU) code from FX graphs. |
| **Triton** | OpenAI's GPU kernel DSL (025d). Inductor's code generation target for CUDA. |
| **Operator Fusion** | Merging ops (matmul+bias+relu) into single kernels. Inductor's primary optimization. |
| **Dynamic Shapes** | Supporting variable batch sizes without recompilation. Uses symbolic shape inference. |
| **Backends** | `eager` (noop), `aot_eager` (AOTAutograd only), `inductor` (full compilation). |
| **Warmup** | First call pays compilation cost. Subsequent calls reuse compiled code. |

### Ecosystem Context

torch.compile replaces PyTorch's previous compilation attempts:

- **TorchScript** (deprecated): Required annotating models with `@torch.jit.script`. Restrictive (no dynamic control flow), hard to debug. Replaced by torch.compile.
- **PyTorch/XLA**: Compiles PyTorch to XLA's HLO (025c). Used for TPUs. torch.compile + XLA backend is newer, preferred approach.
- **ONNX Runtime**: Export PyTorch → ONNX → optimize → run in ONNX Runtime. Requires static graphs. torch.compile compiles eagerly without export.

**Trade-offs vs alternatives**:
- vs **TensorRT**: TensorRT (NVIDIA's optimizer) is faster on NVIDIA GPUs but proprietary and requires ONNX export. torch.compile is open-source and native to PyTorch.
- vs **JAX/XLA** (025c): JAX requires rewriting models in JAX's functional style. torch.compile works with existing PyTorch code.
- vs **TVM** (025e): TVM provides more portable backends (ARM, FPGA) but requires manual integration. torch.compile is native to PyTorch.

**Adoption**: torch.compile is PyTorch's primary optimization path as of 2.x. Meta uses it in production (FAIR, Instagram, WhatsApp). Google uses it for YouTube recommendations. Microsoft uses it in Azure ML. Understanding torch.compile is now table stakes for ML infrastructure roles.

**Platform note:** Phases 1--3 work natively on Windows using the `eager` and `aot_eager` backends. Phases 4--5 require the `inductor` backend with Triton code generation, which only works on Linux --- use Docker for those phases.

## Description

Understand `torch.compile` end-to-end: how TorchDynamo captures Python code into FX graphs, how to debug and eliminate graph breaks, how TorchInductor generates fused Triton/C++ kernels, and how to benchmark compiled models against eager execution.

This practice bridges the gap between "calling torch.compile(model)" and understanding what actually happens underneath --- the same compilation pipeline used by Meta, Google, and every production PyTorch deployment.

### What you'll learn

1. **torch.compile backends** --- what `eager`, `aot_eager`, and `inductor` each do, when to use which, and the compilation cost tradeoff
2. **TorchDynamo graph capture** --- how Python bytecode is rewritten to capture PyTorch operations into an FX graph, and what `torch._dynamo.explain()` reveals
3. **Graph breaks** --- why they happen (print, data-dependent control flow, unsupported ops), how to detect them, and how to eliminate them
4. **TorchInductor codegen** --- inspecting the generated Triton/C++ code, understanding operator fusion, and what `TORCH_LOGS="output_code"` shows
5. **Compilation benchmarking** --- measuring compilation overhead vs runtime speedup, warmup effects, and when torch.compile actually helps

## Instructions

### Phase 1: torch.compile Basics (~15 min) --- Windows OK

Learn what `torch.compile()` does at a high level. Compile a simple MLP with different backends and observe compilation behavior.

1. From this folder: `uv sync`
2. Open `compile_inspect/basics.py` --- read the MLP model and backend descriptions
3. **User implements:** `run_backend_comparison()` --- compile the model with each backend (`eager`, `aot_eager`, `inductor`), run inference, verify outputs match, measure first-call vs subsequent-call latency
4. Run: `uv run python -m compile_inspect.basics`
5. Key question: Why is the first call to a compiled model much slower than subsequent calls? What is happening during that first call?

### Phase 2: TorchDynamo Graph Capture (~20 min) --- Windows OK

Use `torch._dynamo.explain()` to see exactly what TorchDynamo captures from your model.

1. Open `compile_inspect/dynamo_capture.py` --- read the models and the explain() usage
2. **User implements:** `run_dynamo_exploration()` --- trace models with `explain()`, print the captured FX graph, count FX operations, and identify how Python-level operations map to graph nodes
3. Run: `uv run python -m compile_inspect.dynamo_capture`
4. Key question: What is the relationship between FX graph nodes and the original PyTorch operations? Does every Python line become exactly one node?

### Phase 3: Graph Break Analysis (~20 min) --- Windows OK

Understand why graph breaks happen, how to detect them, and how to fix the most common causes.

1. Open `compile_inspect/graph_breaks.py` --- read the models that intentionally cause graph breaks
2. **User implements:** `run_fix_graph_breaks()` --- fix the broken models so they compile without graph breaks, using the provided hints and `torch._dynamo.explain()` to verify
3. Run: `uv run python -m compile_inspect.graph_breaks`
4. Key question: Why do graph breaks matter for performance? What happens when DynAMO encounters a graph break during compilation?

### Phase 4: TorchInductor Code Inspection (~20 min) --- Docker Recommended

Inspect the actual code that TorchInductor generates. On GPU this is Triton kernels; on CPU it is C++/OpenMP.

1. Open `compile_inspect/inductor_codegen.py` --- read the model and the code inspection setup
2. **User implements:** `run_inductor_inspection()` --- trigger inductor compilation, capture/inspect generated code, identify which operations were fused together
3. Run (Docker): `docker compose run --rm torchcompile python -m compile_inspect.inductor_codegen`
4. Run (native, CPU only): `uv run python -m compile_inspect.inductor_codegen`
5. Key question: How does TorchInductor decide which operations to fuse into a single kernel? What fusion patterns do you see in the generated code?

### Phase 5: Benchmarking & Profiling (~15 min) --- Docker Recommended

Measure the real-world impact of `torch.compile` --- compilation time, inference speedup, and how batch size affects the result.

1. Open `compile_inspect/benchmark.py` --- read the model setup and timing infrastructure
2. **User implements:** `run_benchmark_suite()` --- implement warmup + benchmark loops, compare eager vs compiled across different batch sizes, format a results table
3. Run (Docker): `docker compose run --rm torchcompile python -m compile_inspect.benchmark`
4. Run (native, CPU only): `uv run python -m compile_inspect.benchmark`
5. Key question: At what batch size does `torch.compile` start providing meaningful speedup? Why does it not help (or even hurt) for very small batches?

## Motivation

- **torch.compile is the present of PyTorch** --- as of PyTorch 2.x, it is the primary performance optimization path, replacing TorchScript and earlier JIT approaches
- **Understanding the compilation pipeline** (Dynamo -> AOTAutograd -> Inductor -> Triton/C++) is essential for debugging production model performance
- **Graph breaks are the #1 issue** developers hit when adopting torch.compile --- knowing how to find and fix them is a practical, high-value skill
- **Directly builds on 025a-d** --- you built graph IRs (025a), fusion passes (025b), understood XLA compilation (025c), and wrote Triton kernels (025d); this practice shows how PyTorch ties all of them together into `torch.compile`
- **Industry standard** --- Meta, Google, Microsoft, and every major ML team uses torch.compile in production; understanding it is expected for senior ML infrastructure roles

## References

- [torch.compile Official Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [TorchDynamo Deep Dive](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)
- [TorchInductor: A PyTorch-Native Compiler](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)
- [torch._dynamo.explain()](https://pytorch.org/docs/stable/torch.compiler_api.html#torch.compiler.explain)
- [PyTorch 2.0 Announcement](https://pytorch.org/blog/pytorch-2.0-release/)
- [Understanding torch.compile Graph Breaks](https://pytorch.org/docs/stable/torch.compiler_faq.html)

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies (PyTorch CPU) for native Windows use |
| Setup | `docker compose build` | Build Docker image with PyTorch + CUDA + Triton for full Inductor backend |
| Phase 1 | `uv run python -m compile_inspect.basics` | Run torch.compile basics with different backends (Windows OK) |
| Phase 2 | `uv run python -m compile_inspect.dynamo_capture` | Explore TorchDynamo graph capture with explain() (Windows OK) |
| Phase 3 | `uv run python -m compile_inspect.graph_breaks` | Analyze and fix graph breaks (Windows OK) |
| Phase 4 (native) | `uv run python -m compile_inspect.inductor_codegen` | Inspect Inductor codegen on CPU (C++/OpenMP only) |
| Phase 4 (Docker) | `docker compose run --rm torchcompile python -m compile_inspect.inductor_codegen` | Inspect Inductor codegen with GPU (Triton kernels) |
| Phase 5 (native) | `uv run python -m compile_inspect.benchmark` | Benchmark eager vs compiled on CPU |
| Phase 5 (Docker) | `docker compose run --rm torchcompile python -m compile_inspect.benchmark` | Benchmark eager vs compiled with GPU for full speedup |
| All | `uv run python -m compile_inspect.main` | Run all 5 phases sequentially (native) |
| All (Docker) | `docker compose run --rm torchcompile python -m compile_inspect.main` | Run all 5 phases sequentially in Docker |
| Shell | `docker compose run --rm torchcompile bash` | Open interactive shell in Docker container |

## State

`not-started`
