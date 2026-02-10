# Practice 025f: ML Compilers --- torch.compile Deep Dive

## Technologies

- **PyTorch 2.x** --- Deep learning framework with native compilation support via `torch.compile`
- **TorchDynamo** --- Python-level JIT that captures PyTorch operations into FX graphs by rewriting Python bytecode at runtime
- **TorchInductor** --- The default backend compiler for `torch.compile`; generates optimized Triton kernels (GPU) or C++/OpenMP code (CPU)
- **Triton** --- OpenAI's GPU kernel language; TorchInductor's code generation target for CUDA GPUs

## Stack

Python 3.12+ (uv), Docker (for full Inductor/Triton backend)

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
