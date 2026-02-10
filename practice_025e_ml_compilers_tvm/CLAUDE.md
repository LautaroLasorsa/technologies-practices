# Practice 025e: ML Compilers -- TVM Scheduling & Auto-Tuning

## Technologies

- **Apache TVM 0.17+** -- Open-source deep learning compiler that transforms high-level tensor expressions into optimized code for diverse hardware backends (CPU, GPU, FPGA). TVM's key innovation is separating *what* to compute (tensor expressions) from *how* to compute it (schedules).
- **TVM TE (Tensor Expressions)** -- Declarative DSL for specifying tensor computations without specifying execution order. You describe the math; the schedule describes the optimization.
- **TVM Relay** -- High-level graph IR for importing models from PyTorch/ONNX/TF, applying graph-level optimizations (constant folding, operator fusion), and lowering to TVM's tensor-level IR.
- **Ansor (AutoScheduler)** -- TVM's automatic schedule search engine. Ansor generates high-performance schedules by exploring the optimization space (tiling, vectorization, parallelism) using a learned cost model.
- **Docker** -- Container runtime; TVM is complex to install natively (requires LLVM, CMake, build from source). Docker provides a clean, reproducible environment.

## Stack

Python 3.12+ (uv), Docker (TVM container)

> **Platform note:** TVM requires LLVM and is complex to install natively on Windows. All code in this practice runs inside a Docker container. You edit files on your host machine; Docker mounts them and executes inside the container.

## Description

Learn TVM's compilation pipeline -- from tensor expressions to optimized schedules. TVM's central insight is that the same computation (e.g., matrix multiply) can be executed in radically different ways depending on how you tile, reorder, vectorize, and parallelize the loop nest. A naive matmul does `O(n^3)` work with terrible cache behavior; a well-scheduled matmul does the same `O(n^3)` work but 10-100x faster by exploiting memory hierarchy.

You'll write manual schedules using TVM's primitives (`split`, `reorder`, `vectorize`, `parallel`), inspect the generated loop IR, then compare your hand-tuned schedules against auto-generated ones from Ansor.

### What you'll learn

1. **Tensor Expressions (TE)** -- declare computations using `te.placeholder`, `te.compute`, `te.reduce_axis` without specifying execution order
2. **Schedules** -- transform the default loop nest with `split`, `reorder`, `vectorize`, `parallel`, `unroll` to optimize for hardware
3. **TIR (Tensor IR)** -- read the lowered loop IR that TVM generates from your schedule, understand the loop structure and how schedule primitives transform it
4. **Relay** -- import a PyTorch model into TVM's graph IR, apply graph-level passes (fusion, constant folding), compile and execute
5. **AutoScheduler (Ansor)** -- automatically search for optimal schedules, compare against hand-written schedules

## Instructions

### Phase 1: TVM Basics & Tensor Expressions (~15 min)

Learn the fundamental TVM abstraction: tensor expressions. A TE declares *what* to compute (the math) without specifying *how* (the loop order, tiling, etc.). The schedule controls the "how."

1. Build the Docker image: `docker compose build`
2. Open `tvm_practice/te_basics.py` -- read the vector add example (fully implemented)
3. **User implements:** Declare a matrix multiply `C[i,j] = sum_k A[i,k] * B[k,j]` using `te.compute` and `te.reduce_axis`
4. Run: `docker compose run --rm tvm python -m tvm_practice.te_basics`
5. Key question: Why does TVM separate the computation declaration from the schedule? What does this enable that PyTorch's eager mode cannot?

### Phase 2: Manual Schedule Optimization (~25 min)

Apply schedule primitives to transform the naive matmul loop nest into an optimized version. Each primitive transforms the loop structure in a specific way.

1. Open `tvm_practice/manual_schedule.py` -- read the schedule primitive documentation
2. **User implements:** Apply `split`, `reorder`, `vectorize`, `parallel` to the matmul schedule
3. Run: `docker compose run --rm tvm python -m tvm_practice.manual_schedule`
4. Inspect the before/after TIR output to see how each primitive transforms the loop nest
5. Key question: Why does tiling (split + reorder) improve cache performance? Draw the memory access pattern.

### Phase 3: Schedule Analysis & Comparison (~20 min)

Benchmark different schedule variants and understand why some are faster than others.

1. Open `tvm_practice/schedule_analysis.py` -- read the timing infrastructure
2. **User implements:** Build and benchmark naive, tiled, and tiled+parallel schedules; print a comparison table
3. Run: `docker compose run --rm tvm python -m tvm_practice.schedule_analysis`
4. Key question: At what matrix size does parallelization start helping? Why?

### Phase 4: Relay Integration (~15 min)

Import a PyTorch model into TVM's Relay IR, apply graph-level optimizations, compile, and run.

1. Open `tvm_practice/relay_import.py` -- read the model definition and tracing setup
2. **User implements:** Import via `relay.frontend.from_pytorch`, apply passes, compile, and verify against PyTorch output
3. Run: `docker compose run --rm tvm python -m tvm_practice.relay_import`
4. Key question: What graph-level optimizations does Relay apply that individual operator schedules cannot?

### Phase 5: Auto-Tuning with Ansor (~15 min)

Let TVM's AutoScheduler (Ansor) search for optimal schedules automatically and compare with your manual schedules.

1. Open `tvm_practice/auto_tune.py` -- read the task extraction and search helpers
2. **User implements:** Set up auto_scheduler task, create search policy, run tuning, apply best schedule
3. Run: `docker compose run --rm tvm python -m tvm_practice.auto_tune`
4. This phase has a "quick" mode (few trials) if tuning takes too long
5. Key question: Why does Ansor often beat hand-written schedules? What's the tradeoff?

## Motivation

- **TVM is the reference open-source ML compiler** -- understanding its architecture (TE -> schedule -> TIR -> codegen) teaches the compilation pipeline used by all modern ML compilers
- **Schedule-based optimization** is TVM's core innovation and underpins auto-tuning in production (used by Amazon, Alibaba, OctoML for model deployment)
- **Directly builds on 025a-d** -- after learning graph IR (025a), fusion (025b), XLA (025c), and Triton (025d), TVM shows the full end-to-end compiler pipeline with explicit scheduling
- **Auto-tuning** is the future of ML compilation -- writing manual schedules is educational, but production systems use learned cost models to search the optimization space
- **Industry demand** -- TVM is used at AWS (SageMaker Neo), Alibaba (PAI), OctoML, Qualcomm, and many edge deployment companies

## References

- [TVM Documentation](https://tvm.apache.org/docs/)
- [TVM Tensor Expression Tutorial](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html)
- [TVM Schedule Primitives](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html)
- [TVM AutoScheduler (Ansor)](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html)
- [TVM Relay](https://tvm.apache.org/docs/arch/relay_intro.html)
- [Ansor Paper: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762)
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `docker compose build` | Build the Docker image with TVM, LLVM, PyTorch, and NumPy |
| `docker compose run --rm tvm python -c "import tvm; print(tvm.__version__)"` | Verify TVM is installed correctly inside the container |

### Run Individual Phases

| Command | Description |
|---------|-------------|
| `docker compose run --rm tvm python -m tvm_practice.te_basics` | Phase 1: TVM tensor expressions -- vector add and matrix multiply |
| `docker compose run --rm tvm python -m tvm_practice.manual_schedule` | Phase 2: Manual schedule optimization -- split, reorder, vectorize, parallel |
| `docker compose run --rm tvm python -m tvm_practice.schedule_analysis` | Phase 3: Benchmark naive vs tiled vs parallel schedules, print comparison table |
| `docker compose run --rm tvm python -m tvm_practice.relay_import` | Phase 4: Import PyTorch MLP via Relay, compile and verify output |
| `docker compose run --rm tvm python -m tvm_practice.auto_tune` | Phase 5: Auto-tune matmul with Ansor, compare against manual schedules |

### Run All Phases

| Command | Description |
|---------|-------------|
| `docker compose run --rm tvm python -m tvm_practice.main` | Run all 5 phases sequentially |

### Interactive Shell

| Command | Description |
|---------|-------------|
| `docker compose run --rm tvm bash` | Open interactive shell inside the TVM container for experimentation |

### Teardown

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove the TVM container |
| `docker compose down -v` | Stop containers and remove volumes |

## State

`not-started`
