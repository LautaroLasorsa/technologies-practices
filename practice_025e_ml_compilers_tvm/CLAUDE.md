# Practice 025e: ML Compilers -- TVM Scheduling & Auto-Tuning

## Technologies

- **Apache TVM 0.17+** -- Open-source deep learning compiler that transforms high-level tensor expressions into optimized code for diverse hardware backends (CPU, GPU, FPGA). TVM's key innovation is separating *what* to compute (tensor expressions) from *how* to compute it (schedules).
- **TVM TE (Tensor Expressions)** -- Declarative DSL for specifying tensor computations without specifying execution order. You describe the math; the schedule describes the optimization.
- **TVM Relay** -- High-level graph IR for importing models from PyTorch/ONNX/TF, applying graph-level optimizations (constant folding, operator fusion), and lowering to TVM's tensor-level IR.
- **Ansor (AutoScheduler)** -- TVM's automatic schedule search engine. Ansor generates high-performance schedules by exploring the optimization space (tiling, vectorization, parallelism) using a learned cost model.
- **Docker** -- Container runtime; TVM is complex to install natively (requires LLVM, CMake, build from source). Docker provides a clean, reproducible environment.

## Stack

Python 3.12+ (uv), Docker (TVM container)

## Theoretical Context

### What is TVM?

**Apache TVM** is an open-source compiler that transforms deep learning models into optimized code for diverse hardware (CPUs, GPUs, FPGAs, ASICs). TVM's core innovation is **separating computation declaration from execution strategy**: you describe *what* to compute (tensor expressions), then specify *how* to execute it (schedules). This separation enables hardware-agnostic model authoring — the same computation can be scheduled differently for ARM CPUs, NVIDIA GPUs, or custom accelerators.

TVM solves the **performance portability problem** in ML deployment. PyTorch and TensorFlow achieve high performance via cuDNN/cuBLAS on NVIDIA GPUs, but these libraries don't exist for ARM, RISC-V, or custom AI chips. Writing hand-optimized kernels for each backend is prohibitively expensive. TVM generates optimized code automatically via **auto-scheduling** (exploring tiling, vectorization, parallelism with learned cost models) or allows manual schedule tuning when experts want control.

### How TVM Works Internally

TVM's compilation pipeline has five stages:

1. **Frontend Import** (Relay): Models from PyTorch/TensorFlow/ONNX are imported into **Relay IR**, TVM's high-level graph representation. Relay is similar to XLA's HLO or torch.fx's IR — a dataflow graph of tensor operations.

2. **Graph-Level Optimization** (Relay Passes): Apply graph transformations: operator fusion (merge element-wise ops), constant folding, dead code elimination, layout optimization. These passes mirror XLA's optimizations (025c) but are hardware-independent.

3. **Tensor Expression (TE)**: Each Relay operator is lowered to a **Tensor Expression** — a mathematical specification of what to compute. `C[i,j] = sum_k A[i,k] * B[k,j]` for matmul. This is *declarative* — no loops, no execution order.

4. **Scheduling**: A **schedule** transforms the TE's default loop nest (naive nested loops) into an optimized version. Schedule primitives:
   - **split**: Tile a loop into inner/outer loops (for cache blocking)
   - **reorder**: Change loop nesting order (outer product → inner product)
   - **vectorize**: Map inner loop to SIMD/vector instructions
   - **parallel**: Distribute outer loop across CPU cores or GPU blocks
   - **unroll**: Unroll inner loop (reduce branch overhead)

5. **Code Generation** (TIR → LLVM/CUDA): The scheduled TE is lowered to **TIR (Tensor IR)**, TVM's low-level loop IR (similar to LLVM IR but tensor-aware). TIR is then compiled to machine code via LLVM (CPU) or CUDA/PTX (GPU).

**Auto-Scheduling (Ansor)**: Instead of manually writing schedules, TVM's AutoScheduler explores the space of possible schedules (millions of combinations) using a **learned cost model**. It generates random schedules, measures their performance, trains a neural net to predict performance from schedule features, then uses the model to guide search toward fast schedules. Production deployments use Ansor; manual scheduling is educational but doesn't scale.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Tensor Expression (TE)** | Declarative math spec (what to compute). No loops, no execution order. |
| **Schedule** | Transformation of loop nest (how to compute). Maps TE to optimized code. |
| **Schedule Primitives** | split, reorder, vectorize, parallel, unroll — transformations on loop nests. |
| **Compute/Schedule Separation** | Decouple algorithm (TE) from optimization (schedule). Enables portability. |
| **TIR (Tensor IR)** | TVM's low-level loop IR. Explicit loops, memory allocations, synchronization. |
| **Relay IR** | High-level graph IR for imported models. Enables graph-level optimization. |
| **AutoScheduler (Ansor)** | Learned cost model + search algorithm for finding optimal schedules automatically. |
| **Tiling (split)** | Break loops into tiles (cache blocks) for locality. Key to matmul performance. |
| **Vectorization** | Map inner loop to SIMD (AVX, NEON). 4-16x speedup for element-wise ops. |
| **Parallelization** | Distribute outer loop across threads/cores. Scales to multi-core CPUs/GPUs. |

### Ecosystem Context

TVM competes with and complements other ML compilers:

- **XLA**: Google's compiler (025c). Similar graph optimization, but less flexible scheduling. XLA targets Google hardware (TPU). TVM is hardware-agnostic.
- **TorchInductor** (torch.compile): PyTorch's compiler. Uses Triton for GPU code generation (025d). TVM predates Inductor and is more mature for edge devices.
- **TensorRT**: NVIDIA's inference optimizer. Proprietary, GPU-only. Faster than TVM on NVIDIA GPUs, but not portable.
- **MLIR**: Compiler infrastructure for building domain-specific compilers. TVM is adopting MLIR (TVM Unity refactor). Future TVM versions use MLIR dialects.

**Trade-offs**: TVM's manual scheduling provides full control but requires expertise. Auto-scheduling (Ansor) is easier but requires tuning time (hours of measurement for production models). XLA and TorchInductor optimize automatically without user intervention but are less portable. TVM excels at **edge deployment** (mobile, IoT, custom hardware) where portability matters more than peak performance on one device.

**Adoption**: AWS (SageMaker Neo uses TVM), Alibaba (PAI), OctoML (TVM-as-a-service), Qualcomm (Hexagon DSP compiler uses TVM), many academic research groups. TVM is the de facto standard for researching ML compiler techniques.

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
