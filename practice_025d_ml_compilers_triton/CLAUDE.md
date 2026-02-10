# Practice 025d: ML Compilers — Triton Custom GPU Kernels

## Technologies

- **Triton** — OpenAI's language and compiler for writing GPU kernels at block level, where you program tiles of data rather than individual threads
- **PyTorch** — Reference implementations and tensor utilities for comparing against Triton kernels
- **CUDA** — NVIDIA GPU computing platform; Triton compiles down to PTX/CUDA assembly targeting this runtime

## Stack

Python 3.12+ (uv), NVIDIA GPU required, **Docker with GPU access on Windows**

**Platform requirement:** Triton requires Linux + NVIDIA GPU. On Windows, use Docker with NVIDIA Container Toolkit:

1. **Install Docker Desktop for Windows** (if not already) and enable WSL2 backend in Docker settings.
2. **Install NVIDIA Container Toolkit** — follow [NVIDIA's Docker guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). This enables `--gpus` flag in Docker.
3. **Verify GPU access in Docker:**
   ```
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see your GPU listed. If not, check Docker Desktop settings and NVIDIA driver.
4. **Build the practice image:**
   ```
   docker compose build
   ```
5. **Run any phase inside the container:**
   ```
   docker compose run --rm triton python -m kernels.main
   ```

**Prerequisite:** Practice 019a (CUDA basics) recommended but not required. Triton abstracts away most CUDA complexity.

## Description

Write custom GPU kernels using Triton's block-level programming model. Unlike CUDA where you program individual threads and must manually manage shared memory, coalescing, and synchronization, Triton operates on **blocks of data** — the compiler handles thread mapping, memory coalescing, and shared memory usage. This dramatically simplifies GPU programming while achieving near-CUDA performance.

Build three kernels of increasing complexity:
1. **Vector add** — the "hello world" of GPU programming; learn `tl.load`, `tl.store`, masking
2. **Softmax** — reduction operations within a block; numerically-stable online softmax
3. **Fused MatMul+ReLU** — tiled matrix multiplication with fused activation; the most important pattern in ML

Benchmark each against PyTorch eager execution to understand when custom kernels provide meaningful speedup.

### What you'll learn

1. **Triton programming model** — `@triton.jit`, `program_id`, `tl.load`/`tl.store`, `tl.constexpr`
2. **Block-level parallelism** — how Triton maps blocks to GPU SMs (streaming multiprocessors), and why you think in tiles not threads
3. **Memory access patterns** — coalesced loads via `tl.arange`, masking for boundary conditions when tensor size is not a multiple of BLOCK_SIZE
4. **Reduction operations** — `tl.max` and `tl.sum` for tree reduction within a block (used in softmax normalization)
5. **Tiled matrix multiplication** — blocking for shared memory reuse, accumulator pattern, inner K-loop
6. **Kernel fusion** — combining matmul + ReLU into a single kernel to avoid memory round-trips
7. **Benchmarking & auto-tuning** — `triton.testing.do_bench`, `@triton.autotune` for sweeping BLOCK_SIZE, num_warps, num_stages

## Instructions

### Phase 1: Vector Add Kernel (~15 min)

The simplest possible Triton kernel. Learn the fundamental building blocks: `program_id`, `tl.arange`, `tl.load`, `tl.store`, and masking.

1. Open `kernels/vector_add.py` — read the scaffold and launch code (fully provided)
2. **User implements:** `triton_vector_add_kernel` — the `@triton.jit` decorated kernel function
3. Run: `docker compose run --rm triton python -m kernels.vector_add`
4. Verify output matches PyTorch reference
5. Key question: Why do we need the `mask = offsets < n` check? What happens without it?

### Phase 2: Softmax Kernel (~25 min)

A numerically-stable row-wise softmax. This introduces **reduction operations** — computing max and sum across elements within a single block.

1. Open `kernels/softmax.py` — read the scaffold and understand the row-wise processing model
2. **User implements:** `triton_softmax_kernel` — load a row, compute max, subtract, exp, sum, normalize, store
3. Run: `docker compose run --rm triton python -m kernels.softmax`
4. Verify output matches `torch.softmax(x, dim=-1)`
5. Key question: Why must we subtract the row max before computing exp? (Hint: numerical stability, overflow)

### Phase 3: Fused MatMul+ReLU (~25 min)

The crown jewel: a tiled matrix multiplication with fused ReLU activation. This is the same pattern that production ML compilers (TorchInductor, TensorRT) generate for linear layers.

1. Open `kernels/matmul_relu.py` — read the scaffold, understand the tile layout and accumulator pattern
2. **User implements:** `triton_matmul_relu_kernel` — tiled matmul with K-loop accumulation and fused ReLU
3. Run: `docker compose run --rm triton python -m kernels.matmul_relu`
4. Verify output matches `torch.relu(A @ B)`
5. Key question: Why do we iterate over K in blocks instead of loading entire rows/columns? (Hint: shared memory size, data reuse)

### Phase 4: Benchmarking & Analysis (~15 min)

Compare all three kernels against their PyTorch equivalents. Understand where custom kernels win and where PyTorch's built-ins are already well-optimized.

1. Open `kernels/benchmark.py` — read the benchmark framework (fully provided)
2. **User implements:** The benchmark runner that calls each kernel and its PyTorch equivalent, collects timings, and prints a comparison table
3. Run: `docker compose run --rm triton python -m kernels.benchmark`
4. Key question: For which operation is the speedup largest? Why? (Hint: fusion eliminates memory round-trips)

### Phase 5: Occupancy & Tuning (~15 min)

Use Triton's `@triton.autotune` decorator to sweep over BLOCK_SIZE, num_warps, and num_stages configurations. See how the optimal configuration depends on problem size.

1. Read the auto-tune section at the bottom of each kernel file
2. Experiment with different `BLOCK_SIZE` values (64, 128, 256, 512, 1024)
3. Try different `num_warps` values (1, 2, 4, 8)
4. Key question: Why does the optimal BLOCK_SIZE depend on the problem? (Hint: occupancy, register pressure, shared memory)

## Motivation

- **Triton is the code generation backend** for `torch.compile` (TorchInductor) — understanding Triton means understanding the code your compiler generates
- **Writing custom kernels** is essential for pushing beyond what compilers auto-generate; production teams at OpenAI, Meta, and Google regularly write Triton kernels for custom attention, quantization, and fusion patterns
- **Block-level programming** is the future of GPU development — it's the right abstraction between "too low" (CUDA threads) and "too high" (Python operators)
- **Directly builds on 025a-b** — you designed the graph IR and fusion passes; now you write the actual GPU code that fused operators compile to
- **High demand** — Triton kernel development is one of the most sought-after skills at NVIDIA, OpenAI, Meta, Google, and ML infrastructure startups

## References

- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials — Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton Tutorials — Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton Tutorials — Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (paper)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [PyTorch TorchInductor — how torch.compile uses Triton](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `docker compose build` | Build the Docker image with PyTorch, Triton, and CUDA runtime |
| Setup | `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` | Verify GPU access from Docker before building |
| Phase 1 | `docker compose run --rm triton python -m kernels.vector_add` | Run vector add kernel and verify against PyTorch reference |
| Phase 2 | `docker compose run --rm triton python -m kernels.softmax` | Run softmax kernel and verify against PyTorch reference |
| Phase 3 | `docker compose run --rm triton python -m kernels.matmul_relu` | Run fused matmul+ReLU kernel and verify against PyTorch reference |
| Phase 4 | `docker compose run --rm triton python -m kernels.benchmark` | Benchmark all kernels vs PyTorch, print comparison table |
| All | `docker compose run --rm triton python -m kernels.main` | Run all phases sequentially |
| Shell | `docker compose run --rm triton bash` | Open interactive shell inside the container for experimentation |

## State

`not-started`
