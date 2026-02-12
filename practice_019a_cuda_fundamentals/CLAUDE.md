# Practice 019a: CUDA Fundamentals for HPC & HFT

## Technologies
- CUDA Runtime API (CUDA 12.x)
- CMake with CUDA language support
- MSVC 19.x + nvcc

## Stack
- C++17 (host), CUDA C++ (device)
- Windows, VS 2022, CUDA Toolkit 12.6

## Theoretical Context

### What CUDA Is and What Problem It Solves

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that exposes GPUs for general-purpose computing (GPGPU). Traditional CPUs excel at sequential tasks with complex control flow; GPUs excel at data-parallel tasks (applying the same operation to thousands of elements). CUDA solves the problem of harnessing GPU parallelism for workloads beyond graphics: scientific computing (weather simulation, molecular dynamics), machine learning (training neural networks), and finance (Monte Carlo option pricing, risk aggregation).

The core abstraction is **SIMT (Single Instruction, Multiple Threads)**: thousands of lightweight threads execute the same kernel code on different data. Unlike CPU threads (OS-scheduled, context-switching overhead), GPU threads are hardware-scheduled with near-zero context-switch cost. A modern GPU has thousands of CUDA cores organized into **streaming multiprocessors (SMs)**, each capable of running 1024-2048 threads concurrently. This architecture trades off single-thread performance (lower clock speeds, simpler cores) for massive parallelism—perfect for embarrassingly parallel workloads.

### How CUDA Works Internally

CUDA's execution model is **hierarchical**: a **grid** of **blocks**, each containing **threads**. When you launch a kernel with `<<<gridDim, blockDim>>>`, you specify this hierarchy. For example, `<<<256, 512>>>` launches 256 blocks of 512 threads each (131,072 threads total). The GPU schedules blocks onto SMs; each SM executes blocks' threads in groups of 32 called **warps**. Warps are the fundamental execution unit—all 32 threads in a warp execute the same instruction in lockstep (SIMT). Divergent control flow (if/else branches) within a warp causes serialization: both paths execute, with inactive threads masked out.

CUDA's **memory hierarchy** is explicit and tiered for latency/bandwidth tradeoffs:

| Memory | Scope | Latency | Bandwidth | Typical Size | Use Case |
|--------|-------|---------|-----------|--------------|----------|
| **Registers** | Per-thread | 1 cycle | ~TB/s | 64 KB per SM (256 KB per thread) | Loop counters, temp variables |
| **Shared Memory** | Per-block | ~20 cycles | ~TB/s | 48-164 KB per SM | Inter-thread communication, user-managed cache |
| **L1/L2 Cache** | Per-SM / Global | ~30 / ~200 cycles | ~TB/s / ~GB/s | 128 KB / 40 MB (A100) | Automatic caching of global memory |
| **Global Memory** | All threads | ~400 cycles | ~GB/s (HBM2e: 2 TB/s on A100) | 8-80 GB | Input/output data, large arrays |
| **Constant Memory** | All threads | cached ~20 cycles | ~GB/s | 64 KB | Read-only parameters broadcast to all threads |

**Memory coalescing** is critical for performance: when threads in a warp access consecutive addresses in global memory, the hardware coalesces requests into a single transaction. Strided or random access patterns cause multiple transactions, drastically reducing bandwidth utilization (from 900 GB/s to 50 GB/s on an RTX 3090).

**Shared memory** acts as a user-managed cache—up to 48-164 KB per block, shared by all threads in that block. Algorithms use it for **tiling**: load a tile of global memory data into shared memory, perform computations using fast shared memory accesses, write results back to global memory. This pattern (load → compute → store) is ubiquitous in matrix multiplication, convolution, and reduction kernels.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Kernel** | Function marked `__global__` that runs on the GPU. Callable from host (CPU), executes on device (GPU). Each thread runs the kernel independently. |
| **Grid** | 1D/2D/3D array of blocks. Configured via `<<<gridDim, blockDim>>>`. Total threads = gridDim * blockDim (per dimension, multiplied together). |
| **Block** | 1D/2D/3D array of threads within a grid. All threads in a block execute on the same SM and can share data via shared memory and synchronize. |
| **Thread** | Smallest execution unit. Identified by `threadIdx` (within block) and `blockIdx` (within grid). Each thread computes a unique global index. |
| **Warp** | Group of 32 threads executed in lockstep on an SM. The fundamental SIMT execution unit. Divergent control flow causes warp serialization. |
| **Shared Memory** | Fast on-chip memory (~TB/s bandwidth) shared by all threads in a block. Used for tiling, inter-thread communication. Requires `__syncthreads()`. |
| **Global Memory** | Large off-chip DRAM (8-80 GB). High latency (~400 cycles), high bandwidth (~GB/s HBM). Accessed via `cudaMalloc`, `cudaMemcpy`. |
| **Memory Coalescing** | Hardware optimization that combines adjacent memory accesses from threads in a warp into a single transaction. Critical for bandwidth utilization. |
| **`__syncthreads()`** | Block-level barrier. All threads in a block wait until all reach this point. Used to synchronize shared memory writes before reads. |
| **Occupancy** | Ratio of active warps to maximum possible warps on an SM. Higher occupancy hides latency. Controlled by register usage, shared memory, block size. |

### Ecosystem Context

**CUDA vs alternatives**: CUDA is NVIDIA-exclusive. **OpenCL** (open standard) works on NVIDIA, AMD, and Intel GPUs but has lower adoption and performance. **ROCm** (AMD) and **oneAPI** (Intel) are vendor-specific alternatives. For production ML/HPC, CUDA dominates due to ecosystem maturity (cuDNN, cuBLAS, TensorRT). For portability, teams write high-level code (PyTorch, TensorFlow) that compiles to CUDA/ROCm/oneAPI under the hood.

**CUDA vs CPU**: CUDA shines when (1) data parallelism is high (millions of independent operations), (2) memory access patterns are coalesced, (3) arithmetic intensity is high (many FLOPs per byte). CUDA loses when (1) serial logic dominates, (2) irregular memory access (random indexing), (3) frequent host-device transfers. For HFT, CUDA handles batch pricing (1000s of options in parallel) but not order matching (serial, latency-sensitive).

**CUDA in the ML stack**: Modern ML frameworks (PyTorch, JAX) abstract CUDA via **kernels** (cuDNN for convolution, cuBLAS for matmul). Users write Python; the framework compiles to CUDA kernels. Custom CUDA kernels (via Triton, CuPy, or raw CUDA) are written for novel operations (fused attention, custom activations). Understanding CUDA fundamentals is essential for writing high-performance custom layers.

## Description

Hands-on introduction to GPU programming with CUDA. Six progressive phases covering the core mental model: thousands of threads executing the same kernel in parallel, with an explicit memory hierarchy you control. Each phase builds a working program that verifies GPU results against a CPU reference.

### Phase 1 — Hello CUDA (~15 min)
Device query, first kernel launch, `__global__`, `<<<blocks, threads>>>`, `cudaDeviceSynchronize()`.
**Analogy:** Think of GPU threads like running 10,000 independent CP solutions in parallel — each thread gets a unique ID and executes the same code on different data (SIMT).

### Phase 2 — Memory Model (~20 min)
Host vs device memory. `cudaMalloc`/`cudaMemcpy`/`cudaFree` (explicit transfers) vs `cudaMallocManaged` (unified memory, automatic transfers at a performance cost). Build a **vector addition** kernel.

### Phase 3 — Thread Hierarchy & Indexing (~20 min)
Grid, blocks, threads, warps. `threadIdx.x`, `blockIdx.x`, `blockDim.x`. 1D/2D/3D grids. Implement a **naive matrix multiplication** kernel — the "hello world" of GPU computing. Compute global thread indices and handle boundary conditions.

### Phase 4 — Shared Memory & Tiling (~20 min)
`__shared__` memory as user-managed L1 cache. Bank conflicts. Memory coalescing. Re-implement matmul with **shared memory tiling** — the classic optimization showing 5-10x speedup.
**HFT relevance:** Shared memory is how you build fast GPU order books and option pricing grids.

### Phase 5 — Reduction & Parallel Patterns (~15 min)
Parallel reduction (sum N numbers in O(log N) steps). Warp-level primitives (`__shfl_down_sync`). The GPU equivalent of segment trees in CP — hierarchical aggregation.
**HFT relevance:** Fast portfolio risk aggregation, real-time P&L across thousands of positions.

### Phase 6 — Error Handling & Profiling (~10 min)
`cudaGetLastError()`, `cudaError_t` checking macro. CUDA events for kernel timing. Brief intro to Nsight Compute.

## Instructions

### Prerequisites
- CUDA Toolkit 12.x installed (`nvcc --version` works)
- CMake 3.18+ (bundled with VS 2022)
- NVIDIA GPU with compute capability >= 5.0

### Build & Run

```powershell
# From practice root directory
.\build.bat

# Run individual phases
.\build\Release\phase1_hello.exe
.\build\Release\phase2_memory.exe
.\build\Release\phase3_matmul.exe
.\build\Release\phase4_shared.exe
.\build\Release\phase5_reduction.exe
.\build\Release\phase6_profiling.exe
```

### Workflow
1. Read the phase source file — understand the boilerplate and CUDA setup
2. Find the `TODO(human)` markers — these are the kernel bodies you implement
3. Build and run — the program verifies your GPU output against a CPU reference
4. Move to the next phase once verification passes

## Motivation

GPU computing is essential for:
- **HFT:** Options pricing (Monte Carlo, Black-Scholes on thousands of strikes), risk calculations, market data processing, order book management
- **HPC:** Scientific computing, ML training/inference, simulation
- **Industry demand:** CUDA is the dominant GPU framework; understanding the memory hierarchy and parallel patterns translates directly to writing low-latency GPU pipelines

This complements C++17 skills from practice 012a. The memory model concepts (explicit allocation, data locality, cache management) are the same principles that make low-latency C++ fast — CUDA just makes them more visible and more critical.

## Commands

All commands are run from `practice_019a_cuda_fundamentals/`. Uses VS 2022 bundled CMake with the Visual Studio 17 2022 generator for CUDA support.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -G 'Visual Studio 17 2022' 2>&1"` | Configure the project (auto-detects CUDA toolkit and GPU architecture) |

### Build All Targets

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release 2>&1"` | Build all 6 phase executables (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Debug 2>&1"` | Build all 6 phase executables (Debug) |

### Build Individual Targets

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_hello 2>&1"` | Build Phase 1 only — device query + first kernel |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_memory 2>&1"` | Build Phase 2 only — vector addition, manual vs managed memory |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_matmul 2>&1"` | Build Phase 3 only — naive matrix multiplication |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_shared 2>&1"` | Build Phase 4 only — tiled matmul with shared memory |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase5_reduction 2>&1"` | Build Phase 5 only — parallel reduction + warp shuffles |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase6_profiling 2>&1"` | Build Phase 6 only — error handling + CUDA events profiling |

### Run Executables

| Command | Description |
|---------|-------------|
| `.\build\Release\phase1_hello.exe` | Phase 1: Device query + first kernel launch |
| `.\build\Release\phase2_memory.exe` | Phase 2: Vector addition with manual and managed memory |
| `.\build\Release\phase3_matmul.exe` | Phase 3: Naive matrix multiplication (verifies against CPU) |
| `.\build\Release\phase4_shared.exe` | Phase 4: Tiled matmul with shared memory (compare speedup vs Phase 3) |
| `.\build\Release\phase5_reduction.exe` | Phase 5: Parallel reduction + warp shuffle primitives |
| `.\build\Release\phase6_profiling.exe` | Phase 6: Error handling macros + CUDA event timing |

### Cleanup

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "if (Test-Path build) { Remove-Item -Recurse -Force build }"` | Remove build directory |

### Helper Script (alternative)

| Command | Description |
|---------|-------------|
| `.\build.bat` | Configure + build all targets (Release) in one step |

## State

`not-started`
