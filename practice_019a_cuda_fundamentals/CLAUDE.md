# Practice 019a: CUDA Fundamentals for HPC & HFT

## Technologies
- CUDA Runtime API (CUDA 12.x)
- CMake with CUDA language support
- MSVC 19.x + nvcc

## Stack
- C++17 (host), CUDA C++ (device)
- Windows, VS 2022, CUDA Toolkit 12.6

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
