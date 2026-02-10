# Practice 019b: CUDA for HPC/HFT — Streams, Pinned Memory & Low-Latency Pipelines

## Technologies
- CUDA Runtime API (CUDA 12.x)
- CUDA Cooperative Groups
- CMake with CUDA language support
- MSVC 19.x + nvcc

## Stack
- C++17 (host), CUDA C++ (device)
- Windows, VS 2022, CUDA Toolkit 12.6

## Description

Advanced CUDA patterns used in real HPC and HFT (high-frequency trading) GPU systems. Builds on 019a fundamentals (kernels, memory model, thread hierarchy, shared memory, reductions) to cover the techniques that separate tutorial code from production GPU pipelines.

### Phase 1 — CUDA Streams & Async Operations (~15 min)
Default stream vs created streams. `cudaStreamCreate`, `cudaMemcpyAsync`, kernel launches on non-default streams. Demonstrate overlapping computation and data transfer — the foundation of all high-throughput GPU pipelines.
**HFT context:** Processing multiple order books simultaneously.

### Phase 2 — Pinned (Page-Locked) Memory (~20 min)
`cudaMallocHost` / `cudaHostAlloc` for pinned memory. Why it's faster for transfers (DMA without staging). Benchmark pageable vs pinned transfer speeds. `cudaHostAlloc` flags: `cudaHostAllocPortable`, `cudaHostAllocMapped` (zero-copy).
**HFT context:** Pinned buffers for market data ingestion from NIC to GPU.

### Phase 3 — Multi-Stream Pipeline (~20 min)
Build a complete pipeline: divide work into chunks, overlap H2D transfer of chunk N+1 with computation of chunk N and D2H transfer of chunk N-1. The canonical GPU pipeline pattern. Measure throughput vs single-stream.
**HFT context:** Streaming tick data processing with continuous GPU utilization.

### Phase 4 — Cooperative Groups & Grid Sync (~20 min)
`cooperative_groups` namespace for flexible thread synchronization. `thread_block`, `thread_block_tile`, `tiled_partition`. Grid-wide synchronization with `cooperative_groups::grid_group`. Single-kernel multi-pass algorithm (vs multiple kernel launches).
**HFT context:** Atomic order book updates requiring grid-wide consistency.

### Phase 5 — Unified Memory Deep Dive (~15 min)
`cudaMallocManaged` internals: page faulting, prefetching with `cudaMemPrefetchAsync`, memory advice with `cudaMemAdvise`. Compare naive managed memory vs prefetched vs explicit transfers.
**HFT context:** Rapid prototyping vs production latency requirements.

### Phase 6 — Low-Latency Patterns for HFT (~15 min)
Persistent kernels (kernel that stays running and polls for work), spin-waiting on GPU, synchronization latency comparison. Building a minimal GPU task queue.
**HFT context:** Sub-microsecond kernel launch overhead elimination.

## Instructions

### Prerequisites
- Completed practice 019a (CUDA fundamentals)
- CUDA Toolkit 12.x installed (`nvcc --version` works)
- CMake 3.18+ (bundled with VS 2022)
- NVIDIA GPU with compute capability >= 7.0 (cooperative groups require sm_70+)

### Build & Run

```powershell
# From practice root directory
.\build.bat

# Run individual phases
.\build\Release\phase1_streams.exe
.\build\Release\phase2_pinned.exe
.\build\Release\phase3_pipeline.exe
.\build\Release\phase4_cooperative.exe
.\build\Release\phase5_unified.exe
.\build\Release\phase6_low_latency.exe
```

### Workflow
1. Read the phase source file — understand the boilerplate and CUDA setup
2. Find the `TODO(human)` markers — these are the core patterns you implement
3. Build and run — the program measures bandwidth/latency and prints comparisons
4. Move to the next phase once results show expected speedups

## Motivation

Real HFT GPU systems don't just launch kernels — they build continuous pipelines with overlapped transfers, use pinned memory for NIC-to-GPU paths, and employ persistent kernels to eliminate launch overhead. These patterns are what separate CUDA tutorials from production GPU code. Understanding them is essential for roles at HFT firms (Citadel, Jane Street, Tower Research) and HPC shops.

This builds directly on 019a's fundamentals. Where 019a taught "how to write a kernel," 019b teaches "how to build a system around kernels."

## Commands

All commands are run from `practice_019b_cuda_hpc_hft/`. Uses VS 2022 bundled CMake with the Visual Studio 17 2022 generator. Requires GPU with compute capability >= 7.0 (Phase 4 cooperative groups need sm_70+).

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
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_streams 2>&1"` | Build Phase 1 only — CUDA streams + async operations |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_pinned 2>&1"` | Build Phase 2 only — pinned (page-locked) memory |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_pipeline 2>&1"` | Build Phase 3 only — multi-stream pipeline |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_cooperative 2>&1"` | Build Phase 4 only — cooperative groups + grid sync (requires sm_70+) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase5_unified 2>&1"` | Build Phase 5 only — unified memory deep dive |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase6_low_latency 2>&1"` | Build Phase 6 only — low-latency HFT patterns (persistent kernels) |

### Run Executables

| Command | Description |
|---------|-------------|
| `.\build\Release\phase1_streams.exe` | Phase 1: CUDA streams, async memcpy, overlapping compute + transfer |
| `.\build\Release\phase2_pinned.exe` | Phase 2: Pageable vs pinned memory bandwidth benchmark |
| `.\build\Release\phase3_pipeline.exe` | Phase 3: Multi-stream pipeline — overlapped H2D/compute/D2H |
| `.\build\Release\phase4_cooperative.exe` | Phase 4: Cooperative groups, thread_block_tile, grid-wide sync |
| `.\build\Release\phase5_unified.exe` | Phase 5: Managed memory, prefetching, cudaMemAdvise comparison |
| `.\build\Release\phase6_low_latency.exe` | Phase 6: Persistent kernels, spin-waiting, GPU task queue |

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
