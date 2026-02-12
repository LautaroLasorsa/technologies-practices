# Practice 019b: CUDA for HPC/HFT — Streams, Pinned Memory & Low-Latency Pipelines

## Technologies
- CUDA Runtime API (CUDA 12.x)
- CUDA Cooperative Groups
- CMake with CUDA language support
- MSVC 19.x + nvcc

## Stack
- C++17 (host), CUDA C++ (device)
- Windows, VS 2022, CUDA Toolkit 12.6

## Theoretical Context

### What Advanced CUDA Patterns Solve

Practice 019a taught kernel basics (memory model, thread hierarchy, shared memory). Real HPC/HFT GPU systems require **pipeline optimization**: overlapping data transfers and computation to maximize GPU utilization. A naive approach (copy data → compute → copy result back) leaves the GPU idle during transfers. **CUDA streams** enable concurrent execution of transfers and kernels on different data chunks, achieving continuous GPU utilization—critical for throughput-bound HFT systems (processing millions of ticks/sec) and HPC simulations (multi-hour runs where every wasted cycle matters).

**Pinned (page-locked) memory** eliminates OS paging overhead during DMA transfers between host and device. Normal pageable memory requires the CUDA driver to first copy data to a staging buffer before DMA; pinned memory enables direct DMA, achieving 2-3x faster transfer speeds (PCIe Gen4: ~25 GB/s pinned vs ~10 GB/s pageable). For HFT market data ingestion (network → CPU → GPU), pinned buffers are essential.

**Persistent kernels** eliminate kernel launch overhead (~5-10 µs per launch). Instead of launching a kernel for each task, a persistent kernel stays running and polls a work queue. This reduces latency from milliseconds to microseconds—crucial for HFT where microseconds matter. Production HFT GPU systems use persistent kernels for pricing engines, risk calculations, and order book updates.

### How CUDA Streams and Async Operations Work

CUDA operations execute in **streams** (command queues). The **default stream (stream 0)** is synchronous with host and blocking between kernels. **Non-default streams** enable concurrent execution: kernels/memcopies in different streams can run simultaneously if hardware resources (SMs, DMA engines) are available. Modern GPUs have 2+ DMA engines (one for H2D, one for D2H), enabling bidirectional transfers while kernels execute.

A typical pipeline pattern:
```
Stream 0: [H2D chunk 1] → [Kernel chunk 1] → [D2H chunk 1]
Stream 1:                 [H2D chunk 2] → [Kernel chunk 2] → [D2H chunk 2]
Stream 2:                                 [H2D chunk 3] → [Kernel chunk 3] → [D2H chunk 3]
```
By staggering operations across streams, you achieve **overlapped execution**: while chunk 1 is transferring results back, chunk 2 is computing, and chunk 3 is transferring in. This saturates GPU resources, achieving 2-3x throughput vs single-stream.

**`cudaMemcpyAsync`** returns immediately; the transfer happens asynchronously. **`cudaStreamSynchronize(stream)`** blocks host until all work in that stream completes. **`cudaEventRecord` / `cudaEventSynchronize`** provide fine-grained synchronization: record an event in a stream, wait on it from another stream or host.

### Pinned Memory and Zero-Copy

**Pageable memory** (`malloc`, `new`) can be swapped to disk by the OS. CUDA must lock pages before DMA, requiring a staging copy. **Pinned memory** (`cudaMallocHost`, `cudaHostAlloc`) is guaranteed resident in physical RAM, enabling direct DMA. Trade-off: pinned memory reduces OS flexibility (can't be paged out) and counts against system RAM limits. Best practice: use pinned memory only for active transfer buffers, not bulk storage.

**Zero-copy memory** (`cudaHostAllocMapped`) allows GPU kernels to directly access host memory via PCIe. Latency is high (~400ns vs ~100ns for device memory), but it eliminates explicit `cudaMemcpy` for small data. Use case: configuration structs, rare-access metadata. Not suitable for bulk data (bandwidth limited to PCIe ~25 GB/s vs device HBM ~900 GB/s).

### Cooperative Groups and Grid-Wide Synchronization

**Cooperative Groups** (`cooperative_groups` namespace) generalize thread synchronization beyond `__syncthreads()`. **`thread_block`** represents all threads in a block (equivalent to `__syncthreads()`). **`thread_block_tile<N>`** represents a subgroup of N threads (e.g., warp-level primitives). **`grid_group`** enables grid-wide synchronization across all blocks—impossible with `__syncthreads()`.

Grid-wide sync requires special kernel launch:
```cpp
cudaLaunchCooperativeKernel(&kernel, gridDim, blockDim, args, sharedMem, stream);
```
Hardware must support cooperative groups (compute capability ≥7.0). Use case: multi-pass algorithms (e.g., histogram where one pass counts, next pass writes) implemented in a single kernel launch (faster than multiple kernel launches with implicit sync).

### Unified Memory and Prefetching

**Unified Memory** (`cudaMallocManaged`) presents a single pointer accessible from host and device. Internally, CUDA uses demand paging: when the GPU accesses a page not in device memory, a page fault occurs, triggering migration. Naive unified memory is **slow** (page faults stall kernels). **`cudaMemPrefetchAsync`** migrates pages to the GPU before the kernel, eliminating faults. **`cudaMemAdvise`** provides hints (read-mostly, preferred location) for the CUDA runtime's heuristic.

Unified memory simplifies development (no explicit `cudaMemcpy`) but requires tuning for performance. For production HFT/HPC, explicit transfers with pinned memory and streams yield better control and predictability.

### Persistent Kernels and GPU Task Queues

**Persistent kernels** loop indefinitely, polling a work queue:
```cuda
__global__ void persistent_kernel(WorkQueue* queue) {
    while (!queue->should_stop()) {
        Task task = queue->pop();
        if (task.valid()) process(task);
    }
}
```
Launched once, they handle thousands of tasks without re-launch overhead. Synchronization via atomic operations or lock-free queues. Use case: HFT pricing engine where tasks arrive asynchronously (new market data, order submission).

Trade-off: persistent kernels consume GPU resources (SMs, registers) even when idle. For bursty workloads (HFT: idle 99% of time, burst at market open), dynamic kernel launches may be better. For continuous workloads (HPC simulation), persistent kernels eliminate overhead.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **CUDA Stream** | Command queue for asynchronous execution. Non-default streams enable concurrent kernel/memcpy overlap. Modern GPUs have 2+ DMA engines. |
| **Pinned Memory** | Page-locked host memory (`cudaMallocHost`) enabling direct DMA. 2-3x faster transfers than pageable memory. Counts against RAM limits. |
| **Zero-Copy** | GPU directly accesses host memory (`cudaHostAllocMapped`). High latency (~400ns) but eliminates explicit memcpy for small/rare-access data. |
| **`cudaMemcpyAsync`** | Asynchronous host-device transfer. Returns immediately; completion checked via `cudaStreamSynchronize` or `cudaEventQuery`. |
| **Cooperative Groups** | Namespace for flexible thread synchronization. `thread_block`, `thread_block_tile<N>`, `grid_group` (requires cooperative launch). |
| **Grid-Wide Sync** | `cooperative_groups::grid_group::sync()` synchronizes all threads across all blocks. Requires `cudaLaunchCooperativeKernel`, compute ≥7.0. |
| **Unified Memory** | Single pointer accessible from host/device (`cudaMallocManaged`). Uses demand paging; prefetch with `cudaMemPrefetchAsync` for performance. |
| **`cudaMemPrefetchAsync`** | Migrate unified memory pages to GPU before kernel, eliminating page faults. Essential for unified memory performance. |
| **Persistent Kernel** | Kernel that loops forever, polling a work queue. Eliminates launch overhead (~5-10 µs) for latency-critical workloads (HFT). |
| **Pipeline Pattern** | Overlap H2D/kernel/D2H across streams. Chunk 1 transfers out while chunk 2 computes and chunk 3 transfers in—maximizes utilization. |

### Ecosystem Context

**Streams vs multi-GPU**: CUDA streams handle concurrency on a single GPU. Multi-GPU programming uses **peer-to-peer (P2P) transfers** (`cudaMemcpyPeer`, NVLink) and **multi-stream multi-GPU** patterns (one stream per GPU, synchronized via events). Production ML training (data parallelism) uses NCCL (NVIDIA Collective Communications Library) for multi-GPU allreduce/broadcast, abstracting streams and P2P.

**Pinned memory limits**: OS limits pinned memory to avoid RAM exhaustion. On Linux, `ulimit -l` sets per-process limit. On Windows, driver allocates ~50% of RAM for pinned memory. For HFT systems with 128 GB RAM, allocate 10-20 GB pinned buffers for market data ingestion—enough for sub-millisecond buffering without hitting limits.

**Persistent kernels vs CPU-GPU hybrid**: Some HFT systems use CPU for serial order matching (low latency, simple logic) and GPU for parallel pricing (Monte Carlo, Greeks). Persistent kernels sit in-between: GPU handles both, using atomic queues for task distribution. Trade-off: CPU-GPU hybrid is simpler (familiar CPU code), but persistent kernels eliminate PCIe round-trips (~1 µs overhead per task).

**Unified Memory in production**: Unified memory simplifies prototyping (no manual memcpy) but adds unpredictability (page faults, migration latency). HPC teams use it for algorithm development, then switch to explicit pinned memory + streams for production. HFT avoids unified memory entirely (latency unpredictability is unacceptable).

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
