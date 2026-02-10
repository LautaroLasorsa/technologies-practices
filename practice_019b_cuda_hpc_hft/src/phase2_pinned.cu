// Phase 2: Pinned (Page-Locked) Memory
// =========================================================================
//
// Concepts:
//   - Pageable memory (normal malloc/new): The OS can swap pages to disk.
//     When CUDA transfers data, the driver must first copy it to an
//     internal pinned staging buffer, then DMA to GPU. Two copies.
//
//   - Pinned memory (cudaMallocHost / cudaHostAlloc): Pages are locked
//     in physical RAM — the OS can never swap them. The GPU's DMA engine
//     can transfer directly. One copy. Typically 2-3x faster for large
//     transfers.
//
//   - Trade-offs: Pinned memory reduces available physical RAM for the OS.
//     Over-allocating pinned memory can slow down the entire system.
//     Use it for transfer buffers, not for all allocations.
//
//   - cudaHostAlloc flags:
//       cudaHostAllocDefault    — same as cudaMallocHost
//       cudaHostAllocPortable   — usable from any CUDA context/device
//       cudaHostAllocMapped     — mapped into device address space (zero-copy)
//       cudaHostAllocWriteCombined — write-combining for write-only buffers
//
//   - Zero-copy (mapped pinned memory): The GPU reads/writes host memory
//     directly over PCIe, with no explicit transfer. Useful for small or
//     infrequent access, but PCIe bandwidth is much lower than device
//     memory bandwidth.
//
// Ref: CUDA Programming Guide §3.2.4 "Page-Locked Host Memory"
//
// HFT context: Market data arrives at the NIC and is DMA'd into a pinned
// buffer. The GPU can start processing immediately without waiting for a
// separate memcpy. Zero-copy lets the GPU read the latest tick directly
// from the NIC buffer for ultra-low-latency single-read patterns.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr size_t TRANSFER_SIZE = 256 * 1024 * 1024;  // 256 MB
constexpr int NUM_TRIALS = 5;                          // average over trials
constexpr int BLOCK_SIZE = 256;

// --------------------------------------------------------------------------- //
// Simple kernel for zero-copy test: reads from mapped host pointer
// --------------------------------------------------------------------------- //
__global__ void scale_kernel(const float* input, float* output, int n, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * s;
    }
}

// --------------------------------------------------------------------------- //
// Benchmark: transfer speed for a given host pointer (pageable or pinned)
// --------------------------------------------------------------------------- //
struct TransferResult {
    float h2d_ms;
    float d2h_ms;
};

TransferResult benchmark_transfer(const char* label, void* h_ptr, void* d_ptr,
                                  size_t bytes) {
    CudaTimer timer;
    float h2d_total = 0.0f, d2h_total = 0.0f;

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));

    for (int t = 0; t < NUM_TRIALS; ++t) {
        // H2D
        timer.start();
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
        timer.stop();
        h2d_total += timer.elapsed_ms();

        // D2H
        timer.start();
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));
        timer.stop();
        d2h_total += timer.elapsed_ms();
    }

    float h2d_avg = h2d_total / NUM_TRIALS;
    float d2h_avg = d2h_total / NUM_TRIALS;

    printf("\n--- %s ---\n", label);
    BandwidthMeter::report("H2D", bytes, h2d_avg);
    BandwidthMeter::report("D2H", bytes, d2h_avg);

    return {h2d_avg, d2h_avg};
}

// --------------------------------------------------------------------------- //
// TODO(human): Implement the pinned memory benchmark.
//
// Steps:
//   1. Allocate pinned host memory with cudaMallocHost (size = TRANSFER_SIZE)
//   2. Initialize it (memset or fill with a pattern)
//   3. Call benchmark_transfer("Pinned", pinned_ptr, d_ptr, TRANSFER_SIZE)
//   4. Free with cudaFreeHost
//
// Then implement zero-copy:
//   5. Allocate with cudaHostAlloc(..., cudaHostAllocMapped)
//   6. Get device pointer with cudaHostGetDevicePointer(&d_mapped, h_mapped, 0)
//   7. Launch scale_kernel using d_mapped as input (reads host mem over PCIe)
//   8. Time it and compare with explicit transfer + kernel
//   9. Free with cudaFreeHost
//
// Expected results:
//   - Pinned H2D/D2H should be ~2-3x faster than pageable
//   - Zero-copy kernel will be SLOW for bulk data (PCIe bandwidth << VRAM)
//   - Zero-copy is only useful for small/infrequent access patterns
// --------------------------------------------------------------------------- //
void run_pinned_benchmark(void* d_ptr) {
    printf("\n===== Pinned Memory Benchmark =====\n");

    // TODO(human): Allocate pinned memory, benchmark, free
    // float* h_pinned = nullptr;
    // CUDA_CHECK(cudaMallocHost(&h_pinned, TRANSFER_SIZE));
    // memset(h_pinned, 0x42, TRANSFER_SIZE);
    // auto pinned = benchmark_transfer("Pinned", h_pinned, d_ptr, TRANSFER_SIZE);
    // CUDA_CHECK(cudaFreeHost(h_pinned));

    printf("  (Not yet implemented — fill in TODO(human))\n");
}

void run_zero_copy_benchmark() {
    printf("\n===== Zero-Copy (Mapped Pinned) Memory =====\n");

    int n = 4 * 1024 * 1024;  // 4M floats (~16 MB, small for zero-copy demo)
    size_t bytes = n * sizeof(float);
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // TODO(human): Allocate mapped pinned memory and benchmark zero-copy
    //
    // float* h_mapped = nullptr;
    // CUDA_CHECK(cudaHostAlloc(&h_mapped, bytes, cudaHostAllocMapped));
    //
    // // Initialize
    // for (int i = 0; i < n; ++i) h_mapped[i] = static_cast<float>(i);
    //
    // // Get device-accessible pointer for the mapped host memory
    // float* d_mapped = nullptr;
    // CUDA_CHECK(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));
    //
    // // Output buffer on device
    // float* d_out = nullptr;
    // CUDA_CHECK(cudaMalloc(&d_out, bytes));
    //
    // // Warm-up
    // scale_kernel<<<grid, BLOCK_SIZE>>>(d_mapped, d_out, n, 2.0f);
    // CUDA_CHECK(cudaDeviceSynchronize());
    //
    // // Time zero-copy kernel (GPU reads from host memory over PCIe)
    // CudaTimer timer;
    // timer.start();
    // scale_kernel<<<grid, BLOCK_SIZE>>>(d_mapped, d_out, n, 2.0f);
    // CUDA_CHECK_LAST();
    // timer.stop();
    // float zc_ms = timer.elapsed_ms();
    // printf("  [Zero-copy kernel] %.3f ms\n", zc_ms);
    // BandwidthMeter::report("Zero-copy read", bytes, zc_ms);
    //
    // // Compare: explicit transfer + kernel
    // float* d_in = nullptr;
    // CUDA_CHECK(cudaMalloc(&d_in, bytes));
    //
    // timer.start();
    // CUDA_CHECK(cudaMemcpy(d_in, h_mapped, bytes, cudaMemcpyHostToDevice));
    // scale_kernel<<<grid, BLOCK_SIZE>>>(d_in, d_out, n, 2.0f);
    // CUDA_CHECK_LAST();
    // timer.stop();
    // float explicit_ms = timer.elapsed_ms();
    // printf("  [Explicit transfer + kernel] %.3f ms\n", explicit_ms);
    //
    // BandwidthMeter::compare("Explicit vs Zero-copy", explicit_ms, zc_ms);
    // printf("  (Zero-copy is faster only when data is read once and is small)\n");
    //
    // CUDA_CHECK(cudaFree(d_in));
    // CUDA_CHECK(cudaFree(d_out));
    // CUDA_CHECK(cudaFreeHost(h_mapped));

    printf("  (Not yet implemented — fill in TODO(human))\n");
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 2: Pinned (Page-Locked) Memory ===\n");
    printf("Transfer size: %.0f MB, Trials: %d\n",
           TRANSFER_SIZE / (1024.0 * 1024.0), NUM_TRIALS);

    // Query device
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("  canMapHostMemory: %s\n\n", prop.canMapHostMemory ? "YES" : "NO");

    // Device memory (shared across benchmarks)
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, TRANSFER_SIZE));

    // === Pageable baseline ===
    printf("===== Pageable Memory Baseline =====\n");
    float* h_pageable = static_cast<float*>(malloc(TRANSFER_SIZE));
    memset(h_pageable, 0x42, TRANSFER_SIZE);
    auto pageable = benchmark_transfer("Pageable", h_pageable, d_ptr, TRANSFER_SIZE);
    free(h_pageable);

    // === Pinned memory (TODO(human)) ===
    run_pinned_benchmark(d_ptr);

    // === Zero-copy (TODO(human)) ===
    run_zero_copy_benchmark();

    // === Summary ===
    printf("\n=== Expected Results ===\n");
    printf("Pageable H2D: typical ~6-12 GB/s (limited by staging copy)\n");
    printf("Pinned   H2D: typical ~12-24 GB/s (direct DMA, PCIe limited)\n");
    printf("Zero-copy:    useful only for small, infrequent access\n");
    printf("\nPCIe Gen3 x16 theoretical max: ~16 GB/s\n");
    printf("PCIe Gen4 x16 theoretical max: ~32 GB/s\n");

    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
