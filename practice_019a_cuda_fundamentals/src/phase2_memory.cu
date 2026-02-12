// Phase 2: Memory Model — Vector Addition with Manual & Managed Memory
// =========================================================================
//
// Concepts:
//   - Host memory (CPU RAM) vs Device memory (GPU VRAM) — separate address spaces
//   - cudaMalloc / cudaMemcpy / cudaFree — explicit memory management (like C's malloc
//     but for the GPU). You explicitly control when data moves between CPU and GPU.
//   - cudaMallocManaged — unified memory. The CUDA runtime automatically migrates pages
//     between CPU and GPU on demand. Simpler code, but page faults add latency.
//
// The memory model is THE most important concept in CUDA. Kernel compute is cheap;
// data movement is expensive. A 1024-element vector add is memory-bound, not compute-bound.
//
// HFT relevance: In GPU-accelerated pricing engines, you pre-allocate device buffers
// once and stream market data updates with minimal copies. Understanding the cost of
// each cudaMemcpy is critical for meeting latency targets.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr int N = 1 << 20;  // ~1M elements — big enough to see GPU benefit

// --------------------------------------------------------------------------- //
// Vector addition kernel
// --------------------------------------------------------------------------- //
//
// ── Exercise Context ──────────────────────────────────────────────────
// This exercise teaches explicit memory management (cudaMalloc/cudaMemcpy/cudaFree).
// Understanding the host-device memory split is crucial—GPUs don't share CPU RAM.
// This pattern (allocate → transfer → compute → transfer back → free) is foundational.

// TODO(human): Implement the vector addition kernel.
//   - Compute global thread index
//   - Guard against out-of-bounds access (idx < n)
//   - Write c[idx] = a[idx] + b[idx]
//
// This is the "hello world" of GPU computing — trivially parallel,
// each thread reads two values and writes one.
//
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO(human): Compute idx, guard, write c[idx] = a[idx] + b[idx]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // placeholder
    if (idx < n) {
        c[idx] = 0.0f;  // placeholder — should be a[idx] + b[idx]
    }
}

// --------------------------------------------------------------------------- //
// CPU reference
// --------------------------------------------------------------------------- //
void cpu_vector_add(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// --------------------------------------------------------------------------- //
// Initialize host arrays with test data
// --------------------------------------------------------------------------- //
void fill_test_data(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.3f;
    }
}

// --------------------------------------------------------------------------- //
// Verify GPU result against CPU reference
// --------------------------------------------------------------------------- //
bool verify(const float* gpu, const float* cpu, int n, const char* label) {
    constexpr float EPS = 1e-5f;
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > EPS) {
            printf("[%s] MISMATCH at index %d: GPU=%.6f, CPU=%.6f\n",
                   label, i, gpu[i], cpu[i]);
            return false;
        }
    }
    printf("[%s] PASSED: All %d elements match.\n", label, n);
    return true;
}

// --------------------------------------------------------------------------- //
// Approach 1: Explicit memory management
// --------------------------------------------------------------------------- //
//
// This is the manual approach. You control every byte that moves between
// host and device. More code, but gives you full control over timing.
//
// Memory flow:
//   1. Allocate host arrays (CPU)
//   2. Allocate device arrays (GPU) with cudaMalloc
//   3. Copy input data host -> device with cudaMemcpy
//   4. Launch kernel (GPU computes)
//   5. Copy results device -> host with cudaMemcpy
//   6. Free device memory with cudaFree
//
void run_explicit_memory() {
    printf("\n--- Approach 1: Explicit Memory (cudaMalloc + cudaMemcpy) ---\n");

    // Host allocation
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    float* h_ref = new float[N];
    fill_test_data(h_a, h_b, N);

    // Device allocation
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Host -> Device copy
    CudaTimer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    timer.stop();
    printf("  H2D copy time: %.3f ms\n", timer.elapsed_ms());

    // Launch kernel
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    timer.start();
    vector_add<<<grid, BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();
    timer.stop();
    printf("  Kernel time:   %.3f ms\n", timer.elapsed_ms());

    // Device -> Host copy
    timer.start();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    timer.stop();
    printf("  D2H copy time: %.3f ms\n", timer.elapsed_ms());

    // Verify
    cpu_vector_add(h_a, h_b, h_ref, N);
    verify(h_c, h_ref, N, "Explicit");

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;
}

// --------------------------------------------------------------------------- //
// Approach 2: Unified Memory (cudaMallocManaged)
// --------------------------------------------------------------------------- //
//
// Unified memory simplifies code: no explicit copies needed. The CUDA runtime
// migrates pages on demand. Trade-off: page faults add latency, especially on
// first access. Great for prototyping; explicit memory is better for production
// latency-sensitive code.
//
void run_managed_memory() {
    printf("\n--- Approach 2: Unified Memory (cudaMallocManaged) ---\n");

    // Single allocation visible from both CPU and GPU
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, N * sizeof(float)));

    // Initialize on host — no explicit copy needed
    fill_test_data(a, b, N);

    // Launch kernel — runtime handles migration automatically
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    CudaTimer timer;
    timer.start();
    vector_add<<<grid, BLOCK>>>(a, b, c, N);
    CUDA_CHECK_LAST();
    timer.stop();
    printf("  Kernel time (includes page migration): %.3f ms\n", timer.elapsed_ms());

    // Synchronize before reading on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify — we can read c directly on the host (unified address space)
    float* h_ref = new float[N];
    cpu_vector_add(a, b, h_ref, N);
    verify(c, h_ref, N, "Managed");

    // Cleanup
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    delete[] h_ref;
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 2: Memory Model — Vector Addition ===\n");
    printf("N = %d elements (%.1f MB per array)\n", N, N * sizeof(float) / (1024.0f * 1024.0f));

    run_explicit_memory();
    run_managed_memory();

    printf("\nKey takeaway: Explicit memory gives you control over when copies happen.\n");
    printf("Unified memory is simpler but hides latency in page faults.\n");
    printf("For HFT: always use explicit memory — you need predictable latency.\n");

    return 0;
}
