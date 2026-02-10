// Phase 6: Error Handling & Profiling
// =========================================================================
//
// Concepts:
//   - cudaError_t: every CUDA runtime call returns an error code
//   - cudaGetLastError(): retrieves error from the most recent kernel launch
//     (kernel launches are async, so errors surface here or at next sync)
//   - CUDA events: GPU-side timestamps for measuring kernel execution time
//     without CPU-side timing noise
//   - Nsight Compute: NVIDIA's kernel profiler (ncu) — shows memory
//     throughput, occupancy, warp stalls, instruction mix
//   - Nsight Systems: system-wide profiler (nsys) — shows timeline of
//     CPU/GPU activity, memory transfers, kernel overlaps
//
// Production pattern: Always wrap CUDA calls in error checks (CUDA_CHECK
// macro from cuda_helpers.h). Silent errors are the #1 source of CUDA bugs.
//
// HFT relevance: Profiling is how you find the microseconds. A latency-
// sensitive pricing engine needs profiling at the kernel level to identify
// which memory accesses or sync points are the bottleneck.
//
// Nsight Compute usage (from command line):
//   ncu --set full build\Release\phase6_profiling.exe
//   ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum build\Release\phase6_profiling.exe
//
// Nsight Systems usage:
//   nsys profile build\Release\phase6_profiling.exe
//   (generates a .nsys-rep file, open in Nsight Systems GUI)
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr int N = 1 << 20;       // ~1M elements
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_RUNS = 100;     // Average over many runs for stable timing

// --------------------------------------------------------------------------- //
// Sample kernels to profile
// --------------------------------------------------------------------------- //

// Kernel A: Simple vector scale (memory-bound)
__global__ void vector_scale(const float* input, float* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

// Kernel B: Heavier compute (fused multiply-add chain, compute-bound-ish)
__global__ void compute_heavy(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Simulate compute-heavy work (polynomial evaluation)
        float y = x;
        for (int i = 0; i < 50; ++i) {
            y = y * x + 1.0f;  // fma instruction
        }
        output[idx] = y;
    }
}

// Kernel C: Deliberately bad memory access pattern (non-coalesced)
__global__ void strided_access(const float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_idx = (idx * stride) % n;  // strided read — bad for coalescing
    if (idx < n) {
        output[idx] = input[src_idx];
    }
}

// --------------------------------------------------------------------------- //
// Error handling demonstration
// --------------------------------------------------------------------------- //
void demonstrate_error_handling() {
    printf("--- Error Handling Patterns ---\n\n");

    // Pattern 1: Check allocation errors
    float* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, size_t(1) << 40);  // 1 TB — will fail
    if (err != cudaSuccess) {
        printf("Expected allocation failure: %s\n", cudaGetErrorString(err));
        // Reset error state so subsequent calls work
        cudaGetLastError();
    }

    // Pattern 2: Check kernel launch configuration errors
    // Launching with too many threads per block
    float* d_dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dummy, 100 * sizeof(float)));
    vector_scale<<<1, 2048>>>(d_dummy, d_dummy, 1.0f, 100);  // 2048 > max threads
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Expected launch failure: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaFree(d_dummy));

    printf("\nKey takeaway: Always check errors. Silent failures are the #1 CUDA bug source.\n");
    printf("The CUDA_CHECK macro in cuda_helpers.h does this automatically.\n\n");
}

// --------------------------------------------------------------------------- //
// Profiling with CUDA events
// --------------------------------------------------------------------------- //
//
// TODO(human): Use CUDA events to measure kernel execution time.
//
// The CudaTimer class in cuda_helpers.h wraps CUDA events, but here you'll
// use raw events to understand what's happening underneath.
//
// Steps:
//   1. Create two events: start and stop
//   2. Record start event before kernel launch
//   3. Launch kernel
//   4. Record stop event after kernel launch
//   5. Synchronize on the stop event
//   6. Compute elapsed time with cudaEventElapsedTime
//   7. Destroy events
//
// Why events instead of CPU timers?
//   CPU timers (chrono, QueryPerformanceCounter) measure wall time including
//   CPU overhead, OS scheduling, etc. CUDA events are recorded ON the GPU
//   timeline, so they measure only GPU execution time.
//
void profile_kernels() {
    printf("--- Kernel Profiling with CUDA Events ---\n\n");

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    // Initialize input
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = static_cast<float>(i) / N;
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // TODO(human): Create CUDA events and measure each kernel.
    //
    // For each kernel, do:
    //   cudaEvent_t start, stop;
    //   cudaEventCreate(&start);
    //   cudaEventCreate(&stop);
    //
    //   // Warm-up run
    //   kernel<<<grid, BLOCK_SIZE>>>(args...);
    //   cudaDeviceSynchronize();
    //
    //   cudaEventRecord(start);
    //   for (int run = 0; run < NUM_RUNS; ++run) {
    //       kernel<<<grid, BLOCK_SIZE>>>(args...);
    //   }
    //   cudaEventRecord(stop);
    //   cudaEventSynchronize(stop);
    //
    //   float ms = 0.0f;
    //   cudaEventElapsedTime(&ms, start, stop);
    //   printf("Average: %.4f ms\n", ms / NUM_RUNS);
    //
    //   cudaEventDestroy(start);
    //   cudaEventDestroy(stop);
    //
    // For now, we use CudaTimer as a fallback:

    CudaTimer timer;

    // --- Kernel A: vector_scale (memory-bound) ---
    vector_scale<<<grid, BLOCK_SIZE>>>(d_input, d_output, 2.0f, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    for (int r = 0; r < NUM_RUNS; ++r) {
        vector_scale<<<grid, BLOCK_SIZE>>>(d_input, d_output, 2.0f, N);
    }
    CUDA_CHECK_LAST();
    timer.stop();
    float t_scale = timer.elapsed_ms() / NUM_RUNS;

    // --- Kernel B: compute_heavy ---
    compute_heavy<<<grid, BLOCK_SIZE>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    for (int r = 0; r < NUM_RUNS; ++r) {
        compute_heavy<<<grid, BLOCK_SIZE>>>(d_input, d_output, N);
    }
    CUDA_CHECK_LAST();
    timer.stop();
    float t_heavy = timer.elapsed_ms() / NUM_RUNS;

    // --- Kernel C: strided_access (non-coalesced) ---
    strided_access<<<grid, BLOCK_SIZE>>>(d_input, d_output, N, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Coalesced (stride=1)
    timer.start();
    for (int r = 0; r < NUM_RUNS; ++r) {
        strided_access<<<grid, BLOCK_SIZE>>>(d_input, d_output, N, 1);
    }
    CUDA_CHECK_LAST();
    timer.stop();
    float t_coal = timer.elapsed_ms() / NUM_RUNS;

    // Non-coalesced (stride=32)
    timer.start();
    for (int r = 0; r < NUM_RUNS; ++r) {
        strided_access<<<grid, BLOCK_SIZE>>>(d_input, d_output, N, 32);
    }
    CUDA_CHECK_LAST();
    timer.stop();
    float t_noncoal = timer.elapsed_ms() / NUM_RUNS;

    // --- Results ---
    printf("Kernel timing (avg of %d runs, N=%d):\n\n", NUM_RUNS, N);
    printf("  %-25s %8.4f ms\n", "vector_scale:", t_scale);
    printf("  %-25s %8.4f ms\n", "compute_heavy:", t_heavy);
    printf("  %-25s %8.4f ms\n", "strided (coalesced):", t_coal);
    printf("  %-25s %8.4f ms\n", "strided (non-coalesced):", t_noncoal);

    printf("\nAnalysis:\n");
    printf("  compute_heavy / vector_scale = %.1fx (shows compute vs memory bound)\n",
           t_heavy / t_scale);
    printf("  non-coalesced / coalesced     = %.1fx (shows coalescing impact)\n",
           t_noncoal / t_coal);

    // Bandwidth calculation for vector_scale
    // Reads N floats + writes N floats = 2*N*4 bytes
    double bytes = 2.0 * N * sizeof(float);
    double bw_gbps = bytes / (t_scale * 1e-3) / 1e9;
    printf("\n  vector_scale effective bandwidth: %.1f GB/s\n", bw_gbps);

    // Get theoretical bandwidth for comparison
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    printf("  Theoretical peak bandwidth:      %.1f GB/s\n", peak_bw);
    printf("  Utilization: %.1f%%\n", 100.0 * bw_gbps / peak_bw);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 6: Error Handling & Profiling ===\n\n");

    demonstrate_error_handling();
    profile_kernels();

    printf("\n--- Next Steps ---\n");
    printf("1. Replace CudaTimer usage with raw CUDA events (TODO above)\n");
    printf("2. Run with Nsight Compute for detailed kernel analysis:\n");
    printf("   ncu --set full build\\Release\\phase6_profiling.exe\n");
    printf("3. Run with Nsight Systems for system-wide timeline:\n");
    printf("   nsys profile build\\Release\\phase6_profiling.exe\n");

    return 0;
}
