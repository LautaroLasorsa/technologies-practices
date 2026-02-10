// Phase 5: Parallel Reduction & Warp Primitives
// =========================================================================
//
// Concepts:
//   - Parallel reduction: sum N numbers in O(log N) parallel steps
//     instead of O(N) sequential. The fundamental building block for
//     aggregation on GPUs.
//   - Tree reduction in shared memory: each thread loads one element,
//     then threads pair up and add, halving the active set each step.
//   - Warp-level primitives: __shfl_down_sync() lets threads within a
//     warp exchange values directly through registers (no shared memory
//     needed). Fastest possible communication between threads.
//   - Full warp mask: 0xFFFFFFFF (all 32 threads participate)
//
// CP analogy: This is the GPU equivalent of a segment tree — hierarchical
// aggregation where each level combines pairs of results from the level below.
// But instead of building a tree data structure, the threads ARE the tree.
//
// HFT relevance:
//   - Portfolio risk aggregation: sum VaR contributions across 10K positions
//   - Real-time P&L: aggregate mark-to-market across all positions every tick
//   - Order book: sum volume at each price level
//   - All reduce to "combine N values fast" — this kernel.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr int N = 1 << 22;        // ~4M elements
constexpr int BLOCK_SIZE = 256;   // threads per block (must be power of 2)

// --------------------------------------------------------------------------- //
// Reduction kernel using shared memory (tree reduction)
// --------------------------------------------------------------------------- //
//
// Algorithm:
//   1. Each thread loads one element from global memory into shared memory
//   2. Tree reduction: for stride = blockDim.x/2; stride > 0; stride /= 2:
//        if (threadIdx.x < stride)
//            sdata[threadIdx.x] += sdata[threadIdx.x + stride]
//        __syncthreads()
//   3. Thread 0 of each block writes the block's partial sum to output
//
// After this kernel, you have gridDim.x partial sums. Run the kernel again
// on those partial sums (or finish on CPU if few enough).
//
// TODO(human): Implement the tree reduction.
//   - Load input[global_idx] into shared memory (0 if out of bounds)
//   - Implement the tree reduction loop with __syncthreads()
//   - Thread 0 writes the block result to output[blockIdx.x]
//
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];  // dynamically sized shared memory

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO(human): Load from global memory into shared memory
    //   sdata[tid] = (global_idx < n) ? input[global_idx] : 0.0f;
    sdata[tid] = 0.0f;  // placeholder — load from input

    __syncthreads();

    // TODO(human): Tree reduction loop
    //   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //       if (tid < stride) {
    //           sdata[tid] += sdata[tid + stride];
    //       }
    //       __syncthreads();
    //   }
    // This halves the active threads each iteration: 128, 64, 32, 16, 8, 4, 2, 1

    // TODO(human): Thread 0 writes the block's result
    //   if (tid == 0) output[blockIdx.x] = sdata[0];
}

// --------------------------------------------------------------------------- //
// Optimized reduction with warp shuffle (advanced)
// --------------------------------------------------------------------------- //
//
// Optimization: once we're down to 32 threads (one warp), we don't need
// shared memory or __syncthreads(). Threads in a warp execute in lockstep.
// __shfl_down_sync(mask, val, offset) shifts values between warp lanes:
//   lane i receives the value from lane i+offset
//
// This eliminates shared memory bank conflicts and sync overhead for the
// last 5 reduction steps (32 -> 16 -> 8 -> 4 -> 2 -> 1).
//
// TODO(human): Implement warp-level reduction using __shfl_down_sync.
//   The function takes a value and returns the reduced sum across the warp.
//
__device__ float warp_reduce_sum(float val) {
    // TODO(human): Use __shfl_down_sync to reduce within a warp
    //   for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    //       val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    //   }
    //   return val;
    //
    // How __shfl_down_sync works:
    //   0xFFFFFFFF = mask meaning all 32 lanes participate
    //   __shfl_down_sync(mask, val, 16): lane 0 gets lane 16's value,
    //     lane 1 gets lane 17's, ..., lane 15 gets lane 31's
    //   After adding: lanes 0-15 hold pairwise sums
    //   Repeat with offset 8, 4, 2, 1: lane 0 holds the total
    return val;  // placeholder — should reduce across the warp
}

__global__ void reduce_sum_warp(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (global_idx < n) ? input[global_idx] : 0.0f;
    __syncthreads();

    // Tree reduction down to warp size using shared memory
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction using shuffle (no sync needed)
    float val = (tid < 32) ? sdata[tid] : 0.0f;
    val = warp_reduce_sum(val);

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// --------------------------------------------------------------------------- //
// CPU reference
// --------------------------------------------------------------------------- //
double cpu_sum(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += data[i];
    return sum;
}

// --------------------------------------------------------------------------- //
// Run a reduction kernel and return the final sum
// --------------------------------------------------------------------------- //
float run_reduction(const char* label,
                    void (*kernel)(const float*, float*, int),
                    float* d_input, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int shared_bytes = BLOCK_SIZE * sizeof(float);

    // First pass: N elements -> grid partial sums
    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));

    CudaTimer timer;

    // Warm-up
    kernel<<<grid, BLOCK_SIZE, shared_bytes>>>(d_input, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    timer.start();
    kernel<<<grid, BLOCK_SIZE, shared_bytes>>>(d_input, d_partial, n);
    CUDA_CHECK_LAST();
    timer.stop();
    float kernel_ms = timer.elapsed_ms();

    // Copy partial sums to host and finish on CPU
    // (In production you'd do a second GPU pass, but for clarity we finish here)
    float* h_partial = new float[grid];
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * sizeof(float),
                           cudaMemcpyDeviceToHost));

    double result = 0.0;
    for (int i = 0; i < grid; ++i) result += h_partial[i];

    printf("[%s] GPU kernel: %.3f ms, result: %.2f\n", label, kernel_ms, result);

    CUDA_CHECK(cudaFree(d_partial));
    delete[] h_partial;
    return static_cast<float>(result);
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 5: Parallel Reduction ===\n");
    printf("N = %d elements (%.1f MB)\n\n", N, N * sizeof(float) / (1024.0f * 1024.0f));

    // Initialize with values that sum to a known result
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // sum should be exactly N
    }

    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU reference
    double cpu_result = cpu_sum(h_input, N);
    printf("CPU sum: %.2f (expected: %d)\n\n", cpu_result, N);

    // Run reductions - using function pointers via a lambda workaround
    // (kernel function pointers need matching signatures)
    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int shared_bytes = BLOCK_SIZE * sizeof(float);
        float* d_partial = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));

        CudaTimer timer;

        // --- Shared memory reduction ---
        reduce_sum<<<grid, BLOCK_SIZE, shared_bytes>>>(d_input, d_partial, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        reduce_sum<<<grid, BLOCK_SIZE, shared_bytes>>>(d_input, d_partial, N);
        CUDA_CHECK_LAST();
        timer.stop();
        float t1 = timer.elapsed_ms();

        float* h_partial = new float[grid];
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * sizeof(float),
                               cudaMemcpyDeviceToHost));
        double sum1 = 0;
        for (int i = 0; i < grid; ++i) sum1 += h_partial[i];
        printf("[SharedMem] GPU: %.3f ms, result: %.2f, error: %.2e\n",
               t1, sum1, fabs(sum1 - cpu_result));

        // --- Warp shuffle reduction ---
        reduce_sum_warp<<<grid, BLOCK_SIZE, shared_bytes>>>(d_input, d_partial, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        reduce_sum_warp<<<grid, BLOCK_SIZE, shared_bytes>>>(d_input, d_partial, N);
        CUDA_CHECK_LAST();
        timer.stop();
        float t2 = timer.elapsed_ms();

        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * sizeof(float),
                               cudaMemcpyDeviceToHost));
        double sum2 = 0;
        for (int i = 0; i < grid; ++i) sum2 += h_partial[i];
        printf("[WarpShfl]  GPU: %.3f ms, result: %.2f, error: %.2e\n",
               t2, sum2, fabs(sum2 - cpu_result));

        if (t1 > 0 && t2 > 0) {
            printf("\nWarp shuffle speedup: %.2fx over shared-only\n", t1 / t2);
        }

        // Verify correctness
        float tol = N * 1e-6f;
        bool pass1 = fabs(sum1 - cpu_result) < tol;
        bool pass2 = fabs(sum2 - cpu_result) < tol;
        printf("\n[SharedMem] %s\n", pass1 ? "PASSED" : "FAILED — implement TODO(human)!");
        printf("[WarpShfl]  %s\n", pass2 ? "PASSED" : "FAILED — implement TODO(human)!");

        CUDA_CHECK(cudaFree(d_partial));
        delete[] h_partial;
    }

    CUDA_CHECK(cudaFree(d_input));
    delete[] h_input;
    return 0;
}
