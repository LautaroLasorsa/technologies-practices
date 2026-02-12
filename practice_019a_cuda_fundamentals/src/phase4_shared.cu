// Phase 4: Shared Memory & Tiling — Optimized Matrix Multiplication
// =========================================================================
//
// Concepts:
//   - __shared__ memory: fast on-chip SRAM (~100x faster than global memory)
//     Think of it as a user-managed L1 cache. Each block gets its own copy.
//   - Tiling: load a TILE_SIZE x TILE_SIZE sub-matrix into shared memory,
//     compute partial results, then load the next tile. This reuses data
//     across threads in the same block, reducing global memory reads from
//     O(K) per thread to O(K / TILE_SIZE) per thread.
//   - __syncthreads(): barrier within a block — all threads must reach this
//     point before any can proceed. Essential when writing then reading
//     shared memory.
//   - Memory coalescing: when adjacent threads access adjacent memory
//     addresses, the hardware combines accesses into fewer transactions.
//     Coalesced reads are 10-20x faster than scattered reads.
//   - Bank conflicts: shared memory is divided into 32 banks. If multiple
//     threads in a warp access the same bank (but different addresses),
//     accesses are serialized. Our tiling pattern avoids this naturally.
//
// The tiling optimization typically gives 5-10x speedup over naive matmul.
// This is THE fundamental GPU optimization technique — almost every CUDA
// application uses some form of tiling.
//
// HFT relevance: Shared memory is how you build fast GPU order books —
// load a tile of orders into shared memory, process them in parallel,
// write results back. Option pricing grids use the same tiling pattern.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr int M = 1024;
constexpr int K = 1024;
constexpr int N = 1024;
constexpr int TILE_SIZE = 16;  // Must equal blockDim.x and blockDim.y

// --------------------------------------------------------------------------- //
// Naive matmul (from Phase 3, for comparison)
// --------------------------------------------------------------------------- //
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// --------------------------------------------------------------------------- //
// Tiled matmul with shared memory
// --------------------------------------------------------------------------- //
//
// Algorithm:
//   For each tile t = 0, 1, ..., K/TILE_SIZE - 1:
//     1. Each thread loads one element of A's tile into shared memory:
//          As[threadIdx.y][threadIdx.x] = A[row][t * TILE_SIZE + threadIdx.x]
//     2. Each thread loads one element of B's tile into shared memory:
//          Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y)][col]
//     3. __syncthreads() — wait for all threads to finish loading
//     4. Each thread computes partial dot product from the shared tile:
//          for i in 0..TILE_SIZE: sum += As[threadIdx.y][i] * Bs[i][threadIdx.x]
//     5. __syncthreads() — wait before loading next tile (so we don't overwrite
//        shared memory while others are still reading)
//
//   After all tiles: C[row][col] = sum
//
// Why this is faster:
//   Naive: each thread reads K elements from A and K from B (global memory)
//   Tiled: each thread reads K/TILE elements from A and K/TILE from B (global)
//          plus TILE reads from shared memory per tile (fast)
//   Shared memory reads are ~100x faster, so total time drops significantly.
//
// ── Exercise Context ──────────────────────────────────────────────────
// This exercise teaches shared memory tiling—the core GPU optimization pattern.
// Shared memory (~TB/s) is 10x faster than global memory (~GB/s). Load tiles once, reuse many times.
// This pattern appears in every high-performance GPU algorithm (convolution, FFT, sort).

// TODO(human): Implement the tiled matmul kernel.
//   The scaffolding provides shared memory declarations and the tile loop.
//   You need to fill in:
//     a) Loading A and B tiles into shared memory (with boundary checks)
//     b) The partial dot product accumulation from shared memory
//     c) Writing the final result to C
//
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int m, int k, int n) {
    // Shared memory for one tile of A and one tile of B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles needed to cover the K dimension
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // TODO(human): Load one element from A into As[threadIdx.y][threadIdx.x]
        //   The element is A[row][t * TILE_SIZE + threadIdx.x]
        //   In linear indexing: A[row * k + t * TILE_SIZE + threadIdx.x]
        //   Boundary check: if row < m && (t * TILE_SIZE + threadIdx.x) < k
        //   Otherwise load 0.0f
        As[threadIdx.y][threadIdx.x] = 0.0f;  // placeholder — load from A

        // TODO(human): Load one element from B into Bs[threadIdx.y][threadIdx.x]
        //   The element is B[t * TILE_SIZE + threadIdx.y][col]
        //   In linear indexing: B[(t * TILE_SIZE + threadIdx.y) * n + col]
        //   Boundary check: if (t * TILE_SIZE + threadIdx.y) < k && col < n
        //   Otherwise load 0.0f
        Bs[threadIdx.y][threadIdx.x] = 0.0f;  // placeholder — load from B

        // Barrier: all threads must finish loading before we can read
        __syncthreads();

        // TODO(human): Accumulate partial dot product from shared memory tile
        //   for (int i = 0; i < TILE_SIZE; ++i)
        //       sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        // (no boundary check needed here — out-of-range values are 0.0f)

        // Barrier: wait before loading next tile (don't overwrite while reading)
        __syncthreads();
    }

    // TODO(human): Write result to C
    //   if (row < m && col < n)
    //       C[row * n + col] = sum;
}

// --------------------------------------------------------------------------- //
// CPU reference
// --------------------------------------------------------------------------- //
void cpu_matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int p = 0; p < k; ++p)
                s += A[i * k + p] * B[p * n + j];
            C[i * n + j] = s;
        }
}

void fill_random(float* mat, int sz) {
    for (int i = 0; i < sz; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}

bool verify(const float* gpu, const float* cpu, int sz, const char* label) {
    float eps = K * 1e-5f;
    int bad = 0;
    for (int i = 0; i < sz; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > eps) {
            if (bad < 3) printf("  [%s] MISMATCH at %d: GPU=%.4f CPU=%.4f\n",
                                label, i, gpu[i], cpu[i]);
            ++bad;
        }
    }
    if (bad) { printf("  [%s] %d mismatches\n", label, bad); return false; }
    printf("  [%s] PASSED\n", label);
    return true;
}

// --------------------------------------------------------------------------- //
// Main: run both naive and tiled, compare performance
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 4: Shared Memory Tiled MatMul ===\n");
    printf("C[%d x %d] = A[%d x %d] * B[%d x %d], TILE_SIZE=%d\n\n",
           M, N, M, K, K, N, TILE_SIZE);

    size_t sa = M * K * sizeof(float);
    size_t sb = K * N * sizeof(float);
    size_t sc = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_naive = new float[M * N];
    float* h_C_tiled = new float[M * N];
    float* h_C_cpu   = new float[M * N];

    srand(42);
    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sa));
    CUDA_CHECK(cudaMalloc(&d_B, sb));
    CUDA_CHECK(cudaMalloc(&d_C, sc));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sa, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sb, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    CudaTimer timer;

    // --- Naive ---
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK_LAST();
    timer.stop();
    float t_naive = timer.elapsed_ms();
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, sc, cudaMemcpyDeviceToHost));
    printf("Naive matmul:  %.3f ms\n", t_naive);

    // --- Tiled ---
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK_LAST();
    timer.stop();
    float t_tiled = timer.elapsed_ms();
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, sc, cudaMemcpyDeviceToHost));
    printf("Tiled matmul:  %.3f ms\n", t_tiled);

    printf("Speedup:       %.1fx\n\n", t_naive / t_tiled);

    // --- CPU reference ---
    printf("Computing CPU reference...\n");
    cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);

    verify(h_C_naive, h_C_cpu, M * N, "Naive");
    verify(h_C_tiled, h_C_cpu, M * N, "Tiled");

    // GFLOPS
    double flops = 2.0 * M * N * K;
    printf("\nNaive: %.1f GFLOPS\n", flops / (t_naive * 1e6));
    printf("Tiled: %.1f GFLOPS\n", flops / (t_tiled * 1e6));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B;
    delete[] h_C_naive; delete[] h_C_tiled; delete[] h_C_cpu;

    return 0;
}
