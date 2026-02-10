// Phase 3: Thread Hierarchy & Indexing — Naive Matrix Multiplication
// =========================================================================
//
// Concepts:
//   - Grid = collection of blocks. Block = collection of threads.
//   - 2D grids: blockIdx.x/y, threadIdx.x/y, blockDim.x/y
//   - Global thread index: row = blockIdx.y * blockDim.y + threadIdx.y
//                          col = blockIdx.x * blockDim.x + threadIdx.x
//   - Warp (32 threads): the fundamental execution unit. All threads in a
//     warp execute the same instruction. Divergence within a warp is expensive.
//
// Matrix multiply C = A * B is the canonical GPU benchmark because:
//   - It's O(N^3) compute with O(N^2) data — compute-bound for large N
//   - Each element C[row][col] is independent — perfect parallelism
//   - It exposes the memory hierarchy bottleneck (improved in Phase 4)
//
// CP analogy: Think of the 2D grid as a 2D array where each cell is an
// independent computation. Like running N^2 independent dot products.
//
// HFT relevance: Correlation matrices, covariance estimation, portfolio
// optimization all reduce to matrix multiply. GPU matmul is 100x+ faster
// than CPU for large matrices.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

// Matrix dimensions — 1024x1024 is large enough to show GPU benefit
constexpr int M = 1024;  // rows of A and C
constexpr int K = 1024;  // cols of A, rows of B
constexpr int N = 1024;  // cols of B and C

// Block size for 2D thread blocks (16x16 = 256 threads per block)
constexpr int BLOCK_SIZE = 16;

// --------------------------------------------------------------------------- //
// Naive matrix multiplication kernel
// --------------------------------------------------------------------------- //
//
// Each thread computes ONE element of C: C[row][col] = dot(A[row,:], B[:,col])
//
// TODO(human): Implement the naive matmul kernel.
//   1. Compute the global row and col indices from 2D block/thread IDs:
//        row = blockIdx.y * blockDim.y + threadIdx.y
//        col = blockIdx.x * blockDim.x + threadIdx.x
//   2. Boundary check: if (row < m && col < n)
//   3. Accumulate the dot product:
//        float sum = 0.0f;
//        for (int i = 0; i < k; ++i)
//            sum += A[row * k + i] * B[i * n + col];
//   4. Write: C[row * n + col] = sum;
//
// Note: Matrices are stored in ROW-MAJOR order (C-style).
//   A[row][col] -> A[row * num_cols + col]
//
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int m, int k, int n) {
    // TODO(human): Compute row, col, boundary check, dot product loop
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // placeholder
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // placeholder

    if (row < m && col < n) {
        C[row * n + col] = 0.0f;  // placeholder — should be the dot product
    }
}

// --------------------------------------------------------------------------- //
// CPU reference (naive triple loop)
// --------------------------------------------------------------------------- //
void cpu_matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// --------------------------------------------------------------------------- //
// Initialize matrices with small random values
// --------------------------------------------------------------------------- //
void fill_random(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

// --------------------------------------------------------------------------- //
// Verify GPU vs CPU with tolerance
// --------------------------------------------------------------------------- //
bool verify_matmul(const float* gpu, const float* cpu, int m, int n) {
    // Matmul accumulates K floats, so tolerance scales with K
    float eps = K * 1e-5f;
    int mismatches = 0;
    for (int i = 0; i < m * n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > eps) {
            if (mismatches < 5) {
                int row = i / n, col = i % n;
                printf("  MISMATCH at [%d][%d]: GPU=%.6f, CPU=%.6f (diff=%.6f)\n",
                       row, col, gpu[i], cpu[i], fabsf(gpu[i] - cpu[i]));
            }
            ++mismatches;
        }
    }
    if (mismatches > 0) {
        printf("  Total mismatches: %d / %d\n", mismatches, m * n);
        return false;
    }
    return true;
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 3: Naive Matrix Multiplication ===\n");
    printf("C[%d x %d] = A[%d x %d] * B[%d x %d]\n\n", M, N, M, K, K, N);

    // Host allocation
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_gpu = new float[M * N];
    float* h_C_cpu = new float[M * N];

    srand(42);
    fill_random(h_A, M, K);
    fill_random(h_B, K, N);

    // Device allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // ---- GPU kernel launch ----
    // 2D grid of 2D blocks:
    //   Each block is BLOCK_SIZE x BLOCK_SIZE threads
    //   Grid dimensions cover the entire output matrix C
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);               // 16x16 = 256 threads
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,      // ceil(N/16) blocks in x
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);      // ceil(M/16) blocks in y

    printf("Grid: (%d, %d) blocks, Block: (%d, %d) threads\n",
           grid.x, grid.y, block.x, block.y);
    printf("Total threads: %d (covering %d output elements)\n\n",
           grid.x * block.x * grid.y * block.y, M * N);

    // Warm-up launch (first kernel has overhead from context setup)
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed launch
    CudaTimer timer;
    timer.start();
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK_LAST();
    timer.stop();
    printf("GPU naive matmul: %.3f ms\n", timer.elapsed_ms());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    // CPU reference
    printf("Computing CPU reference (this may take a few seconds)...\n");
    cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);

    // Verify
    if (verify_matmul(h_C_gpu, h_C_cpu, M, N)) {
        printf("PASSED: GPU result matches CPU reference.\n");
    } else {
        printf("FAILED: Implement the TODO(human) kernel!\n");
    }

    // Performance info
    double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
    double gflops = flops / (timer.elapsed_ms() * 1e6);
    printf("\nPerformance: %.1f GFLOPS (naive)\n", gflops);
    printf("Note: cuBLAS achieves 10-50x higher — the gap is what Phase 4 starts to close.\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return 0;
}
