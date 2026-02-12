// Phase 4: Cooperative Groups & Grid Sync
// =========================================================================
//
// Concepts:
//   - cooperative_groups: A CUDA C++ namespace providing flexible,
//     composable thread grouping beyond the fixed block/warp model.
//     Introduced in CUDA 9, essential for modern CUDA code.
//
//   - thread_block: Represents all threads in a block. Replaces
//     __syncthreads() with tb.sync() — same behavior, cleaner API.
//
//   - thread_block_tile<N>: A statically-sized partition of a block.
//     tiled_partition<32>(tb) creates warp-sized tiles. Each tile
//     can sync and shuffle independently.
//
//   - grid_group: ALL threads in the entire grid. grid.sync() is a
//     barrier across the ENTIRE grid. This requires cooperative launch
//     (cudaLaunchCooperativeKernel) and compute capability >= 7.0.
//
//   - Why grid sync matters: Without it, multi-pass algorithms require
//     separate kernel launches (with implicit global sync between them).
//     Grid sync lets you do multiple passes in a SINGLE kernel, avoiding
//     launch overhead (~5-10 μs per launch).
//
//   - Limitation: Grid sync requires that the grid fits in the GPU
//     simultaneously (no oversubscription). Use cudaOccupancyMaxActive-
//     BlocksPerMultiprocessor to determine the max grid size.
//
// Ref: CUDA Programming Guide §B.24 "Cooperative Groups"
// Ref: https://developer.nvidia.com/blog/cooperative-groups/
//
// HFT context: Atomic order book updates requiring grid-wide consistency.
// All threads process order updates, then grid-sync, then all threads
// read the updated book — in a single kernel, no launch overhead.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cuda_helpers.h"

namespace cg = cooperative_groups;

constexpr int N = 1 << 20;      // 1M elements
constexpr int BLOCK_SIZE = 256;

// --------------------------------------------------------------------------- //
// Part A: Tiled reduction using cooperative groups
//
// Instead of raw __shfl_down_sync, use thread_block_tile<32> which
// provides .shfl_down() and .sync() methods — type-safe and composable.
//
// ── Exercise Context ──────────────────────────────────────────────────
// This exercise teaches cooperative groups—modern CUDA's flexible synchronization API.
// Cooperative groups enable composable parallelism (tiles, partitions, grid-wide sync).
// This is how production CUDA code structures complex algorithms (multi-pass, hierarchical reduction).

// TODO(human): Implement a tiled warp reduction.
//
//   1. Get the thread_block: auto tb = cg::this_thread_block();
//   2. Partition into warp-sized tiles: auto warp = cg::tiled_partition<32>(tb);
//   3. Use warp.shfl_down(val, offset) for warp-level reduction
//   4. Use warp.thread_rank() instead of threadIdx.x % 32
//
// The advantage: tiles are composable. You can partition a block into
// tiles of 32, then use each tile independently. The API is cleaner
// than raw __shfl_down_sync with magic mask 0xFFFFFFFF.
// --------------------------------------------------------------------------- //
__global__ void tiled_reduce_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    // TODO(human): Get cooperative group handles
    // auto tb = cg::this_thread_block();
    // auto warp = cg::tiled_partition<32>(tb);

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared
    sdata[tid] = (global_idx < n) ? input[global_idx] : 0.0f;

    // TODO(human): Use tb.sync() instead of __syncthreads()
    __syncthreads();  // Replace with: tb.sync();

    // Block-level reduction down to warp size using shared memory
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        // TODO(human): Use tb.sync() here too
        __syncthreads();  // Replace with: tb.sync();
    }

    // TODO(human): Warp-level reduction using tiled_partition
    //
    // float val = (tid < 32) ? sdata[tid] : 0.0f;
    //
    // // Use warp.shfl_down instead of __shfl_down_sync
    // for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
    //     val += warp.shfl_down(val, offset);
    // }
    //
    // // warp.thread_rank() == 0 is the "leader" of each tile
    // if (warp.thread_rank() == 0 && tid < 32) {
    //     output[blockIdx.x] = val;
    // }

    // Placeholder: just write sdata[0] (will be wrong until TODO is implemented)
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// --------------------------------------------------------------------------- //
// Part B: Grid-wide synchronization — multi-pass single kernel
//
// Classic problem: "normalize an array" = divide each element by the sum.
// Without grid sync, you need 3 kernel launches:
//   Kernel 1: partial sums per block
//   Kernel 2: reduce partial sums to global sum
//   Kernel 3: divide each element by global sum
//
// With grid sync, you can do it in ONE kernel:
//   Pass 1: compute partial sums (block-level reduction)
//   grid.sync()  <-- all blocks wait here
//   Pass 2: thread 0 reduces partial sums to global
//   grid.sync()
//   Pass 3: each thread divides its element by the global sum
//
// This eliminates 2 kernel launch overheads (~10-20 μs saved).
//
// IMPORTANT: This kernel must be launched with cudaLaunchCooperativeKernel,
// and the grid size must not exceed the maximum occupancy.
// --------------------------------------------------------------------------- //
__global__ void normalize_cooperative(float* data, float* partial_sums, int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block tb = cg::this_thread_block();

    extern __shared__ float sdata[];

    int tid = tb.thread_rank();
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // === Pass 1: Block-level partial sum ===
    sdata[tid] = (global_idx < n) ? data[global_idx] : 0.0f;
    tb.sync();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        tb.sync();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }

    // === Grid-wide barrier: all blocks have written their partial sums ===
    grid.sync();

    // === Pass 2: Block 0, thread 0 reduces all partial sums ===
    if (blockIdx.x == 0 && tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < gridDim.x; ++i) {
            total += partial_sums[i];
        }
        partial_sums[0] = total;  // store global sum in [0]
    }

    // === Grid-wide barrier: global sum is ready ===
    grid.sync();

    // === Pass 3: Normalize each element ===
    if (global_idx < n) {
        float total = partial_sums[0];
        if (total != 0.0f) {
            data[global_idx] /= total;
        }
    }
}

// --------------------------------------------------------------------------- //
// CPU reference for normalize
// --------------------------------------------------------------------------- //
void cpu_normalize(float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += data[i];
    if (sum != 0.0) {
        for (int i = 0; i < n; ++i) data[i] /= static_cast<float>(sum);
    }
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 4: Cooperative Groups & Grid Sync ===\n");
    printf("N = %d elements (%.1f MB)\n\n", N,
           N * sizeof(float) / (1024.0f * 1024.0f));

    // --- Part A: Tiled reduction ---
    printf("--- Part A: Tiled Warp Reduction ---\n");
    {
        float* h_input = new float[N];
        for (int i = 0; i < N; ++i) h_input[i] = 1.0f;  // sum = N

        float* d_input = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float),
                              cudaMemcpyHostToDevice));

        int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float* d_partial = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

        CudaTimer timer;
        timer.start();
        tiled_reduce_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_input, d_partial, N);
        CUDA_CHECK_LAST();
        timer.stop();

        float* h_partial = new float[grid_size];
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid_size * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double gpu_sum = 0.0;
        for (int i = 0; i < grid_size; ++i) gpu_sum += h_partial[i];

        printf("  GPU sum: %.2f (expected: %d)\n", gpu_sum, N);
        printf("  Kernel time: %.3f ms\n", timer.elapsed_ms());
        bool pass = fabs(gpu_sum - N) < N * 1e-5;
        printf("  %s\n\n", pass ? "PASSED" : "FAILED — implement TODO(human)!");

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_partial));
        delete[] h_input;
        delete[] h_partial;
    }

    // --- Part B: Grid-wide cooperative normalize ---
    printf("--- Part B: Cooperative Grid Sync (Normalize) ---\n");
    {
        // Determine max grid size for cooperative launch
        int max_blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, normalize_cooperative, BLOCK_SIZE,
            BLOCK_SIZE * sizeof(float)));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int max_grid = max_blocks_per_sm * prop.multiProcessorCount;
        int needed_grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_size = (needed_grid < max_grid) ? needed_grid : max_grid;

        printf("  Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
        printf("  Max blocks for cooperative launch: %d (using %d)\n",
               max_grid, grid_size);

        if (!prop.cooperativeLaunch) {
            printf("  ERROR: Device does not support cooperative launch!\n");
            printf("  Need compute capability >= 7.0 (sm_70+)\n");
            CUDA_CHECK(cudaDeviceReset());
            return 1;
        }

        // Allocate and initialize
        float* h_data = new float[N];
        float* h_cpu  = new float[N];
        for (int i = 0; i < N; ++i) {
            h_data[i] = static_cast<float>(i + 1);
            h_cpu[i]  = static_cast<float>(i + 1);
        }

        float* d_data = nullptr;
        float* d_partial = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_partial, 0, grid_size * sizeof(float)));

        // Cooperative launch
        void* args[] = { &d_data, &d_partial, (void*)&N };
        dim3 grid_dim(grid_size);
        dim3 block_dim(BLOCK_SIZE);
        size_t shared_bytes = BLOCK_SIZE * sizeof(float);

        CudaTimer timer;

        // Multi-kernel approach for comparison
        // (We measure the cooperative single-kernel approach)
        timer.start();
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)normalize_cooperative,
            grid_dim, block_dim, args, shared_bytes, 0));
        CUDA_CHECK_LAST();
        timer.stop();
        float coop_ms = timer.elapsed_ms();

        // Copy back
        CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // CPU reference
        cpu_normalize(h_cpu, N);

        // Verify
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            if (fabsf(h_data[i] - h_cpu[i]) > 1e-4f) {
                if (bad < 3) printf("  MISMATCH at %d: GPU=%.8f CPU=%.8f\n",
                                    i, h_data[i], h_cpu[i]);
                ++bad;
            }
        }

        printf("  Cooperative normalize: %.3f ms\n", coop_ms);
        printf("  %s\n", (bad == 0) ? "PASSED" : "FAILED — check implementation!");
        if (bad > 0) printf("  (%d mismatches)\n", bad);

        // Show a few values
        printf("\n  Sample normalized values (should sum to 1.0):\n");
        double check_sum = 0.0;
        for (int i = 0; i < N; ++i) check_sum += h_data[i];
        printf("  Sum of normalized: %.8f (should be ~1.0)\n", check_sum);
        printf("  data[0]: %.10f, data[N-1]: %.10f\n", h_data[0], h_data[N-1]);

        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_partial));
        delete[] h_data;
        delete[] h_cpu;
    }

    printf("\n=== Key Takeaways ===\n");
    printf("1. cooperative_groups provides a cleaner API than raw __syncthreads/__shfl\n");
    printf("2. tiled_partition<32> replaces 0xFFFFFFFF mask magic\n");
    printf("3. grid.sync() enables single-kernel multi-pass algorithms\n");
    printf("4. Grid sync eliminates kernel launch overhead for iterative algorithms\n");
    printf("5. Grid size is limited by occupancy for cooperative launch\n");

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
