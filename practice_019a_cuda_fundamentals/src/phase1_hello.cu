// Phase 1: Hello CUDA — Device query + first kernel launch
// =========================================================================
//
// Concepts:
//   - __global__ marks a function as a GPU kernel (callable from host, runs on device)
//   - <<<gridDim, blockDim>>> is the execution configuration
//   - cudaDeviceSynchronize() waits for all GPU work to complete
//   - Each thread gets a unique combination of blockIdx and threadIdx
//
// CP analogy: Imagine running 10,000 independent brute-force solutions
// simultaneously — each thread is one solver, and they all execute the
// same code but on different data (SIMT: Single Instruction, Multiple Threads).
//
// HFT relevance: Device query tells you the hardware you're working with —
// number of SMs, clock rate, memory bandwidth — all critical for sizing
// your pricing engine's parallelism.
// =========================================================================

#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

// --------------------------------------------------------------------------- //
// Step 1: Query and print GPU device properties
// --------------------------------------------------------------------------- //
void print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("CUDA devices found: %d\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability:    %d.%d\n", prop.major, prop.minor);
        printf("  SMs (multiprocessors): %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM:    %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp size:             %d\n", prop.warpSize);
        printf("  Global memory:         %.1f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared mem per block:  %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Clock rate:            %.0f MHz\n", prop.clockRate / 1000.0);
        printf("  Memory clock:          %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory bus width:      %d bits\n", prop.memoryBusWidth);
        printf("\n");
    }
}

// --------------------------------------------------------------------------- //
// Step 2: Your first kernel
// --------------------------------------------------------------------------- //
//
// __global__ means: this function runs on the GPU, called from the CPU.
// Every thread executes this function independently.
//
// TODO(human): Fill in the kernel body.
//   - Compute a unique global index: idx = blockIdx.x * blockDim.x + threadIdx.x
//   - If idx < n, write idx * 2 into output[idx]
//   - This is trivially parallel: each thread handles exactly one element
//
__global__ void first_kernel(int* output, int n) {
    // TODO(human): Compute global thread index and write idx * 2 to output[idx]
    // Hint: idx = blockIdx.x * blockDim.x + threadIdx.x
    //       Guard: if (idx < n) { ... }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // placeholder — replace with your code
    if (idx < n) {
        output[idx] = 0;  // placeholder — should be idx * 2
    }
}

// --------------------------------------------------------------------------- //
// CPU reference for verification
// --------------------------------------------------------------------------- //
void cpu_reference(int* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = i * 2;
    }
}

// --------------------------------------------------------------------------- //
// Main: launch kernel and verify
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 1: Hello CUDA ===\n\n");

    // --- Device info ---
    print_device_info();

    // --- Kernel launch ---
    constexpr int N = 256;
    constexpr int BLOCK_SIZE = 64;  // threads per block
    constexpr int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // ceiling division

    printf("Launching kernel: %d blocks x %d threads = %d threads total\n",
           GRID_SIZE, BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE);
    printf("Data size: %d elements\n\n", N);

    // Allocate device memory
    int* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, N * sizeof(int)));

    // Launch the kernel
    // <<<grid_dim, block_dim>>> is the execution configuration:
    //   grid_dim  = number of blocks in the grid
    //   block_dim = number of threads per block
    first_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_output, N);
    CUDA_CHECK_LAST();

    // cudaDeviceSynchronize() — CPU waits here until ALL GPU threads finish.
    // Without this, the CPU would race ahead and read unfinished results.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    int h_output[N];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute CPU reference
    int h_reference[N];
    cpu_reference(h_reference, N);

    // Verify
    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != h_reference[i]) {
            printf("MISMATCH at index %d: GPU=%d, expected=%d\n",
                   i, h_output[i], h_reference[i]);
            pass = false;
            if (i > 5) { printf("  ... (more mismatches)\n"); break; }
        }
    }

    if (pass) {
        printf("PASSED: All %d elements match the CPU reference.\n", N);
    } else {
        printf("FAILED: GPU output does not match. Implement the TODO(human) kernel!\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    return pass ? 0 : 1;
}
