// Phase 5: Unified Memory Deep Dive
// =========================================================================
//
// Concepts:
//   - cudaMallocManaged: Allocates memory accessible from BOTH host and
//     device. The CUDA driver automatically migrates pages on demand.
//     You used this in 019a Phase 2 — now we study its performance.
//
//   - Page faulting: When the GPU accesses a page that's on the CPU
//     (or vice versa), a page fault occurs and the page is migrated.
//     This is similar to virtual memory in an OS, but over PCIe.
//     First-touch penalty can be severe (~100 μs per fault).
//
//   - cudaMemPrefetchAsync: Proactively migrate pages to a device BEFORE
//     they're accessed. Eliminates page faults and bundles the transfer
//     into efficient DMA. The key to making managed memory performant.
//
//   - cudaMemAdvise: Hints to the driver about access patterns:
//       cudaMemAdviseSetReadMostly      — data is read-only (replicate)
//       cudaMemAdviseSetPreferredLocation — prefer this device
//       cudaMemAdviseSetAccessedBy       — will be accessed by this device
//
//   - When to use each approach:
//       Explicit (cudaMalloc + cudaMemcpy): Maximum control, best perf
//       Managed + prefetch: Good perf, simpler code, good for prototyping
//       Managed naive: Easy but slow — only for quick prototypes
//       Zero-copy (mapped pinned): For small, infrequent host access
//
// Ref: CUDA Programming Guide §B.17 "Unified Memory Programming"
// Ref: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
//
// HFT context: Rapid prototyping — use managed memory to quickly test
// a new pricing model, then switch to explicit transfers for production
// once the algorithm is validated. The prefetch API bridges the gap.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr int N = 64 * 1024 * 1024;  // 64M elements (~256 MB)
constexpr int BLOCK_SIZE = 256;

// --------------------------------------------------------------------------- //
// Kernel: simple vector scale (same kernel for all approaches)
// --------------------------------------------------------------------------- //
__global__ void scale_kernel(float* data, int n, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * s + 1.0f;
    }
}

// --------------------------------------------------------------------------- //
// Approach 1: Explicit memory management (baseline, fastest)
// --------------------------------------------------------------------------- //
float run_explicit(const float* h_input, float* h_output, int n) {
    size_t bytes = n * sizeof(float);
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    CudaTimer timer;
    timer.start();

    CUDA_CHECK(cudaMemcpy(d_data, h_input, bytes, cudaMemcpyHostToDevice));
    scale_kernel<<<grid, BLOCK_SIZE>>>(d_data, n, 2.0f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_output, d_data, bytes, cudaMemcpyDeviceToHost));

    timer.stop();
    float ms = timer.elapsed_ms();

    CUDA_CHECK(cudaFree(d_data));
    return ms;
}

// --------------------------------------------------------------------------- //
// Approach 2: Naive managed memory (page faults, slow first access)
// --------------------------------------------------------------------------- //
float run_managed_naive(const float* h_source, float* h_verify, int n) {
    size_t bytes = n * sizeof(float);
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* managed = nullptr;
    CUDA_CHECK(cudaMallocManaged(&managed, bytes));

    // Initialize on CPU — pages are on CPU side
    memcpy(managed, h_source, bytes);

    CudaTimer timer;
    timer.start();

    // GPU access triggers page faults: each 4KB page migrated on demand
    scale_kernel<<<grid, BLOCK_SIZE>>>(managed, n, 2.0f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // CPU read — triggers reverse migration (GPU -> CPU page faults)
    CudaTimer cpu_timer;
    cpu_timer.start();
    memcpy(h_verify, managed, bytes);
    cpu_timer.stop();
    float cpu_ms = cpu_timer.elapsed_ms();

    printf("    GPU kernel (with faults): %.2f ms\n", gpu_ms);
    printf("    CPU readback (with faults): %.2f ms\n", cpu_ms);

    CUDA_CHECK(cudaFree(managed));
    return gpu_ms + cpu_ms;
}

// --------------------------------------------------------------------------- //
// Approach 3: Managed memory with prefetch
//
// TODO(human): Implement prefetch-based managed memory.
//
// Steps:
//   1. cudaMallocManaged(&managed, bytes)
//   2. Initialize on CPU: memcpy(managed, h_source, bytes)
//   3. Prefetch to GPU: cudaMemPrefetchAsync(managed, bytes, device_id, 0)
//   4. cudaDeviceSynchronize() to ensure prefetch completes
//   5. Start timer
//   6. Launch kernel (no page faults — data already on GPU)
//   7. Stop timer for kernel
//   8. Prefetch back to CPU: cudaMemPrefetchAsync(managed, bytes, cudaCpuDeviceId, 0)
//   9. cudaDeviceSynchronize()
//   10. CPU reads the data (no faults — already prefetched to CPU)
//
// cudaCpuDeviceId is a special constant meaning "the CPU."
// device_id is typically 0 (first GPU).
//
// Expected: kernel time should match explicit approach (no faults).
// Total time slightly higher due to prefetch API overhead.
// --------------------------------------------------------------------------- //
float run_managed_prefetch(const float* h_source, float* h_verify, int n) {
    size_t bytes = n * sizeof(float);
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int device_id = 0;

    float* managed = nullptr;
    CUDA_CHECK(cudaMallocManaged(&managed, bytes));
    memcpy(managed, h_source, bytes);

    CudaTimer timer;

    // TODO(human): Prefetch to GPU before kernel launch
    // CUDA_CHECK(cudaMemPrefetchAsync(managed, bytes, device_id, 0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    scale_kernel<<<grid, BLOCK_SIZE>>>(managed, n, 2.0f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // TODO(human): Prefetch back to CPU before host access
    // CUDA_CHECK(cudaMemPrefetchAsync(managed, bytes, cudaCpuDeviceId, 0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer cpu_timer;
    cpu_timer.start();
    memcpy(h_verify, managed, bytes);
    cpu_timer.stop();
    float cpu_ms = cpu_timer.elapsed_ms();

    printf("    GPU kernel (prefetched): %.2f ms\n", gpu_ms);
    printf("    CPU readback (prefetched): %.2f ms\n", cpu_ms);

    CUDA_CHECK(cudaFree(managed));
    return gpu_ms + cpu_ms;
}

// --------------------------------------------------------------------------- //
// Approach 4: Managed memory with MemAdvise hints
//
// TODO(human): Add memory advice before prefetch.
//
// cudaMemAdvise tells the driver about expected access patterns:
//   - cudaMemAdviseSetPreferredLocation(ptr, bytes, device):
//       "This memory will be used mostly on this device."
//       Pages won't migrate away unless needed elsewhere.
//
//   - cudaMemAdviseSetAccessedBy(ptr, bytes, device):
//       "This device will access the memory."
//       Creates a mapping so the device can access without migration.
//
//   - cudaMemAdviseSetReadMostly(ptr, bytes, device):
//       "This data is read-only." The driver may create read-only
//       copies on multiple devices instead of migrating.
//
// Combine with prefetch for best results.
// --------------------------------------------------------------------------- //
float run_managed_advise(const float* h_source, float* h_verify, int n) {
    size_t bytes = n * sizeof(float);
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int device_id = 0;

    float* managed = nullptr;
    CUDA_CHECK(cudaMallocManaged(&managed, bytes));
    memcpy(managed, h_source, bytes);

    // TODO(human): Set memory advice and prefetch
    // CUDA_CHECK(cudaMemAdvise(managed, bytes,
    //                          cudaMemAdviseSetPreferredLocation, device_id));
    // CUDA_CHECK(cudaMemAdvise(managed, bytes,
    //                          cudaMemAdviseSetAccessedBy, device_id));
    // CUDA_CHECK(cudaMemPrefetchAsync(managed, bytes, device_id, 0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    scale_kernel<<<grid, BLOCK_SIZE>>>(managed, n, 2.0f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // TODO(human): Prefetch back to CPU
    // CUDA_CHECK(cudaMemPrefetchAsync(managed, bytes, cudaCpuDeviceId, 0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer cpu_timer;
    cpu_timer.start();
    memcpy(h_verify, managed, bytes);
    cpu_timer.stop();
    float cpu_ms = cpu_timer.elapsed_ms();

    printf("    GPU kernel (advised+prefetch): %.2f ms\n", gpu_ms);
    printf("    CPU readback (advised+prefetch): %.2f ms\n", cpu_ms);

    CUDA_CHECK(cudaFree(managed));
    return gpu_ms + cpu_ms;
}

// --------------------------------------------------------------------------- //
// Verification
// --------------------------------------------------------------------------- //
void cpu_reference(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}

bool verify(const float* result, const float* expected, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (fabsf(result[i] - expected[i]) > 1e-3f) {
            if (bad < 3) printf("    MISMATCH at %d: got %.6f expected %.6f\n",
                                i, result[i], expected[i]);
            ++bad;
        }
    }
    return bad == 0;
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 5: Unified Memory Deep Dive ===\n");
    printf("N = %d elements (%.0f MB)\n\n", N,
           N * sizeof(float) / (1024.0 * 1024.0));

    // Check managed memory support
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  managedMemory: %s\n", prop.managedMemory ? "YES" : "NO");
    printf("  concurrentManagedAccess: %s\n\n",
           prop.concurrentManagedAccess ? "YES" : "NO");

    size_t bytes = N * sizeof(float);

    // Source data and verification buffers
    float* h_source = new float[N];
    float* h_expected = new float[N];
    float* h_result = new float[N];

    for (int i = 0; i < N; ++i) {
        h_source[i] = static_cast<float>(i % 1000) * 0.001f;
    }
    cpu_reference(h_source, h_expected, N);

    // --- Approach 1: Explicit ---
    printf("--- Explicit (cudaMalloc + cudaMemcpy) ---\n");
    memset(h_result, 0, bytes);
    float t_explicit = run_explicit(h_source, h_result, N);
    printf("  Total: %.2f ms\n", t_explicit);
    BandwidthMeter::report("Explicit", bytes * 2, t_explicit);
    printf("  %s\n\n", verify(h_result, h_expected, N) ? "PASSED" : "FAILED");

    // --- Approach 2: Naive managed ---
    printf("--- Managed (naive, page faults) ---\n");
    memset(h_result, 0, bytes);
    float t_naive = run_managed_naive(h_source, h_result, N);
    printf("  Total: %.2f ms\n", t_naive);
    printf("  %s\n", verify(h_result, h_expected, N) ? "PASSED" : "FAILED");
    BandwidthMeter::compare("Explicit vs Naive", t_explicit, t_naive);
    printf("\n");

    // --- Approach 3: Managed + prefetch ---
    printf("--- Managed + Prefetch ---\n");
    memset(h_result, 0, bytes);
    float t_prefetch = run_managed_prefetch(h_source, h_result, N);
    printf("  Total: %.2f ms\n", t_prefetch);
    printf("  %s\n", verify(h_result, h_expected, N) ? "PASSED" : "FAILED — implement TODO(human)!");
    BandwidthMeter::compare("Explicit vs Prefetched", t_explicit, t_prefetch);
    printf("\n");

    // --- Approach 4: Managed + advise + prefetch ---
    printf("--- Managed + MemAdvise + Prefetch ---\n");
    memset(h_result, 0, bytes);
    float t_advise = run_managed_advise(h_source, h_result, N);
    printf("  Total: %.2f ms\n", t_advise);
    printf("  %s\n", verify(h_result, h_expected, N) ? "PASSED" : "FAILED — implement TODO(human)!");
    BandwidthMeter::compare("Explicit vs Advised", t_explicit, t_advise);
    printf("\n");

    // --- Summary ---
    printf("=== Summary ===\n");
    printf("  Explicit:            %8.2f ms (baseline)\n", t_explicit);
    printf("  Managed naive:       %8.2f ms (%.1fx slower)\n",
           t_naive, t_naive / t_explicit);
    printf("  Managed + prefetch:  %8.2f ms (%.1fx vs explicit)\n",
           t_prefetch, t_prefetch / t_explicit);
    printf("  Managed + advise:    %8.2f ms (%.1fx vs explicit)\n",
           t_advise, t_advise / t_explicit);
    printf("\nExpected:\n");
    printf("  Naive >> Explicit (page faults add massive overhead)\n");
    printf("  Prefetch ≈ Explicit (prefetch eliminates faults)\n");
    printf("  Advise ≈ Prefetch (hints help driver optimize)\n");
    printf("\nRule of thumb:\n");
    printf("  Prototype: managed memory (simple, portable)\n");
    printf("  Production: explicit for hot paths, managed+prefetch for the rest\n");

    delete[] h_source;
    delete[] h_expected;
    delete[] h_result;
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
