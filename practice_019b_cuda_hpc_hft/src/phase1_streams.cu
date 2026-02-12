// Phase 1: CUDA Streams & Async Operations
// =========================================================================
//
// Concepts:
//   - Default stream (stream 0): All CUDA calls are serialized on it.
//     Kernel A finishes before kernel B starts. Memcpy finishes before
//     the next kernel. This is the implicit behavior you've used in 019a.
//
//   - Created streams: Independent execution queues. Operations on
//     different streams CAN overlap — this is how you get computation
//     and data transfer happening simultaneously.
//
//   - cudaMemcpyAsync: Non-blocking memcpy that returns immediately to
//     the host. REQUIRES pinned host memory to truly overlap with
//     computation (pageable memory forces the driver to stage-copy).
//
//   - Key rule: Operations within the SAME stream are serialized.
//     Operations on DIFFERENT streams may overlap if the hardware
//     supports it (copy engine + compute engine are separate units).
//
// What you'll see:
//   - Single-stream: total time ≈ memcpy_H2D + kernel + memcpy_D2H
//   - Multi-stream: total time ≈ max(memcpy_H2D, kernel, memcpy_D2H)
//     because the three operations overlap across streams.
//
// Ref: CUDA Programming Guide §3.2.6 "Asynchronous Concurrent Execution"
//
// HFT context: Processing multiple order books simultaneously — each
// order book's data transfer and computation runs on its own stream,
// so the GPU never idles waiting for a single transfer to complete.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

// Data size per stream — large enough to show overlap benefits.
// Each "order book" is 64M floats (~256 MB).
constexpr int ELEMS_PER_STREAM = 64 * 1024 * 1024;  // 64M elements
constexpr int NUM_STREAMS = 4;
constexpr int BLOCK_SIZE = 256;

// --------------------------------------------------------------------------- //
// Kernel: simple computation to simulate "processing an order book."
// Multiplies each element by a scalar and adds a bias — enough to
// keep the GPU busy for a measurable time.
// --------------------------------------------------------------------------- //
__global__ void process_data(const float* input, float* output, int n,
                             float scale, float bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate non-trivial work: multiple FMA operations
        float val = input[idx];
        val = val * scale + bias;
        val = val * scale + bias;
        val = val * scale + bias;
        val = val * scale + bias;
        output[idx] = val;
    }
}

// --------------------------------------------------------------------------- //
// Approach 1: Single stream (default stream, serialized)
//   H2D[0] -> Kernel[0] -> D2H[0] -> H2D[1] -> Kernel[1] -> D2H[1] -> ...
// --------------------------------------------------------------------------- //
float run_single_stream(float* h_in, float* h_out, float* d_in, float* d_out,
                        int n_per_stream, int num_streams) {
    int total = n_per_stream * num_streams;
    size_t bytes = n_per_stream * sizeof(float);
    int grid = (n_per_stream + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CudaTimer timer;
    timer.start();

    for (int s = 0; s < num_streams; ++s) {
        int offset = s * n_per_stream;
        // All on default stream — fully serialized
        CUDA_CHECK(cudaMemcpy(d_in + offset, h_in + offset, bytes,
                              cudaMemcpyHostToDevice));
        process_data<<<grid, BLOCK_SIZE>>>(d_in + offset, d_out + offset,
                                           n_per_stream, 1.01f, 0.001f);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_out + offset, d_out + offset, bytes,
                              cudaMemcpyDeviceToHost));
    }

    timer.stop();
    return timer.elapsed_ms();
}

// --------------------------------------------------------------------------- //
// Approach 2: Multiple streams (async, overlapped)
//
// ── Exercise Context ──────────────────────────────────────────────────
// This exercise teaches stream-based concurrency—the foundation of GPU pipelines.
// Overlapping H2D/compute/D2H achieves 2-3x throughput vs single-stream sequential execution.
// This pattern is ubiquitous in production HPC/HFT: process chunk N while transferring chunk N+1.

// TODO(human): Implement the multi-stream approach.
//
// Steps:
//   1. Create NUM_STREAMS streams with cudaStreamCreate()
//   2. For each stream s:
//      a) cudaMemcpyAsync(d_in + offset, h_in + offset, bytes,
//                          cudaMemcpyHostToDevice, streams[s]);
//      b) process_data<<<grid, BLOCK_SIZE, 0, streams[s]>>>(...);
//      c) cudaMemcpyAsync(h_out + offset, d_out + offset, bytes,
//                          cudaMemcpyDeviceToHost, streams[s]);
//   3. Synchronize all streams (cudaDeviceSynchronize or per-stream sync)
//   4. Destroy streams with cudaStreamDestroy()
//
// Key insight: The 4th argument to <<<grid, block, shared, stream>>> is
// the stream. The 3rd argument to cudaMemcpyAsync is the stream.
// Operations on different streams can execute concurrently.
//
// NOTE: h_in and h_out MUST be pinned memory for async to work!
//       (The caller handles this — check main())
// --------------------------------------------------------------------------- //
float run_multi_stream(float* h_in, float* h_out, float* d_in, float* d_out,
                       int n_per_stream, int num_streams) {
    size_t bytes = n_per_stream * sizeof(float);
    int grid = (n_per_stream + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CudaTimer timer;

    // TODO(human): Create streams
    // cudaStream_t streams[NUM_STREAMS];
    // for (int s = 0; s < num_streams; ++s) {
    //     CUDA_CHECK(cudaStreamCreate(&streams[s]));
    // }

    timer.start();

    // TODO(human): Issue async operations on each stream
    // for (int s = 0; s < num_streams; ++s) {
    //     int offset = s * n_per_stream;
    //     CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, bytes,
    //                                 cudaMemcpyHostToDevice, streams[s]));
    //     process_data<<<grid, BLOCK_SIZE, 0, streams[s]>>>(
    //         d_in + offset, d_out + offset, n_per_stream, 1.01f, 0.001f);
    //     CUDA_CHECK_LAST();
    //     CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, bytes,
    //                                 cudaMemcpyDeviceToHost, streams[s]));
    // }

    CUDA_CHECK(cudaDeviceSynchronize());

    timer.stop();

    // TODO(human): Destroy streams
    // for (int s = 0; s < num_streams; ++s) {
    //     CUDA_CHECK(cudaStreamDestroy(streams[s]));
    // }

    return timer.elapsed_ms();
}

// --------------------------------------------------------------------------- //
// Approach 3: Breadth-first stream scheduling (optimal overlap)
//
// Instead of issuing all 3 ops per stream before moving to the next,
// issue all H2D first, then all kernels, then all D2H. This gives the
// hardware scheduler maximum opportunity to overlap operations.
//
// The GPU has separate copy engines (H2D and D2H) and compute engines.
// Breadth-first issuing lets the copy engine start the next H2D while
// the compute engine runs the first kernel.
// --------------------------------------------------------------------------- //
float run_multi_stream_breadth(float* h_in, float* h_out, float* d_in, float* d_out,
                               int n_per_stream, int num_streams) {
    size_t bytes = n_per_stream * sizeof(float);
    int grid = (n_per_stream + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < num_streams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    CudaTimer timer;
    timer.start();

    // All H2D transfers first
    for (int s = 0; s < num_streams; ++s) {
        int offset = s * n_per_stream;
        CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, bytes,
                                    cudaMemcpyHostToDevice, streams[s]));
    }

    // All kernel launches
    for (int s = 0; s < num_streams; ++s) {
        int offset = s * n_per_stream;
        process_data<<<grid, BLOCK_SIZE, 0, streams[s]>>>(
            d_in + offset, d_out + offset, n_per_stream, 1.01f, 0.001f);
        CUDA_CHECK_LAST();
    }

    // All D2H transfers
    for (int s = 0; s < num_streams; ++s) {
        int offset = s * n_per_stream;
        CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, bytes,
                                    cudaMemcpyDeviceToHost, streams[s]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();

    for (int s = 0; s < num_streams; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    return timer.elapsed_ms();
}

// --------------------------------------------------------------------------- //
// Verification
// --------------------------------------------------------------------------- //
void cpu_reference(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float val = in[i];
        val = val * 1.01f + 0.001f;
        val = val * 1.01f + 0.001f;
        val = val * 1.01f + 0.001f;
        val = val * 1.01f + 0.001f;
        out[i] = val;
    }
}

bool verify(const float* gpu, const float* cpu, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > 1e-3f) {
            if (bad < 3) printf("  MISMATCH at %d: GPU=%.6f CPU=%.6f\n",
                                i, gpu[i], cpu[i]);
            ++bad;
        }
    }
    return bad == 0;
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 1: CUDA Streams & Async Operations ===\n");
    printf("Data: %d streams x %d elements (%.0f MB each, %.0f MB total)\n\n",
           NUM_STREAMS, ELEMS_PER_STREAM,
           ELEMS_PER_STREAM * sizeof(float) / (1024.0 * 1024.0),
           NUM_STREAMS * ELEMS_PER_STREAM * sizeof(float) / (1024.0 * 1024.0));

    int total = NUM_STREAMS * ELEMS_PER_STREAM;
    size_t total_bytes = total * sizeof(float);

    // Pinned host memory (required for cudaMemcpyAsync to truly overlap)
    float *h_in = nullptr, *h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in, total_bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, total_bytes));

    // Initialize input
    for (int i = 0; i < total; ++i) {
        h_in[i] = static_cast<float>(i % 1000) * 0.001f;
    }

    // Device memory
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, total_bytes));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_in, h_in, total_bytes, cudaMemcpyHostToDevice));
    int grid = (ELEMS_PER_STREAM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    process_data<<<grid, BLOCK_SIZE>>>(d_in, d_out, ELEMS_PER_STREAM, 1.01f, 0.001f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Run approaches ---
    printf("--- Single Stream (serialized) ---\n");
    float t_single = run_single_stream(h_in, h_out, d_in, d_out,
                                       ELEMS_PER_STREAM, NUM_STREAMS);
    BandwidthMeter::report("Single stream", total_bytes * 2, t_single);  // *2 for H2D+D2H
    printf("  Total time: %.2f ms\n\n", t_single);

    // Verify single-stream result
    {
        std::vector<float> cpu_out(total);
        cpu_reference(h_in, cpu_out.data(), total);
        printf("  Correctness: %s\n\n", verify(h_out, cpu_out.data(), total) ?
               "PASSED" : "FAILED");
    }

    printf("--- Multi-Stream (depth-first, your implementation) ---\n");
    memset(h_out, 0, total_bytes);
    float t_multi = run_multi_stream(h_in, h_out, d_in, d_out,
                                     ELEMS_PER_STREAM, NUM_STREAMS);
    BandwidthMeter::report("Multi stream", total_bytes * 2, t_multi);
    printf("  Total time: %.2f ms\n", t_multi);
    BandwidthMeter::compare("Multi vs Single", t_multi, t_single);
    printf("\n");

    printf("--- Multi-Stream (breadth-first, reference) ---\n");
    memset(h_out, 0, total_bytes);
    float t_breadth = run_multi_stream_breadth(h_in, h_out, d_in, d_out,
                                               ELEMS_PER_STREAM, NUM_STREAMS);
    BandwidthMeter::report("Breadth-first", total_bytes * 2, t_breadth);
    printf("  Total time: %.2f ms\n", t_breadth);
    BandwidthMeter::compare("Breadth vs Single", t_breadth, t_single);

    // Verify breadth-first result
    {
        std::vector<float> cpu_out(total);
        cpu_reference(h_in, cpu_out.data(), total);
        printf("  Correctness: %s\n\n", verify(h_out, cpu_out.data(), total) ?
               "PASSED" : "FAILED");
    }

    // Summary
    printf("=== Summary ===\n");
    printf("  Single stream: %.2f ms\n", t_single);
    printf("  Multi  (yours): %.2f ms (speedup: %.2fx)\n",
           t_multi, t_single / (t_multi > 0 ? t_multi : 1));
    printf("  Breadth-first:  %.2f ms (speedup: %.2fx)\n",
           t_breadth, t_single / (t_breadth > 0 ? t_breadth : 1));
    printf("\nExpected: multi-stream should be ~2-3x faster than single.\n");
    printf("Breadth-first may be slightly better than depth-first on\n");
    printf("some hardware due to better overlap scheduling.\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
