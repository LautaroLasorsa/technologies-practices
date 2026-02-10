// Phase 3: Multi-Stream Pipeline
// =========================================================================
//
// Concepts:
//   The canonical GPU pipeline pattern — the most important technique
//   in this entire practice. This is how production GPU systems achieve
//   near-100% utilization.
//
//   Problem: A naive GPU workflow is serial:
//     H2D(all) -> Compute(all) -> D2H(all)
//     The GPU idles during transfers. Transfers idle during compute.
//
//   Solution: Divide work into N_CHUNKS. Use N_STREAMS streams to overlap:
//     Stream 0: H2D[0] -> Compute[0] -> D2H[0]
//     Stream 1:          H2D[1] -> Compute[1] -> D2H[1]
//     Stream 2:                   H2D[2] -> Compute[2] -> D2H[2]
//     ...
//
//   At steady state, the copy engine and compute engine run simultaneously:
//     Time T: H2D[N+2] | Compute[N+1] | D2H[N]
//
//   This is a triple-buffer (or double-buffer) pipeline — the same pattern
//   used in graphics (front/back buffer) and network I/O.
//
//   Choosing chunk count:
//     - Too few chunks → not enough overlap
//     - Too many chunks → overhead from stream management, tiny kernels
//     - Sweet spot: 4-16 chunks with 2-4 streams usually works well
//
// Ref: CUDA Programming Guide §3.2.6.5 "Overlapping Data Transfer
//      and Kernel Execution"
//
// HFT context: Streaming tick data processing. Market data arrives
// continuously. While chunk N's results are being read back, chunk N+1
// is being processed, and chunk N+2's data is being transferred.
// The GPU never idles — continuous utilization for real-time analytics.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

// Total data: 256M floats (~1 GB)
constexpr int TOTAL_ELEMS = 256 * 1024 * 1024;
constexpr int NUM_STREAMS = 4;
constexpr int NUM_CHUNKS  = 16;   // divide work into 16 chunks
constexpr int BLOCK_SIZE  = 256;

// --------------------------------------------------------------------------- //
// Simulated "tick processing" kernel: heavier compute to make overlap visible.
// Applies a simple moving-average-like smoothing (each element = average of
// itself and its neighbors).
// --------------------------------------------------------------------------- //
__global__ void process_ticks(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate heavier computation: read neighbors, do FMA chain
        float val = input[idx];
        float left  = (idx > 0)     ? input[idx - 1] : val;
        float right = (idx < n - 1) ? input[idx + 1] : val;
        float avg = (left + val + right) / 3.0f;

        // Extra FMA to increase compute time
        for (int i = 0; i < 8; ++i) {
            avg = avg * 0.999f + 0.001f;
        }
        output[idx] = avg;
    }
}

// --------------------------------------------------------------------------- //
// CPU reference
// --------------------------------------------------------------------------- //
void cpu_process(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float val = in[i];
        float left  = (i > 0)     ? in[i - 1] : val;
        float right = (i < n - 1) ? in[i + 1] : val;
        float avg = (left + val + right) / 3.0f;
        for (int j = 0; j < 8; ++j) {
            avg = avg * 0.999f + 0.001f;
        }
        out[i] = avg;
    }
}

// --------------------------------------------------------------------------- //
// Approach 1: Single-stream (no overlap)
// --------------------------------------------------------------------------- //
float run_single_stream(float* h_in, float* h_out,
                        float* d_in, float* d_out, int total) {
    size_t bytes = total * sizeof(float);
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CudaTimer timer;
    timer.start();

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    process_ticks<<<grid, BLOCK_SIZE>>>(d_in, d_out, total);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    timer.stop();
    return timer.elapsed_ms();
}

// --------------------------------------------------------------------------- //
// Approach 2: Multi-stream chunked pipeline
//
// TODO(human): Implement the triple-buffered pipeline.
//
// This is the hardest exercise in 019b. Here's the plan:
//
//   1. Create NUM_STREAMS streams.
//   2. Divide TOTAL_ELEMS into NUM_CHUNKS equal chunks.
//   3. For each chunk c (0..NUM_CHUNKS-1):
//      a) Pick stream: s = c % NUM_STREAMS
//      b) Compute offset = c * chunk_size, bytes = chunk_size * sizeof(float)
//      c) cudaMemcpyAsync H2D for chunk c on stream s
//      d) Launch process_ticks kernel for chunk c on stream s
//      e) cudaMemcpyAsync D2H for chunk c on stream s
//   4. cudaDeviceSynchronize() — wait for all streams to finish.
//   5. Destroy streams.
//
// Why this works:
//   - Chunk 0 goes on stream 0, chunk 1 on stream 1, etc.
//   - While stream 0 is computing chunk 0, stream 1 is transferring chunk 1
//   - The round-robin assignment (c % NUM_STREAMS) keeps all streams busy
//   - The GPU's copy and compute engines overlap automatically
//
// Edge case: The last chunk may be smaller if TOTAL_ELEMS is not evenly
// divisible by NUM_CHUNKS. Handle this with:
//   int chunk_size = TOTAL_ELEMS / NUM_CHUNKS;
//   int last_chunk = TOTAL_ELEMS - chunk_size * (NUM_CHUNKS - 1);
//   int this_chunk = (c == NUM_CHUNKS - 1) ? last_chunk : chunk_size;
// --------------------------------------------------------------------------- //
float run_pipeline(float* h_in, float* h_out,
                   float* d_in, float* d_out, int total) {
    int chunk_size = total / NUM_CHUNKS;

    CudaTimer timer;

    // TODO(human): Create streams
    // cudaStream_t streams[NUM_STREAMS];
    // for (int s = 0; s < NUM_STREAMS; ++s) {
    //     CUDA_CHECK(cudaStreamCreate(&streams[s]));
    // }

    timer.start();

    // TODO(human): Pipeline loop over chunks
    // for (int c = 0; c < NUM_CHUNKS; ++c) {
    //     int s = c % NUM_STREAMS;
    //     int offset = c * chunk_size;
    //     int this_chunk = (c == NUM_CHUNKS - 1) ? (total - offset) : chunk_size;
    //     size_t bytes = this_chunk * sizeof(float);
    //     int grid = (this_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //
    //     // H2D for this chunk
    //     CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, bytes,
    //                                 cudaMemcpyHostToDevice, streams[s]));
    //
    //     // Compute this chunk
    //     process_ticks<<<grid, BLOCK_SIZE, 0, streams[s]>>>(
    //         d_in + offset, d_out + offset, this_chunk);
    //     CUDA_CHECK_LAST();
    //
    //     // D2H for this chunk
    //     CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, bytes,
    //                                 cudaMemcpyDeviceToHost, streams[s]));
    // }

    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();

    // TODO(human): Destroy streams
    // for (int s = 0; s < NUM_STREAMS; ++s) {
    //     CUDA_CHECK(cudaStreamDestroy(streams[s]));
    // }

    return timer.elapsed_ms();
}

// --------------------------------------------------------------------------- //
// Verification
// --------------------------------------------------------------------------- //
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
    printf("=== Phase 3: Multi-Stream Pipeline ===\n");
    printf("Total: %d elements (%.0f MB)\n",
           TOTAL_ELEMS, TOTAL_ELEMS * sizeof(float) / (1024.0 * 1024.0));
    printf("Chunks: %d, Streams: %d, Chunk size: %d elements (%.0f MB)\n\n",
           NUM_CHUNKS, NUM_STREAMS,
           TOTAL_ELEMS / NUM_CHUNKS,
           (TOTAL_ELEMS / NUM_CHUNKS) * sizeof(float) / (1024.0 * 1024.0));

    size_t total_bytes = TOTAL_ELEMS * sizeof(float);

    // Pinned host memory (required for async overlap)
    float *h_in = nullptr, *h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in, total_bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, total_bytes));

    // Initialize with pattern
    for (int i = 0; i < TOTAL_ELEMS; ++i) {
        h_in[i] = sinf(static_cast<float>(i) * 0.001f) + 1.0f;
    }

    // Device memory
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, total_bytes));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_in, h_in, total_bytes, cudaMemcpyHostToDevice));
    int grid = (TOTAL_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    process_ticks<<<grid, BLOCK_SIZE>>>(d_in, d_out, TOTAL_ELEMS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Single stream ---
    printf("--- Single Stream (no overlap) ---\n");
    memset(h_out, 0, total_bytes);
    float t_single = run_single_stream(h_in, h_out, d_in, d_out, TOTAL_ELEMS);
    BandwidthMeter::report("Single stream", total_bytes * 2, t_single);
    printf("  Total time: %.2f ms\n", t_single);

    // Compute CPU reference (on a subset for speed)
    constexpr int VERIFY_N = 1024 * 1024;  // verify first 1M elements
    std::vector<float> cpu_out(VERIFY_N);
    cpu_process(h_in, cpu_out.data(), VERIFY_N);
    printf("  Correctness (first %dK): %s\n\n",
           VERIFY_N / 1024,
           verify(h_out, cpu_out.data(), VERIFY_N) ? "PASSED" : "FAILED");

    // --- Pipeline ---
    printf("--- Multi-Stream Pipeline (your implementation) ---\n");
    memset(h_out, 0, total_bytes);
    float t_pipe = run_pipeline(h_in, h_out, d_in, d_out, TOTAL_ELEMS);
    BandwidthMeter::report("Pipeline", total_bytes * 2, t_pipe);
    printf("  Total time: %.2f ms\n", t_pipe);
    BandwidthMeter::compare("Pipeline vs Single", t_pipe, t_single);
    printf("  Correctness (first %dK): %s\n\n",
           VERIFY_N / 1024,
           verify(h_out, cpu_out.data(), VERIFY_N) ? "PASSED" : "FAILED — implement TODO(human)!");

    // --- Summary ---
    printf("=== Summary ===\n");
    printf("  Single stream: %.2f ms\n", t_single);
    printf("  Pipeline:      %.2f ms\n", t_pipe);
    if (t_pipe > 0)
        printf("  Speedup:       %.2fx\n", t_single / t_pipe);
    printf("\nExpected speedup: ~1.5-3x depending on compute/transfer ratio.\n");
    printf("The pipeline hides transfer latency behind computation.\n");
    printf("\nVisualization (steady state):\n");
    printf("  Copy engine:    |H2D[2]|     |H2D[5]|     |H2D[8]|     |\n");
    printf("  Compute engine: |Kern[1]|    |Kern[4]|    |Kern[7]|    |\n");
    printf("  Copy engine:    |D2H[0]|    |D2H[3]|     |D2H[6]|    |\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
