// Phase 6: Low-Latency Patterns for HFT
// =========================================================================
//
// Concepts:
//   - Kernel launch overhead: Each cudaLaunchKernel call has ~5-15 μs of
//     CPU-side overhead (driver calls, command buffer submission). In HFT,
//     where the entire trade decision window is 1-5 μs, this is massive.
//
//   - Persistent kernels: A kernel that never terminates — it runs in an
//     infinite loop, polling a flag in global/mapped memory for new work.
//     When work arrives, it processes and signals completion. Launch
//     overhead: zero (kernel is already running).
//
//   - Spin-waiting on GPU: The persistent kernel uses a busy-wait loop
//     (spin) to check for new tasks. This wastes GPU cycles but provides
//     the lowest possible latency (no scheduling delay).
//
//   - Synchronization mechanisms (latency comparison):
//       cudaDeviceSynchronize():     ~10-50 μs (heavyweight, full sync)
//       cudaStreamSynchronize():     ~5-20 μs (per-stream, lighter)
//       cudaEventSynchronize():      ~3-10 μs (event-based, lighter still)
//       Spin-poll (volatile flag):   ~0.5-2 μs (busy-wait, lowest latency)
//
//   - Mapped pinned memory for signaling: Use cudaHostAllocMapped to
//     create memory visible to both CPU and GPU. The CPU writes a flag,
//     the GPU sees it (with appropriate memory ordering / volatile).
//
// *** SAFETY WARNING ***
//   Persistent kernels can hang the GPU if they spin forever (e.g., if
//   the host never signals). Windows has a TDR (Timeout Detection and
//   Recovery) that kills kernels after ~2 seconds by default. For
//   development:
//     - Keep iteration counts bounded
//     - Use a "poison pill" / shutdown flag
//     - Test with small workloads first
//     - If the GPU hangs, the TDR will reset it (you'll lose the context)
//
// Ref: "Persistent Threads" — Gupta et al., 2012
// Ref: CUDA Programming Guide §3.2.6.6 "Streams" (synchronization)
//
// HFT context: A persistent kernel processes trade signals with
// sub-microsecond response. The CPU writes new market data to mapped
// memory, the GPU detects it immediately and computes the response.
// No kernel launch, no driver overhead.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_TASKS  = 100;     // number of tasks to process
constexpr int TASK_SIZE  = 1024;    // elements per task

// --------------------------------------------------------------------------- //
// Part A: Synchronization Latency Comparison
//
// Measures the overhead of different sync mechanisms by launching a
// trivial (near-zero-work) kernel and timing the round-trip.
// --------------------------------------------------------------------------- //
__global__ void trivial_kernel() {
    // Intentionally empty — we're measuring launch + sync overhead
}

struct SyncResult {
    const char* name;
    float avg_us;
};

void benchmark_sync_mechanisms() {
    printf("--- Part A: Synchronization Latency Comparison ---\n\n");

    constexpr int WARMUP = 50;
    constexpr int TRIALS = 200;

    // Warm-up
    for (int i = 0; i < WARMUP; ++i) {
        trivial_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    auto measure_us = [](auto fn) -> float {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::micro>(end - start).count();
    };

    // 1. cudaDeviceSynchronize
    float total_device = 0;
    for (int i = 0; i < TRIALS; ++i) {
        trivial_kernel<<<1, 1>>>();
        total_device += measure_us([&]{ cudaDeviceSynchronize(); });
    }
    float avg_device = total_device / TRIALS;

    // 2. cudaStreamSynchronize
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float total_stream = 0;
    for (int i = 0; i < TRIALS; ++i) {
        trivial_kernel<<<1, 1, 0, stream>>>();
        total_stream += measure_us([&]{ cudaStreamSynchronize(stream); });
    }
    float avg_stream = total_stream / TRIALS;
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 3. cudaEventSynchronize
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    CUDA_CHECK(cudaStreamCreate(&stream));
    float total_event = 0;
    for (int i = 0; i < TRIALS; ++i) {
        trivial_kernel<<<1, 1, 0, stream>>>();
        cudaEventRecord(event, stream);
        total_event += measure_us([&]{ cudaEventSynchronize(event); });
    }
    float avg_event = total_event / TRIALS;
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 4. Spin-poll on mapped flag
    // (CPU writes 1 to flag after kernel sets it to 0, kernel writes 1 back)
    // For this measurement, we just poll cudaEventQuery in a loop.
    CUDA_CHECK(cudaEventCreate(&event));
    CUDA_CHECK(cudaStreamCreate(&stream));
    float total_spin = 0;
    for (int i = 0; i < TRIALS; ++i) {
        trivial_kernel<<<1, 1, 0, stream>>>();
        cudaEventRecord(event, stream);
        total_spin += measure_us([&]{
            while (cudaEventQuery(event) != cudaSuccess) {
                // spin
            }
        });
    }
    float avg_spin = total_spin / TRIALS;
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("  cudaDeviceSynchronize:  %7.1f μs avg\n", avg_device);
    printf("  cudaStreamSynchronize:  %7.1f μs avg\n", avg_stream);
    printf("  cudaEventSynchronize:   %7.1f μs avg\n", avg_event);
    printf("  Spin-poll (EventQuery): %7.1f μs avg\n", avg_spin);
    printf("\n  Spin-poll is typically 2-10x lower latency than blocking sync.\n");
    printf("  Trade-off: CPU burns cycles spinning instead of sleeping.\n\n");
}

// --------------------------------------------------------------------------- //
// Part B: Task Queue with Mapped Memory
//
// Architecture:
//   - Mapped pinned memory shared between CPU and GPU
//   - Task struct: { input data, output data, ready flag, done flag }
//   - CPU writes input + sets ready=1
//   - GPU sees ready=1, processes, writes output, sets done=1
//   - CPU sees done=1, reads output
//
// This is NOT a persistent kernel (that's Part C). This uses normal
// kernel launches but demonstrates the mapped memory communication pattern.
// --------------------------------------------------------------------------- //

struct alignas(64) Task {
    float input[TASK_SIZE];
    float output[TASK_SIZE];
    volatile int ready;    // CPU sets to 1, GPU reads
    volatile int done;     // GPU sets to 1, CPU reads
};

__global__ void process_task_kernel(Task* task) {
    int idx = threadIdx.x;
    if (idx < TASK_SIZE) {
        // Simple processing: square root + scale (simulates pricing calc)
        float val = task->input[idx];
        task->output[idx] = sqrtf(fabsf(val)) * 2.0f + 1.0f;
    }
}

void benchmark_task_queue() {
    printf("--- Part B: Mapped Memory Task Queue ---\n\n");

    // Allocate mapped task (visible to both CPU and GPU)
    Task* h_task = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_task, sizeof(Task), cudaHostAllocMapped));

    Task* d_task = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_task, h_task, 0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float total_us = 0;

    for (int t = 0; t < NUM_TASKS; ++t) {
        // CPU: prepare input
        for (int i = 0; i < TASK_SIZE; ++i) {
            h_task->input[i] = static_cast<float>(t * TASK_SIZE + i + 1);
        }
        h_task->ready = 1;
        h_task->done = 0;

        auto start = std::chrono::high_resolution_clock::now();

        // Launch kernel (still has launch overhead)
        process_task_kernel<<<1, TASK_SIZE, 0, stream>>>(d_task);
        CUDA_CHECK_LAST();

        // Spin-wait for completion
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreate(&ev));
        CUDA_CHECK(cudaEventRecord(ev, stream));
        while (cudaEventQuery(ev) != cudaSuccess) { /* spin */ }
        CUDA_CHECK(cudaEventDestroy(ev));

        auto end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration<float, std::micro>(end - start).count();
    }

    float avg_us = total_us / NUM_TASKS;
    printf("  Tasks processed: %d\n", NUM_TASKS);
    printf("  Avg latency per task: %.1f μs (includes launch overhead)\n", avg_us);
    printf("  This is the baseline — Part C eliminates the launch overhead.\n\n");

    // Verify last task
    bool ok = true;
    for (int i = 0; i < TASK_SIZE; ++i) {
        float expected = sqrtf(fabsf(h_task->input[i])) * 2.0f + 1.0f;
        if (fabsf(h_task->output[i] - expected) > 1e-3f) { ok = false; break; }
    }
    printf("  Correctness: %s\n\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_task));
}

// --------------------------------------------------------------------------- //
// Part C: Persistent Kernel with Spin-Poll
//
// The persistent kernel runs continuously. It polls a "task_ready" flag
// in mapped memory. When the CPU sets task_ready=1, the kernel processes
// the task and sets task_done=1. The kernel exits when it sees
// shutdown=1.
//
// TODO(human): Implement the persistent kernel's work loop.
//
// Pseudocode:
//   while (true) {
//       // Spin-wait for work or shutdown
//       while (control->task_ready == 0 && control->shutdown == 0) {
//           // spin (only thread 0 checks, then broadcasts)
//       }
//       if (control->shutdown) break;
//
//       // Process the task (all threads participate)
//       int idx = threadIdx.x;
//       if (idx < TASK_SIZE) {
//           control->output[idx] = sqrtf(fabsf(control->input[idx])) * 2.0f + 1.0f;
//       }
//       __syncthreads();
//
//       // Signal completion (thread 0 only)
//       if (threadIdx.x == 0) {
//           control->task_done = 1;
//           control->task_ready = 0;  // reset for next task
//           __threadfence_system();   // ensure CPU sees the writes
//       }
//       __syncthreads();  // all threads wait before polling again
//   }
//
// IMPORTANT:
//   - Use volatile reads for flags (compiler must not cache them in registers)
//   - Use __threadfence_system() after writing flags (ensures visibility to CPU)
//   - Only thread 0 should check/modify control flags (avoid races)
//   - Include the shutdown mechanism — otherwise TDR will kill the kernel!
// --------------------------------------------------------------------------- //

struct alignas(64) PersistentControl {
    float input[TASK_SIZE];
    float output[TASK_SIZE];
    volatile int task_ready;    // CPU -> GPU: "new task available"
    volatile int task_done;     // GPU -> CPU: "task complete"
    volatile int shutdown;      // CPU -> GPU: "exit the kernel"
    volatile int tasks_completed;  // Counter for debugging
};

__global__ void persistent_worker(PersistentControl* ctrl) {
    // TODO(human): Implement the persistent kernel work loop
    //
    // Hints:
    //   - Use a while(true) loop
    //   - Thread 0 polls ctrl->task_ready and ctrl->shutdown
    //   - Use __threadfence_system() after writing to ctrl (makes writes
    //     visible to the CPU through the PCIe bus)
    //   - __syncthreads() between poll phases so all threads stay in sync
    //
    // Skeleton:
    //
    // while (true) {
    //     // Only thread 0 polls (avoid thundering herd on PCIe)
    //     if (threadIdx.x == 0) {
    //         while (ctrl->task_ready == 0 && ctrl->shutdown == 0) {
    //             // spin — GPU thread burns cycles waiting
    //         }
    //     }
    //     __syncthreads();
    //
    //     // Check shutdown
    //     if (ctrl->shutdown) return;
    //
    //     // Process
    //     int idx = threadIdx.x;
    //     if (idx < TASK_SIZE) {
    //         ctrl->output[idx] = sqrtf(fabsf(ctrl->input[idx])) * 2.0f + 1.0f;
    //     }
    //     __syncthreads();
    //
    //     // Signal done
    //     if (threadIdx.x == 0) {
    //         ctrl->tasks_completed++;
    //         ctrl->task_done = 1;
    //         ctrl->task_ready = 0;
    //         __threadfence_system();
    //     }
    //     __syncthreads();
    // }

    // Placeholder: process one task and exit (so the kernel doesn't hang)
    int idx = threadIdx.x;
    if (ctrl->task_ready && idx < TASK_SIZE) {
        ctrl->output[idx] = sqrtf(fabsf(ctrl->input[idx])) * 2.0f + 1.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ctrl->task_done = 1;
        ctrl->task_ready = 0;
        ctrl->tasks_completed = 1;
        __threadfence_system();
    }
}

void benchmark_persistent_kernel() {
    printf("--- Part C: Persistent Kernel ---\n\n");

    // *** SAFETY: Persistent kernels can hang the GPU! ***
    // The placeholder implementation processes one task and exits.
    // Once you implement TODO(human), the kernel will loop until shutdown.
    // Always set ctrl->shutdown = 1 before the kernel times out (~2s on Windows TDR).
    printf("  *** WARNING: Persistent kernels risk GPU hangs (TDR ~2s on Windows) ***\n");
    printf("  *** Start with the placeholder, then implement TODO(human) carefully ***\n\n");

    PersistentControl* h_ctrl = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_ctrl, sizeof(PersistentControl),
                             cudaHostAllocMapped));
    memset((void*)h_ctrl, 0, sizeof(PersistentControl));

    PersistentControl* d_ctrl = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_ctrl, h_ctrl, 0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Prepare first task
    for (int i = 0; i < TASK_SIZE; ++i) {
        h_ctrl->input[i] = static_cast<float>(i + 1);
    }
    h_ctrl->task_ready = 1;
    h_ctrl->task_done = 0;
    h_ctrl->shutdown = 0;
    h_ctrl->tasks_completed = 0;

    // Launch persistent kernel (non-blocking)
    persistent_worker<<<1, TASK_SIZE, 0, stream>>>(d_ctrl);
    CUDA_CHECK_LAST();

    // TODO(human): Once the persistent loop is implemented, process multiple tasks:
    //
    // float total_us = 0;
    // for (int t = 0; t < NUM_TASKS; ++t) {
    //     // Prepare input
    //     for (int i = 0; i < TASK_SIZE; ++i) {
    //         h_ctrl->input[i] = static_cast<float>(t * TASK_SIZE + i + 1);
    //     }
    //     h_ctrl->task_done = 0;
    //
    //     auto start = std::chrono::high_resolution_clock::now();
    //
    //     // Signal new task
    //     h_ctrl->task_ready = 1;
    //
    //     // Spin-wait for completion (CPU side)
    //     while (h_ctrl->task_done == 0) { /* spin */ }
    //
    //     auto end = std::chrono::high_resolution_clock::now();
    //     total_us += std::chrono::duration<float, std::micro>(end - start).count();
    // }
    //
    // // Shutdown the persistent kernel
    // h_ctrl->shutdown = 1;
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    //
    // float avg_us = total_us / NUM_TASKS;
    // printf("  Persistent kernel avg latency: %.1f μs per task\n", avg_us);
    // printf("  Tasks completed: %d\n", h_ctrl->tasks_completed);

    // Placeholder: wait for single-task completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  Placeholder: processed %d task(s)\n", h_ctrl->tasks_completed);

    // Verify
    bool ok = true;
    for (int i = 0; i < TASK_SIZE; ++i) {
        float expected = sqrtf(fabsf(h_ctrl->input[i])) * 2.0f + 1.0f;
        if (fabsf(h_ctrl->output[i] - expected) > 1e-3f) { ok = false; break; }
    }
    printf("  Correctness: %s\n\n", ok ? "PASSED" : "FAILED — implement TODO(human)!");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_ctrl));
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
int main() {
    printf("=== Phase 6: Low-Latency Patterns for HFT ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  canMapHostMemory: %s\n\n", prop.canMapHostMemory ? "YES" : "NO");

    if (!prop.canMapHostMemory) {
        printf("ERROR: Device does not support mapped host memory.\n");
        return 1;
    }

    // Part A: Sync latency comparison
    benchmark_sync_mechanisms();

    // Part B: Mapped memory task queue (with kernel launch overhead)
    benchmark_task_queue();

    // Part C: Persistent kernel (eliminates launch overhead)
    benchmark_persistent_kernel();

    // --- Summary ---
    printf("=== Summary ===\n");
    printf("Latency hierarchy (typical):\n");
    printf("  cudaDeviceSynchronize:     10-50 μs\n");
    printf("  cudaStreamSynchronize:      5-20 μs\n");
    printf("  cudaEventSynchronize:       3-10 μs\n");
    printf("  Spin-poll (EventQuery):     1-5 μs\n");
    printf("  Persistent kernel + poll:   0.5-2 μs (no launch overhead)\n");
    printf("\nHFT trade-offs:\n");
    printf("  - Persistent kernels waste GPU resources (spinning thread blocks)\n");
    printf("  - But they provide the absolute lowest CPU-GPU round-trip latency\n");
    printf("  - Use only for the hottest path (e.g., signal generation)\n");
    printf("  - Use streams/events for everything else\n");

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
