#pragma once
// cuda_helpers.h — Error checking macro and CUDA event timer utility.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// --------------------------------------------------------------------------- //
// CUDA_CHECK — wraps any cudaError_t call and aborts on failure.
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
// --------------------------------------------------------------------------- //
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n  Expression: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err_), #call);        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// --------------------------------------------------------------------------- //
// CUDA_CHECK_LAST — checks for errors from the most recent kernel launch.
// Kernel launches are asynchronous and don't return cudaError_t directly.
//
// Usage:
//   my_kernel<<<grid, block>>>(...);
//   CUDA_CHECK_LAST();
// --------------------------------------------------------------------------- //
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                        \
        cudaError_t err_ = cudaGetLastError();                                  \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error at %s:%d — %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err_));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// --------------------------------------------------------------------------- //
// CudaTimer — measures GPU kernel execution time using CUDA events.
//
// CUDA events are recorded on the GPU timeline, so they measure actual
// GPU execution time without being affected by CPU-side overhead.
//
// Usage:
//   CudaTimer timer;
//   timer.start();
//   my_kernel<<<grid, block>>>(...);
//   timer.stop();
//   printf("Kernel took %.3f ms\n", timer.elapsed_ms());
// --------------------------------------------------------------------------- //
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
