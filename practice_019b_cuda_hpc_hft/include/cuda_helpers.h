#pragma once
// cuda_helpers.h — Error checking, timing, bandwidth, and RAII utilities for CUDA.
// Extended from 019a with BandwidthMeter and PinnedBuffer for 019b.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// --------------------------------------------------------------------------- //
// CUDA_CHECK — wraps any cudaError_t call and aborts on failure.
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
// CudaTimer — measures GPU execution time using CUDA events.
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

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    // Record start on a specific stream (default = 0)
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    // Record stop on a specific stream (default = 0)
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
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

// --------------------------------------------------------------------------- //
// BandwidthMeter — computes and prints bandwidth given bytes and time.
//
// Usage:
//   BandwidthMeter bw;
//   bw.report("Pageable H2D", bytes_transferred, elapsed_ms);
//   bw.compare("Pinned vs Pageable", pinned_ms, pageable_ms);
// --------------------------------------------------------------------------- //
class BandwidthMeter {
public:
    // Print bandwidth in GB/s
    static void report(const char* label, size_t bytes, float ms) {
        double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        double sec = ms / 1000.0;
        double gbps = (sec > 0.0) ? gb / sec : 0.0;
        printf("  [%s] %.2f GB in %.3f ms = %.2f GB/s\n", label, gb, ms, gbps);
    }

    // Print speedup comparison
    static void compare(const char* label, float fast_ms, float slow_ms) {
        if (fast_ms > 0.0f && slow_ms > 0.0f) {
            printf("  [%s] Speedup: %.2fx\n", label, slow_ms / fast_ms);
        }
    }
};

// --------------------------------------------------------------------------- //
// PinnedBuffer<T> — RAII wrapper for pinned (page-locked) host memory.
//
// Pinned memory enables DMA transfers (no intermediate staging copy in the
// CUDA driver), which is required for cudaMemcpyAsync to actually be async.
//
// Usage:
//   PinnedBuffer<float> buf(1024);  // 1024 floats, pinned
//   buf[0] = 42.0f;
//   cudaMemcpyAsync(d_ptr, buf.get(), buf.size_bytes(), ...);
// --------------------------------------------------------------------------- //
template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }

    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    // Move-only
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};
