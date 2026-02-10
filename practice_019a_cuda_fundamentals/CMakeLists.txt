cmake_minimum_required(VERSION 3.18)
project(cuda_fundamentals LANGUAGES CXX CUDA)

# --------------------------------------------------------------------------- #
# Standards & global settings
# --------------------------------------------------------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Broad GPU compatibility: sm_50 (Maxwell) as minimum, plus native detection.
# CMake 3.24+ supports "native" to auto-detect the installed GPU.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86)
endif()

# --------------------------------------------------------------------------- #
# Shared include directory (cuda_helpers.h)
# --------------------------------------------------------------------------- #
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --------------------------------------------------------------------------- #
# Helper: add a CUDA phase executable
# --------------------------------------------------------------------------- #
function(add_phase TARGET SOURCE)
    add_executable(${TARGET} src/${SOURCE})
    target_include_directories(${TARGET} PRIVATE ${INCLUDE_DIR})
    # Enable extended lambdas and relaxed constexpr in device code
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )
endfunction()

# --------------------------------------------------------------------------- #
# Phase executables
# --------------------------------------------------------------------------- #

# Phase 1: Device query + first kernel
add_phase(phase1_hello   phase1_hello.cu)

# Phase 2: Vector addition — manual and managed memory
add_phase(phase2_memory  phase2_memory.cu)

# Phase 3: Naive matrix multiplication
add_phase(phase3_matmul  phase3_matmul.cu)

# Phase 4: Tiled matrix multiply with shared memory
add_phase(phase4_shared  phase4_shared.cu)

# Phase 5: Parallel reduction + warp shuffles
add_phase(phase5_reduction phase5_reduction.cu)

# Phase 6: Error handling + CUDA events profiling
add_phase(phase6_profiling phase6_profiling.cu)
