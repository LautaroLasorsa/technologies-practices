cmake_minimum_required(VERSION 3.18)
project(cuda_hpc_hft LANGUAGES CXX CUDA)

# --------------------------------------------------------------------------- #
# Standards & global settings
# --------------------------------------------------------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Phase 4 (cooperative groups) requires sm_70+.
# Use native detection on CMake 3.24+, otherwise target sm_70 and above.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
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
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )
endfunction()

# --------------------------------------------------------------------------- #
# Helper: add a CUDA phase that requires cooperative groups (grid launch)
# --------------------------------------------------------------------------- #
function(add_phase_cooperative TARGET SOURCE)
    add_executable(${TARGET} src/${SOURCE})
    target_include_directories(${TARGET} PRIVATE ${INCLUDE_DIR})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )
endfunction()

# --------------------------------------------------------------------------- #
# Phase executables
# --------------------------------------------------------------------------- #

# Phase 1: CUDA Streams & Async Operations
add_phase(phase1_streams   phase1_streams.cu)

# Phase 2: Pinned (Page-Locked) Memory
add_phase(phase2_pinned    phase2_pinned.cu)

# Phase 3: Multi-Stream Pipeline
add_phase(phase3_pipeline  phase3_pipeline.cu)

# Phase 4: Cooperative Groups & Grid Sync (needs separable compilation)
add_phase_cooperative(phase4_cooperative  phase4_cooperative.cu)

# Phase 5: Unified Memory Deep Dive
add_phase(phase5_unified   phase5_unified.cu)

# Phase 6: Low-Latency Patterns for HFT
add_phase(phase6_low_latency  phase6_low_latency.cu)
