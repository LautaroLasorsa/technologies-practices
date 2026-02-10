cmake_minimum_required(VERSION 3.16)
project(practice_020a_hft_lowlatency_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- MSVC-specific optimizations for Release builds ---
if(MSVC)
    # /O2  = maximize speed
    # /Ob3 = aggressive inlining (MSVC 19.27+, VS 2019 16.7+)
    # /GL  = whole-program optimization
    # /arch:AVX2 = enable AVX2 intrinsics (includes SSE4.2, BMI2)
    # /fp:fast = fast floating point (relaxed precision, better perf)
    # /GS- = disable buffer security checks on hot path
    # /Oi  = enable intrinsic functions (__rdtsc, _mm_prefetch, etc.)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /GL /arch:AVX2 /fp:fast /GS- /Oi /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")

    # Debug still gets AVX2 and intrinsics for consistency
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /arch:AVX2 /Oi")
endif()

# --- Include directory ---
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Phase 1: Cache-Friendly Data Structures ---
add_executable(phase1_cache src/phase1_cache.cpp)

# --- Phase 2: Lock-Free SPSC Ring Buffer ---
add_executable(phase2_spsc src/phase2_spsc.cpp)

# --- Phase 3: Memory Pools & Arena Allocator ---
add_executable(phase3_memory src/phase3_memory.cpp)

# --- Phase 4: Compile-Time Dispatch ---
add_executable(phase4_dispatch src/phase4_dispatch.cpp)

# --- Phase 5: Timestamps & Low-Latency Timing ---
add_executable(phase5_timing src/phase5_timing.cpp)

# --- Phase 6: Integrated Hot-Path Pipeline ---
add_executable(phase6_pipeline src/phase6_pipeline.cpp)

# --- Convenience: build all phases at once ---
add_custom_target(all_phases DEPENDS
    phase1_cache
    phase2_spsc
    phase3_memory
    phase4_dispatch
    phase5_timing
    phase6_pipeline
)
