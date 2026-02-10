cmake_minimum_required(VERSION 3.16)
project(practice_022_concurrent_queues LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── MSVC optimizations ───────────────────────────────────────────────────────
if(MSVC)
    # /O2: maximize speed (critical for accurate benchmarks)
    # /EHsc: standard C++ exception handling
    add_compile_options(/O2 /EHsc /W4)
else()
    add_compile_options(-O2 -Wall -Wextra -pthread)
    add_link_options(-pthread)
endif()

# ── Fetch moodycamel::ConcurrentQueue ────────────────────────────────────────
#
# Header-only library. The repo's CMakeLists.txt defines an INTERFACE target
# named "concurrentqueue". After FetchContent, link with:
#   target_link_libraries(my_target PRIVATE concurrentqueue)
#
# Headers are at the repo root: concurrentqueue.h, blockingconcurrentqueue.h
# Include as: #include "concurrentqueue.h" / #include "blockingconcurrentqueue.h"
# ─────────────────────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG        v1.0.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(concurrentqueue)

# ── Shared include directory ─────────────────────────────────────────────────
# bench_utils.h lives in include/
set(SHARED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Helper: add a phase executable with common setup ─────────────────────────
function(add_phase TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE concurrentqueue)
    target_include_directories(${TARGET_NAME} PRIVATE ${SHARED_INCLUDE_DIR})
endfunction()

# ── Executables (one per phase) ──────────────────────────────────────────────

# Phase 1: ConcurrentQueue basics -- enqueue, dequeue, pre-sizing
add_phase(phase1_basics     src/phase1_basics.cpp)

# Phase 2: Multi-threaded producer/consumer -- throughput, mutex comparison
add_phase(phase2_multithreaded src/phase2_multithreaded.cpp)

# Phase 3: Producer/consumer tokens -- thread-local fast paths
add_phase(phase3_tokens     src/phase3_tokens.cpp)

# Phase 4: Bulk operations -- enqueue_bulk, try_dequeue_bulk
add_phase(phase4_bulk       src/phase4_bulk.cpp)

# Phase 5: BlockingConcurrentQueue -- wait_dequeue, timed wait, shutdown
add_phase(phase5_blocking   src/phase5_blocking.cpp)

# Phase 6: Real-world patterns -- fan-out, fan-in, pipeline, comparison matrix
add_phase(phase6_patterns   src/phase6_patterns.cpp)

# ── Convenience: build all phases at once ────────────────────────────────────
add_custom_target(all_phases DEPENDS
    phase1_basics
    phase2_multithreaded
    phase3_tokens
    phase4_bulk
    phase5_blocking
    phase6_patterns
)
