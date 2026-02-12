"""Phase 2: Manual Schedule Optimization.

TVM's schedule primitives transform the loop nest generated from a tensor
expression. The computation (math) doesn't change -- only the execution
order and parallelism strategy change.

Schedule primitives and what they do to the loop structure:

  split(axis, factor)
  ─────────────────────────────────────────────────────────────────────────
  Splits a loop into an outer and inner loop.
  Before: for i in range(N)
  After:  for i_outer in range(N // factor):
              for i_inner in range(factor):
                  i = i_outer * factor + i_inner

  WHY: Tiling. By splitting both i and j, then reordering, you process
  the matrix in small tiles that fit in L1/L2 cache. Without tiling,
  the inner loop strides through memory and thrashes the cache.

  reorder(*axes)
  ─────────────────────────────────────────────────────────────────────────
  Changes the nesting order of loops.
  Before: for i: for j: for k: ...
  After:  for j: for k: for i: ...  (if you reorder(j, k, i))

  WHY: Loop order determines memory access pattern. Row-major arrays
  want the innermost loop to vary the last index (stride-1 access).
  Reordering after split puts the tile loops outside and the inner
  loops (which touch consecutive memory) inside.

  vectorize(axis)
  ─────────────────────────────────────────────────────────────────────────
  Replaces a loop with SIMD vector instructions.
  Before: for i_inner in range(8): C[i_inner] = A[i_inner] + B[i_inner]
  After:  C[0:8] = A[0:8] + B[0:8]  (single SIMD instruction)

  WHY: Modern CPUs have 128/256/512-bit SIMD units (SSE, AVX, AVX-512).
  Processing 4-16 float32 elements per instruction instead of one gives
  4-16x throughput on element-wise operations.

  parallel(axis)
  ─────────────────────────────────────────────────────────────────────────
  Parallelizes a loop across CPU cores using OpenMP.
  Before: for i_outer in range(N // tile): ... (sequential)
  After:  #pragma omp parallel for
          for i_outer in range(N // tile): ... (multi-threaded)

  WHY: Independent outer tiles can run on different cores simultaneously.
  The outer loop of a tiled matmul is embarrassingly parallel because
  each tile writes to a different region of C.

  unroll(axis)
  ─────────────────────────────────────────────────────────────────────────
  Unrolls a loop -- duplicates the body multiple times.
  Before: for k in range(4): acc += A[k] * B[k]
  After:  acc += A[0]*B[0]; acc += A[1]*B[1]; acc += A[2]*B[2]; acc += A[3]*B[3]

  WHY: Eliminates loop overhead (branch prediction, counter increment).
  For small, known-size inner loops, unrolling can significantly improve
  throughput by keeping the CPU pipeline full.

Typical optimization recipe for matmul:
  1. split i and j by tile_size (e.g., 32)  → creates i_outer, i_inner, j_outer, j_inner
  2. reorder(i_outer, j_outer, k, i_inner, j_inner)  → process in tiles
  3. vectorize(j_inner)  → SIMD on the innermost dimension
  4. parallel(i_outer)   → multi-thread across tile rows
"""

from __future__ import annotations

import numpy as np

try:
    import tvm
    from tvm import te

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    print(
        "WARNING: TVM not found. This practice requires Docker.\n"
        "Run: docker compose run --rm tvm python -m tvm_practice.manual_schedule\n"
    )


def declare_matmul(M: int, K: int, N: int) -> tuple:
    """Declare a matmul computation (reusable across schedule variants).

    Returns (A, B, C) tensor expressions.
    """
    A = te.placeholder((M, K), dtype="float32", name="A")
    B = te.placeholder((K, N), dtype="float32", name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    return A, B, C


def show_tir(sch: te.Schedule, tensors: list, title: str) -> None:
    """Lower a schedule to TIR and print it with a title."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    lowered = tvm.lower(sch, tensors, simple_mode=True)
    print(lowered)


def build_and_verify(
    sch: te.Schedule,
    tensors: list,
    M: int,
    K: int,
    N: int,
    label: str,
) -> None:
    """Build a schedule, run it, and verify against NumPy."""
    A_te, B_te, C_te = tensors
    target = tvm.target.Target("llvm")
    func = tvm.build(sch, [A_te, B_te, C_te], target=target, name=label)
    dev = tvm.cpu(0)

    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    func(a_tvm, b_tvm, c_tvm)

    c_expected = a_np @ b_np
    np.testing.assert_allclose(c_tvm.numpy(), c_expected, rtol=1e-3)
    print(f"\n  [{label}] Verification: PASSED")


# ---------------------------------------------------------------------------
# Naive schedule (reference -- fully implemented)
# ---------------------------------------------------------------------------

def demo_naive_schedule(M: int, K: int, N: int) -> None:
    """Show the default naive schedule: three nested loops."""
    if not TVM_AVAILABLE:
        return

    print("\n" + "=" * 60)
    print("NAIVE SCHEDULE (default -- three nested loops)")
    print("=" * 60)

    A, B, C = declare_matmul(M, K, N)
    sch = te.create_schedule(C.op)

    show_tir(sch, [A, B, C], "Naive matmul loop nest")
    build_and_verify(sch, [A, B, C], M, K, N, "naive")


# ---------------------------------------------------------------------------
# Tiled schedule -- TODO(human)
# ---------------------------------------------------------------------------

def demo_tiled_schedule(M: int, K: int, N: int) -> None:
    """Apply split + reorder to tile the matmul.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches loop tiling via TVM schedule primitives. Tiling (split + reorder)
    # is the #1 optimization for cache locality in matmul. Understanding how split/reorder
    # transform loop nests is essential for manual scheduling and reading TIR.

    TODO(human): Implement the tiled schedule using split and reorder.

    The goal is to transform the naive loop nest:
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C[i,j] += A[i,k] * B[k,j]

    Into a tiled loop nest:
        for i_outer in range(M // tile):
            for j_outer in range(N // tile):
                for k in range(K):
                    for i_inner in range(tile):
                        for j_inner in range(tile):
                            C[i_outer*tile+i_inner, j_outer*tile+j_inner] +=
                                A[...] * B[...]

    Why tiling matters:
        In the naive version, the inner loop over k accesses B[k,j] where k changes
        each iteration. Since B is stored row-major, B[0,j], B[1,j], B[2,j], ...
        are NOT consecutive in memory -- they're N elements apart. This thrashes
        the CPU cache.

        With tiling, we process a small TILE_SIZE x TILE_SIZE block at a time.
        If the tile fits in L1 cache (~32KB), we reuse every loaded cache line
        TILE_SIZE times instead of once. This is the single most important
        optimization for matmul.

    Steps:
        1. Get the loop axes from the schedule:
               i, j, k = sch[C].op.axis[0], sch[C].op.axis[1], sch[C].op.reduce_axis[0]
           - axis[0] is the 'i' loop (rows of C)
           - axis[1] is the 'j' loop (columns of C)
           - reduce_axis[0] is the 'k' loop (reduction dimension)

        2. Split i by TILE_SIZE:
               i_outer, i_inner = sch[C].split(i, factor=TILE_SIZE)
           - This turns `for i in range(M)` into:
             `for i_outer in range(M // TILE): for i_inner in range(TILE):`

        3. Split j by TILE_SIZE:
               j_outer, j_inner = sch[C].split(j, factor=TILE_SIZE)

        4. Reorder to put outer loops outside, inner loops inside:
               sch[C].reorder(i_outer, j_outer, k, i_inner, j_inner)
           - The reorder is crucial: it makes the outer loops iterate over tiles,
             and the inner loops iterate within a tile.
           - k is between outer and inner: for each tile, we do the full reduction.
           - Without reorder, the split alone doesn't help -- you'd still access
             memory in the same bad pattern.

    Args:
        M, K, N: matrix dimensions (should be divisible by TILE_SIZE for simplicity)
    """
    if not TVM_AVAILABLE:
        return

    TILE_SIZE = 32

    print("\n" + "=" * 60)
    print(f"TILED SCHEDULE (tile_size={TILE_SIZE})")
    print("=" * 60)

    A, B, C = declare_matmul(M, K, N)
    sch = te.create_schedule(C.op)

    # TODO(human): implement the 4 steps above
    # 1. Get axes: i, j from sch[C].op.axis and k from sch[C].op.reduce_axis
    # 2. Split i and j by TILE_SIZE
    # 3. Reorder to: i_outer, j_outer, k, i_inner, j_inner

    show_tir(sch, [A, B, C], f"Tiled matmul (tile={TILE_SIZE})")
    build_and_verify(sch, [A, B, C], M, K, N, "tiled")


# ---------------------------------------------------------------------------
# Tiled + vectorized + parallel -- TODO(human)
# ---------------------------------------------------------------------------

def demo_optimized_schedule(M: int, K: int, N: int) -> None:
    """Apply split + reorder + vectorize + parallel to the matmul.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches combining schedule primitives (tiling + vectorization + parallelization).
    # Understanding how primitives compose to exploit cache, SIMD, and multi-core parallelism
    # is key to writing high-performance manual schedules in TVM.

    TODO(human): Implement the fully optimized schedule.

    This combines all four schedule primitives for maximum performance:
        - split: create tiles for cache reuse
        - reorder: ensure good memory access pattern
        - vectorize: use SIMD on the innermost loop
        - parallel: multi-thread across outer tiles

    Steps (building on the tiled schedule):
        1. Get axes and split i, j by TILE_SIZE (same as demo_tiled_schedule)
               i, j, k = sch[C].op.axis[0], sch[C].op.axis[1], sch[C].op.reduce_axis[0]
               i_outer, i_inner = sch[C].split(i, factor=TILE_SIZE)
               j_outer, j_inner = sch[C].split(j, factor=TILE_SIZE)

        2. Reorder to put outer loops first:
               sch[C].reorder(i_outer, j_outer, k, i_inner, j_inner)

        3. Vectorize the innermost loop (j_inner):
               sch[C].vectorize(j_inner)
           - j_inner iterates over consecutive columns in C and B.
           - Since arrays are row-major, consecutive j values are consecutive
             in memory -- perfect for SIMD.
           - This replaces the scalar inner loop with vector instructions:
             instead of C[i,j] += A[i,k]*B[k,j] one element at a time,
             it does C[i,j:j+SIMD_WIDTH] += A[i,k]*B[k,j:j+SIMD_WIDTH].

        4. Parallelize the outermost loop (i_outer):
               sch[C].parallel(i_outer)
           - Each i_outer iteration processes a different TILE_SIZE-row
             strip of C. These strips are independent (no data dependencies
             between different rows of C), so they can run on separate cores.
           - TVM generates OpenMP pragmas for this.

    Expected TIR structure:
        parallel for i_outer in range(M // TILE):           ← parallel
            for j_outer in range(N // TILE):
                for k in range(K):
                    for i_inner in range(TILE):
                        vectorized for j_inner in range(TILE):  ← vectorized
                            C[...] += A[...] * B[...]

    Performance impact:
        - Tiling: ~3-5x over naive (cache reuse)
        - Vectorize: ~2-4x additional (SIMD)
        - Parallel: ~Nx additional (N = number of cores)
        - Combined: often 10-50x over naive for large matrices

    Args:
        M, K, N: matrix dimensions (should be divisible by TILE_SIZE)
    """
    if not TVM_AVAILABLE:
        return

    TILE_SIZE = 32

    print("\n" + "=" * 60)
    print(f"OPTIMIZED SCHEDULE (tile={TILE_SIZE} + vectorize + parallel)")
    print("=" * 60)

    A, B, C = declare_matmul(M, K, N)
    sch = te.create_schedule(C.op)

    # TODO(human): implement the 4 steps above
    # 1. Get axes, split i and j
    # 2. Reorder
    # 3. Vectorize j_inner
    # 4. Parallelize i_outer

    show_tir(sch, [A, B, C], f"Optimized matmul (tile={TILE_SIZE}, vectorize, parallel)")
    build_and_verify(sch, [A, B, C], M, K, N, "optimized")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run Phase 2: Manual Schedule Optimization."""
    print("\n" + "#" * 60)
    print("  PHASE 2: Manual Schedule Optimization")
    print("#" * 60)

    M, K, N = 512, 512, 512

    demo_naive_schedule(M, K, N)
    demo_tiled_schedule(M, K, N)
    demo_optimized_schedule(M, K, N)

    print("\n" + "-" * 60)
    print("KEY TAKEAWAYS:")
    print("-" * 60)
    print("""
  1. split creates tiles -- small blocks that fit in cache.
  2. reorder changes loop nesting -- puts tile loops outside, work loops inside.
  3. vectorize uses SIMD -- processes multiple elements per instruction.
  4. parallel distributes across CPU cores -- outer tiles are independent.
  5. The SAME computation (matmul) runs 10-50x faster with good scheduling.
  6. Compare the TIR output of each schedule to see exactly what changed.
""")


if __name__ == "__main__":
    run_phase()
