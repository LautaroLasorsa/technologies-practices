"""Phase 1: TVM Tensor Expressions (TE) -- declaring computations.

TVM's tensor expression (TE) API lets you declare *what* to compute without
specifying *how* to compute it. The "what" is the mathematical formula; the
"how" is the schedule (Phase 2).

Key concepts:
  - te.placeholder(shape, dtype, name): declare an input tensor (like a function parameter)
  - te.compute(shape, fcompute, name): declare a new tensor by a lambda over indices
  - te.reduce_axis(dom, name): declare a reduction dimension (for sum, max, etc.)
  - te.create_schedule(ops): create a default schedule (naive loop nest)
  - tvm.lower(sch, tensors, simple_mode=True): lower to TIR (loop IR) for inspection

The separation of computation from schedule is TVM's core insight:
  - PyTorch: you write the computation AND the execution order in one place (eager)
  - TVM: you declare the math (TE), then separately optimize the execution (schedule)
  - This separation lets you try hundreds of different schedules for the same math

Analogy to competitive programming:
  Think of TE as writing the recurrence relation for a DP problem:
    dp[i][j] = min(dp[i-1][k] + cost[k][j]) for k in range(n)
  The recurrence says WHAT to compute. Whether you iterate row-by-row, column-by-column,
  or use divide-and-conquer is the SCHEDULE. Same answer, vastly different performance.
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
        "Run: docker compose run --rm tvm python -m tvm_practice.te_basics\n"
    )


# ---------------------------------------------------------------------------
# Part 1: Vector Add (fully implemented reference)
# ---------------------------------------------------------------------------

def demo_vector_add() -> None:
    """Demonstrate TVM tensor expressions with a simple vector add.

    This is the "hello world" of TVM: C[i] = A[i] + B[i].

    The pipeline is:
        1. Declare inputs (te.placeholder)
        2. Declare computation (te.compute)
        3. Create default schedule (te.create_schedule)
        4. Lower to TIR (tvm.lower) -- inspect the loop structure
        5. Build (tvm.build) -- compile to executable code
        6. Run -- execute with real data and verify
    """
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return

    print("=" * 60)
    print("PART 1: Vector Add (fully implemented reference)")
    print("=" * 60)

    n = 1024

    # Step 1: Declare input tensors
    # te.placeholder creates a symbolic tensor -- no data yet, just shape and dtype.
    # Think of it as declaring a function parameter: "this function takes an array A
    # of shape (n,) with float32 elements."
    A = te.placeholder((n,), dtype="float32", name="A")
    B = te.placeholder((n,), dtype="float32", name="B")

    # Step 2: Declare the computation
    # te.compute says: "create a tensor C of shape (n,), where element C[i] is
    # computed by the lambda." The lambda receives the index tuple (i,).
    # This is DECLARATIVE -- TVM records the formula, it doesn't execute it yet.
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")

    # Step 3: Create a default schedule
    # The default schedule is a naive loop nest: for i in range(n): C[i] = A[i] + B[i]
    # We'll optimize this in Phase 2.
    sch = te.create_schedule(C.op)

    # Step 4: Lower to TIR (Tensor IR)
    # tvm.lower translates the schedule into a loop-based IR (TIR).
    # simple_mode=True gives a cleaner, more readable output.
    print("\n--- Lowered TIR (default schedule) ---")
    lowered = tvm.lower(sch, [A, B, C], simple_mode=True)
    print(lowered)

    # Step 5: Build -- compile to executable machine code
    # "llvm" target means compile for CPU using LLVM backend.
    target = tvm.target.Target("llvm")
    func = tvm.build(sch, [A, B, C], target=target, name="vector_add")

    # Step 6: Run with real data
    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=n).astype("float32")
    b_np = np.random.uniform(size=n).astype("float32")
    c_np = np.zeros(n, dtype="float32")

    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    func(a_tvm, b_tvm, c_tvm)

    # Verify
    c_expected = a_np + b_np
    np.testing.assert_allclose(c_tvm.numpy(), c_expected, rtol=1e-5)
    print(f"\nVector add (n={n}): PASSED")
    print(f"  C[0:5] = {c_tvm.numpy()[:5]}")
    print(f"  Expected = {c_expected[:5]}")


# ---------------------------------------------------------------------------
# Part 2: Matrix Multiply -- TODO(human)
# ---------------------------------------------------------------------------

def declare_matmul(M: int, K: int, N: int) -> tuple:
    """Declare a matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j].

    TODO(human): Implement the matrix multiply tensor expression.

    This is the most fundamental computation in deep learning: every linear
    layer, attention head, and convolution (via im2col) is a matmul. Writing
    it as a tensor expression teaches you exactly what TVM optimizes.

    The math is simple: C[i,j] = sum over k of A[i,k] * B[k,j]
    But expressing this in TVM requires understanding reduce_axis:

    Steps:
        1. Declare input tensors A and B using te.placeholder:
               A = te.placeholder((M, K), dtype="float32", name="A")
               B = te.placeholder((K, N), dtype="float32", name="B")
           - A has shape (M, K): M rows, K columns
           - B has shape (K, N): K rows, N columns
           - The result C will have shape (M, N)

        2. Declare the reduction axis using te.reduce_axis:
               k = te.reduce_axis((0, K), name="k")
           - This tells TVM that 'k' is NOT a spatial dimension of the output.
             It's a dimension we SUM over (reduce).
           - In loop terms: for each (i,j), we iterate k from 0 to K and
             accumulate A[i,k] * B[k,j].
           - Analogy: in NumPy, np.dot(A, B) sums over the shared dimension K.
             te.reduce_axis is TVM's way of expressing that shared dimension.

        3. Declare the computation using te.compute:
               C = te.compute(
                   (M, N),
                   lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                   name="C",
               )
           - Shape (M, N): the output is M x N
           - Lambda receives spatial indices (i, j) -- one for each output dimension
           - te.sum(expr, axis=k) says: "sum this expression over the reduction axis k"
           - The full expression reads: C[i,j] = SUM_k(A[i,k] * B[k,j])

    Why te.sum and reduce_axis?
        TVM needs to distinguish between:
        - Spatial axes (i, j): these map to output dimensions, each computes independently
        - Reduction axes (k): these are summed/accumulated within each output element
        This distinction is critical for scheduling: spatial axes can be parallelized
        freely, but reduction axes need accumulation (think parallel reduction).

    Why this matters for scheduling (Phase 2):
        The default schedule for this matmul is:
            for i in range(M):
                for j in range(N):
                    C[i,j] = 0
                    for k in range(K):
                        C[i,j] += A[i,k] * B[k,j]
        This is correct but has terrible cache behavior: B[k,j] jumps through
        memory as k changes (column access in row-major layout). In Phase 2,
        you'll fix this with tiling.

    Args:
        M: number of rows in A / output C
        K: shared dimension (columns of A, rows of B)
        N: number of columns in B / output C

    Returns:
        Tuple of (A, B, C) tensor expressions.
        A: te.placeholder of shape (M, K)
        B: te.placeholder of shape (K, N)
        C: te.compute of shape (M, N)
    """
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return None, None, None

    # TODO(human): implement the three steps above
    # Hint: it's only 4 lines of code -- the concept is more important than the amount of code

    # Stub: return placeholders so the file parses (replace with real implementation)
    A = te.placeholder((M, K), dtype="float32", name="A")
    B = te.placeholder((K, N), dtype="float32", name="B")
    # Stub C: a zero matrix (incorrect -- replace with te.compute + te.reduce_axis)
    C = te.compute((M, N), lambda i, j: tvm.tir.const(0.0, "float32"), name="C")
    return A, B, C


def demo_matmul() -> None:
    """Demonstrate matrix multiply tensor expression.

    Declares the matmul, creates a default schedule, lowers to TIR,
    builds, and verifies against NumPy.
    """
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return

    print("\n" + "=" * 60)
    print("PART 2: Matrix Multiply (TODO -- implement declare_matmul)")
    print("=" * 60)

    M, K, N = 128, 256, 128
    A, B, C = declare_matmul(M, K, N)

    if A is None:
        return

    # Create default (naive) schedule and lower to TIR
    sch = te.create_schedule(C.op)
    print(f"\n--- Lowered TIR for matmul ({M}x{K} @ {K}x{N}) ---")
    print("--- Default (naive) schedule: three nested loops ---")
    lowered = tvm.lower(sch, [A, B, C], simple_mode=True)
    print(lowered)

    # Build and run
    target = tvm.target.Target("llvm")
    func = tvm.build(sch, [A, B, C], target=target, name="matmul_naive")
    dev = tvm.cpu(0)

    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    func(a_tvm, b_tvm, c_tvm)

    # Verify against NumPy
    c_expected = a_np @ b_np
    try:
        np.testing.assert_allclose(c_tvm.numpy(), c_expected, rtol=1e-3)
        print(f"\nMatrix multiply ({M}x{K} @ {K}x{N}): PASSED")
    except AssertionError:
        print(f"\nMatrix multiply ({M}x{K} @ {K}x{N}): FAILED (implement declare_matmul)")
        print(f"  Got C[0,0:5] = {c_tvm.numpy()[0, :5]}")
        print(f"  Expected     = {c_expected[0, :5]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run Phase 1: TVM Tensor Expressions."""
    print("\n" + "#" * 60)
    print("  PHASE 1: TVM Basics & Tensor Expressions")
    print("#" * 60)

    demo_vector_add()
    demo_matmul()

    print("\n" + "-" * 60)
    print("KEY TAKEAWAYS:")
    print("-" * 60)
    print("""
  1. te.placeholder declares inputs -- shape and dtype, no data yet.
  2. te.compute declares the output -- a lambda over indices.
  3. te.reduce_axis declares reduction dimensions (sum, max, etc.).
  4. te.create_schedule gives a naive loop nest -- the starting point.
  5. tvm.lower shows the loop IR (TIR) -- this is what you'll optimize in Phase 2.
  6. tvm.build compiles to machine code -- you can run it with real data.
""")


if __name__ == "__main__":
    run_phase()
